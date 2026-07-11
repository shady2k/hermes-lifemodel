"""``observability.sqlite`` — the disposable durable trace store + async writer.

The second law of the observability invariant (spec §4.2/§4.3): the being's
``State`` DB is precious and fail-*closed*; the durable trace is a **separate,
disposable** SQLite file that is fail-*open*. Losing it never changes the
being's behaviour — so its writes are asynchronous, non-blocking, and swallow
their own errors. A tick never waits on this I/O.

Shape (all stdlib — ``sqlite3``/``threading``/``queue``; the plugin runs in
Hermes' own interpreter, no third-party deps):

* **One writer thread + one bounded queue.** The SQLite connection lives ONLY
  in the writer thread (thread-affine, its own WAL). Callers only ever
  ``put_nowait`` a record; on a full queue the record is dropped and
  :attr:`TraceWriter.dropped_count` bumped — the tick is never blocked.
  ``agent.log``/``deque`` projections are the *caller's* job and happen only
  when the enqueue succeeds (durable-first, §4.2) — that lives in
  :class:`~lifemodel.log.SpanLogger`.
* **Singleton per db-path** with refcount (:func:`acquire_trace_writer` /
  :func:`release_trace_writer`): idempotent start, reconnect-safe (a fresh
  acquire after the last release spins up a brand-new thread + connection).
* **Retention** (:func:`prune_traces`) prunes by age / count / size, always a
  WHOLE trace at a time, and NEVER an in-flight/unresolved one (spec §4.3):
  a trace with a live state anchor, an unresolved correlation, or one still
  inside the post-resolve grace window is protected from the axe.

The store is self-contained: ``/lifemodel trace <id>`` (a later phase) answers
any "why" by reading only this file.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol, overload

from ..core.timeutil import from_iso, to_iso
from ..ports.clock import ClockPort

#: The trace DB's filename, a sibling of ``lifemodel.sqlite`` in the state dir.
_DB_FILENAME: Final = "observability.sqlite"

#: The trace schema version. Even a disposable store carries one (codex fix) so
#: a future shape change is detectable rather than silently misread.
SCHEMA_VERSION: Final = 1

#: Bounded queue depth: past this the writer is falling behind and new records
#: are dropped (counted) rather than blocking the tick.
_DEFAULT_MAX_QUEUE: Final = 10_000
#: Commit after this many applied records (batch for throughput).
_DEFAULT_BATCH: Final = 128
#: Idle wake so a partial batch still commits promptly for read-your-writes.
_IDLE_COMMIT_SECONDS: Final = 0.2
#: Run retention roughly every this-many commits (rare, from the writer thread).
_DEFAULT_PRUNE_EVERY_COMMITS: Final = 50

_module_logger = logging.getLogger("lifemodel.trace_store")


def observability_db_path(base_dir: Path) -> Path:
    """Return the trace DB path under *base_dir* (sibling of ``lifemodel.sqlite``)."""
    return base_dir / _DB_FILENAME


# --------------------------------------------------------------------------- #
# Process-local monotonic record id (spec §4.2: dedup key for the deque overlay)
# --------------------------------------------------------------------------- #

_record_id_lock = threading.Lock()
_record_id_counter = 0


def next_record_id() -> int:
    """Return the next monotonic, process-local ``record_id`` (spec §4.2).

    It is the ``trace_events`` primary key AND the dedup key the viewer uses to
    overlay the in-memory ring on the flushed rows without double-counting. A
    single process-wide counter (shared across writers) keeps ids unique even
    when several trace DBs are open.
    """
    global _record_id_counter
    with _record_id_lock:
        _record_id_counter += 1
        return _record_id_counter


def _ensure_record_id_floor(value: int) -> None:
    """Raise the counter floor to *value* — so ids stay monotonic over a DB that
    survived a restart (whose ``trace_events`` already holds ids up to *value*)."""
    global _record_id_counter
    with _record_id_lock:
        if _record_id_counter < value:
            _record_id_counter = value


# --------------------------------------------------------------------------- #
# Schema + connection
# --------------------------------------------------------------------------- #


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the trace schema (spec §4.3) if absent; idempotent."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    if conn.execute("SELECT version FROM schema_version").fetchone() is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_spans ("
        "  trace_id TEXT NOT NULL, span_id TEXT NOT NULL, parent_span_id TEXT,"
        "  component TEXT, tick INTEGER, started_at TEXT, ended_at TEXT,"
        "  status TEXT, attrs_json TEXT,"
        "  PRIMARY KEY (trace_id, span_id))"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_spans_trace ON trace_spans(trace_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_spans_tick ON trace_spans(tick)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_events ("
        "  record_id INTEGER PRIMARY KEY,"
        "  trace_id TEXT NOT NULL, span_id TEXT, tick INTEGER,"
        "  event TEXT NOT NULL, ts TEXT NOT NULL, fields_json TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_events_trace ON trace_events(trace_id)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_correlations ("
        "  correlation_id TEXT PRIMARY KEY, origin_trace_id TEXT NOT NULL,"
        "  origin_traceparent TEXT, kind TEXT,"
        "  created_at TEXT NOT NULL, resolved_at TEXT)"
    )
    conn.commit()


def connect(db_path: Path, *, create_parent: bool = True) -> sqlite3.Connection:
    """Open a trace-DB connection with WAL + incremental auto-vacuum + schema.

    Used by BOTH the writer thread and retention tests, so the on-disk shape is
    identical either way. ``auto_vacuum=INCREMENTAL`` is set before any table
    exists (a fresh file) so :func:`prune_traces` can actually shrink the file.
    """
    path = Path(db_path)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
    initialize_schema(conn)
    return conn


def _dumps(value: Mapping[str, Any] | None) -> str | None:
    """JSON-encode an attribute/field bag, degrading odd values rather than raising."""
    if value is None:
        return None
    try:
        return json.dumps(dict(value), ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_unserializable": True})


@overload
def _normalize_ts(value: str) -> str: ...
@overload
def _normalize_ts(value: None) -> None: ...
def _normalize_ts(value: str | None) -> str | None:
    """Canonicalize a trace timestamp at the write boundary (spec §4 codex #9).

    Every ``started_at``/``ended_at``/``ts``/``created_at``/``resolved_at`` is run
    through :func:`~lifemodel.core.timeutil.to_iso` at the ``submit_*`` (enqueue)
    boundary, so ``observability.sqlite`` only ever holds normalized, fixed-width
    ISO — which is what makes retention's lexical ``MIN(ts)`` / ``< cutoff`` a
    correct chronological compare.

    A tz-*aware* value is re-serialized to canonical UTC. A tz-naive or malformed
    value is NOT silently coerced to UTC on the write path (the reader's
    :func:`_parse_ts` stays defensive for *legacy* rows, but a NEW write never
    relies on that): because this store is disposable and fail-open (§4.2), such a
    value is kept raw and a WARNING is logged, never dropped and never guessed-UTC.
    """
    if value is None:
        return None
    try:
        return to_iso(from_iso(value))
    except ValueError:
        _module_logger.warning("trace_ts_not_normalized value=%r; storing raw", value)
        return value


# --------------------------------------------------------------------------- #
# Queue records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SpanWrite:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    component: str | None
    tick: int | None
    started_at: str | None
    ended_at: str | None
    status: str | None
    attrs_json: str | None


@dataclass(frozen=True)
class _EventWrite:
    record_id: int
    trace_id: str
    span_id: str | None
    tick: int | None
    event: str
    ts: str
    fields_json: str | None


@dataclass(frozen=True)
class _CorrelationWrite:
    correlation_id: str
    origin_trace_id: str
    origin_traceparent: str | None
    kind: str | None
    created_at: str
    resolved_at: str | None


@dataclass(frozen=True)
class _Flush:
    event: threading.Event


@dataclass(frozen=True)
class _Stop:
    pass


_QueueItem = _SpanWrite | _EventWrite | _CorrelationWrite | _Flush | _Stop


# --------------------------------------------------------------------------- #
# Retention (spec §4.3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetentionPolicy:
    """The three retention axes (any may be ``None`` to disable it) + grace.

    Defaults are deliberately conservative. Pruning always removes a WHOLE trace
    and never an in-flight/unresolved one (see :func:`prune_traces`).
    """

    max_age_days: int | None = 14
    max_traces: int | None = 5_000
    max_bytes: int | None = 256 * 1024 * 1024
    #: Days after a correlation resolves during which its origin trace is still
    #: protected — a cushion for late async writes (spec §4.3, NOT tied to
    #: ``max_age_days``).
    resolved_grace_days: int = 1


def _parse_ts(value: str) -> datetime | None:
    """Defensively parse a stored timestamp for retention (spec §4 codex #9).

    New writes are normalized at ingress (:func:`_normalize_ts`), so this is only
    ever tz-naive for a *legacy* row written before that; for those the reader
    stays tolerant (assume UTC) rather than dropping the row from the retention
    scan. The write path never relies on this coercion.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _trace_starts(conn: sqlite3.Connection) -> dict[str, datetime]:
    """Map each known ``trace_id`` to its earliest timestamp (span start or event)."""
    starts: dict[str, datetime] = {}
    rows = conn.execute(
        "SELECT trace_id, MIN(t) FROM ("
        "  SELECT trace_id, started_at AS t FROM trace_spans WHERE started_at IS NOT NULL"
        "  UNION ALL"
        "  SELECT trace_id, ts AS t FROM trace_events"
        ") GROUP BY trace_id"
    ).fetchall()
    for trace_id, earliest in rows:
        if earliest is None:
            continue
        parsed = _parse_ts(earliest)
        if parsed is not None:
            starts[trace_id] = parsed
    return starts


def _protected_trace_ids(
    conn: sqlite3.Connection, policy: RetentionPolicy, now: datetime, extra: set[str]
) -> set[str]:
    """Trace ids that must NEVER be pruned (spec §4.3, codex fix #1).

    Union of: (a) the caller-supplied live state anchors (*extra*), (b) every
    origin trace of an UNRESOLVED correlation, (c) origins resolved within the
    grace window.
    """
    protected = set(extra)
    for origin, resolved_at in conn.execute(
        "SELECT origin_trace_id, resolved_at FROM trace_correlations"
    ).fetchall():
        if resolved_at is None:
            protected.add(origin)  # (b) still in flight
            continue
        resolved = _parse_ts(resolved_at)  # (c) grace window
        if resolved is None:
            protected.add(origin)  # unparseable — protect conservatively
        elif resolved > now - timedelta(days=policy.resolved_grace_days):
            protected.add(origin)
    return protected


def _delete_trace(conn: sqlite3.Connection, trace_id: str) -> None:
    conn.execute("DELETE FROM trace_spans WHERE trace_id = ?", (trace_id,))
    conn.execute("DELETE FROM trace_events WHERE trace_id = ?", (trace_id,))
    conn.execute("DELETE FROM trace_correlations WHERE origin_trace_id = ?", (trace_id,))


def _db_size_bytes(conn: sqlite3.Connection) -> int:
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    return int(page_count) * int(page_size)


def _prune_by_size(
    conn: sqlite3.Connection, policy: RetentionPolicy, protected: set[str], now: datetime
) -> None:
    assert policy.max_bytes is not None
    if _db_size_bytes(conn) <= policy.max_bytes:
        return
    ordered = sorted(_trace_starts(conn).items(), key=lambda kv: kv[1])
    for trace_id, _ in ordered:
        if trace_id in protected:
            continue
        if _db_size_bytes(conn) <= policy.max_bytes:
            break
        _delete_trace(conn, trace_id)
        conn.commit()
        with contextlib.suppress(sqlite3.Error):
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()


def prune_traces(
    conn: sqlite3.Connection,
    *,
    policy: RetentionPolicy,
    protected_ids: set[str],
    now: datetime,
) -> set[str]:
    """Prune whole traces past the policy, never a protected one. Returns the ids
    deleted by the age/count axes (size runs afterward, oldest-first).

    A pure function over *conn* so it is unit-testable off the writer thread and
    reused by it. Fail-open is the caller's concern (the writer swallows).
    """
    starts = _trace_starts(conn)
    if not starts:
        return set()
    protected = _protected_trace_ids(conn, policy, now, protected_ids)
    ordered = [tid for tid, _ in sorted(starts.items(), key=lambda kv: kv[1])]  # oldest first
    marked: set[str] = set()

    if policy.max_age_days is not None:
        cutoff = now - timedelta(days=policy.max_age_days)
        for trace_id in ordered:
            if trace_id not in protected and starts[trace_id] < cutoff:
                marked.add(trace_id)

    if policy.max_traces is not None and len(starts) > policy.max_traces:
        excess = len(starts) - policy.max_traces
        prunable = [tid for tid in ordered if tid not in protected]
        marked.update(prunable[:excess])

    for trace_id in marked:
        _delete_trace(conn, trace_id)
    if marked:
        conn.commit()
        with contextlib.suppress(sqlite3.Error):
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()

    if policy.max_bytes is not None:
        _prune_by_size(conn, policy, protected, now)

    return marked


# --------------------------------------------------------------------------- #
# Async writer
# --------------------------------------------------------------------------- #


class TraceWriter:
    """One daemon thread draining a bounded queue into ``observability.sqlite``.

    Fail-open by construction (spec §4.2): submitting never blocks and never
    raises — a full queue drops the record and bumps :attr:`dropped_count`; a
    per-record SQL error is swallowed and counted in :attr:`write_errors`; the
    thread itself never dies on a bad record. :meth:`flush` drains + commits the
    queue for read-your-writes. Prefer :func:`acquire_trace_writer` /
    :func:`release_trace_writer` over constructing directly — they enforce the
    singleton-per-db-path lifecycle.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        batch_size: int = _DEFAULT_BATCH,
        retention: RetentionPolicy | None = None,
        protected_trace_ids: Callable[[], set[str]] | None = None,
        prune_every_commits: int = _DEFAULT_PRUNE_EVERY_COMMITS,
        clock: ClockPort | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=max_queue)
        self._batch_size = max(1, batch_size)
        self._retention = retention or RetentionPolicy()
        self._protected = protected_trace_ids
        self._prune_every_commits = max(1, prune_every_commits)
        # The ONE source of "now" for the writer thread's retention pass (spec §3.1):
        # the injected clock, never ``datetime.now``. The live being injects the SAME
        # ``SystemClock`` the tick/stores use; a bare test/CLI writer with no clock
        # simply skips the periodic prune (fail-open — the store is disposable).
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._started = False
        self._lifecycle_lock = threading.Lock()
        self._dropped = 0
        self._write_errors = 0
        self._commits_since_prune = 0

    # ---- lifecycle ------------------------------------------------------- #

    def start(self) -> None:
        """Spawn the writer thread; idempotent and double-start safe (spec §4.2)."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run, name="lifemodel-trace-writer", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Drain, commit and join the writer thread; idempotent."""
        with self._lifecycle_lock:
            thread = self._thread
            self._started = False
            self._thread = None
        if thread is None or not thread.is_alive():
            return
        with contextlib.suppress(queue.Full):
            self._queue.put(_Stop(), timeout=timeout)
        thread.join(timeout)

    @property
    def dropped_count(self) -> int:
        """Records dropped because the queue was full (spec §4.2 overload signal)."""
        return self._dropped

    @property
    def write_errors(self) -> int:
        """Records that raised on write and were swallowed (fail-open, §4.2)."""
        return self._write_errors

    # ---- submit (non-blocking, fail-open) -------------------------------- #

    def _submit(self, item: _QueueItem) -> bool:
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def submit_event(
        self,
        *,
        record_id: int,
        trace_id: str,
        span_id: str | None,
        tick: int | None,
        event: str,
        ts: str,
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        """Enqueue one ``trace_events`` row. Returns ``False`` (dropped) if full.

        ``ts`` is normalized to canonical ISO here at the enqueue boundary (§4
        codex #9), so the DB only ever holds normalized time.
        """
        return self._submit(
            _EventWrite(
                record_id, trace_id, span_id, tick, event, _normalize_ts(ts), _dumps(fields)
            )
        )

    def submit_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        component: str | None = None,
        tick: int | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Enqueue one ``trace_spans`` upsert. Returns ``False`` (dropped) if full.

        ``started_at``/``ended_at`` are normalized to canonical ISO here at the
        enqueue boundary (§4 codex #9), so the DB only ever holds normalized time.
        """
        return self._submit(
            _SpanWrite(
                trace_id,
                span_id,
                parent_span_id,
                component,
                tick,
                _normalize_ts(started_at),
                _normalize_ts(ended_at),
                status,
                _dumps(attrs),
            )
        )

    def submit_correlation(
        self,
        *,
        correlation_id: str,
        origin_trace_id: str,
        created_at: str,
        origin_traceparent: str | None = None,
        kind: str | None = None,
        resolved_at: str | None = None,
    ) -> bool:
        """Enqueue one ``trace_correlations`` upsert (index only — §4.4).

        ``created_at``/``resolved_at`` are normalized to canonical ISO here at the
        enqueue boundary (§4 codex #9), so the DB only ever holds normalized time.
        """
        return self._submit(
            _CorrelationWrite(
                correlation_id,
                origin_trace_id,
                origin_traceparent,
                kind,
                _normalize_ts(created_at),
                _normalize_ts(resolved_at),
            )
        )

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every queued record is committed (read-your-writes, §4.2).

        Returns ``True`` once the queue is drained + committed, ``False`` on
        timeout (or if the queue was too full to even enqueue the marker). A
        never-started writer flushes trivially to ``True``.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        marker = threading.Event()
        try:
            self._queue.put(_Flush(marker), timeout=timeout)
        except queue.Full:
            return False
        return marker.wait(timeout)

    # ---- writer thread --------------------------------------------------- #

    def _run(self) -> None:
        try:
            conn = connect(self._db_path)
            _ensure_record_id_floor(_max_event_record_id(conn))
        except sqlite3.Error:
            _module_logger.warning("trace_store_open_failed path=%s", self._db_path, exc_info=True)
            return
        try:
            self._loop(conn)
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.commit()
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    def _loop(self, conn: sqlite3.Connection) -> None:
        pending = 0
        while True:
            try:
                item = self._queue.get(timeout=_IDLE_COMMIT_SECONDS)
            except queue.Empty:
                if pending:
                    self._commit(conn)
                    pending = 0
                continue
            try:
                if isinstance(item, _Stop):
                    if pending:
                        self._commit(conn)
                    return
                if isinstance(item, _Flush):
                    if pending:
                        self._commit(conn)
                        pending = 0
                    item.event.set()
                    continue
                try:
                    self._apply(conn, item)
                    pending += 1
                except sqlite3.Error:
                    self._write_errors += 1
                if pending >= self._batch_size:
                    self._commit(conn)
                    pending = 0
            except Exception:  # never let the writer thread die (fail-open, §4.2)
                self._write_errors += 1
            finally:
                self._queue.task_done()

    def _commit(self, conn: sqlite3.Connection) -> None:
        try:
            conn.commit()
        except sqlite3.Error:
            self._write_errors += 1
            return
        self._commits_since_prune += 1
        if self._commits_since_prune >= self._prune_every_commits:
            self._commits_since_prune = 0
            self._maybe_prune(conn)

    def _maybe_prune(self, conn: sqlite3.Connection) -> None:
        if self._clock is None:
            # No injected clock (a bare test/CLI writer): retention needs a "now" and
            # must NOT self-source wall time (spec §3.1), so skip the periodic prune.
            # The disposable store simply grows until a clock-backed writer reopens it.
            return
        extra: set[str] = set()
        if self._protected is not None:
            with contextlib.suppress(Exception):
                extra = set(self._protected())
        try:
            prune_traces(conn, policy=self._retention, protected_ids=extra, now=self._clock.now())
        except sqlite3.Error:
            self._write_errors += 1

    @staticmethod
    def _apply(conn: sqlite3.Connection, item: _QueueItem) -> None:
        if isinstance(item, _EventWrite):
            conn.execute(
                "INSERT INTO trace_events "
                "(record_id, trace_id, span_id, tick, event, ts, fields_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.record_id,
                    item.trace_id,
                    item.span_id,
                    item.tick,
                    item.event,
                    item.ts,
                    item.fields_json,
                ),
            )
        elif isinstance(item, _SpanWrite):
            conn.execute(
                "INSERT INTO trace_spans "
                "(trace_id, span_id, parent_span_id, component, tick, started_at, "
                " ended_at, status, attrs_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(trace_id, span_id) DO UPDATE SET "
                "  parent_span_id=excluded.parent_span_id, component=excluded.component, "
                "  tick=excluded.tick, "
                "  started_at=COALESCE(trace_spans.started_at, excluded.started_at), "
                "  ended_at=excluded.ended_at, status=excluded.status, "
                "  attrs_json=excluded.attrs_json",
                (
                    item.trace_id,
                    item.span_id,
                    item.parent_span_id,
                    item.component,
                    item.tick,
                    item.started_at,
                    item.ended_at,
                    item.status,
                    item.attrs_json,
                ),
            )
        elif isinstance(item, _CorrelationWrite):
            conn.execute(
                "INSERT INTO trace_correlations "
                "(correlation_id, origin_trace_id, origin_traceparent, kind, "
                " created_at, resolved_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(correlation_id) DO UPDATE SET "
                "  origin_trace_id=excluded.origin_trace_id, "
                "  origin_traceparent="
                "    COALESCE(excluded.origin_traceparent, trace_correlations.origin_traceparent), "
                "  kind=COALESCE(excluded.kind, trace_correlations.kind), "
                "  resolved_at=COALESCE(excluded.resolved_at, trace_correlations.resolved_at)",
                (
                    item.correlation_id,
                    item.origin_trace_id,
                    item.origin_traceparent,
                    item.kind,
                    item.created_at,
                    item.resolved_at,
                ),
            )


def _max_event_record_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(record_id) FROM trace_events").fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


# --------------------------------------------------------------------------- #
# The tick-path durable sink seam (spec §4.2)
# --------------------------------------------------------------------------- #


class TraceSink(Protocol):
    """The durable-write surface the tick path (CoreLoop / SpanLogger) depends on.

    :class:`TraceWriter` is the real, async implementation; :data:`NULL_TRACE_SINK`
    is the no-op used off the live being (a bare unit test / a CLI path with no
    :class:`~lifemodel.adapters.being_platform.BeingAdapter` to
    :func:`acquire_trace_writer`). Both ``submit_*`` return ``True`` on a
    successful enqueue and ``False`` when the record was dropped (fail-open) —
    the durable-first signal :class:`~lifemodel.log.SpanLogger` gates its
    projections on.
    """

    def submit_event(
        self,
        *,
        record_id: int,
        trace_id: str,
        span_id: str | None,
        tick: int | None,
        event: str,
        ts: str,
        fields: Mapping[str, Any] | None = None,
    ) -> bool: ...

    def submit_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        component: str | None = None,
        tick: int | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> bool: ...

    def submit_correlation(
        self,
        *,
        correlation_id: str,
        origin_trace_id: str,
        created_at: str,
        origin_traceparent: str | None = None,
        kind: str | None = None,
        resolved_at: str | None = None,
    ) -> bool: ...


class _NullTraceWriter:
    """A :class:`TraceSink` that accepts and discards — the off-being default.

    Returns ``True`` (a trivially "successful" enqueue) so a
    :class:`~lifemodel.log.SpanLogger` built over it still projects onto its ring
    + human tail, without a durable store wired. The live being always injects a
    real :class:`TraceWriter`; this only backs bare test / CLI ticks where no
    ``observability.sqlite`` exists to persist to.
    """

    def submit_event(
        self,
        *,
        record_id: int,
        trace_id: str,
        span_id: str | None,
        tick: int | None,
        event: str,
        ts: str,
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        return True

    def submit_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        component: str | None = None,
        tick: int | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> bool:
        return True

    def submit_correlation(
        self,
        *,
        correlation_id: str,
        origin_trace_id: str,
        created_at: str,
        origin_traceparent: str | None = None,
        kind: str | None = None,
        resolved_at: str | None = None,
    ) -> bool:
        return True


#: The process-wide no-op sink used when no durable trace writer is wired.
NULL_TRACE_SINK: Final[TraceSink] = _NullTraceWriter()


# --------------------------------------------------------------------------- #
# Singleton per db-path (spec §4.2: refcount, idempotent start, reconnect-safe)
# --------------------------------------------------------------------------- #


@dataclass
class _WriterHandle:
    writer: TraceWriter
    refcount: int


_registry_lock = threading.Lock()
_registry: dict[str, _WriterHandle] = {}


def _registry_key(db_path: Path) -> str:
    return str(Path(db_path).resolve())


def acquire_trace_writer(
    db_path: Path,
    *,
    max_queue: int = _DEFAULT_MAX_QUEUE,
    batch_size: int = _DEFAULT_BATCH,
    retention: RetentionPolicy | None = None,
    protected_trace_ids: Callable[[], set[str]] | None = None,
    prune_every_commits: int = _DEFAULT_PRUNE_EVERY_COMMITS,
    clock: ClockPort | None = None,
) -> TraceWriter:
    """Return the started :class:`TraceWriter` for *db_path*, refcounted (§4.2).

    The FIRST acquire constructs + starts the singleton; later acquires bump the
    refcount and return the same instance (idempotent start). After the matching
    :func:`release_trace_writer` calls drop the count to zero the writer is
    stopped and forgotten, so a subsequent acquire is reconnect-safe — it spins
    a brand-new thread + connection. Construction options are honoured only on
    the first acquire of a path. *clock* is the writer thread's injected source of
    "now" for retention (spec §3.1) — the thread never reads system time.
    """
    key = _registry_key(db_path)
    with _registry_lock:
        handle = _registry.get(key)
        if handle is None:
            writer = TraceWriter(
                Path(db_path),
                max_queue=max_queue,
                batch_size=batch_size,
                retention=retention,
                protected_trace_ids=protected_trace_ids,
                prune_every_commits=prune_every_commits,
                clock=clock,
            )
            writer.start()
            _registry[key] = _WriterHandle(writer, 1)
            return writer
        handle.refcount += 1
        handle.writer.start()  # idempotent — covers a caller that stopped it directly
        return handle.writer


def peek_trace_writer(db_path: Path) -> TraceWriter | None:
    """Return the LIVE singleton writer for *db_path* WITHOUT taking a refcount.

    For best-effort out-of-band index touches from a code path that has no writer
    of its own but runs in the SAME process as the live being — e.g. the admin
    ``/lifemodel force-wake`` command marking a cleared correlation ``resolved_at``
    so retention can eventually reclaim its origin trace (§4.4). ``None`` when no
    writer is running for that path (a bare CLI process): the disposable index
    simply goes untouched — the *precious* state anchor was already cleared."""
    with _registry_lock:
        handle = _registry.get(_registry_key(db_path))
        return handle.writer if handle is not None else None


def release_trace_writer(db_path: Path, *, flush: bool = True) -> None:
    """Drop one refcount; on the last release flush + stop the writer (§4.2)."""
    key = _registry_key(db_path)
    with _registry_lock:
        handle = _registry.get(key)
        if handle is None:
            return
        handle.refcount -= 1
        if handle.refcount > 0:
            return
        del _registry[key]
        writer = handle.writer
    if flush:
        writer.flush(timeout=5.0)
    writer.stop()

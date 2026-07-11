"""``metrics.sqlite`` — the periodic ``MetricRegistry`` sampler + time-series store.

The time-series half of telemetry-core (design §4.4): a separate, disposable
SQLite file — a sibling of ``lifemodel.sqlite`` / ``observability.sqlite`` — into
which a daemon thread periodically snapshots the process-local
:class:`~lifemodel.core.metrics.MetricRegistry`. Distinct from
``observability.sqlite`` on purpose (§0): that store answers *"why did THIS
happen"* per-trace; this one answers *"how loaded, over time"* as aggregates.
Losing it never changes the being's behaviour, so — exactly like
:mod:`~lifemodel.state.trace_store` — every write is **fail-open**: the sampler
swallows its own errors and a tick never waits on this I/O.

Shape (all stdlib — ``sqlite3``/``threading``/``uuid``; the plugin runs inside
Hermes' own interpreter, no third-party deps):

* **Snapshot, not stream.** Unlike the trace writer's per-event queue, the
  sampler reads the whole registry every ``interval`` seconds via the metric
  types' ``.items()`` reads and writes the changed series. A ``Counter``/``Gauge``
  series is one row; a ``Histogram`` decomposes into Prometheus-shaped
  ``name_bucket{le}`` / ``name_count`` / ``name_sum`` rows.
* **run_id** — minted once per process (:func:`process_run_id`) and stamped on
  every sample, so a restart is a visible boundary (design §4.4/§4.5): rates are
  only ever computed *within* one ``run_id``, never glued across a restart.
* **export whitelist + skip-unchanged** — only ``export=1`` specs are sampled,
  and a series whose value is unchanged since the last write is skipped. A
  **heartbeat** every ``M`` cycles re-writes even unchanged series so the
  read-side (bead 7.7) always finds a recent point and can prove the sampler is
  alive.
* **Singleton per db-path** with refcount (:func:`acquire_metrics_sampler` /
  :func:`release_metrics_sampler`): idempotent start, reconnect-safe — mirrors
  :func:`~lifemodel.state.trace_store.acquire_trace_writer`.
* **Retention** (:func:`prune_metric_samples`) prunes by age / row-count / size,
  fail-open — mirrors :func:`~lifemodel.state.trace_store.prune_traces`.

Wiring the sampler's lifecycle into the being (composition / being_platform) is
a later integration; this module is the store + sampler + acquire/release API.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..core.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricRegistry,
    MetricSpec,
    label_key,
)

#: The metrics DB's filename, a sibling of ``lifemodel.sqlite`` in the state dir.
_DB_FILENAME: Final = "metrics.sqlite"

#: The metrics schema version. Even a disposable store carries one (mirrors the
#: trace store) so a future shape change is detectable rather than silently misread.
SCHEMA_VERSION: Final = 1

#: Default sampling period in seconds (design §4.4). Coarse: the being ticks
#: ~every 60s, so 15s over-samples enough to catch intra-tick change.
_DEFAULT_INTERVAL_SECONDS: Final = 15.0
#: Force-write even an unchanged series once every this-many cycles (heartbeat,
#: §4.4). At the 15s default that is a full snapshot ~every 5 minutes.
_DEFAULT_HEARTBEAT_EVERY: Final = 20
#: Run retention roughly every this-many sample cycles (rare, from the sampler).
_DEFAULT_PRUNE_EVERY_CYCLES: Final = 40

_module_logger = logging.getLogger("lifemodel.metrics_store")


def metrics_db_path(base_dir: Path) -> Path:
    """Return the metrics DB path under *base_dir* (sibling of ``lifemodel.sqlite``)."""
    return base_dir / _DB_FILENAME


# --------------------------------------------------------------------------- #
# Process run id (design §4.4 — the restart boundary stamped on every sample)
# --------------------------------------------------------------------------- #

_run_id_lock = threading.Lock()
_run_id: str | None = None


def process_run_id() -> str:
    """Return this process's ``run_id`` — minted once, stable thereafter (§4.4).

    A process-local unique id (stdlib :func:`uuid.uuid4`) stamped on every
    sample so a restart is a visible boundary: rates are computed only *within*
    one ``run_id``, never glued across the reset a restart implies. Shared by all
    samplers in the process (a restart boundary is process-wide, not per-profile).
    """
    global _run_id
    with _run_id_lock:
        if _run_id is None:
            _run_id = uuid.uuid4().hex
        return _run_id


# --------------------------------------------------------------------------- #
# Schema + connection
# --------------------------------------------------------------------------- #


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the metrics schema (design §4.4) if absent; idempotent."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    if conn.execute("SELECT version FROM schema_version").fetchone() is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metric_defs ("
        "  name TEXT PRIMARY KEY, kind TEXT NOT NULL,"
        "  unit TEXT, help TEXT,"
        "  label_keys_json TEXT NOT NULL,"
        "  export INTEGER NOT NULL DEFAULT 1,"
        "  created_at INTEGER, updated_at INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metric_samples ("
        "  ts INTEGER NOT NULL, run_id TEXT NOT NULL,"
        "  name TEXT NOT NULL,"
        "  label_key TEXT NOT NULL,"
        "  value REAL NOT NULL, labels_json TEXT)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_samples_name_label_ts ON metric_samples(name, label_key, ts)"
    )
    conn.commit()


def connect(db_path: Path, *, create_parent: bool = True) -> sqlite3.Connection:
    """Open a metrics-DB connection with WAL + incremental auto-vacuum + schema.

    Used by BOTH the sampler thread and retention tests, so the on-disk shape is
    identical either way. ``auto_vacuum=INCREMENTAL`` is set before any table
    exists (a fresh file) so :func:`prune_metric_samples` can actually shrink it.
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


# --------------------------------------------------------------------------- #
# Snapshot → rows (pure over the registry)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SampleRow:
    """One prospective ``metric_samples`` row derived from a registry read."""

    name: str
    label_key: str
    value: float
    labels_json: str | None


def _format_le(bound: float) -> str:
    """Render a histogram bucket bound as the ``le`` label value (Prometheus-style)."""
    return repr(bound)


def _labels_json(labels: dict[str, str]) -> str | None:
    """Deterministic JSON for a label set (``None`` for the empty set)."""
    return json.dumps(labels, sort_keys=True) if labels else None


def _sample_row(name: str, labels: dict[str, str], value: float) -> _SampleRow:
    return _SampleRow(
        name=name, label_key=label_key(labels), value=value, labels_json=_labels_json(labels)
    )


def snapshot_rows(registry: MetricRegistry) -> list[_SampleRow]:
    """Snapshot every ``export=1`` metric in *registry* into prospective rows (§4.4).

    ``Counter``/``Gauge`` series map one-to-one; a ``Histogram`` series decomposes
    into cumulative ``name_bucket{le}`` rows plus ``name_count`` and ``name_sum``
    (the implicit ``+Inf`` bucket equals ``name_count`` and is recovered on read,
    so it is not stored twice). Reads go through the metric types' own locked
    ``.items()`` — this never mutates the registry.
    """
    rows: list[_SampleRow] = []
    for metric in registry.metrics():
        spec = metric.spec
        if not spec.export:
            continue
        if isinstance(metric, Histogram):
            for _lk, labels, snap in metric.items():
                for bound, cumulative in snap.buckets:
                    rows.append(
                        _sample_row(
                            f"{spec.name}_bucket",
                            {**labels, "le": _format_le(bound)},
                            float(cumulative),
                        )
                    )
                rows.append(_sample_row(f"{spec.name}_count", dict(labels), float(snap.count)))
                rows.append(_sample_row(f"{spec.name}_sum", dict(labels), float(snap.sum)))
        elif isinstance(metric, (Counter, Gauge)):
            for _lk, labels, value in metric.items():
                rows.append(_sample_row(spec.name, dict(labels), float(value)))
    return rows


def sync_metric_defs(conn: sqlite3.Connection, specs: list[MetricSpec], ts: int) -> None:
    """Upsert ``metric_defs`` from *specs* — records EVERY metric, export or not."""
    conn.executemany(
        "INSERT INTO metric_defs "
        "(name, kind, unit, help, label_keys_json, export, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "  kind=excluded.kind, unit=excluded.unit, help=excluded.help, "
        "  label_keys_json=excluded.label_keys_json, export=excluded.export, "
        "  updated_at=excluded.updated_at",
        [
            (
                spec.name,
                spec.kind,
                spec.unit,
                spec.help,
                json.dumps(list(spec.label_keys)),
                1 if spec.export else 0,
                ts,
                ts,
            )
            for spec in specs
        ],
    )


# --------------------------------------------------------------------------- #
# Retention (design §4.4 — by age / row-count / size, fail-open)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricsRetentionPolicy:
    """The three retention axes (any may be ``None`` to disable it).

    Defaults are conservative. Pruning drops whole sample rows oldest-first; the
    heartbeat (design §4.4) guarantees a recent point per series survives so the
    read-side still resolves a window even after old points are reclaimed.
    """

    max_age_seconds: int | None = 14 * 24 * 60 * 60
    max_rows: int | None = 1_000_000
    max_bytes: int | None = 128 * 1024 * 1024


def _db_size_bytes(conn: sqlite3.Connection) -> int:
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    return int(page_count) * int(page_size)


def _vacuum(conn: sqlite3.Connection) -> None:
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA incremental_vacuum")
        conn.commit()


def _delete_oldest_ts_cohort(conn: sqlite3.Connection) -> int:
    """Delete every row sharing the OLDEST ``ts`` (a whole sample-snapshot).

    Pruning a whole ``ts`` cohort at a time (never an arbitrary rowid batch)
    keeps each snapshot intact — a histogram's ``_bucket``/``_count``/``_sum``
    rows for one instant are reclaimed together, never split at the boundary
    (design §4.4: retention целыми). Returns rows deleted (0 when empty)."""
    row = conn.execute("SELECT MIN(ts) FROM metric_samples").fetchone()
    if row is None or row[0] is None:
        return 0
    return conn.execute("DELETE FROM metric_samples WHERE ts = ?", (int(row[0]),)).rowcount


def _prune_by_size(conn: sqlite3.Connection, max_bytes: int) -> int:
    """Delete oldest whole ts-snapshots until the file is under *max_bytes*."""
    deleted = 0
    while _db_size_bytes(conn) > max_bytes:
        n = _delete_oldest_ts_cohort(conn)
        if n == 0:
            break
        conn.commit()
        deleted += n
        _vacuum(conn)
    return deleted


def prune_metric_samples(
    conn: sqlite3.Connection, *, policy: MetricsRetentionPolicy, now_ts: int
) -> int:
    """Prune samples past the policy (age → rows → size). Returns rows deleted.

    A pure function over *conn* so it is unit-testable off the sampler thread and
    reused by it. Fail-open is the caller's concern (the sampler swallows).
    """
    deleted = 0

    if policy.max_age_seconds is not None:
        cutoff = now_ts - policy.max_age_seconds
        deleted += conn.execute("DELETE FROM metric_samples WHERE ts < ?", (cutoff,)).rowcount

    if policy.max_rows is not None:
        # Drop whole oldest ts-snapshots (not arbitrary rowids) until under the cap,
        # so a snapshot is never left partial at the boundary (design §4.4).
        while (
            int(conn.execute("SELECT COUNT(*) FROM metric_samples").fetchone()[0]) > policy.max_rows
        ):
            n = _delete_oldest_ts_cohort(conn)
            if n == 0:
                break
            deleted += n

    if deleted:
        conn.commit()
        _vacuum(conn)

    if policy.max_bytes is not None:
        deleted += _prune_by_size(conn, policy.max_bytes)

    return deleted


# --------------------------------------------------------------------------- #
# Reader helper (minimal — for tests / bead 7.7 to build on)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricSample:
    """One decoded ``metric_samples`` row (labels parsed back from JSON)."""

    ts: int
    run_id: str
    name: str
    label_key: str
    value: float
    labels: dict[str, str]


def read_samples(
    db_path: Path,
    *,
    name: str | None = None,
    latest_run: bool = False,
    limit: int | None = None,
) -> list[MetricSample]:
    """Read ``metric_samples`` (optionally for one *name*), oldest first.

    ``latest_run`` restricts to the run of the newest sample (rates are only ever
    computed within one ``run_id``, so a windowed reader never needs older runs).
    ``limit`` caps the read to the most-recent rows — the row set is fetched
    newest-first under the cap and then reversed to oldest-first, so a ``stats``
    window over a huge store never loads the whole DB into memory (design §4.4).
    The cap is generous enough to span many heartbeats, so every series still
    carries a baseline point ``<= t0`` for its windowed delta.
    """
    clauses: list[str] = []
    params: list[object] = []
    if latest_run:
        clauses.append(
            "run_id = (SELECT run_id FROM metric_samples ORDER BY ts DESC, rowid DESC LIMIT 1)"
        )
    if name is not None:
        clauses.append("name = ?")
        params.append(name)
    sql = "SELECT ts, run_id, name, label_key, value, labels_json FROM metric_samples"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if limit is not None:
        sql += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
    else:
        sql += " ORDER BY ts ASC, rowid ASC"
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if limit is not None:
        rows = list(reversed(rows))  # newest-first read → present oldest-first
    samples: list[MetricSample] = []
    for ts, run_id, mname, lk, value, labels_json in rows:
        labels: dict[str, str] = json.loads(labels_json) if labels_json else {}
        samples.append(
            MetricSample(
                ts=int(ts),
                run_id=str(run_id),
                name=str(mname),
                label_key=str(lk),
                value=float(value),
                labels=labels,
            )
        )
    return samples


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


class MetricsSampler:
    """One daemon thread snapshotting a :class:`MetricRegistry` into ``metrics.sqlite``.

    Fail-open by construction (design §4.4/§7): the sampler swallows its own
    errors (counted in :attr:`sample_errors`) and never lets a bad sample kill
    the thread or the tick. Each cycle writes only *changed* series
    (skip-unchanged), with a heartbeat every ``heartbeat_every`` cycles forcing a
    full write. :meth:`sample_once` is the synchronous unit-testable core;
    :meth:`start`/:meth:`stop` run it on a daemon thread. Prefer
    :func:`acquire_metrics_sampler` / :func:`release_metrics_sampler` over
    constructing directly — they enforce the singleton-per-db-path lifecycle.

    A single instance samples on ONE thread at a time (either the daemon OR a
    direct :meth:`sample_once` caller): its SQLite connection is thread-affine
    and the daemon path owns its own connection, closed when the thread exits.
    """

    def __init__(
        self,
        registry: MetricRegistry,
        db_path: Path,
        *,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        heartbeat_every: int = _DEFAULT_HEARTBEAT_EVERY,
        retention: MetricsRetentionPolicy | None = None,
        run_id: str | None = None,
        prune_every_cycles: int = _DEFAULT_PRUNE_EVERY_CYCLES,
    ) -> None:
        self._registry = registry
        self._db_path = Path(db_path)
        self._interval_seconds = max(0.0, interval_seconds)
        self._heartbeat_every = max(1, heartbeat_every)
        self._retention = retention or MetricsRetentionPolicy()
        self._run_id = run_id or process_run_id()
        self._prune_every_cycles = max(1, prune_every_cycles)

        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None  # the direct sample_once path's conn
        self._last_written: dict[tuple[str, str], float] = {}
        self._cycles = 0
        self._commits_since_prune = 0
        self.sample_errors = 0

    @property
    def run_id(self) -> str:
        """The ``run_id`` stamped on this sampler's rows."""
        return self._run_id

    # ---- synchronous core (unit-testable) ------------------------------- #

    def sample_once(self, *, ts: int | None = None) -> int:
        """Snapshot the registry once, writing changed rows. Returns rows written.

        For direct/testing use — lazily opens (and reuses) a thread-affine
        connection on the calling thread. The daemon path uses its OWN connection
        (do not mix the two on one instance). May raise (so tests see failures);
        the daemon wraps it fail-open.
        """
        if self._conn is None:
            self._conn = connect(self._db_path)
        return self._sample_into(self._conn, _now_ts() if ts is None else ts)

    def _sample_into(self, conn: sqlite3.Connection, ts: int) -> int:
        """Do one sample against *conn*: sync defs, write changed rows, maybe prune."""
        with self._lock:
            force = (self._cycles % self._heartbeat_every) == 0
            self._cycles += 1
            specs = self._registry.specs()
            rows = snapshot_rows(self._registry)

            sync_metric_defs(conn, specs, ts)

            to_write: list[tuple[int, str, str, str, float, str | None]] = []
            for row in rows:
                key = (row.name, row.label_key)
                previous = self._last_written.get(key)
                if not force and previous is not None and previous == row.value:
                    continue  # skip-unchanged (§4.4)
                to_write.append(
                    (ts, self._run_id, row.name, row.label_key, row.value, row.labels_json)
                )
                self._last_written[key] = row.value

            if to_write:
                conn.executemany(
                    "INSERT INTO metric_samples "
                    "(ts, run_id, name, label_key, value, labels_json) VALUES (?, ?, ?, ?, ?, ?)",
                    to_write,
                )
            conn.commit()
            self._maybe_prune(conn, ts)
            return len(to_write)

    def _maybe_prune(self, conn: sqlite3.Connection, ts: int) -> None:
        self._commits_since_prune += 1
        if self._commits_since_prune < self._prune_every_cycles:
            return
        self._commits_since_prune = 0
        try:
            prune_metric_samples(conn, policy=self._retention, now_ts=ts)
        except sqlite3.Error:
            self.sample_errors += 1

    def close(self) -> None:
        """Close the direct-path connection (for the non-daemon :meth:`sample_once` flow)."""
        conn = self._conn
        self._conn = None
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # ---- daemon lifecycle ----------------------------------------------- #

    def start(self) -> None:
        """Spawn the sampler thread; idempotent and double-start safe."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._first_sample.clear()
            self._thread = threading.Thread(
                target=self._run, name="lifemodel-metrics-sampler", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the sampler thread to finish its cycle and join it; idempotent."""
        with self._lifecycle_lock:
            thread = self._thread
            self._thread = None
            self._stop.set()
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout)

    def wait_first_sample(self, timeout: float | None = None) -> bool:
        """Block until the daemon has completed (or attempted) its first sample."""
        return self._first_sample.wait(timeout)

    def _run(self) -> None:
        try:
            conn = connect(self._db_path)
        except sqlite3.Error:
            _module_logger.warning(
                "metrics_store_open_failed path=%s", self._db_path, exc_info=True
            )
            self._first_sample.set()
            return
        try:
            while not self._stop.is_set():
                try:
                    self._sample_into(conn, _now_ts())
                except Exception:  # never let the sampler thread die (fail-open, §4.4)
                    self.sample_errors += 1
                finally:
                    self._first_sample.set()
                self._stop.wait(self._interval_seconds)
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.commit()
            with contextlib.suppress(sqlite3.Error):
                conn.close()


def _now_ts() -> int:
    return int(time.time())


# --------------------------------------------------------------------------- #
# Singleton per db-path (design §4.4: refcount, idempotent start, reconnect-safe)
# --------------------------------------------------------------------------- #


@dataclass
class _SamplerHandle:
    sampler: MetricsSampler
    refcount: int


_samplers_lock = threading.Lock()
_samplers: dict[str, _SamplerHandle] = {}


def _sampler_key(db_path: Path) -> str:
    return str(Path(db_path).resolve())


def acquire_metrics_sampler(
    registry: MetricRegistry,
    base_dir: Path,
    *,
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
    heartbeat_every: int = _DEFAULT_HEARTBEAT_EVERY,
    retention: MetricsRetentionPolicy | None = None,
    run_id: str | None = None,
    prune_every_cycles: int = _DEFAULT_PRUNE_EVERY_CYCLES,
) -> MetricsSampler:
    """Return the started :class:`MetricsSampler` for *base_dir*, refcounted (§4.4).

    The FIRST acquire constructs + starts the singleton; later acquires bump the
    refcount and return the same instance (idempotent start). After the matching
    :func:`release_metrics_sampler` calls drop the count to zero the sampler is
    stopped and forgotten, so a subsequent acquire is reconnect-safe. Construction
    options are honoured only on the first acquire of a path.
    """
    db_path = metrics_db_path(base_dir)
    key = _sampler_key(db_path)
    with _samplers_lock:
        handle = _samplers.get(key)
        if handle is None:
            sampler = MetricsSampler(
                registry,
                db_path,
                interval_seconds=interval_seconds,
                heartbeat_every=heartbeat_every,
                retention=retention,
                run_id=run_id,
                prune_every_cycles=prune_every_cycles,
            )
            sampler.start()
            _samplers[key] = _SamplerHandle(sampler, 1)
            return sampler
        handle.refcount += 1
        handle.sampler.start()  # idempotent — covers a caller that stopped it directly
        return handle.sampler


def peek_metrics_sampler(base_dir: Path) -> MetricsSampler | None:
    """Return the LIVE singleton sampler for *base_dir* WITHOUT taking a refcount."""
    with _samplers_lock:
        handle = _samplers.get(_sampler_key(metrics_db_path(base_dir)))
        return handle.sampler if handle is not None else None


def release_metrics_sampler(base_dir: Path, *, timeout: float = 5.0) -> None:
    """Drop one refcount; on the last release stop the sampler (§4.4)."""
    key = _sampler_key(metrics_db_path(base_dir))
    with _samplers_lock:
        handle = _samplers.get(key)
        if handle is None:
            return
        handle.refcount -= 1
        if handle.refcount > 0:
            return
        del _samplers[key]
        sampler = handle.sampler
    sampler.stop(timeout=timeout)

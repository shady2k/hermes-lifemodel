"""Tests for ``observability.sqlite`` — the disposable trace store + async writer.

Contract under test (spec §4.2/§4.3):
* schema is created (schema_version + the three tables);
* the async writer drops on a full queue without raising, and is fail-open on a
  per-record write error;
* ``flush`` gives read-your-writes determinism;
* the singleton-per-db-path is refcounted, idempotent to start, reconnect-safe;
* retention prunes by age / count / size, a whole trace at a time, and NEVER an
  in-flight / unresolved / within-grace trace.

Stdlib only; every test uses ``tmp_path`` and stops any writer it starts.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.timeutil import from_iso, to_iso
from lifemodel.state.trace_store import (
    RetentionPolicy,
    TraceWriter,
    _ensure_record_id_floor,
    _max_event_record_id,
    acquire_trace_writer,
    connect,
    initialize_schema,
    next_record_id,
    observability_db_path,
    prune_traces,
    release_trace_writer,
)
from lifemodel.testing.fakes import FakeClock

_NOW = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)

#: The canonical, fixed-width normalized-ISO shape every stored trace timestamp
#: must take (spec §3 invariant): always 6-digit µs, always ``+00:00``.
_CANONICAL_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")

#: A tz-aware but DELIBERATELY non-normalized instant (a ``+03:00`` offset, whole
#: second, no fractional part) — its canonical UTC form is 12:00:00.000000+00:00.
_UNNORMALIZED_PLUS3 = "2026-07-09T15:00:00+03:00"
_UNNORMALIZED_PLUS3_CANONICAL = "2026-07-09T12:00:00.000000+00:00"


def _db(tmp_path: Path) -> Path:
    return observability_db_path(tmp_path)


def _read(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with sqlite3.connect(str(path)) as conn:
        return conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_connect_creates_schema_and_version(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"schema_version", "trace_spans", "trace_events", "trace_correlations"} <= tables
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 1
    finally:
        conn.close()


def test_initialize_schema_is_idempotent(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        initialize_schema(conn)  # again — must not duplicate the version row
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# record_id
# --------------------------------------------------------------------------- #


def test_next_record_id_is_monotonic() -> None:
    a = next_record_id()
    b = next_record_id()
    c = next_record_id()
    assert a < b < c


def test_record_id_floor_seeds_above_existing_max(tmp_path: Path) -> None:
    # A DB that survived a restart already holds ids up to N; the counter floor
    # must jump above them so a fresh insert never collides.
    conn = connect(_db(tmp_path))
    huge = 10_000_000
    conn.execute(
        "INSERT INTO trace_events (record_id, trace_id, event, ts) VALUES (?, 't', 'e', ?)",
        (huge, _NOW.isoformat()),
    )
    conn.commit()
    assert _max_event_record_id(conn) == huge
    conn.close()

    _ensure_record_id_floor(huge)
    assert next_record_id() > huge


# --------------------------------------------------------------------------- #
# async writer: drop-on-full, fail-open, flush determinism
# --------------------------------------------------------------------------- #


def test_writer_drops_on_full_queue_without_raising(tmp_path: Path) -> None:
    # NOT started → nothing drains → the bounded queue fills and further submits
    # are dropped (counted), never raised.
    writer = TraceWriter(_db(tmp_path), max_queue=2)
    results = [
        writer.submit_event(
            record_id=next_record_id(),
            trace_id="t",
            span_id="s",
            tick=i,
            event="e",
            ts=_NOW.isoformat(),
        )
        for i in range(5)
    ]
    assert results[:2] == [True, True]  # first two fit
    assert results[2:] == [False, False, False]  # rest dropped
    assert writer.dropped_count == 3


def test_writer_is_fail_open_on_write_error(tmp_path: Path) -> None:
    # Two events with the SAME record_id → the second's INSERT raises a PK
    # conflict on the writer thread; it is swallowed (counted), the thread lives,
    # and the first row still lands.
    path = _db(tmp_path)
    writer = TraceWriter(path, batch_size=1)
    writer.start()
    try:
        assert writer.submit_event(
            record_id=777_001, trace_id="t", span_id="s", tick=1, event="a", ts=_NOW.isoformat()
        )
        assert writer.submit_event(
            record_id=777_001, trace_id="t", span_id="s", tick=2, event="b", ts=_NOW.isoformat()
        )
        assert writer.flush(timeout=5.0)
        assert writer.write_errors >= 1
        rows = _read(path, "SELECT event FROM trace_events WHERE record_id=?", (777_001,))
        assert rows == [("a",)]  # only the first survived; the dup was dropped
    finally:
        writer.stop()


def test_flush_gives_read_your_writes(tmp_path: Path) -> None:
    path = _db(tmp_path)
    writer = TraceWriter(path)
    writer.start()
    try:
        for i in range(20):
            assert writer.submit_event(
                record_id=next_record_id(),
                trace_id="trace-A",
                span_id="s",
                tick=i,
                event="tick",
                ts=_NOW.isoformat(),
            )
        assert writer.flush(timeout=5.0)
        count = _read(path, "SELECT COUNT(*) FROM trace_events WHERE trace_id='trace-A'")[0][0]
        assert count == 20  # every enqueued record is durably readable after flush
    finally:
        writer.stop()


def test_submit_span_attrs_serialize_and_round_trip(tmp_path: Path) -> None:
    path = _db(tmp_path)
    writer = TraceWriter(path)
    writer.start()
    try:
        assert writer.submit_span(
            trace_id="trace-S",
            span_id="span-1",
            parent_span_id=None,
            component="cognition",
            tick=4,
            started_at=_NOW.isoformat(),
            ended_at=_NOW.isoformat(),
            status="suppressed",
            attrs={"u": 0.42, "gate": "silent", "reason": "act_gate"},
        )
        assert writer.flush(timeout=5.0)
        rows = _read(path, "SELECT status, attrs_json FROM trace_spans WHERE trace_id='trace-S'")
        assert rows[0][0] == "suppressed"
        import json

        assert json.loads(rows[0][1]) == {"u": 0.42, "gate": "silent", "reason": "act_gate"}
    finally:
        writer.stop()


def test_flush_on_unstarted_writer_is_trivially_true(tmp_path: Path) -> None:
    writer = TraceWriter(_db(tmp_path))
    assert writer.flush(timeout=0.1) is True


# --------------------------------------------------------------------------- #
# singleton lifecycle: refcount, idempotent start, reconnect-safe
# --------------------------------------------------------------------------- #


def test_acquire_is_singleton_per_path_and_refcounted(tmp_path: Path) -> None:
    path = _db(tmp_path)
    w1 = acquire_trace_writer(path)
    try:
        w2 = acquire_trace_writer(path)
        assert w1 is w2  # same instance for the same path
    finally:
        release_trace_writer(path)  # refcount 2 -> 1, still alive
    assert w1._thread is not None and w1._thread.is_alive()
    release_trace_writer(path)  # refcount 1 -> 0, stopped
    assert w1._thread is None or not w1._thread.is_alive()


def test_start_is_idempotent(tmp_path: Path) -> None:
    writer = TraceWriter(_db(tmp_path))
    writer.start()
    thread = writer._thread
    try:
        writer.start()  # must NOT spawn a second thread
        assert writer._thread is thread
    finally:
        writer.stop()


def test_reconnect_after_release_is_a_fresh_working_writer(tmp_path: Path) -> None:
    path = _db(tmp_path)
    w1 = acquire_trace_writer(path)
    release_trace_writer(path)  # fully released -> stopped + forgotten
    w2 = acquire_trace_writer(path)
    try:
        assert w2 is not w1  # a brand-new writer (new thread + connection)
        assert w2.submit_event(
            record_id=next_record_id(),
            trace_id="reconnect",
            span_id="s",
            tick=1,
            event="e",
            ts=_NOW.isoformat(),
        )
        assert w2.flush(timeout=5.0)
        assert (
            _read(path, "SELECT COUNT(*) FROM trace_events WHERE trace_id='reconnect'")[0][0] == 1
        )
    finally:
        release_trace_writer(path)


def test_writer_thread_seeds_record_id_from_surviving_db(tmp_path: Path) -> None:
    path = _db(tmp_path)
    seeded = 20_000_000
    conn = connect(path)
    conn.execute(
        "INSERT INTO trace_events (record_id, trace_id, event, ts) VALUES (?, 't', 'e', ?)",
        (seeded, _NOW.isoformat()),
    )
    conn.commit()
    conn.close()

    writer = acquire_trace_writer(path)
    try:
        assert writer.flush(timeout=5.0)  # thread has opened + seeded before this returns
        assert next_record_id() > seeded
    finally:
        release_trace_writer(path)


# --------------------------------------------------------------------------- #
# retention (pure prune_traces over a test connection)
# --------------------------------------------------------------------------- #


def _seed_trace(
    conn: sqlite3.Connection,
    trace_id: str,
    start: datetime,
    *,
    correlation: str | None = None,
    resolved_at: datetime | None = None,
) -> None:
    conn.execute(
        "INSERT INTO trace_spans (trace_id, span_id, started_at, status) VALUES (?, ?, ?, 'ok')",
        (trace_id, f"{trace_id}-root", start.isoformat()),
    )
    conn.execute(
        "INSERT INTO trace_events (record_id, trace_id, event, ts) VALUES (?, ?, 'tick', ?)",
        (next_record_id(), trace_id, start.isoformat()),
    )
    if correlation is not None:
        conn.execute(
            "INSERT INTO trace_correlations "
            "(correlation_id, origin_trace_id, created_at, resolved_at) VALUES (?, ?, ?, ?)",
            (
                correlation,
                trace_id,
                start.isoformat(),
                resolved_at.isoformat() if resolved_at is not None else None,
            ),
        )
    conn.commit()


def _remaining(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT DISTINCT trace_id FROM trace_spans").fetchall()
    return {r[0] for r in rows}


def test_retention_prunes_by_age(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        _seed_trace(conn, "old", _NOW - timedelta(days=30))
        _seed_trace(conn, "fresh", _NOW - timedelta(days=1))
        prune_traces(
            conn,
            policy=RetentionPolicy(max_age_days=14, max_traces=None, max_bytes=None),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == {"fresh"}
    finally:
        conn.close()


def test_retention_prunes_by_count_keeping_newest(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        _seed_trace(conn, "t1", _NOW - timedelta(days=3))
        _seed_trace(conn, "t2", _NOW - timedelta(days=2))
        _seed_trace(conn, "t3", _NOW - timedelta(days=1))
        prune_traces(
            conn,
            policy=RetentionPolicy(max_age_days=None, max_traces=1, max_bytes=None),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == {"t3"}  # only the newest kept
    finally:
        conn.close()


def test_retention_prunes_by_size_keeping_protected(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        _seed_trace(conn, "s1", _NOW - timedelta(days=3))
        # 's2' has a LIVE (unresolved) correlation -> protected even under size pressure.
        _seed_trace(conn, "s2", _NOW - timedelta(days=2), correlation="c-s2")
        _seed_trace(conn, "s3", _NOW - timedelta(days=1))
        prune_traces(
            conn,
            policy=RetentionPolicy(max_age_days=None, max_traces=None, max_bytes=1),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == {"s2"}  # everything unprotected pruned under size=1
    finally:
        conn.close()


def test_retention_never_prunes_unresolved_trace(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        # Old-past-limit but with an UNRESOLVED correlation -> protected (codex #1).
        _seed_trace(conn, "inflight", _NOW - timedelta(days=30), correlation="c-1")
        prune_traces(
            conn,
            policy=RetentionPolicy(max_age_days=1, max_traces=None, max_bytes=None),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == {"inflight"}  # not pruned while unresolved
    finally:
        conn.close()


def test_retention_prunes_after_resolve_plus_grace(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        # Resolved 2 days ago, grace is 1 day -> past the window -> prunable.
        _seed_trace(
            conn,
            "resolved",
            _NOW - timedelta(days=30),
            correlation="c-2",
            resolved_at=_NOW - timedelta(days=2),
        )
        prune_traces(
            conn,
            policy=RetentionPolicy(
                max_age_days=1, max_traces=None, max_bytes=None, resolved_grace_days=1
            ),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == set()  # resolved + past grace -> pruned
    finally:
        conn.close()


def test_retention_within_grace_is_protected(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        # Resolved 12h ago, grace is 1 day -> still inside the window -> protected.
        _seed_trace(
            conn,
            "recent",
            _NOW - timedelta(days=30),
            correlation="c-3",
            resolved_at=_NOW - timedelta(hours=12),
        )
        prune_traces(
            conn,
            policy=RetentionPolicy(
                max_age_days=1, max_traces=None, max_bytes=None, resolved_grace_days=1
            ),
            protected_ids=set(),
            now=_NOW,
        )
        assert _remaining(conn) == {"recent"}
    finally:
        conn.close()


def test_retention_protects_live_state_anchor(tmp_path: Path) -> None:
    conn = connect(_db(tmp_path))
    try:
        _seed_trace(conn, "anchored", _NOW - timedelta(days=30))
        prune_traces(
            conn,
            policy=RetentionPolicy(max_age_days=1, max_traces=None, max_bytes=None),
            protected_ids={"anchored"},  # a live pending_proactive_origin_traceparent
            now=_NOW,
        )
        assert _remaining(conn) == {"anchored"}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# ingress normalization (spec §4 codex #9): every trace timestamp is serialized
# via ``to_iso`` at the ``submit_*`` boundary, so the DB only ever holds
# normalized ISO and retention's lexical ``MIN(ts)`` / ``< cutoff`` is correct.
# --------------------------------------------------------------------------- #


def test_submit_event_normalizes_ts_to_canonical_iso(tmp_path: Path) -> None:
    # A tz-aware but non-normalized ``ts`` (+03:00 offset, no µs) must be stored as
    # its canonical UTC form, not verbatim — so TEXT order == chronological order.
    path = _db(tmp_path)
    writer = TraceWriter(path, batch_size=1)
    writer.start()
    try:
        assert writer.submit_event(
            record_id=next_record_id(),
            trace_id="t",
            span_id="s",
            tick=1,
            event="e",
            ts=_UNNORMALIZED_PLUS3,
        )
        assert writer.flush(timeout=5.0)
        (stored,) = _read(path, "SELECT ts FROM trace_events WHERE trace_id='t'")[0]
        assert _CANONICAL_ISO_RE.match(str(stored))
        assert stored == _UNNORMALIZED_PLUS3_CANONICAL
        assert stored == to_iso(from_iso(_UNNORMALIZED_PLUS3))
    finally:
        writer.stop()


def test_submit_span_normalizes_started_and_ended(tmp_path: Path) -> None:
    path = _db(tmp_path)
    writer = TraceWriter(path, batch_size=1)
    writer.start()
    try:
        assert writer.submit_span(
            trace_id="t",
            span_id="s",
            component="cognition",
            tick=1,
            started_at=_UNNORMALIZED_PLUS3,
            ended_at="2026-07-09T15:00:01+03:00",
            status="ok",
        )
        assert writer.flush(timeout=5.0)
        (started, ended) = _read(
            path, "SELECT started_at, ended_at FROM trace_spans WHERE trace_id='t'"
        )[0]
        assert _CANONICAL_ISO_RE.match(str(started))
        assert _CANONICAL_ISO_RE.match(str(ended))
        assert started == _UNNORMALIZED_PLUS3_CANONICAL
        assert ended == "2026-07-09T12:00:01.000000+00:00"
    finally:
        writer.stop()


def test_submit_correlation_normalizes_created_and_resolved(tmp_path: Path) -> None:
    path = _db(tmp_path)
    writer = TraceWriter(path, batch_size=1)
    writer.start()
    try:
        assert writer.submit_correlation(
            correlation_id="c",
            origin_trace_id="t",
            created_at=_UNNORMALIZED_PLUS3,
            resolved_at="2026-07-09T18:00:00+03:00",
        )
        assert writer.flush(timeout=5.0)
        (created, resolved) = _read(
            path,
            "SELECT created_at, resolved_at FROM trace_correlations WHERE correlation_id='c'",
        )[0]
        assert _CANONICAL_ISO_RE.match(str(created))
        assert _CANONICAL_ISO_RE.match(str(resolved))
        assert created == _UNNORMALIZED_PLUS3_CANONICAL
        assert resolved == "2026-07-09T15:00:00.000000+00:00"
    finally:
        writer.stop()


def test_writer_thread_retention_uses_injected_clock(tmp_path: Path) -> None:
    # The writer thread must source retention's "now" from the INJECTED clock, not
    # wall time. Pin the clock far from real now (2020) and seed traces relative to
    # it: 'old' is >14d before the injected now, 'fresh' is 1d before. If the writer
    # (wrongly) used wall-clock now (2026) both would be ancient and BOTH pruned;
    # sourcing the injected clock keeps 'fresh'.
    injected = datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC)
    path = _db(tmp_path)
    conn = connect(path)
    try:
        _seed_trace(conn, "old", injected - timedelta(days=31))
        _seed_trace(conn, "fresh", injected - timedelta(days=1))
    finally:
        conn.close()
    writer = TraceWriter(
        path,
        clock=FakeClock(injected),
        batch_size=1,
        prune_every_commits=1,
        retention=RetentionPolicy(max_age_days=14, max_traces=None, max_bytes=None),
    )
    writer.start()
    try:
        assert writer.submit_event(
            record_id=next_record_id(),
            trace_id="new",
            span_id="s",
            tick=1,
            event="e",
            ts=to_iso(injected),
        )
        assert writer.flush(timeout=5.0)
    finally:
        writer.stop()
    remaining = {r[0] for r in _read(path, "SELECT DISTINCT trace_id FROM trace_spans")}
    assert "old" not in remaining  # >14d before the injected now -> pruned
    assert "fresh" in remaining  # 1d before the injected now -> kept


def test_submit_does_not_silently_coerce_naive_ts(tmp_path: Path, caplog: object) -> None:
    # A tz-naive ``ts`` must NEVER be silently coerced to an aware +00:00 form on the
    # write path (spec §4: normalize at ingress, don't naive->UTC on writes). The
    # disposable store is fail-open, so the record is kept (raw) + a WARNING logged,
    # never dropped and never silently coerced.
    import pytest  # local: keep the module's top imports minimal

    assert isinstance(caplog, pytest.LogCaptureFixture)
    path = _db(tmp_path)
    writer = TraceWriter(path, batch_size=1)
    writer.start()
    try:
        with caplog.at_level(logging.WARNING, logger="lifemodel.trace_store"):
            assert writer.submit_event(
                record_id=next_record_id(),
                trace_id="t",
                span_id="s",
                tick=1,
                event="e",
                ts="2026-07-09T12:00:00",  # naive — no offset
            )
        assert writer.flush(timeout=5.0)
        (stored,) = _read(path, "SELECT ts FROM trace_events WHERE trace_id='t'")[0]
        assert stored == "2026-07-09T12:00:00"  # kept raw, NOT coerced to +00:00
        assert not _CANONICAL_ISO_RE.match(str(stored))
        assert any("trace_ts_not_normalized" in r.getMessage() for r in caplog.records)
    finally:
        writer.stop()

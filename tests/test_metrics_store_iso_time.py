"""Tests for ``metrics.sqlite`` unified-time migration (spec §4, lm-fib.10.3).

The metrics store used to keep time as INTEGER epoch (``metric_samples.ts``,
``metric_defs.created_at/updated_at``). Slice 3 flips those columns to
normalized ISO-8601 UTC TEXT — the SAME :func:`~lifemodel.core.timeutil.to_iso`
serializer as every other store — WITHOUT breaking the sampler's whole-snapshot
pruning (one ``ts`` per sample cycle) or the retention arithmetic (ISO cutoff).

Contract under test:

* DDL: ``ts`` / ``created_at`` / ``updated_at`` are TEXT; the index survives;
  ``SCHEMA_VERSION`` is bumped and an OLD-shape file is destructively recreated
  (an existing INTEGER-column file must NOT be silently kept).
* one ISO ``ts`` per sample cycle (µs must not split a snapshot into cohorts);
* retention deletes ``ts < cutoff_iso`` where ``cutoff = now - max_age``;
* ``read_samples`` returns ISO ``ts`` strings, chronologically ordered;
* the sampler sources "now" from an injected clock, never system time.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.timeutil import from_iso, to_iso
from lifemodel.state.metrics_store import (
    SCHEMA_VERSION,
    MetricsRetentionPolicy,
    MetricsSampler,
    connect,
    metrics_db_path,
    prune_metric_samples,
    read_samples,
)
from lifemodel.testing.fakes import FakeClock

_RUN = "run-iso-0001"
#: A fixed aware-UTC anchor so a test can name an instant by "seconds since anchor"
#: and preserve the old integer tests' 60s spacing / boundary relationships.
_ANCHOR = datetime(2026, 7, 11, 0, 0, 0, tzinfo=UTC)


def _iso(offset_seconds: float) -> str:
    """The canonical ISO string for ``anchor + offset_seconds``."""
    return to_iso(_ANCHOR + timedelta(seconds=offset_seconds))


def _sampler(registry: MetricRegistry, tmp_path: Path, **kwargs: object) -> MetricsSampler:
    kwargs.setdefault("run_id", _RUN)
    kwargs.setdefault("heartbeat_every", 10_000)
    return MetricsSampler(registry, metrics_db_path(tmp_path), **kwargs)  # type: ignore[arg-type]


def _col_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {row[1]: row[2] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# --------------------------------------------------------------------------- #
# DDL: time columns are TEXT; index survives; version bumped
# --------------------------------------------------------------------------- #


def test_time_columns_are_text_after_bootstrap(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        assert _col_types(conn, "metric_samples")["ts"] == "TEXT"
        defs = _col_types(conn, "metric_defs")
        assert defs["created_at"] == "TEXT"
        assert defs["updated_at"] == "TEXT"
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "ix_samples_name_label_ts" in indexes
    finally:
        conn.close()


def test_schema_version_is_bumped_past_the_integer_era(tmp_path: Path) -> None:
    # The INTEGER-epoch era was SCHEMA_VERSION 1; the ISO era must be strictly newer
    # so an on-disk v1 file is detected as a shape mismatch and recreated.
    assert SCHEMA_VERSION >= 2
    conn = connect(metrics_db_path(tmp_path))
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_old_integer_shape_file_is_destructively_recreated(tmp_path: Path) -> None:
    # Simulate a pre-migration file: the OLD DDL (INTEGER ts/created_at/updated_at)
    # at schema_version 1, carrying a row. CREATE TABLE IF NOT EXISTS would keep the
    # INTEGER columns; the version-mismatch fresh-DB path must DROP and recreate.
    path = metrics_db_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        raw.execute("INSERT INTO schema_version (version) VALUES (1)")
        raw.execute(
            "CREATE TABLE metric_defs ("
            "  name TEXT PRIMARY KEY, kind TEXT NOT NULL, unit TEXT, help TEXT,"
            "  label_keys_json TEXT NOT NULL, export INTEGER NOT NULL DEFAULT 1,"
            "  created_at INTEGER, updated_at INTEGER)"
        )
        raw.execute(
            "CREATE TABLE metric_samples ("
            "  ts INTEGER NOT NULL, run_id TEXT NOT NULL, name TEXT NOT NULL,"
            "  label_key TEXT NOT NULL, value REAL NOT NULL, labels_json TEXT)"
        )
        raw.execute(
            "INSERT INTO metric_samples (ts, run_id, name, label_key, value, labels_json) "
            "VALUES (1000, 'OLD', 'g', '', 1.0, NULL)"
        )
        raw.commit()
    finally:
        raw.close()

    conn = connect(path)
    try:
        assert _col_types(conn, "metric_samples")["ts"] == "TEXT"
        assert _col_types(conn, "metric_defs")["created_at"] == "TEXT"
        # Destructive: the stale INTEGER row is gone, the file was recreated fresh.
        assert conn.execute("SELECT COUNT(*) FROM metric_samples").fetchone()[0] == 0
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# one ISO ts per sample cycle (whole-snapshot pruning stays intact)
# --------------------------------------------------------------------------- #


def test_one_iso_ts_per_cycle_shared_by_every_row(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("a").set(1.0)
    reg.gauge("b").set(2.0)
    reg.gauge("c").set(3.0)
    sampler = _sampler(reg, tmp_path)
    now = _ANCHOR + timedelta(seconds=5)
    try:
        written = sampler.sample_once(now=now)
    finally:
        sampler.close()
    assert written == 3
    stamps = {s.ts for s in read_samples(metrics_db_path(tmp_path))}
    # Exactly ONE ts for the whole cycle — µs did not split rows into cohorts.
    assert stamps == {to_iso(now)}


def test_later_cycle_gets_a_strictly_later_iso_ts(tmp_path: Path) -> None:
    reg = MetricRegistry()
    gauge = reg.gauge("g")
    sampler = _sampler(reg, tmp_path)
    try:
        gauge.set(1.0)
        sampler.sample_once(now=_ANCHOR + timedelta(seconds=1))
        gauge.set(2.0)
        sampler.sample_once(now=_ANCHOR + timedelta(seconds=2))
    finally:
        sampler.close()
    rows = read_samples(metrics_db_path(tmp_path), name="g")
    tss = [r.ts for r in rows]
    assert tss == [_iso(1), _iso(2)]
    assert tss == sorted(tss)  # TEXT order == chronological order


def test_read_samples_returns_iso_strings_not_ints(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(4.0)
    sampler = _sampler(reg, tmp_path)
    try:
        sampler.sample_once(now=_ANCHOR + timedelta(seconds=7))
    finally:
        sampler.close()
    (sample,) = read_samples(metrics_db_path(tmp_path), name="g")
    assert isinstance(sample.ts, str)
    assert sample.ts == _iso(7)
    # Round-trips through the strict parser → proves it is canonical, not a raw int.
    assert from_iso(sample.ts) == _ANCHOR + timedelta(seconds=7)


# --------------------------------------------------------------------------- #
# retention: cutoff = now - max_age, deleted where ts < cutoff (ISO compare)
# --------------------------------------------------------------------------- #


def _seed(conn: sqlite3.Connection, ts_iso: str, value: float) -> None:
    conn.execute(
        "INSERT INTO metric_samples (ts, run_id, name, label_key, value, labels_json) "
        "VALUES (?, ?, 'g', '', ?, NULL)",
        (ts_iso, _RUN, value),
    )


def test_prune_by_age_cuts_at_the_iso_boundary(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        for secs in (898, 899, 900, 901):
            _seed(conn, _iso(secs), float(secs))
        conn.commit()
        deleted = prune_metric_samples(
            conn,
            policy=MetricsRetentionPolicy(max_age_seconds=100, max_rows=None, max_bytes=None),
            now_iso=_iso(1000),  # cutoff = anchor+900; ts < that pruned, the boundary kept
        )
        remaining = sorted(r[0] for r in conn.execute("SELECT ts FROM metric_samples"))
        assert remaining == [_iso(900), _iso(901)]
        assert deleted == 2
    finally:
        conn.close()


def test_whole_snapshot_cohort_prune_removes_exactly_the_oldest_iso_ts(tmp_path: Path) -> None:
    # Two rows per cycle (a snapshot). max_rows must drop the WHOLE oldest ISO ts,
    # never a partial cohort — the per-cycle ISO string is the cohort key.
    conn = connect(metrics_db_path(tmp_path))
    try:
        for secs in (1, 2, 3):
            _seed(conn, _iso(secs), float(secs))
            _seed(conn, _iso(secs), float(secs) + 0.5)
        conn.commit()
        prune_metric_samples(
            conn,
            policy=MetricsRetentionPolicy(max_age_seconds=None, max_rows=3, max_bytes=None),
            now_iso=_iso(1000),
        )
        rows_per_ts = {
            ts: n for ts, n in conn.execute("SELECT ts, COUNT(*) FROM metric_samples GROUP BY ts")
        }
        assert rows_per_ts == {_iso(3): 2}  # only the newest cohort, intact
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# the sampler sources "now" from the injected clock (no system-time read)
# --------------------------------------------------------------------------- #


def test_sample_once_uses_injected_clock_when_no_now_given(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    clock = FakeClock(_ANCHOR + timedelta(seconds=42))
    sampler = _sampler(reg, tmp_path, clock=clock)
    try:
        sampler.sample_once()  # no explicit now → must read the injected clock
    finally:
        sampler.close()
    (sample,) = read_samples(metrics_db_path(tmp_path), name="g")
    assert sample.ts == _iso(42)


def test_daemon_thread_stamps_iso_from_injected_clock(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(7.0)
    clock = FakeClock(_ANCHOR + timedelta(seconds=99))
    sampler = MetricsSampler(
        reg, metrics_db_path(tmp_path), interval_seconds=0.01, run_id=_RUN, clock=clock
    )
    sampler.start()
    try:
        assert sampler.wait_first_sample(timeout=5.0)
    finally:
        sampler.stop()
    rows = read_samples(metrics_db_path(tmp_path), name="g")
    assert rows and rows[0].ts == _iso(99)


def test_sync_metric_defs_stores_iso_created_and_updated(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    sampler = _sampler(reg, tmp_path)
    try:
        sampler.sample_once(now=_ANCHOR + timedelta(seconds=3))
    finally:
        sampler.close()
    with sqlite3.connect(str(metrics_db_path(tmp_path))) as conn:
        created, updated = conn.execute(
            "SELECT created_at, updated_at FROM metric_defs WHERE name='g'"
        ).fetchone()
    assert created == _iso(3)
    assert updated == _iso(3)
    assert from_iso(created) == _ANCHOR + timedelta(seconds=3)  # canonical, not int

"""Tests for ``metrics.sqlite`` — the periodic ``MetricRegistry`` sampler (§4.4).

Contract under test (telemetry-core design §4.4/§7):

* the schema is created (``schema_version`` + ``metric_defs`` + ``metric_samples``
  + the ``(name, label_key, ts)`` index);
* :meth:`MetricsSampler.sample_once` snapshots the registry into ``metric_samples``
  rows carrying the process ``run_id``, and syncs ``metric_defs``;
* only ``export=1`` metrics are sampled (the def is still recorded, with export=0);
* skip-unchanged: an unchanged value writes no new row;
* heartbeat: every ``M`` cycles an unchanged value is re-written anyway;
* a ``Histogram`` decomposes into ``name_bucket{le}`` / ``name_count`` / ``name_sum``;
* retention prunes by age (at the boundary) and by max rows, fail-open;
* the sampler is a singleton-per-db-path, refcounted, and the daemon thread
  samples on its own.

Stdlib only; every test uses ``tmp_path`` and closes/stops what it opens.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lifemodel.core.metrics import MetricRegistry, MetricSpec
from lifemodel.state.metrics_store import (
    SCHEMA_VERSION,
    MetricsRetentionPolicy,
    MetricsSampler,
    acquire_metrics_sampler,
    connect,
    initialize_schema,
    metrics_db_path,
    process_run_id,
    prune_metric_samples,
    read_samples,
    release_metrics_sampler,
)

_RUN = "run-test-0001"


def _sampler(registry: MetricRegistry, tmp_path: Path, **kwargs: object) -> MetricsSampler:
    kwargs.setdefault("run_id", _RUN)
    kwargs.setdefault("heartbeat_every", 10_000)  # effectively disable forced writes
    return MetricsSampler(registry, metrics_db_path(tmp_path), **kwargs)  # type: ignore[arg-type]


def _read(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with sqlite3.connect(str(path)) as conn:
        return conn.execute(sql, params).fetchall()


def _seed_sample(conn: sqlite3.Connection, ts: int, value: float) -> None:
    conn.execute(
        "INSERT INTO metric_samples (ts, run_id, name, label_key, value, labels_json) "
        "VALUES (?, ?, 'g', '', ?, NULL)",
        (ts, _RUN, value),
    )


# --------------------------------------------------------------------------- #
# schema + path
# --------------------------------------------------------------------------- #


def test_metrics_db_path_is_sibling(tmp_path: Path) -> None:
    assert metrics_db_path(tmp_path) == tmp_path / "metrics.sqlite"


def test_connect_creates_schema_and_version(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"schema_version", "metric_defs", "metric_samples"} <= tables
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "ix_samples_name_label_ts" in indexes
    finally:
        conn.close()


def test_initialize_schema_is_idempotent(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        initialize_schema(conn)  # again — must not duplicate the version row
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# sampling: rows carry run_id, defs are synced
# --------------------------------------------------------------------------- #


def test_sample_writes_counter_and_gauge_rows_with_run_id(tmp_path: Path) -> None:
    reg = MetricRegistry()
    counter = reg.counter("lifemodel_runs_total", label_keys=("component",))
    gauge = reg.gauge("lifemodel_drive_u")
    counter.inc(component="neuron")
    counter.inc(2.0, component="neuron")
    gauge.set(0.5)

    sampler = _sampler(reg, tmp_path)
    try:
        sampler.sample_once(ts=1000)
    finally:
        sampler.close()

    by = {(s.name, s.label_key): s for s in read_samples(metrics_db_path(tmp_path))}
    run = by[("lifemodel_runs_total", "component=neuron")]
    assert run.value == 3.0
    assert run.run_id == _RUN
    assert run.ts == 1000
    assert run.labels == {"component": "neuron"}
    assert by[("lifemodel_drive_u", "")].value == 0.5

    # metric_defs recorded for both, with kind + label_keys_json.
    kinds = dict(_read(metrics_db_path(tmp_path), "SELECT name, kind FROM metric_defs"))
    assert kinds["lifemodel_runs_total"] == "counter"
    assert kinds["lifemodel_drive_u"] == "gauge"


def test_only_exported_metrics_are_sampled_but_all_defs_recorded(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="secret_gauge", kind="gauge", export=False))
    reg.set("secret_gauge", 9.0)
    reg.gauge("public_gauge").set(1.0)

    sampler = _sampler(reg, tmp_path)
    try:
        sampler.sample_once(ts=1000)
    finally:
        sampler.close()

    names = {s.name for s in read_samples(metrics_db_path(tmp_path))}
    assert "public_gauge" in names
    assert "secret_gauge" not in names  # export=0 → not sampled
    # …but its def IS recorded, flagged export=0.
    assert _read(
        metrics_db_path(tmp_path), "SELECT export FROM metric_defs WHERE name=?", ("secret_gauge",)
    ) == [(0,)]
    assert _read(
        metrics_db_path(tmp_path), "SELECT export FROM metric_defs WHERE name=?", ("public_gauge",)
    ) == [(1,)]


# --------------------------------------------------------------------------- #
# skip-unchanged + heartbeat
# --------------------------------------------------------------------------- #


def test_unchanged_value_is_not_re_sampled(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    sampler = _sampler(reg, tmp_path, heartbeat_every=10_000)
    try:
        sampler.sample_once(ts=1000)
        sampler.sample_once(ts=1001)  # value unchanged, no heartbeat → skipped
    finally:
        sampler.close()
    rows = read_samples(metrics_db_path(tmp_path), name="g")
    assert [r.ts for r in rows] == [1000]


def test_changed_value_writes_a_new_row(tmp_path: Path) -> None:
    reg = MetricRegistry()
    gauge = reg.gauge("g")
    gauge.set(1.0)
    sampler = _sampler(reg, tmp_path, heartbeat_every=10_000)
    try:
        sampler.sample_once(ts=1000)
        gauge.set(2.0)
        sampler.sample_once(ts=1001)
    finally:
        sampler.close()
    rows = sorted(read_samples(metrics_db_path(tmp_path), name="g"), key=lambda s: s.ts)
    assert [(r.ts, r.value) for r in rows] == [(1000, 1.0), (1001, 2.0)]


def test_heartbeat_forces_write_every_m_cycles(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    sampler = _sampler(reg, tmp_path, heartbeat_every=2)
    try:
        for ts in (1000, 1001, 1002):  # cycle 0 (force), 1 (skip), 2 (force)
            sampler.sample_once(ts=ts)
    finally:
        sampler.close()
    rows = sorted(read_samples(metrics_db_path(tmp_path), name="g"), key=lambda s: s.ts)
    assert [r.ts for r in rows] == [1000, 1002]  # 1001 skipped, 1002 re-written by heartbeat


# --------------------------------------------------------------------------- #
# histogram decomposition
# --------------------------------------------------------------------------- #


def test_histogram_decomposes_into_bucket_count_sum(tmp_path: Path) -> None:
    reg = MetricRegistry()
    hist = reg.histogram("lat_seconds", buckets=(0.1, 1.0))
    hist.observe(0.05)  # <= 0.1 and <= 1.0
    hist.observe(0.5)  # <= 1.0
    hist.observe(2.0)  # overflow (> last bound): only count + sum

    sampler = _sampler(reg, tmp_path)
    try:
        sampler.sample_once(ts=1000)
    finally:
        sampler.close()

    by = {(s.name, s.label_key): s for s in read_samples(metrics_db_path(tmp_path))}
    assert by[("lat_seconds_bucket", "le=0.1")].value == 1.0  # cumulative: only 0.05
    assert by[("lat_seconds_bucket", "le=1.0")].value == 2.0  # cumulative: 0.05 + 0.5
    assert by[("lat_seconds_bucket", "le=0.1")].labels == {"le": "0.1"}
    assert by[("lat_seconds_count", "")].value == 3.0
    assert by[("lat_seconds_sum", "")].value == pytest.approx(2.55)


# --------------------------------------------------------------------------- #
# retention (pure prune over a test connection)
# --------------------------------------------------------------------------- #


def test_prune_by_age_cuts_at_the_boundary(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        for ts in (898, 899, 900, 901):
            _seed_sample(conn, ts, float(ts))
        conn.commit()
        deleted = prune_metric_samples(
            conn,
            policy=MetricsRetentionPolicy(max_age_seconds=100, max_rows=None, max_bytes=None),
            now_ts=1000,  # cutoff = 900; ts < 900 pruned, ts >= 900 kept
        )
        remaining = sorted(int(r[0]) for r in conn.execute("SELECT ts FROM metric_samples"))
        assert remaining == [900, 901]
        assert deleted == 2
    finally:
        conn.close()


def test_prune_by_max_rows_keeps_newest(tmp_path: Path) -> None:
    conn = connect(metrics_db_path(tmp_path))
    try:
        for ts in range(1, 6):
            _seed_sample(conn, ts, float(ts))
        conn.commit()
        prune_metric_samples(
            conn,
            policy=MetricsRetentionPolicy(max_age_seconds=None, max_rows=2, max_bytes=None),
            now_ts=1000,
        )
        remaining = sorted(int(r[0]) for r in conn.execute("SELECT ts FROM metric_samples"))
        assert remaining == [4, 5]  # only the two newest survive
    finally:
        conn.close()


def test_prune_by_max_rows_drops_whole_ts_snapshots(tmp_path: Path) -> None:
    # Two rows per ts (a snapshot). max_rows must drop a WHOLE oldest snapshot,
    # never leave a ts with only some of its rows (design §4.4: retention целыми).
    conn = connect(metrics_db_path(tmp_path))
    try:
        for ts in (1, 2, 3):
            _seed_sample(conn, ts, float(ts))
            _seed_sample(conn, ts, float(ts) + 0.5)
        conn.commit()
        prune_metric_samples(
            conn,
            policy=MetricsRetentionPolicy(max_age_seconds=None, max_rows=3, max_bytes=None),
            now_ts=1000,
        )
        rows_per_ts = {
            int(ts): int(n)
            for ts, n in conn.execute("SELECT ts, COUNT(*) FROM metric_samples GROUP BY ts")
        }
        # 6 rows > 3: drop whole ts=1 (→4), still >3: drop whole ts=2 (→2) ≤3 stop.
        assert rows_per_ts == {3: 2}  # only the newest snapshot, intact — never split
    finally:
        conn.close()


def test_read_samples_latest_run_and_limit(tmp_path: Path) -> None:
    path = metrics_db_path(tmp_path)
    conn = connect(path)
    try:
        conn.execute(
            "INSERT INTO metric_samples (ts, run_id, name, label_key, value, labels_json) "
            "VALUES (10,'OLD','g','',1.0,NULL),(1000,'NEW','g','',2.0,NULL),"
            "(1060,'NEW','g','',3.0,NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    # latest_run excludes the OLD run entirely (rates are per-run anyway).
    assert {s.run_id for s in read_samples(path, latest_run=True)} == {"NEW"}
    # limit bounds the read to the most-recent rows, presented oldest-first.
    assert [s.ts for s in read_samples(path, limit=2)] == [1000, 1060]


# --------------------------------------------------------------------------- #
# lifecycle: singleton, refcount, daemon
# --------------------------------------------------------------------------- #


def test_process_run_id_is_stable(tmp_path: Path) -> None:
    assert process_run_id() == process_run_id()


def test_acquire_is_singleton_per_path_and_refcounted(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    s1 = acquire_metrics_sampler(reg, tmp_path, interval_seconds=0.01, run_id=_RUN)
    try:
        s2 = acquire_metrics_sampler(reg, tmp_path)
        assert s1 is s2  # same instance for the same path
    finally:
        release_metrics_sampler(tmp_path)  # refcount 2 -> 1, still alive
    assert s1._thread is not None and s1._thread.is_alive()
    release_metrics_sampler(tmp_path)  # refcount 1 -> 0, stopped
    assert s1._thread is None or not s1._thread.is_alive()


def test_daemon_thread_samples_and_writes_rows(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(7.0)
    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), interval_seconds=0.01, run_id=_RUN)
    sampler.start()
    try:
        assert sampler.wait_first_sample(timeout=5.0)  # event-driven, no fixed sleep
    finally:
        sampler.stop()
    rows = read_samples(metrics_db_path(tmp_path), name="g")
    assert rows and rows[0].value == 7.0
    assert rows[0].run_id == _RUN


def test_start_is_idempotent(tmp_path: Path) -> None:
    reg = MetricRegistry()
    reg.gauge("g").set(1.0)
    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), interval_seconds=0.01, run_id=_RUN)
    sampler.start()
    thread = sampler._thread
    try:
        sampler.start()  # must NOT spawn a second thread
        assert sampler._thread is thread
    finally:
        sampler.stop()

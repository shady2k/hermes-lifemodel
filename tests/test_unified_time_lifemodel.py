"""Acceptance tests for SLICE 2 of the unified-time epic (lm-fib.10.2).

``lifemodel.sqlite`` stores every instant ONCE, as normalized ISO-8601 UTC TEXT
(spec ``docs/superpowers/specs/2026-07-11-unified-time-single-helper-iso-design.md``
§4 + §6 items 1/3). The epoch mirror columns are gone; ISO TEXT is the
ordering/expiry key; every ``_at`` value is passed through
:func:`~lifemodel.core.timeutil.to_iso` BEFORE storage (normalize-on-write), so
no raw caller string ever reaches a column.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.timeutil import from_iso, to_iso
from lifemodel.domain.memory import MemoryDraft, PressureIndex, summarize_pressure_index
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock, FakeMemoryStore

BASE_TIME = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
DB_FILENAME = "lifemodel.sqlite"

_EPOCH_COLUMNS = {"created_at_epoch", "updated_at_epoch", "expires_at_epoch"}


def _draft(**overrides: object) -> MemoryDraft:
    base = dict(kind="desire", id="d1", state="active", payload={"note": "hi"}, source="test")
    base.update(overrides)
    return MemoryDraft(**base)  # type: ignore[arg-type]


def _columns(base_dir: Path, table: str) -> set[str]:
    with closing(sqlite3.connect(str(base_dir / DB_FILENAME))) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(base_dir: Path) -> set[str]:
    with closing(sqlite3.connect(str(base_dir / DB_FILENAME))) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


# ---- DDL: no epoch columns, ISO index (acceptance §6.1) ---------------------


def test_memory_records_has_no_epoch_columns(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    cols = _columns(tmp_path, "memory_records")
    assert cols & _EPOCH_COLUMNS == set()
    assert {"expires_at", "created_at", "updated_at"} <= cols


def test_runtime_state_has_no_epoch_columns(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    cols = _columns(tmp_path, "runtime_state")
    assert "updated_at_epoch" not in cols
    assert "updated_at" in cols


def test_expires_at_index_is_on_the_text_column(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    indexes = _indexes(tmp_path)
    assert "idx_memory_records_expires_at" in indexes
    assert "idx_memory_records_expires_at_epoch" not in indexes


# ---- normalize-on-write (spec §4 codex #1, acceptance §6.5/§6 round-trip) ----


def test_created_and_updated_at_stored_as_normalized_iso(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())

    with closing(sqlite3.connect(str(tmp_path / DB_FILENAME))) as conn:
        created_at, updated_at = conn.execute(
            "SELECT created_at, updated_at FROM memory_records WHERE id = 'd1'"
        ).fetchone()

    assert created_at == to_iso(BASE_TIME)
    assert updated_at == to_iso(BASE_TIME)
    # fixed-width normalization is load-bearing for TEXT ordering
    assert created_at == "2026-07-06T12:00:00.000000+00:00"


def test_caller_expires_at_is_normalized_to_utc_before_storage(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # A +03:00 offset with whole-second precision — a raw store would corrupt
    # TEXT ordering/expiry. It must land as normalized UTC via to_iso.
    store.put(_draft(expires_at="2026-07-11T12:00:00+03:00"))

    record = store.get("desire", "d1")
    assert record is not None
    assert record.expires_at == "2026-07-11T09:00:00.000000+00:00"

    with closing(sqlite3.connect(str(tmp_path / DB_FILENAME))) as conn:
        (raw,) = conn.execute("SELECT expires_at FROM memory_records WHERE id = 'd1'").fetchone()
    assert raw == "2026-07-11T09:00:00.000000+00:00"


def test_null_expires_at_stays_null(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.put(_draft(expires_at=None))
    with closing(sqlite3.connect(str(tmp_path / DB_FILENAME))) as conn:
        (raw,) = conn.execute("SELECT expires_at FROM memory_records WHERE id = 'd1'").fetchone()
    assert raw is None


# ---- ordering by normalized TEXT (acceptance §6.3) --------------------------


def test_updated_desc_orders_by_normalized_text_across_subsecond(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # Three rows a few MICROSECONDS apart — only a fixed-width TEXT column sorts
    # these right; an un-padded ".isoformat()" would misorder whole-second rows.
    store.put(_draft(id="a"))
    clock.advance(timedelta(microseconds=1))
    store.put(_draft(id="b"))
    clock.advance(timedelta(microseconds=1))
    store.put(_draft(id="c"))

    assert [r.id for r in store.find(order_by="updated_desc")] == ["c", "b", "a"]


def test_created_desc_orders_by_normalized_text_across_subsecond(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft(id="a"))
    clock.advance(timedelta(microseconds=500))
    store.put(_draft(id="b"))
    clock.advance(timedelta(microseconds=500))
    store.put(_draft(id="c"))

    assert [r.id for r in store.find(order_by="created_desc")] == ["c", "b", "a"]


# ---- expiry / pressure boundary preserved (acceptance §6.3) -----------------


def test_read_pressure_index_boundary_is_strict_gt(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    now = clock.now()
    # exactly-at-now == expired (<= now); strictly-after-now == active (> now)
    store.put(_draft(id="at_now", salience=0.9, expires_at=to_iso(now)))
    store.put(_draft(id="after", salience=0.4, expires_at=to_iso(now + timedelta(microseconds=1))))

    index = store.read_pressure_index(now)
    assert index.active_desire_count == 1
    assert index.max_desire_salience == 0.4


def test_summarize_matches_read_pressure_index_on_same_fixtures(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    real = SQLiteRuntimeStore(tmp_path, clock=clock)
    fake = FakeMemoryStore(clock=clock)
    now = clock.now()
    fixtures = [
        _draft(id="expired", salience=0.9, expires_at=to_iso(now - timedelta(minutes=1))),
        _draft(id="boundary", salience=0.8, expires_at=to_iso(now)),
        _draft(id="alive", salience=0.5, expires_at=to_iso(now + timedelta(minutes=1))),
        _draft(id="never", salience=0.3, expires_at=None),
    ]
    for draft in fixtures:
        real.put(draft)
        fake.put(draft)

    from_sql = real.read_pressure_index(now)
    from_python = fake.read_pressure_index(now)
    assert from_sql == from_python
    assert from_sql == PressureIndex(
        active_desire_count=2, max_desire_salience=0.5, contact_frame_available=True
    )


def test_summarize_pressure_index_boundary_is_strict_gt() -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    at_now = _record(id="at_now", expires_at=to_iso(now))
    after = _record(id="after", expires_at=to_iso(now + timedelta(microseconds=1)))
    idx = summarize_pressure_index([at_now, after], now)
    assert idx.active_desire_count == 1


def _record(**overrides: object) -> object:
    from lifemodel.domain.memory import MemoryRecord

    base = dict(
        kind="desire",
        id="d1",
        state="active",
        payload={},
        source="test",
        recipient_id="owner",
        salience=0.0,
        confidence=None,
        expires_at=None,
        created_at=to_iso(BASE_TIME),
        updated_at=to_iso(BASE_TIME),
        revision=0,
        schema_version=1,
    )
    base.update(overrides)
    return MemoryRecord(**base)  # type: ignore[arg-type]


# ---- destructive fresh-DB path on a superseded shape (spec §4 codex #4/#10) --


def test_old_shape_db_is_reset_fresh_on_construction(tmp_path: Path) -> None:
    db_path = tmp_path / DB_FILENAME
    # Hand-build a DB in the OLD shape: memory_records + runtime_state carrying
    # the retired *_epoch columns, migrations 1+2 recorded, and a store_meta with
    # NO schema_version key (exactly what a pre-cutover file looks like).
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE memory_records ("
            "kind TEXT NOT NULL, id TEXT NOT NULL, state TEXT NOT NULL, "
            "recipient_id TEXT NOT NULL DEFAULT 'owner', payload_json TEXT NOT NULL, "
            "salience REAL NOT NULL DEFAULT 0, confidence REAL, expires_at TEXT, "
            "expires_at_epoch INTEGER, source TEXT NOT NULL, created_at TEXT NOT NULL, "
            "created_at_epoch INTEGER NOT NULL, updated_at TEXT NOT NULL, "
            "updated_at_epoch INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0, "
            "schema_version INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (kind, id))"
        )
        conn.execute(
            "CREATE TABLE runtime_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
            "state_json TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "updated_at_epoch INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            [(1, "2026-01-01T00:00:00+00:00"), (2, "2026-01-01T00:00:00+00:00")],
        )

    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    # The stale file was moved aside (forensics) and a fresh, new-shape DB built.
    assert list(tmp_path.glob(f"{DB_FILENAME}.superseded.*"))
    assert _columns(tmp_path, "memory_records") & _EPOCH_COLUMNS == set()
    assert "updated_at_epoch" not in _columns(tmp_path, "runtime_state")


def test_round_trip_from_iso_of_stored_created_at(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())
    record = store.get("desire", "d1")
    assert record is not None
    assert from_iso(record.created_at) == BASE_TIME
    assert from_iso(record.updated_at) == BASE_TIME

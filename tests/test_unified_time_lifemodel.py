"""Acceptance tests for SLICE 2 of the unified-time epic (lm-fib.10.2).

``lifemodel.sqlite`` stores every instant ONCE, as normalized ISO-8601 UTC TEXT
(spec ``docs/superpowers/specs/2026-07-11-unified-time-single-helper-iso-design.md``
§4 + §6 items 1/3). The epoch mirror columns are gone; ISO TEXT is the
ordering/expiry key; every ``_at`` value is passed through
:func:`~lifemodel.core.timeutil.to_iso` BEFORE storage (normalize-on-write), so
no raw caller string ever reaches a column.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lifemodel.core.timeutil import from_iso, to_iso
from lifemodel.domain.memory import MemoryDraft, PressureIndex, summarize_pressure_index
from lifemodel.state.model import State
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


# ---- non-destructive MIGRATION of an old dual-column shape (lm-fib.10.5) ----
# PRINCIPLE: MIGRATE THE SELF, RECREATE DERIVED. lifemodel.sqlite IS the being's
# self (drive u, energy, memory records, the UserModel/relationship) — a schema
# change must PRESERVE it, not reset it. (Reset stays only for genuine corruption.)


def _build_old_shape_db(
    db_path: Path,
    *,
    state_json: str,
    mem_created_at: str,
    mem_updated_at: str,
    mem_expires_at: str | None = None,
) -> None:
    """Hand-build a pre-10.2 lifemodel.sqlite: memory_records + runtime_state carrying
    the retired ``*_epoch`` mirror columns, with migrations 1+2 recorded (so v3 is the
    single pending one) and seeded data whose ISO stamps lack the fixed-µs width."""
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO store_meta (key, value) VALUES ('schema_version', '2')")
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
        conn.execute("CREATE INDEX idx_memory_records_kind_state ON memory_records (kind, state)")
        conn.execute(
            "CREATE INDEX idx_memory_records_expires_at_epoch ON memory_records (expires_at_epoch)"
        )
        conn.execute(
            "INSERT INTO memory_records (kind, id, state, recipient_id, payload_json, salience, "
            "confidence, expires_at, expires_at_epoch, source, created_at, created_at_epoch, "
            "updated_at, updated_at_epoch, revision, schema_version) "
            "VALUES ('desire','d1','active','owner',?,0.5,NULL,?,NULL,'seed',?,?,?,?,0,1)",
            (
                '{"note": "hi"}',
                mem_expires_at,
                mem_created_at,
                1_752_000_000,
                mem_updated_at,
                1_752_000_000,
            ),
        )
        conn.execute(
            "CREATE TABLE runtime_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
            "state_json TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "updated_at_epoch INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, updated_at_epoch, revision) "
            "VALUES (1, ?, '2026-07-11T12:00:00+00:00', 1752000000, 4)",
            (state_json,),
        )
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            [(1, "2026-01-01T00:00:00+00:00"), (2, "2026-01-01T00:00:00+00:00")],
        )


def test_old_shape_db_migrates_preserving_the_self(tmp_path: Path) -> None:
    db_path = tmp_path / DB_FILENAME
    # The being's self, seeded into the OLD-shape file: a known u/energy in
    # runtime_state and a memory record with an UN-normalized created_at (no µs).
    old_state = State(u=2.5, energy=0.7, tick_count=42)
    _build_old_shape_db(
        db_path,
        state_json=json.dumps(old_state.to_dict()),
        mem_created_at="2026-07-11T12:00:00+00:00",  # lacks the fixed 6-µs width
        mem_updated_at="2026-07-11T12:00:00+00:00",
    )

    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    # The self was PRESERVED, not wiped: u/energy unchanged.
    loaded = store.load()
    assert loaded.u == 2.5
    assert loaded.energy == 0.7
    assert loaded.tick_count == 42

    # The memory record survived and its created_at is now NORMALIZED fixed-width ISO.
    record = store.get("desire", "d1")
    assert record is not None
    assert record.created_at == "2026-07-11T12:00:00.000000+00:00"
    assert record.updated_at == "2026-07-11T12:00:00.000000+00:00"
    assert json.loads(json.dumps(record.payload)) == {"note": "hi"}

    # The shape is now ISO-only, the ISO index exists, and NOTHING was moved aside
    # (no destructive .superseded.* reset) — a *.bak.* backup was taken instead.
    assert _columns(tmp_path, "memory_records") & _EPOCH_COLUMNS == set()
    assert "updated_at_epoch" not in _columns(tmp_path, "runtime_state")
    assert "idx_memory_records_expires_at" in _indexes(tmp_path)
    assert not any("epoch" in name for name in _indexes(tmp_path))
    assert list(tmp_path.glob(f"{DB_FILENAME}.superseded.*")) == []
    assert len(list(tmp_path.glob(f"{DB_FILENAME}.bak.*"))) == 1

    # quick_check passes and schema_migrations records the new versions.
    with closing(sqlite3.connect(str(db_path))) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert versions == [1, 2, 3, 4]


def test_ordering_and_expiry_correct_after_migration(tmp_path: Path) -> None:
    # A MIGRATED record (its ISO stamps re-normalized to fixed width) must sort and
    # expire correctly ALONGSIDE a freshly-written one — the whole point of
    # re-normalizing on migration (lexical TEXT order == chronological only at fixed µs).
    db_path = tmp_path / DB_FILENAME
    _build_old_shape_db(
        db_path,
        state_json=json.dumps(State(u=1.0).to_dict()),
        mem_created_at="2026-07-11T10:00:00+00:00",  # migrated desire d1, un-normalized
        mem_updated_at="2026-07-11T10:00:00+00:00",
        mem_expires_at="2026-07-11T18:00:00+00:00",  # a future expiry, un-normalized
    )

    at_noon = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(at_noon))
    # A freshly-written desire, stamped by the store's clock (later than d1, normalized).
    store.put(MemoryDraft(kind="desire", id="d2", state="active", payload={}, source="fresh"))

    # Ordering: created_desc puts the newer d2 (12:00) before the migrated d1 (10:00);
    # both created_at values are now fixed-width, so the TEXT sort is chronological.
    ordered = [r.id for r in store.find(kind="desire", order_by="created_desc")]
    assert ordered == ["d2", "d1"]
    d1 = store.get("desire", "d1")
    assert d1 is not None
    assert d1.created_at == "2026-07-11T10:00:00.000000+00:00"

    # Expiry: at noon d1 (expires 18:00) is still active -> both desires pressure;
    # after d1's expiry only d2 (no expiry) remains active.
    assert store.read_pressure_index(at_noon).active_desire_count == 2
    after_expiry = datetime(2026, 7, 11, 19, 0, tzinfo=UTC)
    assert store.read_pressure_index(after_expiry).active_desire_count == 1


def test_migration_fails_loud_on_unnormalizable_time_value(tmp_path: Path) -> None:
    # FAIL-CLOSED on the WRITE path (codex MAJOR): a legacy value that cannot be
    # normalized to fixed-width ISO must NEVER be persisted raw — it would silently rot
    # the lexical ordering/expiry invariant forever ('not-a-timestamp' > '2026-…' is
    # true, an immortal desire) and, since v3 is then recorded, never be revisited. So
    # the migration RAISES; the framework restores the *.bak.* backup and the being's
    # self stays intact on disk. (Contrast core/timeutil.to_display — the READ path —
    # is fail-OPEN.) It can't happen for data our own code wrote, so failing loud here
    # is the right stop, not a silent shrug.
    db_path = tmp_path / DB_FILENAME
    old_state = State(u=2.5, energy=0.7, tick_count=42)
    expected_state_json = json.dumps(old_state.to_dict())
    _build_old_shape_db(
        db_path,
        state_json=expected_state_json,
        mem_created_at="2026-07-11T12:00:00+00:00",
        mem_updated_at="2026-07-11T12:00:00+00:00",
        mem_expires_at="not-a-timestamp",  # unparseable -> fail CLOSED, loud
    )

    with pytest.raises(ValueError, match="cannot normalize"):
        SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    # The self was PRESERVED on disk (backup restored to the original path), NOT wiped
    # and NOT half-migrated: the file is back to the OLD shape and no raw value ever
    # reached an ISO-only column. v3 was rolled back (not recorded).
    assert _columns(tmp_path, "memory_records") & _EPOCH_COLUMNS == _EPOCH_COLUMNS
    assert "updated_at_epoch" in _columns(tmp_path, "runtime_state")
    assert list(tmp_path.glob(f"{DB_FILENAME}.superseded.*")) == []
    with closing(sqlite3.connect(str(db_path))) as conn:
        (state_json,) = conn.execute("SELECT state_json FROM runtime_state").fetchone()
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert state_json == expected_state_json  # self intact, byte-for-byte
    assert versions == [1, 2]  # v3 rolled back by the restore


def test_migration_removes_stale_store_meta_schema_version(tmp_path: Path) -> None:
    # NIT (codex): a migrated DB must be indistinguishable from a freshly-bootstrapped
    # one, whose store_meta carries NO schema_version key (schema_migrations is the sole
    # version authority since lm-fib.10.5). The pre-cutover marker left by the retired
    # _stamp_store_schema_version guard is dropped on migration.
    db_path = tmp_path / DB_FILENAME
    _build_old_shape_db(
        db_path,
        state_json=json.dumps(State(u=1.0).to_dict()),
        mem_created_at="2026-07-11T12:00:00+00:00",
        mem_updated_at="2026-07-11T12:00:00+00:00",
    )
    # sanity: the old file DID carry the stale marker (see _build_old_shape_db)
    with closing(sqlite3.connect(str(db_path))) as conn:
        assert conn.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone() == ("2",)

    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute("SELECT value FROM store_meta WHERE key = 'schema_version'").fetchone()
    assert row is None  # stale marker gone -> migrated == fresh


def test_migration_preserves_multiple_rows_and_is_idempotent_on_normalized(
    tmp_path: Path,
) -> None:
    # Codex test-gap: several memory rows — one whose legacy stamp ALREADY has the fixed
    # µs width (normalization must be idempotent, not double-shift it), one NON-NULL
    # un-normalized expires_at, one NULL expires_at (stays NULL). All survive migration,
    # all land normalized.
    db_path = tmp_path / DB_FILENAME
    _build_old_shape_db(
        db_path,
        state_json=json.dumps(State(u=3.0).to_dict()),
        mem_created_at="2026-07-11T09:00:00.500000+00:00",  # already fixed-width µs
        mem_updated_at="2026-07-11T09:00:00.500000+00:00",
        mem_expires_at="2026-07-11T20:00:00+00:00",  # non-null, un-normalized
    )
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            "INSERT INTO memory_records (kind, id, state, recipient_id, payload_json, salience, "
            "confidence, expires_at, expires_at_epoch, source, created_at, created_at_epoch, "
            "updated_at, updated_at_epoch, revision, schema_version) "
            "VALUES ('desire','d2','active','owner','{}',0.4,NULL,NULL,NULL,'seed',"
            "'2026-07-11T08:00:00+00:00',1752000000,'2026-07-11T08:00:00+00:00',1752000000,0,1)"
        )
        conn.execute(
            "INSERT INTO memory_records (kind, id, state, recipient_id, payload_json, salience, "
            "confidence, expires_at, expires_at_epoch, source, created_at, created_at_epoch, "
            "updated_at, updated_at_epoch, revision, schema_version) "
            "VALUES ('opinion','o1','active','owner','{}',0.2,NULL,'2026-07-11T22:00:00+00:00',"
            "NULL,'seed','2026-07-11T07:00:00+00:00',1752000000,"
            "'2026-07-11T07:00:00+00:00',1752000000,0,1)"
        )

    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    d1 = store.get("desire", "d1")
    d2 = store.get("desire", "d2")
    o1 = store.get("opinion", "o1")
    assert d1 is not None and d2 is not None and o1 is not None
    # already-µs value is unchanged (idempotent); un-normalized ones gain fixed width
    assert d1.created_at == "2026-07-11T09:00:00.500000+00:00"
    assert d1.expires_at == "2026-07-11T20:00:00.000000+00:00"
    assert d2.expires_at is None  # NULL stayed NULL
    assert o1.expires_at == "2026-07-11T22:00:00.000000+00:00"
    assert {r.id for r in store.find()} == {"d1", "d2", "o1"}  # every row survived
    assert store.load().u == 3.0  # self preserved


def _build_epoch_scaffold(conn: sqlite3.Connection) -> None:
    """Create ``store_meta`` + ``schema_migrations`` (v1/v2 recorded) shared by the
    empty-tables and partial-epoch old-shape fixtures below."""
    conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
        [(1, "2026-01-01T00:00:00+00:00"), (2, "2026-01-01T00:00:00+00:00")],
    )


_OLD_MEMORY_DDL = (
    "CREATE TABLE memory_records ("
    "kind TEXT NOT NULL, id TEXT NOT NULL, state TEXT NOT NULL, "
    "recipient_id TEXT NOT NULL DEFAULT 'owner', payload_json TEXT NOT NULL, "
    "salience REAL NOT NULL DEFAULT 0, confidence REAL, expires_at TEXT, "
    "expires_at_epoch INTEGER, source TEXT NOT NULL, created_at TEXT NOT NULL, "
    "created_at_epoch INTEGER NOT NULL, updated_at TEXT NOT NULL, "
    "updated_at_epoch INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0, "
    "schema_version INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (kind, id))"
)
_OLD_RUNTIME_DDL = (
    "CREATE TABLE runtime_state (id INTEGER PRIMARY KEY CHECK (id = 1), "
    "state_json TEXT NOT NULL, updated_at TEXT NOT NULL, "
    "updated_at_epoch INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0)"
)


def test_migration_handles_empty_old_shape_tables(tmp_path: Path) -> None:
    # Codex test-gap: an old-shape file with NO rows still migrates to the ISO-only
    # shape (empty rebuild) and records v3 (+ v4) — no crash on empty SELECT/executemany.
    db_path = tmp_path / DB_FILENAME
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        _build_epoch_scaffold(conn)
        conn.execute(_OLD_MEMORY_DDL)
        conn.execute(_OLD_RUNTIME_DDL)  # empty runtime_state (no self row yet)

    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    assert _columns(tmp_path, "memory_records") & _EPOCH_COLUMNS == set()
    assert "updated_at_epoch" not in _columns(tmp_path, "runtime_state")
    with closing(sqlite3.connect(str(db_path))) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
        (mem_count,) = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()
    assert versions == [1, 2, 3, 4]
    assert mem_count == 0  # empty rebuild preserved zero rows


def test_migration_rebuilds_only_the_table_that_still_has_epoch_columns(tmp_path: Path) -> None:
    # Codex test-gap: v3 checks each table independently. Build a file where
    # memory_records is ALREADY iso-only but runtime_state still carries
    # updated_at_epoch -> only runtime_state is rebuilt; the memory row is untouched.
    db_path = tmp_path / DB_FILENAME
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        _build_epoch_scaffold(conn)
        conn.execute(
            "CREATE TABLE memory_records ("
            "kind TEXT NOT NULL, id TEXT NOT NULL, state TEXT NOT NULL, "
            "recipient_id TEXT NOT NULL DEFAULT 'owner', payload_json TEXT NOT NULL, "
            "salience REAL NOT NULL DEFAULT 0, confidence REAL, expires_at TEXT, "
            "source TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "revision INTEGER NOT NULL DEFAULT 0, schema_version INTEGER NOT NULL DEFAULT 1, "
            "PRIMARY KEY (kind, id))"
        )
        conn.execute("CREATE INDEX idx_memory_records_kind_state ON memory_records (kind, state)")
        conn.execute("CREATE INDEX idx_memory_records_expires_at ON memory_records (expires_at)")
        conn.execute(
            "INSERT INTO memory_records (kind,id,state,recipient_id,payload_json,salience,"
            "confidence,expires_at,source,created_at,updated_at,revision,schema_version) "
            "VALUES ('desire','d1','active','owner','{}',0.5,NULL,NULL,'seed',"
            "'2026-07-11T12:00:00.000000+00:00','2026-07-11T12:00:00.000000+00:00',0,1)"
        )
        conn.execute(_OLD_RUNTIME_DDL)
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, updated_at_epoch, revision) "
            "VALUES (1, ?, '2026-07-11T12:00:00+00:00', 1752000000, 4)",
            (json.dumps(State(u=1.5).to_dict()),),
        )

    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    assert "updated_at_epoch" not in _columns(tmp_path, "runtime_state")  # rebuilt
    assert store.load().u == 1.5  # self preserved through the runtime_state rebuild
    record = store.get("desire", "d1")
    assert record is not None  # the already-iso memory row was left untouched
    assert record.created_at == "2026-07-11T12:00:00.000000+00:00"


def test_round_trip_from_iso_of_stored_created_at(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())
    record = store.get("desire", "d1")
    assert record is not None
    assert from_iso(record.created_at) == BASE_TIME
    assert from_iso(record.updated_at) == BASE_TIME

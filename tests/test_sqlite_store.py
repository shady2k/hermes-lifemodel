"""SQLite-only tests for :class:`SQLiteRuntimeStore` (lm-fib.6.1/6.2, HLA §4.1/D7).

Contract behavior shared with the fake lives in ``test_memory_contract.py``;
this file covers what only a real database file can exercise: corruption
recovery, migrations, epoch/PRAGMA storage details, fail-soft, and — since
lm-fib.6.2 — the ``StatePort`` cutover (``runtime_state``, the v2 migration).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import lifemodel.state.sqlite_store as sqlite_store_module
from lifemodel.domain.memory import MemoryDraft, PressureIndex
from lifemodel.state.errors import StateCorruptError, StateSchemaError, StateSerializationError
from lifemodel.state.model import SCHEMA_VERSION, State
from lifemodel.state.port import StatePort
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock

BASE_TIME = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
DB_FILENAME = "lifemodel.sqlite"


def _draft(**overrides: object) -> MemoryDraft:
    base = dict(kind="desire", id="d1", state="active", payload={"note": "hi"}, source="test")
    base.update(overrides)
    return MemoryDraft(**base)  # type: ignore[arg-type]


def _db_path(base_dir: Path) -> Path:
    return base_dir / DB_FILENAME


# ---- recovery ---------------------------------------------------------------


def test_recovery_quarantines_garbage_trio_and_bootstraps_fresh(tmp_path: Path) -> None:
    _db_path(tmp_path).write_bytes(b"this is not a sqlite database")
    (tmp_path / f"{DB_FILENAME}-wal").write_bytes(b"junk")
    (tmp_path / f"{DB_FILENAME}-shm").write_bytes(b"junk")

    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))  # must not raise

    assert len(list(tmp_path.glob(f"{DB_FILENAME}.corrupt.*"))) == 1
    assert len(list(tmp_path.glob(f"{DB_FILENAME}-wal.corrupt.*"))) == 1
    assert len(list(tmp_path.glob(f"{DB_FILENAME}-shm.corrupt.*"))) == 1

    # fresh bootstrap is fully usable
    store.put(_draft())
    record = store.get("desire", "d1")
    assert record is not None
    assert record.payload == {"note": "hi"}


def test_recovery_logs_quarantine_incident(tmp_path: Path) -> None:
    _db_path(tmp_path).write_bytes(b"garbage")
    with capture_logs() as logs:
        SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    events = [e for e in logs if e.get("event") == "sqlite_quarantined"]
    assert len(events) == 1
    assert events[0]["path"] == str(_db_path(tmp_path))


def test_valid_store_reopens_without_quarantine(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    SQLiteRuntimeStore(tmp_path, clock=clock).put(_draft())

    SQLiteRuntimeStore(tmp_path, clock=clock)  # reopen a healthy store

    assert list(tmp_path.glob("*.corrupt.*")) == []


def test_data_survives_store_reopen(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    SQLiteRuntimeStore(tmp_path, clock=clock).put(_draft())

    reopened = SQLiteRuntimeStore(tmp_path, clock=clock)

    record = reopened.get("desire", "d1")
    assert record is not None
    assert record.payload == {"note": "hi"}


def test_quarantine_unlinks_corrupt_main_db_when_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the corrupt main DB cannot be renamed aside, it must NOT be left in place
    # (recovery would then migrate against corruption and could raise, restart-
    # looping the being). It is unlinked instead, and construction still succeeds.
    _db_path(tmp_path).write_bytes(b"this is not a sqlite database")

    real_rename = Path.rename

    def flaky_rename(self: Path, target: Path) -> Path:
        if self.name == DB_FILENAME:  # only the main DB rename fails
            raise OSError("simulated rename failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))  # must not raise

    # No .corrupt.* was produced for the main DB (rename failed), and no leftover
    # file still holds the garbage bytes — it was force-removed.
    assert list(tmp_path.glob(f"{DB_FILENAME}.corrupt.*")) == []
    assert all(p.read_bytes() != b"this is not a sqlite database" for p in tmp_path.iterdir())
    # fresh bootstrap works
    store.put(_draft())
    assert store.get("desire", "d1") is not None


# ---- migrations ---------------------------------------------------------


def test_fresh_db_gets_a_schema_migrations_row(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert rows == [(1,), (2,)]


def test_constructing_twice_is_idempotent(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    SQLiteRuntimeStore(tmp_path, clock=clock)
    SQLiteRuntimeStore(tmp_path, clock=clock)  # no re-apply, no error

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert rows == [(1,), (2,)]
    # a brand-new DB has nothing to back up, and the second construction found
    # no pending migrations either, so no backup file is ever created.
    assert list(tmp_path.glob("*.bak.*")) == []


def test_v1_db_upgrades_to_v2_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a pre-6.2 DB that only ever applied v1 (lm-fib.6.1), then reopen
    # with the real migration set — v2 must apply cleanly on top of existing data.
    clock = FakeClock(BASE_TIME)
    monkeypatch.setattr(sqlite_store_module, "_MIGRATIONS", [sqlite_store_module._MIGRATIONS[0]])
    SQLiteRuntimeStore(tmp_path, clock=clock).put(_draft())  # a v1-only DB with data

    monkeypatch.undo()  # restore the real (v1+v2) migration set
    upgraded = SQLiteRuntimeStore(tmp_path, clock=clock)

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert rows == [(1,), (2,)]
    assert "runtime_state" in tables
    assert upgraded.get("desire", "d1") is not None  # pre-existing v1 data survived
    assert len(list(tmp_path.glob(f"{DB_FILENAME}.bak.*"))) == 1  # backup taken before the upgrade


# ---- StatePort over SQLite (lm-fib.6.2) -------------------------------------


def test_sqlite_store_satisfies_the_state_port(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    assert isinstance(store, StatePort)


def test_load_missing_row_returns_default_state(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    assert store.load() == State()


def test_commit_then_load_round_trips_full_fidelity(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    state = State(
        tick_count=5,
        u=2.75,
        energy=0.5,
        fatigue=0.3,
        duration_over_theta=4.0,
        last_exchange_at="2026-07-03T12:00:00+00:00",
        desire_status="active",
        declined_at="2026-07-02T00:00:00+00:00",
        decline_count=2,
        pending_proactive_id="corr-1",
        pending_proactive_since="2026-07-03T11:00:00+00:00",
        last_tick_at="2026-07-03T12:30:00+00:00",
        last_contact_at="2026-07-03T11:30:00+00:00",
        action_pending_since="2026-07-03T11:45:00+00:00",
        proactive_send_log=["2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00"],
    )
    store.commit(state)
    assert store.load() == state


def test_commit_upserts_the_singleton_row_and_bumps_revision(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=1.0))
    store.commit(State(u=2.0))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT id, revision FROM runtime_state").fetchall()
    assert rows == [(1, 1)]  # one singleton row, revision bumped on the 2nd commit


def test_commit_rejects_non_finite_float_before_writing(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    with pytest.raises(StateSerializationError):
        store.commit(State(u=float("nan")))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT * FROM runtime_state").fetchall()
    assert rows == []  # nothing written
    assert store.load() == State()  # still the default


def test_load_rejects_newer_schema_version(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, json.dumps({"schema_version": SCHEMA_VERSION + 1, "u": 0.0}))
    with pytest.raises(StateSchemaError):
        store.load()


def test_load_rejects_unknown_older_schema_version(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, json.dumps({"schema_version": 0, "u": 0.0}))
    with pytest.raises(StateSchemaError):
        store.load()


def test_load_raises_corrupt_on_unparseable_json(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, "{ not json")
    with pytest.raises(StateCorruptError):
        store.load()


def test_load_raises_corrupt_on_non_object_json(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, "[1, 2, 3]")
    with pytest.raises(StateCorruptError):
        store.load()


def test_load_raises_corrupt_on_missing_schema_version(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, json.dumps({"u": 0.0}))
    with pytest.raises(StateCorruptError):
        store.load()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_finite_tokens(tmp_path: Path, token: str) -> None:
    # json.loads accepts these non-standard tokens by default; the store must
    # reject the resulting non-finite floats as corrupt (mirrors JsonStateStore).
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, f'{{"schema_version": {SCHEMA_VERSION}, "u": {token}}}')
    with pytest.raises(StateCorruptError):
        store.load()


def test_reset_overwrites_and_returns_a_fresh_state(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.commit(State(tick_count=7, u=3.0, desire_status="active"))

    fresh = store.reset()

    assert fresh == State()
    assert store.load() == State()


def test_reset_works_with_no_row_present(tmp_path: Path) -> None:
    # No prior commit at all -- reset must not require a successful load() first.
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    fresh = store.reset()
    assert fresh == State()
    assert store.load() == State()


def test_reset_works_when_the_existing_row_is_unreadable_garbage(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    _write_raw_state_json(tmp_path, "{ not json at all")
    with pytest.raises(StateCorruptError):
        store.load()  # sanity: the row really is unreadable beforehand

    fresh = store.reset()

    assert fresh == State()
    assert store.load() == State()  # reset replaced the garbage row cleanly


def _write_raw_state_json(base_dir: Path, raw: str) -> None:
    """Directly UPSERT ``raw`` into the ``runtime_state`` singleton row.

    Bypasses :meth:`SQLiteRuntimeStore.commit` (which would refuse malformed
    payloads) so tests can plant an already-corrupt row, mirroring how
    ``tests/test_state_store.py`` used to hand-write a bad ``state.json``.
    """
    with closing(sqlite3.connect(str(base_dir / DB_FILENAME))) as conn, conn:
        conn.execute("DELETE FROM runtime_state WHERE id = 1")
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, updated_at_epoch, revision) "
            "VALUES (1, ?, ?, ?, 0)",
            (raw, BASE_TIME.isoformat(), int(BASE_TIME.timestamp() * 1000)),
        )


def test_migration_failure_restores_backup_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock(BASE_TIME)
    # seed a healthy v1+v2 DB (both real migrations, incl. runtime_state, lm-fib.6.2)
    SQLiteRuntimeStore(tmp_path, clock=clock).put(_draft())

    def _v3(conn: sqlite3.Connection, _strict: bool) -> None:
        conn.execute("CREATE TABLE v3_marker (x INTEGER)")

    monkeypatch.setattr(
        sqlite_store_module, "_MIGRATIONS", [*sqlite_store_module._MIGRATIONS, (3, _v3)]
    )

    real_quick_check_ok = SQLiteRuntimeStore._quick_check_ok
    calls = {"n": 0}

    def fake_quick_check_ok(self: SQLiteRuntimeStore, path: Path) -> bool:
        calls["n"] += 1
        if calls["n"] == 2:  # the check right after applying the (fake) v3 migration
            return False
        return real_quick_check_ok(self, path)

    monkeypatch.setattr(SQLiteRuntimeStore, "_quick_check_ok", fake_quick_check_ok)

    with pytest.raises(sqlite_store_module.MigrationFailed):
        SQLiteRuntimeStore(tmp_path, clock=clock)

    assert len(list(tmp_path.glob(f"{DB_FILENAME}.bak.*"))) == 1  # backup was taken

    monkeypatch.undo()  # restore the real _quick_check_ok / _MIGRATIONS for the reopen below
    recovered = SQLiteRuntimeStore(tmp_path, clock=clock)
    record = recovered.get("desire", "d1")
    assert record is not None  # pre-v3 data survived the restore

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert rows == [(1,), (2,)]  # the failed v3 migration was rolled back by the restore


def test_migration_that_raises_restores_backup_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A migration fn that raises (a code bug) must not leave the DB partially
    # advanced. Python's sqlite3 auto-commits DDL, so the CREATE below is NOT
    # rolled back by the transaction — only the backup restore reverts it.
    clock = FakeClock(BASE_TIME)
    # seed a healthy v1+v2 DB (both real migrations, incl. runtime_state, lm-fib.6.2)
    SQLiteRuntimeStore(tmp_path, clock=clock).put(_draft())

    class _MigrationBug(Exception):
        pass

    def _v3(conn: sqlite3.Connection, _strict: bool) -> None:
        conn.execute("CREATE TABLE half_applied (x INTEGER)")  # auto-commits
        raise _MigrationBug("bug in migration code")

    monkeypatch.setattr(
        sqlite_store_module, "_MIGRATIONS", [*sqlite_store_module._MIGRATIONS, (3, _v3)]
    )

    with pytest.raises(_MigrationBug):
        SQLiteRuntimeStore(tmp_path, clock=clock)

    assert len(list(tmp_path.glob(f"{DB_FILENAME}.bak.*"))) == 1  # backup was taken

    monkeypatch.undo()
    recovered = SQLiteRuntimeStore(tmp_path, clock=clock)
    assert recovered.get("desire", "d1") is not None  # pre-v3 data survived

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert "half_applied" not in tables  # the restore reverted the auto-committed DDL
    assert rows == [(1,), (2,)]


def test_migration_creates_expected_tables_and_indexes(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
    assert {"store_meta", "schema_migrations", "memory_records", "runtime_state"} <= tables
    assert "outbound_ledger" not in tables  # out of scope for this bead (6.4)
    assert any("kind" in name and "state" in name for name in indexes)
    assert any("expires_at_epoch" in name for name in indexes)


# ---- STRICT feature-detection -------------------------------------------


def test_strict_table_detection_matches_this_sqlite_build(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(sqlite3.connect(":memory:")) as conn:
        try:
            conn.execute("CREATE TABLE t (x INTEGER) STRICT")
            expected_supported = True
        except sqlite3.OperationalError:
            expected_supported = False

    assert store._strict_supported is expected_supported  # noqa: SLF001

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_records'"
        ).fetchone()
    assert row is not None
    assert ("STRICT" in row[0]) is expected_supported


# ---- PRAGMA / connection settings ----------------------------------------


def test_journal_mode_is_wal(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_connect_sets_expected_per_connection_pragmas(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))

    with closing(store._connect()) as conn:  # noqa: SLF001
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


# ---- epoch storage --------------------------------------------------------


def test_epoch_columns_stored_alongside_iso_text_no_lossy_reconstruction(
    tmp_path: Path,
) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    expires = (clock.now() + timedelta(days=1)).isoformat()
    store.put(_draft(expires_at=expires))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        row = conn.execute(
            "SELECT created_at, created_at_epoch, updated_at, updated_at_epoch, "
            "expires_at, expires_at_epoch FROM memory_records WHERE id = 'd1'"
        ).fetchone()

    created_at, created_at_epoch, updated_at, updated_at_epoch, expires_at, expires_at_epoch = row
    assert created_at == clock.now().isoformat()
    assert created_at_epoch == int(clock.now().timestamp() * 1000)
    assert updated_at == clock.now().isoformat()
    assert updated_at_epoch == int(clock.now().timestamp() * 1000)
    assert expires_at == expires
    assert expires_at_epoch == int(datetime.fromisoformat(expires).timestamp() * 1000)


def test_null_expires_at_leaves_epoch_column_null(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.put(_draft(expires_at=None))

    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        row = conn.execute(
            "SELECT expires_at, expires_at_epoch FROM memory_records WHERE id = 'd1'"
        ).fetchone()
    assert row == (None, None)


# ---- atomic UPSERT / concurrent writers (ON CONFLICT) -----------------------


def test_put_upsert_is_atomic_across_two_store_instances(tmp_path: Path) -> None:
    # Two SQLiteRuntimeStore handles over the SAME file — a stand-in for the
    # 60s tick and a separate-process /lifemodel command. Both put() the same
    # FRESH key with NO preceding get(): a SELECT-then-INSERT/UPDATE would raise
    # a PRIMARY KEY IntegrityError on the second insert (or undercount revision);
    # the ON CONFLICT UPSERT must instead land last-writer-wins, bump revision,
    # and preserve the ORIGINAL created_at.
    clock = FakeClock(BASE_TIME)
    store_a = SQLiteRuntimeStore(tmp_path, clock=clock)
    store_b = SQLiteRuntimeStore(tmp_path, clock=clock)

    store_a.put(_draft(payload={"note": "from-a"}))  # fresh insert
    created_stamp = clock.now().isoformat()
    clock.advance(timedelta(minutes=5))
    store_b.put(_draft(payload={"note": "from-b"}))  # conflict path, no prior read

    record = store_b.get("desire", "d1")
    assert record is not None
    assert record.payload == {"note": "from-b"}  # last writer wins
    assert record.revision == 1  # bumped atomically, not undercounted
    assert record.created_at == created_stamp  # original creation stamp preserved
    assert record.updated_at == clock.now().isoformat()  # refreshed on update


def test_commit_upsert_is_atomic_across_two_store_instances(tmp_path: Path) -> None:
    # Same guarantee for the runtime_state singleton row: two handles both
    # commit() a State with NO preceding load(). No IntegrityError, and the
    # revision reflects the second (conflicting) write.
    clock = FakeClock(BASE_TIME)
    store_a = SQLiteRuntimeStore(tmp_path, clock=clock)
    store_b = SQLiteRuntimeStore(tmp_path, clock=clock)

    store_a.commit(State(u=1.0))  # fresh insert
    store_b.commit(State(u=2.0))  # conflict path, no prior load

    assert store_b.load().u == 2.0  # last writer wins
    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn:
        rows = conn.execute("SELECT id, revision FROM runtime_state").fetchall()
    assert rows == [(1, 1)]  # one singleton row, revision bumped atomically


# ---- fail-soft --------------------------------------------------------------


def test_read_pressure_index_fails_soft_on_operational_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())  # would otherwise report a non-default index

    def boom(_self: SQLiteRuntimeStore) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SQLiteRuntimeStore, "_connect", boom)

    assert store.read_pressure_index(clock.now()) == PressureIndex()


def test_fresh_healthy_store_returns_real_pressure_numbers(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())

    index = store.read_pressure_index(clock.now())

    assert index.active_desire_count == 1
    assert index.contact_frame_available is True


def test_read_pressure_index_fails_soft_on_runtime_corrupt_database_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A runtime-corrupt image surfaces as sqlite3.DatabaseError ("database disk
    # image is malformed"), which is NOT an OperationalError — it must still
    # fail-soft, not crash the live pressure read.
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.put(_draft())

    class _CorruptConn:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            raise sqlite3.DatabaseError("database disk image is malformed")

        def close(self) -> None:
            pass

    monkeypatch.setattr(SQLiteRuntimeStore, "_connect", lambda _self: _CorruptConn())

    assert store.read_pressure_index(clock.now()) == PressureIndex()


def test_read_pressure_index_does_not_fail_soft_schema_errors(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TIME)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    with closing(sqlite3.connect(str(_db_path(tmp_path)))) as conn, conn:
        conn.execute("DROP TABLE memory_records")

    with pytest.raises(sqlite3.OperationalError):
        store.read_pressure_index(clock.now())

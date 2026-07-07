"""SQLite adapter for :class:`MemoryPort` + :class:`PressureSensorPort` (HLA §4.1/D7).

Writes ``<base_dir>/lifemodel.sqlite`` — the plugin's first durable SQLite
runtime store. Purely additive (lm-fib.6.1): nothing in the live tick,
composition root, or ``core/proactive.py`` constructs this yet; the state
package's safety-critical store stays :class:`~lifemodel.state.json_store.JsonStateStore`
until the ``StatePort`` cutover (lm-fib.6.2). Imports nothing from Hermes.

**Connection-per-operation.** Every public method opens a short-lived
connection (:meth:`_connect`), sets per-connection PRAGMA (``busy_timeout``,
``synchronous=NORMAL``, ``foreign_keys=ON``, ``wal_autocheckpoint``), does its
work, and closes — mirroring ``json_store``'s "no long-lived handle" posture.
``journal_mode=WAL`` is a *database-level* property (persisted in the file
header), so it is set once, in :meth:`_ensure_ready`, not per connection.
Writes run inside ``with conn:`` so a raised exception rolls back rather than
leaving a half-applied change.

**Recovery runs once, in ``__init__``, before any read/write** — never inside
:meth:`read_pressure_index` (a corruption check on every pressure read would
be needless overhead and, worse, could itself raise mid-tick). If the DB file
exists but fails ``PRAGMA quick_check``, the trio (``lifemodel.sqlite`` +
``-wal`` + ``-shm``) is quarantined — each existing file renamed to
``*.corrupt.<epoch_ms>`` — and construction falls through to a fresh bootstrap.
This step never raises: a raise here would restart-loop the being (the same
failure mode ``JsonStateStore``'s corruption handling was designed to avoid,
generalized to a real database file that cannot simply be temp+replace'd).

**Migrations** are tracked in ``schema_migrations`` (integer versions, applied
in order, each in its own transaction). Before applying a migration to a DB
that already has *some* applied migration (i.e. it is not brand new), a
``sqlite3.Connection.backup()`` snapshot is taken to
``lifemodel.sqlite.bak.<epoch_ms>``; after applying, ``PRAGMA quick_check``
must still pass, or the backup is restored and the migration raises
(a genuine migration-code bug, as opposed to on-disk corruption — the
*next* construction attempt's recovery step is what would quarantine a
still-bad file). Migration v1 creates ``store_meta``, ``memory_records``, and
its two indexes; no destructive migration ever runs silently.

**STRICT tables** are used when the host's SQLite build supports them
(feature-detected once, at construction, via a throwaway ``:memory:`` table);
older builds fall back to ordinary column-typed tables. ``fts5`` is out of
scope for this bead.

**Epochs** are stored as ``INTEGER`` milliseconds UTC
(:func:`~lifemodel.domain.memory.epoch_ms`) *alongside* the ISO-8601 text
column they were derived from — never reconstructed lossily from one or the
other. All ordering/expiry comparisons use the epoch columns.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing, suppress
from datetime import datetime
from pathlib import Path
from typing import Final

from ..domain.memory import (
    JsonObject,
    MemoryDraft,
    MemoryPatch,
    MemoryRecord,
    PressureIndex,
    StaleTransition,
    coalesce_patch,
    describe_stale_transition,
    ensure_json_serializable,
    epoch_ms,
    merge_payload,
    parse_expires_at_epoch_ms,
    stamp_iso_utc,
)
from ..log import EventLogger, get_logger
from ..ports.clock import ClockPort
from ..ports.memory import OrderBy

_DB_FILENAME = "lifemodel.sqlite"
_BUSY_TIMEOUT_MS = 5_000
_WAL_AUTOCHECKPOINT_PAGES = 1_000

_SELECT_COLUMNS = (
    "SELECT kind, id, state, payload_json, source, recipient_id, salience, "
    "confidence, expires_at, created_at, updated_at, revision, schema_version "
    "FROM memory_records"
)

_ORDER_SQL: Final[dict[OrderBy, str]] = {
    "updated_desc": "updated_at_epoch DESC, id ASC",
    "created_desc": "created_at_epoch DESC, id ASC",
    "salience_desc": "salience DESC, id ASC",
}


class MigrationFailed(Exception):
    """Raised when a schema migration fails its post-apply ``quick_check``.

    Adapter-internal (unlike :class:`~lifemodel.domain.memory.StaleTransition`,
    it is not part of the ``MemoryPort``/``PressureSensorPort`` contract fakes
    must also honor — fakes have no migrations to fail). A previously-applied
    DB's backup is restored before this is raised; a still-bad file is left
    for the *next* construction attempt's :meth:`SQLiteRuntimeStore._ensure_ready`
    to quarantine.
    """


class SQLiteRuntimeStore:
    """A :class:`MemoryPort` + :class:`PressureSensorPort` over one SQLite file (HLA §4.1/D7)."""

    def __init__(
        self, base_dir: Path, *, clock: ClockPort, logger: EventLogger | None = None
    ) -> None:
        self._base_dir = base_dir
        self._path = base_dir / _DB_FILENAME
        self._clock = clock
        self._log = logger or get_logger("lifemodel.state.sqlite")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._strict_supported = self._detect_strict_support()
        self._ensure_ready()

    # ---- construction-time recovery + schema -------------------------------

    def _ensure_ready(self) -> None:
        """Recovery, then schema — run once, before any read/write (§4.1)."""
        if self._path.exists() and not self._quick_check_ok(self._path):
            self._quarantine()
        self._ensure_wal_mode()
        self._run_migrations()

    def _quick_check_ok(self, path: Path) -> bool:
        # Opened read-only (a URI connection) so merely *checking* an invalid
        # or WAL-inconsistent file never mutates or deletes its "-wal"/"-shm"
        # siblings as a side effect — a read-write connection's own recovery
        # logic can do exactly that, which would defeat quarantine's attempt
        # to preserve the corrupt trio for forensics.
        uri = f"{path.resolve().as_uri()}?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                row = conn.execute("PRAGMA quick_check").fetchone()
        except sqlite3.Error:
            return False
        return row is not None and row[0] == "ok"

    def _quarantine(self) -> None:
        """Move the corrupt trio aside and log an incident. Never raises.

        Preferred outcome: each existing file is renamed to ``*.corrupt.<ms>``
        for forensics. But if a rename fails, the corrupt file *must not* remain
        in place — ``_run_migrations`` would then run against it and could raise,
        restart-looping the being (the very failure recovery exists to prevent),
        and a stale ``-wal``/``-shm`` left pointing at the fresh DB we bootstrap
        next would re-corrupt it. Such a file is already deemed unrecoverable, so
        availability beats forensics: force it out with a best-effort ``unlink``,
        logging if even that fails.
        """
        stamp = epoch_ms(self._clock.now())
        trio = [Path(f"{self._path}{suffix}") for suffix in ("", "-wal", "-shm")]
        for src in trio:
            if src.exists():
                with suppress(OSError):
                    src.rename(Path(f"{src}.corrupt.{stamp}"))
        for src in trio:
            if src.exists():  # rename failed above — drop it so bootstrap is clean
                try:
                    src.unlink()
                except OSError as exc:
                    self._log.info("sqlite_quarantine_unlink_failed", path=str(src), error=str(exc))
        self._log.info("sqlite_quarantined", path=str(self._path), epoch_ms=stamp)

    def _ensure_wal_mode(self) -> None:
        with suppress(sqlite3.Error), closing(sqlite3.connect(str(self._path))) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

    @staticmethod
    def _detect_strict_support() -> bool:
        try:
            with closing(sqlite3.connect(":memory:")) as conn:
                conn.execute("CREATE TABLE t (x INTEGER) STRICT")
            return True
        except sqlite3.OperationalError:
            return False

    def _run_migrations(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

        pending = [(version, fn) for version, fn in _MIGRATIONS if version not in applied]
        if not pending:
            return

        is_brand_new = not applied
        backup_path = None if is_brand_new else self._backup()

        # Any failure past this point — a migration fn that raises (a code bug),
        # or a post-apply quick_check that fails — must not leave the DB partially
        # advanced. Python's sqlite3 auto-commits DDL, so a mid-migration raise is
        # NOT rolled back by the surrounding transaction; restore the pre-migration
        # backup, then re-raise. A brand-new DB has no backup to restore, so it just
        # re-raises — the next construction's recovery quarantines a still-bad file.
        try:
            for version, migrate in pending:
                with closing(self._connect()) as conn, conn:
                    migrate(conn, self._strict_supported)
                    conn.execute(
                        # applied_at is an internal audit column, not a memory-record
                        # text column, so it keeps the raw clock text (fix 4's
                        # canonicalization/naive-clock guard is scoped to put/transition):
                        # rejecting a naive clock here would fail construction rather
                        # than the first write, diverging from the fake.
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, self._clock.now().isoformat()),
                    )
                if not self._quick_check_ok(self._path):
                    raise MigrationFailed(
                        f"schema migration to v{version} failed PRAGMA quick_check"
                    )
        except Exception:
            if backup_path is not None:
                self._restore_backup(backup_path)
            raise

    def _backup(self) -> Path:
        backup_path = Path(f"{self._path}.bak.{epoch_ms(self._clock.now())}")
        with (
            closing(sqlite3.connect(str(self._path))) as src,
            closing(sqlite3.connect(str(backup_path))) as dst,
        ):
            src.backup(dst)
        return backup_path

    def _restore_backup(self, backup_path: Path) -> None:
        for suffix in ("-wal", "-shm"):
            stale = Path(f"{self._path}{suffix}")
            if stale.exists():
                stale.unlink()
        shutil.copyfile(backup_path, self._path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
        return conn

    # ---- MemoryPort ---------------------------------------------------------

    def put(self, draft: MemoryDraft) -> str:
        ensure_json_serializable(draft.payload)
        payload_json = json.dumps(draft.payload, allow_nan=False)
        expires_at_epoch = parse_expires_at_epoch_ms(draft.expires_at)
        now = self._clock.now()
        now_iso = stamp_iso_utc(now)  # canonical UTC text; rejects a naive clock
        now_epoch = epoch_ms(now)

        with closing(self._connect()) as conn, conn:
            existing = conn.execute(
                "SELECT created_at, revision FROM memory_records WHERE kind = ? AND id = ?",
                (draft.kind, draft.id),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO memory_records ("
                    "kind, id, state, recipient_id, payload_json, salience, confidence, "
                    "expires_at, expires_at_epoch, source, created_at, created_at_epoch, "
                    "updated_at, updated_at_epoch, revision, schema_version) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,1)",
                    (
                        draft.kind,
                        draft.id,
                        draft.state,
                        draft.recipient_id,
                        payload_json,
                        draft.salience,
                        draft.confidence,
                        draft.expires_at,
                        expires_at_epoch,
                        draft.source,
                        now_iso,
                        now_epoch,
                        now_iso,
                        now_epoch,
                    ),
                )
            else:
                _created_at, revision = existing
                conn.execute(
                    "UPDATE memory_records SET state=?, recipient_id=?, payload_json=?, "
                    "salience=?, confidence=?, expires_at=?, expires_at_epoch=?, source=?, "
                    "updated_at=?, updated_at_epoch=?, revision=? WHERE kind=? AND id=?",
                    (
                        draft.state,
                        draft.recipient_id,
                        payload_json,
                        draft.salience,
                        draft.confidence,
                        draft.expires_at,
                        expires_at_epoch,
                        draft.source,
                        now_iso,
                        now_epoch,
                        revision + 1,
                        draft.kind,
                        draft.id,
                    ),
                )
        return draft.id

    def get(self, kind: str, id: str) -> MemoryRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"{_SELECT_COLUMNS} WHERE kind = ? AND id = ?", (kind, id)
            ).fetchone()
        return None if row is None else _row_to_record(row)

    def find(
        self,
        kind: str | None = None,
        state: str | None = None,
        limit: int | None = None,
        order_by: OrderBy = "updated_desc",
    ) -> list[MemoryRecord]:
        # SQLite treats `LIMIT -1` as "no limit"; reject a negative limit so the
        # contract is unambiguous and identical to the fake (which would slice).
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        clauses: list[str] = []
        params: list[str | int] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)

        sql = _SELECT_COLUMNS
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY " + _ORDER_SQL[order_by]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def transition(
        self,
        kind: str,
        id: str,
        from_state: str,
        to_state: str,
        patch: MemoryPatch | None = None,
    ) -> MemoryRecord:
        patch = patch if patch is not None else MemoryPatch()
        if patch.payload_merge is not None:
            ensure_json_serializable(patch.payload_merge)

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT payload_json, salience, confidence, expires_at, source "
                "FROM memory_records WHERE kind = ? AND id = ? AND state = ?",
                (kind, id, from_state),
            ).fetchone()
            if row is None:
                actual = conn.execute(
                    "SELECT state FROM memory_records WHERE kind = ? AND id = ?", (kind, id)
                ).fetchone()
                actual_state = actual[0] if actual is not None else None
                raise StaleTransition(describe_stale_transition(kind, id, from_state, actual_state))

            payload_json, salience, confidence, expires_at, source = row
            payload: JsonObject = merge_payload(json.loads(payload_json), patch.payload_merge)
            new_expires_at = coalesce_patch(patch.expires_at, expires_at)
            new_expires_epoch = parse_expires_at_epoch_ms(new_expires_at)
            now = self._clock.now()
            now_iso = stamp_iso_utc(now)  # canonical UTC text; rejects a naive clock

            cursor = conn.execute(
                "UPDATE memory_records SET state = ?, payload_json = ?, salience = ?, "
                "confidence = ?, expires_at = ?, expires_at_epoch = ?, source = ?, "
                "updated_at = ?, updated_at_epoch = ?, revision = revision + 1 "
                "WHERE kind = ? AND id = ? AND state = ?",
                (
                    to_state,
                    json.dumps(payload, allow_nan=False),
                    coalesce_patch(patch.salience, salience),
                    coalesce_patch(patch.confidence, confidence),
                    new_expires_at,
                    new_expires_epoch,
                    coalesce_patch(patch.source, source),
                    now_iso,
                    epoch_ms(now),
                    kind,
                    id,
                    from_state,
                ),
            )
            # Defensive: the guarded UPDATE ran in the same transaction as the
            # SELECT above, so it should always match exactly the one row. Once a
            # later bead adds a second writer, this is what keeps the guarded-
            # transition contract honest — a lost race raises StaleTransition and
            # the surrounding ``with conn:`` rolls the (no-op) UPDATE back.
            if cursor.rowcount != 1:
                raise StaleTransition(
                    f"guarded transition for kind={kind!r} id={id!r} from_state={from_state!r} "
                    f"matched {cursor.rowcount} rows (expected 1)"
                )

        record = self.get(kind, id)
        if record is None:  # pragma: no cover - defensive: we just wrote this row
            raise StaleTransition(f"record kind={kind!r} id={id!r} vanished during transition")
        return record

    # ---- PressureSensorPort ---------------------------------------------------

    def read_pressure_index(self, now: datetime) -> PressureIndex:
        now_epoch = epoch_ms(now)
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), MAX(salience) FROM memory_records "
                    "WHERE kind = 'desire' AND state = 'active' "
                    "AND (expires_at_epoch IS NULL OR expires_at_epoch > ?)",
                    (now_epoch,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            # Fail-soft the transient/operational cases (locked DB) AND a
            # runtime-corrupt image ("database disk image is malformed" surfaces
            # as sqlite3.DatabaseError, NOT OperationalError) — a stalled or
            # damaged pressure read must never crash the live tick. A schema
            # error ("no such table"/"no such column") is a genuine init bug, not
            # a transient condition, so it still surfaces.
            if _is_schema_error(exc):
                raise
            self._log.info("pressure_read_failed_soft", error=str(exc))
            return PressureIndex()

        count = row[0] if row is not None else 0
        if count == 0:
            return PressureIndex()
        max_salience = row[1] if row[1] is not None else 0.0
        return PressureIndex(
            active_desire_count=count,
            max_desire_salience=max_salience,
            contact_frame_available=True,
        )


def _row_to_record(row: tuple[object, ...]) -> MemoryRecord:
    (
        kind,
        id_,
        state,
        payload_json,
        source,
        recipient_id,
        salience,
        confidence,
        expires_at,
        created_at,
        updated_at,
        revision,
        schema_version,
    ) = row
    assert isinstance(kind, str)
    assert isinstance(id_, str)
    assert isinstance(state, str)
    assert isinstance(payload_json, str)
    assert isinstance(source, str)
    assert isinstance(recipient_id, str)
    assert isinstance(salience, int | float)
    assert confidence is None or isinstance(confidence, int | float)
    assert expires_at is None or isinstance(expires_at, str)
    assert isinstance(created_at, str)
    assert isinstance(updated_at, str)
    assert isinstance(revision, int)
    assert isinstance(schema_version, int)
    return MemoryRecord(
        kind=kind,
        id=id_,
        state=state,
        payload=json.loads(payload_json),
        source=source,
        recipient_id=recipient_id,
        salience=float(salience),
        confidence=None if confidence is None else float(confidence),
        expires_at=expires_at,
        created_at=created_at,
        updated_at=updated_at,
        revision=revision,
        schema_version=schema_version,
    )


def _is_schema_error(exc: sqlite3.DatabaseError) -> bool:
    """True for a bad-schema failure (init bug); false for a transient/corrupt one.

    ``sqlite3.DatabaseError`` covers "no such table"/"no such column" (a real bug
    — must surface), "database is locked"/other transient contention, and a
    runtime-corrupt image ("database disk image is malformed"). Only the first is
    a schema error; the rest fail-soft (per
    :meth:`SQLiteRuntimeStore.read_pressure_index`).
    """
    message = str(exc).lower()
    return "no such table" in message or "no such column" in message


def _migrate_v1(conn: sqlite3.Connection, strict: bool) -> None:
    """Create ``store_meta``, ``memory_records``, and its indexes (§4.1)."""
    strict_kw = " STRICT" if strict else ""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS store_meta ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)" + strict_kw
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_records ("
        "kind TEXT NOT NULL, "
        "id TEXT NOT NULL, "
        "state TEXT NOT NULL, "
        "recipient_id TEXT NOT NULL DEFAULT 'owner', "
        "payload_json TEXT NOT NULL, "
        "salience REAL NOT NULL DEFAULT 0, "
        "confidence REAL, "
        "expires_at TEXT, "
        "expires_at_epoch INTEGER, "
        "source TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "created_at_epoch INTEGER NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "updated_at_epoch INTEGER NOT NULL, "
        "revision INTEGER NOT NULL DEFAULT 0, "
        "schema_version INTEGER NOT NULL DEFAULT 1, "
        "PRIMARY KEY (kind, id))" + strict_kw
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_records_kind_state ON memory_records (kind, state)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_records_expires_at_epoch "
        "ON memory_records (expires_at_epoch)"
    )


_MIGRATIONS: Final[list[tuple[int, Callable[[sqlite3.Connection, bool], None]]]] = [
    (1, _migrate_v1),
]

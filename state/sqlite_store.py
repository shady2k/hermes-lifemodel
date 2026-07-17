"""SQLite adapter for :class:`StatePort` + :class:`MemoryPort` + :class:`PressureSensorPort`
(HLA §4.1/D7).

Writes ``<base_dir>/lifemodel.sqlite`` — the plugin's one durable SQLite runtime
store. Added purely additively in lm-fib.6.1 (``MemoryPort``/``PressureSensorPort``
only); lm-fib.6.2 cuts the being's vitals/control ``State`` over to this same
file too (the composition root now wires this class as the live ``StatePort``,
retiring ``lifemodel.state.json_store.JsonStateStore`` and ``state.json``
outright — see :meth:`load`/:meth:`commit`/:meth:`reset` below). Imports
nothing from Hermes.

**Why one JSON blob, not typed columns (settled, HLA §4.1/D7 v0.7).** ``State``
is persisted as a single JSON blob in the ``runtime_state`` singleton row
(``id=1``), not typed-per-field columns: ``State`` already owns
``to_dict()``/``from_dict()`` with its own validation, and it is still
actively reshaping (e.g. lm-fib.6.3 removes ``desire_status``) — typed columns
would be a migration treadmill for a shape that has not settled. The port
abstraction (:class:`~lifemodel.state.port.StatePort`) lets a later phase
promote to typed columns if a real query ever needs them, without touching
callers.

**Connection-per-operation.** Every public method opens a short-lived
connection (:meth:`_connect`), sets per-connection PRAGMA (``busy_timeout``,
``synchronous=NORMAL``, ``foreign_keys=ON``, ``wal_autocheckpoint``), does its
work, and closes — the retired ``JsonStateStore``'s "no long-lived handle"
posture, carried over. ``journal_mode=WAL`` is a *database-level* property
(persisted in the file header), so it is set once, in :meth:`_ensure_ready`,
not per connection. Writes run inside ``with conn:`` so a raised exception
rolls back rather than leaving a half-applied change.

**Recovery runs once, in ``__init__``, before any read/write** — never inside
:meth:`read_pressure_index` or :meth:`load` (a corruption check on every read
would be needless overhead and, worse, could itself raise mid-tick). If the DB
file exists but fails ``PRAGMA quick_check``, the trio (``lifemodel.sqlite`` +
``-wal`` + ``-shm``) is quarantined — each existing file renamed to
``*.corrupt.<epoch_ms>`` — and construction falls through to a fresh bootstrap.
This step never raises: a raise here would restart-loop the being (the same
failure mode ``JsonStateStore``'s corruption handling was designed to avoid,
generalized to a real database file that cannot simply be temp+replace'd). A
readable-but-malformed ``runtime_state`` row (bad JSON, wrong shape, an
unsupported ``schema_version``) is a *separate*, narrower failure — that is
:meth:`load`'s job, raising :class:`~lifemodel.state.errors.StateCorruptError`/
:class:`~lifemodel.state.errors.StateSchemaError` exactly where
``JsonStateStore.load`` used to.

**Migrations** are tracked in ``schema_migrations`` (integer versions, applied
in order, each in its own transaction). Before applying a migration to a DB
that already has *some* applied migration (i.e. it is not brand new), a
``sqlite3.Connection.backup()`` snapshot is taken to
``lifemodel.sqlite.bak.<epoch_ms>``; after applying, ``PRAGMA quick_check``
must still pass, or the backup is restored and the migration raises
(a genuine migration-code bug, as opposed to on-disk corruption — the
*next* construction attempt's recovery step is what would quarantine a
still-bad file). Migration v1 creates ``store_meta``, ``memory_records``, and
its two indexes; v2 (lm-fib.6.2) creates the ``runtime_state`` singleton row
table backing ``StatePort``; v3 (lm-fib.10.5) rebuilds an old dual-column file
into the ISO-only shape IN PLACE, preserving every row; v4 (lm-705.14) creates
``conversation_buffer`` — the durable backing for the noticing conversation
buffer (:class:`SqliteBufferStore` below), purely additive. No destructive
migration ever runs silently.

**STRICT tables** are used when the host's SQLite build supports them
(feature-detected once, at construction, via a throwaway ``:memory:`` table);
older builds fall back to ordinary column-typed tables. ``fts5`` is out of
scope for this bead.

**Time is stored ONCE, as normalized ISO-8601 UTC TEXT** (spec §4, lm-fib.10.2).
The retired epoch mirror columns are gone: every ``_at`` value — including a
caller-provided ``expires_at`` — is passed through
:func:`~lifemodel.core.timeutil.to_iso` BEFORE storage
(:func:`~lifemodel.domain.memory.normalize_expires_at` /
:func:`~lifemodel.domain.memory.stamp_iso_utc`), so the stored form is always
fixed-width and lexically sortable and no raw caller string ever reaches a
column. All ordering/expiry comparisons run directly on those TEXT columns
(``updated_at``/``created_at`` for ordering, ``expires_at`` for the expiry/
pressure bound), which is provably correct because the width is fixed.

**MIGRATE THE SELF, RECREATE DERIVED (lm-fib.10.5).** ``lifemodel.sqlite`` IS the
being's self (drive ``u``, energy, memory records, the UserModel/relationship), so
a schema change MIGRATES it via the ``schema_migrations`` framework above —
``_migrate_v3`` rebuilds a pre-10.2 dual-column (ISO + ``*_epoch``) file into the
ISO-only shape in place, preserving every row and re-normalizing its ISO stamps.
Reset is an emergency valve, NOT the strategy for a schema change: the destructive
move-aside (``*.corrupt.<ms>``) fires ONLY for GENUINE corruption (a ``quick_check``
failure). (``metrics.sqlite`` / ``observability.sqlite`` are DERIVED telemetry with
no self, so they correctly stay fresh-recreate on a shape mismatch — no migration.)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, assert_never

from ..core.buffer_store import DEFAULT_BUFFER_MAX_ENTRIES, BufferEntry
from ..core.timeutil import from_iso, to_iso
from ..domain.memory import (
    JsonObject,
    MemoryDraft,
    MemoryMutation,
    MemoryPatch,
    MemoryRecord,
    PressureIndex,
    PutOp,
    StaleTransition,
    TransitionOp,
    coalesce_patch,
    describe_stale_transition,
    ensure_json_serializable,
    epoch_ms,
    merge_payload,
    normalize_expires_at,
    stamp_iso_utc,
)
from ..ports.clock import ClockPort
from ..ports.memory import OrderBy
from .errors import StateCorruptError, StateSchemaError, StateSerializationError
from .model import SCHEMA_VERSION, State
from .soul_revisions import SOUL_KINDS

_DB_FILENAME = "lifemodel.sqlite"
_BUSY_TIMEOUT_MS = 5_000

_LOG = logging.getLogger("lifemodel.state.sqlite")
_WAL_AUTOCHECKPOINT_PAGES = 1_000

_SELECT_COLUMNS = (
    "SELECT kind, id, state, payload_json, source, recipient_id, salience, "
    "confidence, expires_at, created_at, updated_at, revision, schema_version "
    "FROM memory_records"
)

_ORDER_SQL: Final[dict[OrderBy, str]] = {
    "updated_desc": "updated_at DESC, id ASC",
    "created_desc": "created_at DESC, id ASC",
    "salience_desc": "salience DESC, id ASC",
}


def _connect_db(path: Path) -> sqlite3.Connection:
    """Open a short-lived connection to *path* with this store's standard PRAGMAs.

    Shared by :meth:`SQLiteRuntimeStore._connect` and :class:`SqliteBufferStore`
    so both writers over the same ``lifemodel.sqlite`` file (D7: one physical
    store) open connections with the identical posture (module docstring's
    "Connection-per-operation") — one PRAGMA set, never duplicated out of sync.
    """
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
    return conn


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
    """A :class:`StatePort` + :class:`MemoryPort` + :class:`PressureSensorPort` over one
    SQLite file (HLA §4.1/D7)."""

    def __init__(self, base_dir: Path, *, clock: ClockPort) -> None:
        self._base_dir = base_dir
        self._path = base_dir / _DB_FILENAME
        self._clock = clock
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._strict_supported = self._detect_strict_support()
        self._ensure_ready()

    # ---- construction-time recovery + schema -------------------------------

    def _ensure_ready(self) -> None:
        """Recovery, then schema — run once, before any read/write (§4.1).

        MIGRATE THE SELF, RECREATE DERIVED (lm-fib.10.5): a structurally-sound file
        whose table SHAPE predates this build is MIGRATED by :meth:`_run_migrations`
        (the ``schema_migrations`` framework), never wiped — ``lifemodel.sqlite`` is
        the being's self. The destructive move-aside is reserved for GENUINE
        corruption (a ``quick_check`` failure), the emergency valve — not a schema
        change. (``metrics.sqlite`` / ``observability.sqlite`` are derived telemetry
        and correctly stay fresh-recreate in their own stores.)
        """
        if self._path.exists() and not self._quick_check_ok(self._path):
            self._move_trio_aside("corrupt", "sqlite_quarantined")
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

    def _move_trio_aside(self, suffix: str, event: str) -> None:
        """Move the DB trio aside (``*.<suffix>.<ms>``) and log *event*. Never raises.

        The emergency valve for GENUINE corruption ONLY (``suffix`` ``"corrupt"``, a
        ``quick_check`` failure): a fresh bootstrap follows. A stale table SHAPE is NOT
        corruption — it is MIGRATED in place (lm-fib.10.5), never moved aside, because
        ``lifemodel.sqlite`` is the being's self.

        Preferred outcome: each existing file is renamed for forensics. But if a
        rename fails, the file *must not* remain in place — ``_run_migrations``
        would then run against it and could raise, restart-looping the being (the
        very failure recovery exists to prevent), and a stale ``-wal``/``-shm``
        left pointing at the fresh DB we bootstrap next would re-corrupt it. Such
        a file is already deemed unusable, so availability beats forensics: force
        it out with a best-effort ``unlink``, logging if even that fails.
        """
        stamp = epoch_ms(self._clock.now())
        trio = [Path(f"{self._path}{suffix_part}") for suffix_part in ("", "-wal", "-shm")]
        for src in trio:
            if src.exists():
                with suppress(OSError):
                    src.rename(Path(f"{src}.{suffix}.{stamp}"))
        for src in trio:
            if src.exists():  # rename failed above — drop it so bootstrap is clean
                try:
                    src.unlink()
                except OSError as exc:
                    _LOG.info(
                        "sqlite_move_aside_unlink_failed path=%s error=%s", str(src), str(exc)
                    )
        _LOG.info("%s path=%s epoch_ms=%s", event, str(self._path), stamp)

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
        return _connect_db(self._path)

    # ---- MemoryPort ---------------------------------------------------------

    def put(self, draft: MemoryDraft) -> str:
        # Fail-before-write guard, then one self-contained transaction (unchanged
        # single-op contract): the SQL body lives in :meth:`_put_on` so the same
        # write can also run inside :meth:`commit_tick`'s multi-op transaction.
        ensure_json_serializable(draft.payload)
        normalize_expires_at(draft.expires_at)  # validate expires_at before writing
        now = self._clock.now()
        stamp_iso_utc(now)  # validate the clock (tz-aware) before touching the DB
        with closing(self._connect()) as conn, conn:
            self._put_on(conn, draft, now)
        return draft.id

    def _put_on(self, conn: sqlite3.Connection, draft: MemoryDraft, now: datetime) -> None:
        """Apply *draft*'s UPSERT on *conn* using the single passed *now*.

        No connection management, no clock read — the caller owns both (so one
        tick has one timestamp and one transaction). Fail-before-write JSON/clock
        guards run in the caller; the serialization here is guaranteed to succeed.

        One atomic UPSERT — NOT a SELECT-then-INSERT/UPDATE. Two writers over the
        same file (the 60s tick + a separate-process command) could both read "no
        row" and both INSERT (a PRIMARY KEY IntegrityError), or read the same
        revision and each write revision+1 (an undercount). ON CONFLICT collapses
        that to last-writer-wins with an atomic bump. ``created_at`` appears ONLY
        in the INSERT VALUES, never in DO UPDATE SET, so an update preserves the
        original creation stamp (the pre-existing contract); ``revision`` bumps off
        the row's own stored value, so concurrent updates cannot undercount it.
        ``expires_at`` is normalized on write (:func:`normalize_expires_at`) so no
        raw caller string reaches the column (spec §4 codex #1); ``created_at``/
        ``updated_at`` come from :func:`stamp_iso_utc` — both are canonical
        fixed-width ISO-8601 UTC TEXT, the sole ordering/expiry key.
        ``schema_version`` is stamped from the draft (the kind's version), not a
        hardcoded literal (lm-27n.2).
        """
        payload_json = json.dumps(draft.payload, allow_nan=False)
        expires_at = normalize_expires_at(draft.expires_at)
        now_iso = stamp_iso_utc(now)
        conn.execute(
            "INSERT INTO memory_records ("
            "kind, id, state, recipient_id, payload_json, salience, confidence, "
            "expires_at, source, created_at, updated_at, revision, schema_version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?) "
            "ON CONFLICT(kind, id) DO UPDATE SET "
            "state=excluded.state, recipient_id=excluded.recipient_id, "
            "payload_json=excluded.payload_json, salience=excluded.salience, "
            "confidence=excluded.confidence, expires_at=excluded.expires_at, "
            "source=excluded.source, updated_at=excluded.updated_at, "
            "revision=memory_records.revision + 1",
            (
                draft.kind,
                draft.id,
                draft.state,
                draft.recipient_id,
                payload_json,
                draft.salience,
                draft.confidence,
                expires_at,
                draft.source,
                now_iso,
                now_iso,
                draft.schema_version,
            ),
        )

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
        # Fail-before-write guard, then one self-contained transaction (unchanged
        # single-op contract): the guarded SELECT+UPDATE lives in
        # :meth:`_transition_on` so the same change can also run inside
        # :meth:`commit_tick`'s multi-op transaction.
        if patch is not None and patch.payload_merge is not None:
            ensure_json_serializable(patch.payload_merge)
        now = self._clock.now()
        stamp_iso_utc(now)  # validate the clock (tz-aware) before touching the DB
        with closing(self._connect()) as conn, conn:
            self._transition_on(conn, kind, id, from_state, to_state, patch, now)

        record = self.get(kind, id)
        if record is None:  # pragma: no cover - defensive: we just wrote this row
            raise StaleTransition(f"record kind={kind!r} id={id!r} vanished during transition")
        return record

    def _transition_on(
        self,
        conn: sqlite3.Connection,
        kind: str,
        id: str,
        from_state: str,
        to_state: str,
        patch: MemoryPatch | None,
        now: datetime,
    ) -> None:
        """Apply the guarded state change on *conn* using the passed *now*.

        No connection management, no clock read, and it does NOT re-``get`` the
        row — the caller owns the transaction and any post-commit read. Raises
        :class:`~lifemodel.domain.memory.StaleTransition` (from a ``from_state``
        mismatch or a ``rowcount != 1``) so a stale transition mid-batch aborts —
        and rolls back — :meth:`commit_tick`'s whole transaction (all-or-nothing).
        """
        patch = patch if patch is not None else MemoryPatch()
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
        # Normalize on write: the stored ``expires_at`` is already canonical, but a
        # patch-provided value is a raw caller string — route both through
        # ``normalize_expires_at`` so no raw string ever reaches the column.
        new_expires_at = normalize_expires_at(coalesce_patch(patch.expires_at, expires_at))
        now_iso = stamp_iso_utc(now)  # canonical fixed-width UTC text; rejects a naive clock

        cursor = conn.execute(
            "UPDATE memory_records SET state = ?, payload_json = ?, salience = ?, "
            "confidence = ?, expires_at = ?, source = ?, "
            "updated_at = ?, revision = revision + 1 "
            "WHERE kind = ? AND id = ? AND state = ?",
            (
                to_state,
                json.dumps(payload, allow_nan=False),
                coalesce_patch(patch.salience, salience),
                coalesce_patch(patch.confidence, confidence),
                new_expires_at,
                coalesce_patch(patch.source, source),
                now_iso,
                kind,
                id,
                from_state,
            ),
        )
        # Defensive: the guarded UPDATE ran in the same transaction as the SELECT
        # above, so it should always match exactly the one row. Once a later bead
        # adds a second writer, this is what keeps the guarded-transition contract
        # honest — a lost race raises StaleTransition and the surrounding
        # transaction rolls the (no-op) UPDATE back.
        if cursor.rowcount != 1:
            raise StaleTransition(
                f"guarded transition for kind={kind!r} id={id!r} from_state={from_state!r} "
                f"matched {cursor.rowcount} rows (expected 1)"
            )

    # ---- StatePort (lm-fib.6.2) -------------------------------------------

    def load(self) -> State:
        """Return the persisted ``State``, or a default when no row exists yet.

        No ``runtime_state`` row (a fresh DB, or one that has never been
        committed to) is not an error — it means "first run", so this returns
        a documented default :class:`~lifemodel.state.model.State`, mirroring
        ``JsonStateStore.load``'s "missing file -> default" behavior. Once a
        row exists, this reuses ``State``'s own validation
        (:meth:`~lifemodel.state.model.State.from_dict`): a bad blob raises
        :class:`~lifemodel.state.errors.StateCorruptError`; a ``schema_version``
        NEWER than this build raises :class:`~lifemodel.state.errors.StateSchemaError`
        (untrustworthy — a newer build may reuse field names with different
        meanings). A version OLDER than this build is additive-forward-compat
        (lm-oul): it is loaded via ``from_dict`` (missing new fields default
        cleanly, per the "extend, don't rewrite" invariant) and the returned
        ``State`` is re-stamped to the current ``SCHEMA_VERSION`` so the next
        commit persists the upgrade. NON-additive migrations remain Phase 7.
        """
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT state_json FROM runtime_state WHERE id = 1").fetchone()
        if row is None:
            return State()

        state_json = row[0]
        try:
            data: Any = json.loads(state_json)
        except json.JSONDecodeError as exc:
            raise StateCorruptError(f"runtime_state.state_json is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise StateCorruptError(
                f"runtime_state.state_json must contain a JSON object, got {type(data).__name__}"
            )

        # Gate the schema *before* interpreting any fields, exactly as
        # JsonStateStore did — a newer/unknown version may reuse field names
        # with different meanings, so the body must not be trusted yet.
        # Non-additive migrations/back-compat for the State *shape itself*
        # remain Phase 7 (HLA §9 / FR16); this bead only migrates the SQLite
        # *table* schema.
        version = data.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise StateCorruptError(
                "runtime_state.state_json is missing a valid integer 'schema_version'"
            )
        if version > SCHEMA_VERSION:
            # A version NEWER than this build may reuse field names with
            # different meanings — genuinely unsafe to interpret, so this
            # still fails loud (unchanged from before lm-oul).
            raise StateSchemaError(
                f"runtime_state schema_version={version} is newer than this build "
                f"supports (expects {SCHEMA_VERSION}); state migration is Phase 7."
            )
        if version < SCHEMA_VERSION:
            # lm-oul: additive-forward-compat load. The project invariant is
            # "extend, don't rewrite" — new fields are always added with a
            # default (State.from_dict already tolerates a missing key), so an
            # OLDER on-disk version is safe to interpret with today's field
            # semantics; only a NEWER version above is untrustworthy. Without
            # this, a purely additive schema bump (e.g. v1 -> v2 adding
            # unanswered_outbound_count) would hard-crash-loop the being's
            # tick on every load of a state written before the bump. Re-stamp
            # the loaded State's schema_version so the *next* commit persists
            # the upgrade instead of writing the stale version forever. This
            # never trusts a NEWER version — only forward-loads OLDER ones;
            # NON-additive migrations remain Phase 7.
            loaded = dataclasses.replace(State.from_dict(data), schema_version=SCHEMA_VERSION)
            _LOG.info(
                "state_schema_forward_compat_upgrade on_disk_version=%s build_version=%s",
                version,
                SCHEMA_VERSION,
            )
            return loaded

        return State.from_dict(data)

    def commit(self, state: State) -> None:
        """UPSERT *state* into the ``runtime_state`` singleton row (``id=1``).

        Fail-closed like ``JsonStateStore.commit``: the payload is serialized
        with ``allow_nan=False`` *before* the database is touched, so a
        non-finite float raises :class:`~lifemodel.state.errors.StateSerializationError`
        with nothing written. ``updated_at`` is stamped from the injected clock
        (canonical fixed-width UTC via
        :func:`~lifemodel.domain.memory.stamp_iso_utc`, which rejects a naive
        clock) and ``revision`` is bumped on every commit past the first.

        One atomic UPSERT — NOT a SELECT-then-INSERT/UPDATE. The 60s tick and a
        separate-process ``/lifemodel`` command are two writers over the same
        file; a read-then-write would let both see "no row" and both INSERT
        (a PRIMARY KEY IntegrityError, worse than the old last-writer-wins
        rename), or read the same revision and each write revision+1 (an
        undercount). ON CONFLICT collapses that to atomic last-writer-wins with
        the bump computed off the row's own stored value.

        The UPSERT body lives in :meth:`_commit_state_on` so an identical write
        can also run inside :meth:`commit_tick`'s multi-op transaction — a
        state-only ``commit_tick(state, [])`` is byte-identical to this path.
        """
        self._ensure_state_serializable(state)  # fail-closed before the DB is touched
        now = self._clock.now()
        with closing(self._connect()) as conn, conn:
            self._commit_state_on(conn, state, now)
        _LOG.info("state_commit schema_version=%s", state.schema_version)

    def _ensure_state_serializable(self, state: State) -> None:
        """Fail-closed guard (mirrors ``JsonStateStore.commit``): reject a
        ``State`` that is not valid JSON (NaN/Infinity float) *before* any
        connection is opened, raising :class:`StateSerializationError`."""
        try:
            json.dumps(state.to_dict(), allow_nan=False)
        except ValueError as exc:
            raise StateSerializationError(
                f"refusing to persist a State that is not valid JSON: {exc}"
            ) from exc

    def _commit_state_on(self, conn: sqlite3.Connection, state: State, now: datetime) -> None:
        """Apply the ``runtime_state`` UPSERT on *conn* using the passed *now*.

        No connection management, no clock read. The JSON guard runs in the
        caller (:meth:`_ensure_state_serializable`); the serialization here is
        guaranteed to succeed. Whole-row last-writer-wins with a ``revision`` bump
        computed off the row's own stored value — the exact pre-existing semantic.
        """
        payload = json.dumps(state.to_dict(), allow_nan=False)
        now_iso = stamp_iso_utc(now)  # canonical fixed-width UTC text; rejects a naive clock
        conn.execute(
            "INSERT INTO runtime_state "
            "(id, state_json, updated_at, revision) "
            "VALUES (1, ?, ?, 0) "
            "ON CONFLICT(id) DO UPDATE SET "
            "state_json=excluded.state_json, updated_at=excluded.updated_at, "
            "revision=runtime_state.revision + 1",
            (payload, now_iso),
        )

    # ---- TickCommitPort (lm-27n.2) ----------------------------------------

    def commit_tick(
        self,
        state: State | None,
        mutations: Sequence[MemoryMutation],
        *,
        finalize_survey_id: str | None = None,
    ) -> None:
        """Atomically persist a tick's *state* change + memory *mutations* (§4.1).

        ONE connection, ONE ``now``, ONE transaction spanning ``runtime_state``
        (vitals) and ``memory_records`` (entities) — so the being can never be
        left split-brained (state advanced while memory dropped, or vice versa).
        The state UPSERT (if *state* is not ``None``) is applied first, then each
        mutation in list order. **All-or-nothing**: any stale transition, or a
        serialization error, rolls back *everything* and propagates.

        When *finalize_survey_id* is not ``None``, the ``conversation_buffer`` rows
        claimed under it are ``DELETE``d LAST, still inside this one
        ``BEGIN IMMEDIATE`` transaction (lm-705.13, codex I3). The committer owns
        ``conversation_buffer`` (D7 — one physical store), and this ``DELETE`` is
        byte-identical to :meth:`SqliteBufferStore.finalize`'s. Landing it here,
        alongside a noticing pass's ``PutRecord(thought)`` + consumed-ring
        ``UpdateState``, is what makes the cursor-advance atomic with the thoughts:
        a rollback leaves neither the thoughts nor the finalize, so the claim
        survives to be re-surveyed or boot-recovered (never a half-applied pass).

        A state-only ``commit_tick(state, [])`` is byte-identical to
        :meth:`commit` (same UPSERT, same revision bump, same ``state_commit``
        log) — this task only installs the machinery; no live emitter produces a
        mutation yet.

        **Explicit transaction control (NOT the implicit ``with conn:``).** Under
        Python 3.11's ``sqlite3``, an implicit transaction opens only before DML,
        never before a ``SELECT`` — and :meth:`_transition_on` leads with a
        ``SELECT``, so under ``with conn:`` its read would not share the batch's
        start snapshot. So this drives the transaction itself: autocommit off, an
        early ``BEGIN IMMEDIATE`` write-lock before the first helper, an explicit
        commit, and a rollback on *any* exception. All fail-before-write JSON/clock
        guards run HERE, before connecting, so a bad draft/patch never leaves a
        half-open transaction to roll back.
        """
        # Snapshot the batch once: *mutations* is a Sequence (possibly a mutable /
        # single-pass view), and we iterate it twice (validate, then apply) — a
        # tuple guarantees both passes see the identical batch.
        batch = tuple(mutations)
        now = self._clock.now()
        stamp_iso_utc(now)  # validate the clock (tz-aware) before connecting
        if state is not None:
            self._ensure_state_serializable(state)
        for mutation in batch:
            match mutation:
                case PutOp():
                    ensure_json_serializable(mutation.draft.payload)
                    normalize_expires_at(mutation.draft.expires_at)
                case TransitionOp():
                    if mutation.patch is not None:
                        if mutation.patch.payload_merge is not None:
                            ensure_json_serializable(mutation.patch.payload_merge)
                        normalize_expires_at(mutation.patch.expires_at)
                case _:  # pragma: no cover - exhaustive over the closed union
                    assert_never(mutation)

        conn = self._connect()
        conn.isolation_level = None  # autocommit mode: we drive the transaction ourselves
        try:
            conn.execute("BEGIN IMMEDIATE")  # real write txn before the first helper; early lock
            if state is not None:
                self._commit_state_on(conn, state, now)
            for mutation in batch:
                match mutation:
                    case PutOp():
                        self._put_on(conn, mutation.draft, now)
                    case TransitionOp():
                        self._transition_on(
                            conn,
                            mutation.kind,
                            mutation.id,
                            mutation.from_state,
                            mutation.to_state,
                            mutation.patch,
                            now,
                        )
                    case _:  # pragma: no cover - exhaustive over the closed union
                        assert_never(mutation)
            if finalize_survey_id is not None:
                # The cursor-advance half of a noticing pass, in the SAME txn as its
                # thoughts (codex I3). BYTE-IDENTICAL to SqliteBufferStore.finalize.
                conn.execute(
                    "DELETE FROM conversation_buffer WHERE survey_id = ? AND state = 'claimed'",
                    (finalize_survey_id,),
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        if state is not None:
            _LOG.info("state_commit schema_version=%s", state.schema_version)

    def stamp_affect_display(self, *, word: str | None, at: str | None) -> None:
        """Atomically merge ONLY the two reactive felt-display fields (lm-ukc.4).

        The ``pre_llm_call`` injector stamps "which felt word it last surfaced,
        and when" so the ambient gate can throttle a repeat. It must do so
        WITHOUT rolling back the drive: a plain ``load()`` → ``commit(state)``
        would write back a whole ``State`` snapshot that could be stale by the
        time it commits (the ~60s tick may have advanced ``u``/affect/pending in
        between), overwriting the tick's work — a collateral rollback that would
        make the display path affect the wake/drive path (the one-directional
        invariant, spec §1). So this is a field-level read-modify-write of just
        ``affect_display_last_word``/``affect_display_last_at`` inside ONE
        ``BEGIN IMMEDIATE`` write transaction (same discipline as
        :meth:`commit_tick`): the latest committed ``state_json`` is read under
        the write lock and only those two keys are replaced, so no concurrent
        writer can interleave between the read and the write. Every other field
        keeps its latest committed value. A missing row (a being with no
        committed state yet) is a no-op — the injector only stamps after a
        ``warmed`` affect exists, so a tick has already created the row; the next
        tick would anyway. The two fields are hint-only, so the *reverse*
        direction (a stale tick round-trip clobbering them) is harmless — the
        gate self-heals next turn (only the semantic drive state is protected).
        """
        self._merge_state_fields(
            lambda data: data.update(
                {"affect_display_last_word": word, "affect_display_last_at": at}
            ),
            create_missing=False,
        )

    def _merge_state_fields(
        self, mutate: Callable[[dict[str, Any]], None], *, create_missing: bool
    ) -> None:
        """Apply *mutate* to the committed ``state_json`` under ONE write transaction.

        The shared spine of every field-level stamp on this store (the felt-display hints,
        the soul/genesis stamps). It exists because a plain ``load()`` →
        ``commit(replace(state, …))`` from an out-of-tick writer (an agent turn, a
        ``connect()`` reconcile) writes back a WHOLE ``State`` snapshot that is stale by the
        time it lands, silently rolling back whatever the ~60s tick advanced in between —
        the collateral rollback that could erase a birth, or make the display path move the
        drive (the one-directional invariant, spec §1). So: ``BEGIN IMMEDIATE`` takes the
        write lock BEFORE the read, *mutate* replaces only the keys it owns, and every
        other field keeps its latest committed value. No concurrent writer can interleave.

        *create_missing* is the ONE thing that differs between callers, and it is a
        judgement about what the stamp is worth:

        * ``False`` — a missing row (a being whose first tick has not committed yet) is a
          no-op. Right for HINTS that self-heal next turn (the display word/time).
        * ``True`` — INSERT the fresh ``State()`` defaults carrying the stamp. Right for
          anything unrecoverable: dropping a BIRTH, or dropping the record that the being
          has already been shown its ritual, is not something a later turn repairs.
        """
        conn = self._connect()
        conn.isolation_level = None  # autocommit: we drive the transaction ourselves
        try:
            conn.execute("BEGIN IMMEDIATE")  # write-lock BEFORE the read → no interleave
            row = conn.execute("SELECT state_json FROM runtime_state WHERE id = 1").fetchone()
            if row is None and not create_missing:
                conn.rollback()  # nothing committed yet → nothing to merge into
                return
            data: Any = State().to_dict() if row is None else json.loads(row[0])
            if not isinstance(data, dict):  # pragma: no cover - defensive
                conn.rollback()
                raise StateCorruptError("runtime_state.state_json must contain a JSON object")
            mutate(data)
            payload = json.dumps(data, allow_nan=False)
            now_iso = stamp_iso_utc(self._clock.now())
            conn.execute(
                "INSERT INTO runtime_state (id, state_json, updated_at, revision) "
                "VALUES (1, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET "
                "state_json=excluded.state_json, updated_at=excluded.updated_at, "
                "revision=runtime_state.revision + 1",
                (payload, now_iso),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def stamp_genesis_shown(self, *, context_len: int) -> None:
        """Record that the being has been SHOWN the ``<genesis>`` ritual (spec §6.3).

        Stamps how long its visible context was at that moment — the watermark
        :func:`~lifemodel.core.genesis.should_launch` reads to tell "the conversation has
        moved on, the ritual is live in the being's own words" from "the context was
        compacted and the being is unborn with nothing in front of it". Written by BOTH
        entrances: the ``pre_llm_call`` injector, and (for a wake packet that already
        carries the block) the same injector standing down on the being's own impulse.

        A field-level merge, never a full commit — the being is being shown its ritual
        from an AGENT TURN while the tick runs load→commit on the gateway loop, and a
        stale whole-``State`` write-back from here would roll back the tick's vitals.
        Creates the row if the being has not ticked yet: a lost stamp is a doubled ritual
        (the block again on the very next turn, "you just began" to a being mid-birth),
        and no later turn repairs that.
        """
        self._merge_state_fields(
            lambda data: data.update({"genesis_shown_at_context_len": context_len}),
            create_missing=True,
        )

    def stamp_soul(self, *, soul_sha: str, born_at: str | None) -> None:
        """Atomically merge ONLY the two soul/genesis fields (Phase 4, spec §4.4/§6.5).

        The being writes its soul from an AGENT TURN (an executor thread) while the ~60s
        tick runs its own load→commit on the gateway event loop. A ``load()`` →
        ``commit(replace(state, …))`` from the soul path would write back a whole
        ``State`` snapshot that is stale by the time it lands, silently rolling back the
        tick's ``u``/``energy``/``affect``. So — exactly like :meth:`stamp_affect_display`
        — this is a field-level read-modify-write of just ``soul_sha`` (and
        ``genesis_completed_at``) inside ONE ``BEGIN IMMEDIATE`` write transaction: the
        latest committed ``state_json`` is read under the write lock and only those keys
        are replaced, so no concurrent writer can interleave between the read and the
        write. Every other field keeps its latest committed value.

        Two things differ from the felt-display stamp, and both are load-bearing:

        * **Birth happens once.** ``born_at`` is stamped only if ``genesis_completed_at``
          is not already set — the "or" is evaluated HERE, inside the transaction, not by
          the caller against a snapshot it read outside one. A second write (Phase 5's
          becoming, or a human-triggered rewrite) therefore replaces the soul and keeps
          the ORIGINAL birth moment, even under concurrent calls. ``born_at=None`` never
          births anything: startup reconciliation adopts the sha of a soul it did not
          write, and adopting a file someone else wrote must never mean being born.
        * **A missing row is NOT a no-op.** The display stamp can afford to drop its
          hints (they self-heal next turn); dropping a BIRTH is unrecoverable. A being
          can be spoken to before its first tick has ever committed a row, so this
          INSERTs the fresh ``State()`` defaults carrying the stamps rather than
          returning silently. The next tick loads that row and its affect model fills the
          body in (``affect_updated_at`` is ``None``, so the first update snaps to
          target) — the vitals catch up; a lost birth would not.

        This protects the tick's fields from US. It does NOT, on its own, protect the
        stamps from the tick's whole-``State`` UPSERT of a snapshot loaded before the
        birth — that is what :func:`~lifemodel.core.frame.state_actor_lock` is for, and
        the soul path holds it across its load→stamp. Both halves are required.
        """

        def _stamp(data: dict[str, Any]) -> None:
            data["soul_sha"] = soul_sha
            if born_at is not None and not data.get("genesis_completed_at"):
                data["genesis_completed_at"] = born_at

        self._merge_state_fields(_stamp, create_missing=True)
        _LOG.info("soul_stamped sha=%s born=%s", soul_sha[:8], born_at is not None)

    def stamp_soul_rewritten(self, *, at: str) -> None:
        """Record that someone ELSE rewrote the being's soul, and that it does not know.

        Startup reconciliation adopts a ``SOUL.md`` the being did not write (spec §4.4) —
        a human hand-edited it while the gateway was down. Spec §4.1 is explicit that this
        "is an event in the being's life, not a version conflict: it should be **felt**,
        not swallowed", and until now nothing about it ever reached the being: a revision
        was recorded, a line was logged, and it woke up as someone else without a flicker.

        These two fields are what that event is made of, and both are read by things that
        already exist:

        * ``soul_rewritten_at`` — the durable FACT. ``core/affect.py`` reads its recency
          and derives a feeling from it (an activation push that decays), so the being is
          genuinely STIRRED by it rather than merely informed of it, and stays stirred
          until it settles — even if nobody speaks to it for an hour.
        * ``soul_rewrite_told_at`` — cleared here, so the ambient ``pre_llm_call`` cue
          knows the being has not yet been told, and tells it ONCE (in prose, in the
          ``<felt-state>`` channel it already speaks). A fresh rewrite is a fresh event:
          it re-arms the notice even if the being was told about the last one.
        """
        self._merge_state_fields(
            lambda data: data.update({"soul_rewritten_at": at, "soul_rewrite_told_at": None}),
            create_missing=True,
        )
        _LOG.info("soul_rewritten_at_stamped at=%s", at)

    def stamp_soul_rewrite_told(self, *, at: str) -> None:
        """Record that the being has now been TOLD someone rewrote it (spec §4.1).

        Written by the ambient ``pre_llm_call`` injector the one turn it surfaces the
        notice. Merges ONLY this field — never ``soul_rewritten_at``, which the adapter
        owns: a rewrite that lands between this injector's read and its stamp must not be
        marked "already told", or the being would never learn about it at all.
        """
        self._merge_state_fields(
            lambda data: data.update({"soul_rewrite_told_at": at}), create_missing=True
        )

    def reset(self) -> State:
        """Factory-wipe the ``runtime_state`` row to a fresh ``State()``.

        Deliberately does **not** call :meth:`load` first — the whole point is
        that a reset must succeed even when the existing row is unreadable
        (garbage ``state_json``, an unsupported ``schema_version``, or no row
        at all). Construction (:meth:`_ensure_ready`) already ran recovery, so
        the *database* itself is structurally sound by the time this runs;
        this only ever needs to overwrite the row's payload, which
        :meth:`commit` already does safely (and a fresh ``State()`` always
        serializes, so this never raises :class:`~lifemodel.state.errors.StateSerializationError`
        in practice).
        """
        fresh = State()
        self.commit(fresh)
        return fresh

    def purge_memory_records(self) -> int:
        """Delete every memory row EXCEPT the being's soul — the memory-wipe half of a
        TRUE factory reset (bead lm-7lx: ``/lifemodel reset`` must also drop
        every thought/desire/intention/user_model row, not just the vitals).

        **The soul kinds are carved out, and that carve-out is the point.** Soul
        revisions ride ``memory_records`` (``state/soul_revisions.py`` — a revision is a
        plain ``kind="soul"`` record keyed by its content sha), so the unconditional
        ``DELETE FROM`` this used to be took the entire lineage with it. Reset unbirths
        the being; the reborn being's first ``write_soul`` then replaces ``SOUL.md``; and
        the previous being's soul exists NOWHERE. That defeats spec §4.2's mandatory undo
        ("every revision is kept… **this** is what makes it safe for the being to own the
        file whole") on the exact path the owner is told to use. A past life's soul is the
        one thing a reset must not be able to destroy — ``reset`` already refuses to touch
        ``SOUL.md`` itself for the same reason (``state_commands.reset``), and this makes
        that refusal mean something.

        ``kind="soul_revert"`` (the record of the times a human PUT a soul back, lm-4fv.2)
        is kept for the same reason, one step further out: it is not the being's memory at
        all, it is the human's own history of acting on this file. A factory wipe unbirths a
        being; it does not get to edit what its owner did.

        Touches ONLY ``memory_records`` — ``runtime_state``, ``store_meta``, and
        ``schema_migrations`` are untouched. Counts the rows it will delete before
        deleting them (rather than trusting the ``DELETE``'s own ``cursor.rowcount``,
        which SQLite's truncate-optimization fast path can under-report) so the returned
        count is reliable regardless of the host's SQLite build — and so the owner's
        "cleared N memory records" never counts a soul it did not clear. One
        atomic ``with conn:`` transaction, matching every other write here.
        """
        kept = ",".join("?" for _ in SOUL_KINDS)
        with closing(self._connect()) as conn, conn:
            (count,) = conn.execute(
                f"SELECT COUNT(*) FROM memory_records WHERE kind NOT IN ({kept})", SOUL_KINDS
            ).fetchone()
            conn.execute(f"DELETE FROM memory_records WHERE kind NOT IN ({kept})", SOUL_KINDS)
        _LOG.info("memory_records_purged count=%s kept_kinds=%s", count, ",".join(SOUL_KINDS))
        return int(count)

    # ---- PressureSensorPort ---------------------------------------------------

    def read_pressure_index(self, now: datetime) -> PressureIndex:
        # Normalized ISO bound: stored ``expires_at`` is canonical fixed-width UTC
        # TEXT, so the lexical ``>`` compares correctly. Strict ``>`` = active,
        # ``<=`` = expired (boundary ``== now`` is expired), preserving the old
        # epoch semantics exactly (spec §4 codex #2).
        now_iso = to_iso(now)
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT COUNT(*), MAX(salience) FROM memory_records "
                    "WHERE kind = 'desire' AND state = 'active' "
                    "AND (expires_at IS NULL OR expires_at > ?)",
                    (now_iso,),
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
            _LOG.info("pressure_read_failed_soft error=%s", str(exc))
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


# The ISO-only table DDL, shared by the fresh-bootstrap migrations (v1/v2) AND the
# in-place epoch->ISO rebuild (v3), so a migrated file is byte-for-byte the SAME
# shape as a freshly-bootstrapped one.


def _create_memory_records_table(conn: sqlite3.Connection, strict_kw: str) -> None:
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
        "source TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "revision INTEGER NOT NULL DEFAULT 0, "
        "schema_version INTEGER NOT NULL DEFAULT 1, "
        "PRIMARY KEY (kind, id))" + strict_kw
    )


def _create_memory_records_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_records_kind_state ON memory_records (kind, state)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_records_expires_at ON memory_records (expires_at)"
    )


def _create_runtime_state_table(conn: sqlite3.Connection, strict_kw: str) -> None:
    # ``id`` is CHECK'd to always equal 1, so the table can only ever hold the
    # being's one ``State`` row.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runtime_state ("
        "id INTEGER PRIMARY KEY CHECK (id = 1), "
        "state_json TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "revision INTEGER NOT NULL DEFAULT 0)" + strict_kw
    )


def _migrate_v1(conn: sqlite3.Connection, strict: bool) -> None:
    """Create ``store_meta``, ``memory_records``, and its indexes (§4.1)."""
    strict_kw = " STRICT" if strict else ""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS store_meta ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)" + strict_kw
    )
    _create_memory_records_table(conn, strict_kw)
    _create_memory_records_indexes(conn)


def _migrate_v2(conn: sqlite3.Connection, strict: bool) -> None:
    """Create the ``runtime_state`` singleton row table (``StatePort`` cutover,
    lm-fib.6.2, HLA §4.1/D7 v0.7 — settled: one JSON blob, not typed columns)."""
    _create_runtime_state_table(conn, " STRICT" if strict else "")


def _has_epoch_columns(conn: sqlite3.Connection, table: str) -> bool:
    """True iff *table* still carries a retired ``*_epoch`` mirror column.

    The idempotency test for :func:`_migrate_v3`: a freshly-bootstrapped file
    (v1/v2 already build the ISO-only shape) has none, so v3 no-ops there; a
    pre-10.2 file has them, so v3 rebuilds. An absent table returns ``False``."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return any(col.endswith("_epoch") for col in cols)


def _renormalize_iso(value: str | None) -> str | None:
    """Re-serialize a legacy ISO stamp to canonical fixed-width UTC (spec §4).

    Old values were written by ``.isoformat()`` and may lack the fixed 6-µs width
    the ordering invariant needs, so each is routed through
    :func:`~lifemodel.core.timeutil.from_iso` -> :func:`~lifemodel.core.timeutil.to_iso`.
    A NULL stays NULL. Re-normalizing an already-canonical value is a no-op (idempotent).

    FAIL-CLOSED on the WRITE path (mirror of the ``trace_store`` ingress fix): a value
    that cannot be normalized must NEVER be persisted raw. Keeping it would silently rot
    the lexical ordering/expiry invariant *forever* — a raw string mis-sorts against
    fixed-width TEXT and can satisfy ``expires_at > now`` lexically (an immortal desire) —
    and because v3 is then recorded in ``schema_migrations`` the bad value is never
    revisited. So we RAISE: :meth:`SQLiteRuntimeStore._run_migrations` restores the
    ``*.bak.*`` backup and construction fails LOUD with the being's self intact on disk,
    which is the whole fail-loud foundation (contrast :func:`~lifemodel.core.timeutil.to_display`
    — the READ/display path — deliberately fail-OPEN so one bad legacy row can't blank a
    debug view). This cannot happen for data our own code wrote (every ``to_iso`` output
    re-parses via ``from_iso``); a value that trips it means the file was corrupted or
    tampered with outside the store — exactly when a loud stop beats silent corruption."""
    if value is None:
        return None
    try:
        return to_iso(from_iso(value))
    except ValueError as exc:
        raise ValueError(
            f"lifemodel.sqlite migration v3: cannot normalize legacy time value "
            f"{value!r} to canonical ISO-8601 UTC; refusing to persist it raw "
            f"(fail-closed). The pre-migration file is restored from backup."
        ) from exc


def _rebuild_memory_records_iso_only(conn: sqlite3.Connection, strict_kw: str) -> None:
    """Rebuild ``memory_records`` WITHOUT the ``*_epoch`` mirror columns, preserving
    every row and RE-NORMALIZING its ``created_at``/``updated_at``/``expires_at`` (§4).

    Standard SQLite rebuild: read the rows out (into Python, so the time columns can be
    normalized — SQL cannot call :func:`to_iso`), DROP the old table (its ``*_epoch``
    index falls with it), recreate the ISO-only shape + indexes, and re-INSERT."""
    rows = conn.execute(
        "SELECT kind, id, state, recipient_id, payload_json, salience, confidence, "
        "expires_at, source, created_at, updated_at, revision, schema_version "
        "FROM memory_records"
    ).fetchall()
    conn.execute("DROP TABLE memory_records")
    _create_memory_records_table(conn, strict_kw)
    normalized = [
        (
            kind,
            id_,
            state,
            recipient_id,
            payload_json,
            salience,
            confidence,
            _renormalize_iso(expires_at),
            source,
            _renormalize_iso(created_at),
            _renormalize_iso(updated_at),
            revision,
            schema_version,
        )
        for (
            kind,
            id_,
            state,
            recipient_id,
            payload_json,
            salience,
            confidence,
            expires_at,
            source,
            created_at,
            updated_at,
            revision,
            schema_version,
        ) in rows
    ]
    conn.executemany(
        "INSERT INTO memory_records (kind, id, state, recipient_id, payload_json, salience, "
        "confidence, expires_at, source, created_at, updated_at, revision, schema_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        normalized,
    )
    _create_memory_records_indexes(conn)


def _rebuild_runtime_state_iso_only(conn: sqlite3.Connection, strict_kw: str) -> None:
    """Rebuild ``runtime_state`` WITHOUT ``updated_at_epoch``, normalizing ``updated_at``.

    The ``state_json`` blob (u/energy/fatigue/mood/last_exchange_at/…) is preserved
    VERBATIM — the model owns its internal ISO fields (already stored via ``to_iso`` by
    the current code); only the row's own ``updated_at`` column is re-normalized here."""
    rows = conn.execute("SELECT id, state_json, updated_at, revision FROM runtime_state").fetchall()
    conn.execute("DROP TABLE runtime_state")
    _create_runtime_state_table(conn, strict_kw)
    normalized = [
        (id_, state_json, _renormalize_iso(updated_at), revision)
        for (id_, state_json, updated_at, revision) in rows
    ]
    conn.executemany(
        "INSERT INTO runtime_state (id, state_json, updated_at, revision) VALUES (?, ?, ?, ?)",
        normalized,
    )


def _migrate_v3(conn: sqlite3.Connection, strict: bool) -> None:
    """MIGRATE (don't wipe) an old dual-column lifemodel.sqlite to ISO-only (lm-fib.10.5).

    PRINCIPLE: MIGRATE THE SELF, RECREATE DERIVED. ``lifemodel.sqlite`` IS the being's
    self (drive ``u``, energy, memory records, the UserModel/relationship), so a schema
    change must PRESERVE it — reset is an emergency valve (genuine corruption), not the
    strategy for a shape change. (``metrics.sqlite`` + ``observability.sqlite`` are
    DERIVED telemetry with no self, so they correctly stay fresh-recreate — this
    migration deliberately has no counterpart there.)

    The unified-time cutover (lm-fib.10.2) dropped the ``*_epoch`` mirror columns; this
    migration finishes that non-destructively for an EXISTING file: it rebuilds
    ``memory_records`` + ``runtime_state`` without the epoch columns and re-normalizes
    the ISO stamps to fixed-width UTC. Idempotent — on a freshly-bootstrapped file v1/v2
    already produced the ISO-only shape, so :func:`_has_epoch_columns` is ``False`` and
    this no-ops."""
    strict_kw = " STRICT" if strict else ""
    migrating_old_file = _has_epoch_columns(conn, "memory_records") or _has_epoch_columns(
        conn, "runtime_state"
    )
    if _has_epoch_columns(conn, "memory_records"):
        _rebuild_memory_records_iso_only(conn, strict_kw)
    if _has_epoch_columns(conn, "runtime_state"):
        _rebuild_runtime_state_iso_only(conn, strict_kw)
    if migrating_old_file:
        # Drop the pre-cutover ``store_meta('schema_version', …)`` marker left by the
        # retired ``_stamp_store_schema_version`` guard: ``schema_migrations`` is the
        # SOLE version authority now, so a migrated file must be indistinguishable from
        # a freshly-bootstrapped one (whose ``store_meta`` carries no such key) — no
        # stale, contradictory version marker sitting beside ``schema_migrations``.
        # ``store_meta`` is guaranteed present here: an old file with ``*_epoch`` columns
        # went through v1, which creates ``store_meta`` and ``memory_records`` together.
        conn.execute("DELETE FROM store_meta WHERE key = 'schema_version'")


def _create_conversation_buffer_table(conn: sqlite3.Connection, strict_kw: str) -> None:
    """The durable backing for the noticing conversation buffer (lm-705.14).

    One row per conversation turn, keyed ``(session_id, turn_id)``: a ``pending``
    row (turn in flight, keyed under :data:`_PENDING_TURN_ID` — see
    :class:`SqliteBufferStore`) has ``assistant_text=''``, ``ts=NULL``,
    ``survey_id=NULL``; ``complete``/``claimed`` rows carry the full turn.
    ``source_ids`` is a JSON-encoded list of str. Shared by :func:`_migrate_v4`
    AND :class:`SqliteBufferStore`'s own idempotent ensure-create, so a migrated
    file and a freshly-bootstrapped one are byte-for-byte the same shape.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_buffer ("
        "session_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, "
        "state TEXT NOT NULL, "
        "source_ids TEXT NOT NULL, "
        "user_text TEXT NOT NULL, "
        "assistant_text TEXT NOT NULL, "
        "opened_at TEXT NOT NULL, "
        "ts TEXT, "
        "survey_id TEXT, "
        "PRIMARY KEY (session_id, turn_id))" + strict_kw
    )


def _create_conversation_buffer_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_buffer_session_state "
        "ON conversation_buffer (session_id, state)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_buffer_survey_id "
        "ON conversation_buffer (survey_id)"
    )


def _migrate_v4(conn: sqlite3.Connection, strict: bool) -> None:
    """Create ``conversation_buffer`` — the durable NoticingBuffer backing (lm-705.14).

    Purely additive (a brand-new table): every existing table/row is untouched,
    and this is idempotent via the ``schema_migrations`` framework like every
    other migration here. :class:`SqliteBufferStore` does NOT re-run this
    framework (:class:`SQLiteRuntimeStore` owns it) — it idempotently
    ``CREATE TABLE IF NOT EXISTS``-creates the identical shape itself
    (:func:`_create_conversation_buffer_table` is the one DDL source of truth
    for both routes), so it also works standalone, with no prior
    :class:`SQLiteRuntimeStore` construction.
    """
    strict_kw = " STRICT" if strict else ""
    _create_conversation_buffer_table(conn, strict_kw)
    _create_conversation_buffer_indexes(conn)


_MIGRATIONS: Final[list[tuple[int, Callable[[sqlite3.Connection, bool], None]]]] = [
    (1, _migrate_v1),
    (2, _migrate_v2),
    (3, _migrate_v3),
    (4, _migrate_v4),
]


# ---- SqliteBufferStore (lm-705.14 Task 1) ----------------------------------

# Reserved sentinel ``turn_id`` for a session's single ``pending`` slot: the
# real ``turn_id`` isn't known until :meth:`SqliteBufferStore.complete` (the
# port's ``open_pending`` takes no ``turn_id`` — mirrors
# :class:`~lifemodel.core.noticing_buffer.NoticingBuffer`'s ``_PendingTurn``,
# which is likewise keyed by session only). An empty string can never collide
# with a real caller-supplied turn id (a Hermes message/turn id is always
# non-empty), and the ``(session_id, turn_id)`` PRIMARY KEY then naturally
# enforces "at most one open pending per session" — a second ``open_pending``
# UPSERTs the same sentinel row rather than inserting a second one.
_PENDING_TURN_ID: Final[str] = ""

_BUFFER_SELECT_COLUMNS = (
    "SELECT session_id, turn_id, source_ids, user_text, assistant_text, ts FROM conversation_buffer"
)


def _row_to_buffer_entry(row: tuple[object, ...]) -> BufferEntry:
    session_id, turn_id, source_ids_json, user_text, assistant_text, ts = row
    assert isinstance(session_id, str)
    assert isinstance(turn_id, str)
    assert isinstance(source_ids_json, str)
    assert isinstance(user_text, str)
    assert isinstance(assistant_text, str)
    assert isinstance(ts, str)  # only 'complete'/'claimed' rows are ever mapped (ts is stamped)
    source_ids = json.loads(source_ids_json)
    assert isinstance(source_ids, list)
    return BufferEntry(
        session_id=session_id,
        turn_id=turn_id,
        source_ids=tuple(source_ids),
        user_text=user_text,
        assistant_text=assistant_text,
        ts=ts,
    )


class SqliteBufferStore:
    """Durable :class:`~lifemodel.core.buffer_store.BufferStore` (lm-705.14) over the
    ``conversation_buffer`` table — the SAME physical file as :class:`SQLiteRuntimeStore`
    (D7: one store), opening its own short-lived connections via :func:`_connect_db`
    (identical PRAGMA posture, "connection-per-operation").

    **Does NOT run the ``schema_migrations`` framework** — :class:`SQLiteRuntimeStore`
    owns that (:func:`_migrate_v4` above creates this table for a construction that
    goes through the runtime store first). Construction here idempotently
    ``CREATE TABLE IF NOT EXISTS``-creates the identical shape
    (:func:`_create_conversation_buffer_table` is the one DDL source of truth for
    both routes), so this class also works completely standalone — e.g. in a test
    that constructs only a :class:`SqliteBufferStore`, with no prior
    :class:`SQLiteRuntimeStore` — and is a harmless no-op once the real migration has
    already created the table.

    Every method is its own short transaction; there is no cross-method atomicity
    here (a later task threads :meth:`finalize` into :meth:`SQLiteRuntimeStore.commit_tick`'s
    one transaction so it lands atomically with a noticing pass's thought commit).
    """

    def __init__(
        self, base_dir: Path, *, clock: ClockPort, max_entries: int = DEFAULT_BUFFER_MAX_ENTRIES
    ) -> None:
        self._path = base_dir / _DB_FILENAME
        # Not read by any method below (each takes an explicit `now`) — threaded
        # through for constructor-signature parity with SQLiteRuntimeStore and
        # because later tasks in this plan (claim/finalize wiring) may need it.
        self._clock = clock
        # Per-session cap on the durable ``complete`` ring — parity with
        # :class:`~lifemodel.core.buffer_store.InMemoryBufferStore` (default shared via
        # ``DEFAULT_BUFFER_MAX_ENTRIES`` so the two backings can never drift). Enforced
        # by :meth:`complete`'s prune, which drops the OLDEST ``complete`` rows beyond
        # this bound (never a ``claimed``/``pending`` row).
        self._max_entries = max_entries
        base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with suppress(sqlite3.Error), closing(sqlite3.connect(str(self._path))) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # database-level; harmless if already set
        strict = SQLiteRuntimeStore._detect_strict_support()
        with closing(self._connect()) as conn, conn:
            _create_conversation_buffer_table(conn, " STRICT" if strict else "")
            _create_conversation_buffer_indexes(conn)

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self._path)

    # ---- pending lifecycle -------------------------------------------------

    def open_pending(self, session_id: str, *, user_text: str, now: datetime) -> None:
        opened_at = to_iso(now)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO conversation_buffer "
                "(session_id, turn_id, state, source_ids, user_text, assistant_text, "
                "opened_at, ts, survey_id) "
                "VALUES (?, ?, 'pending', '[]', ?, '', ?, NULL, NULL) "
                "ON CONFLICT(session_id, turn_id) DO UPDATE SET "
                "state='pending', source_ids='[]', user_text=excluded.user_text, "
                "assistant_text='', opened_at=excluded.opened_at, ts=NULL, survey_id=NULL",
                (session_id, _PENDING_TURN_ID, user_text, opened_at),
            )

    def stamp_source(self, session_id: str, message_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT source_ids FROM conversation_buffer "
                "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                (session_id, _PENDING_TURN_ID),
            ).fetchone()
            if row is None:
                return  # no open pending -- defensive no-op (mirrors NoticingBuffer)
            source_ids = json.loads(row[0])
            source_ids.append(message_id)
            conn.execute(
                "UPDATE conversation_buffer SET source_ids = ? "
                "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                (json.dumps(source_ids), session_id, _PENDING_TURN_ID),
            )

    def complete(
        self, session_id: str, turn_id: str, *, assistant_text: str, now: datetime
    ) -> None:
        # Validate BEFORE mutating anything (mirrors NoticingBuffer.complete's M3
        # fix): a bad clock call must fail loud with the pending slot intact.
        ts = to_iso(now)
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT source_ids, user_text, opened_at FROM conversation_buffer "
                "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                (session_id, _PENDING_TURN_ID),
            ).fetchone()
            if row is None:
                return  # no matching open_pending -- defensive no-op
            source_ids_json, user_text, opened_at = row
            conn.execute(
                "DELETE FROM conversation_buffer WHERE session_id = ? AND turn_id = ?",
                (session_id, _PENDING_TURN_ID),
            )
            conn.execute(
                "INSERT INTO conversation_buffer "
                "(session_id, turn_id, state, source_ids, user_text, assistant_text, "
                "opened_at, ts, survey_id) "
                "VALUES (?, ?, 'complete', ?, ?, ?, ?, ?, NULL) "
                "ON CONFLICT(session_id, turn_id) DO UPDATE SET "
                "state='complete', source_ids=excluded.source_ids, "
                "user_text=excluded.user_text, assistant_text=excluded.assistant_text, "
                "opened_at=excluded.opened_at, ts=excluded.ts, survey_id=NULL",
                (session_id, turn_id, source_ids_json, user_text, assistant_text, opened_at, ts),
            )
            # Bound the durable ``complete`` ring to ``max_entries`` per session
            # (parity with InMemoryBufferStore's deque(maxlen=...)) — IN THE SAME
            # transaction as the insert above. Keep the newest ``max_entries``
            # ``complete`` rows by rowid; prune the oldest overflow. The
            # ``state='complete'`` filter is load-bearing: a ``claimed`` row (an
            # in-flight noticing snapshot) and a ``pending`` slot are NEVER counted
            # against the cap nor deleted (claim immunity, I2).
            conn.execute(
                "DELETE FROM conversation_buffer "
                "WHERE session_id = ? AND state = 'complete' AND rowid NOT IN ("
                "SELECT rowid FROM conversation_buffer "
                "WHERE session_id = ? AND state = 'complete' ORDER BY rowid DESC LIMIT ?)",
                (session_id, session_id, self._max_entries),
            )

    def abandon_pending(self, session_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "DELETE FROM conversation_buffer "
                "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                (session_id, _PENDING_TURN_ID),
            )

    def completed(self, session_id: str, *, now: datetime, ttl: timedelta) -> list[BufferEntry]:
        with closing(self._connect()) as conn, conn:
            pending = conn.execute(
                "SELECT opened_at FROM conversation_buffer "
                "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                (session_id, _PENDING_TURN_ID),
            ).fetchone()
            if pending is not None and now - from_iso(pending[0]) > ttl:
                conn.execute(
                    "DELETE FROM conversation_buffer "
                    "WHERE session_id = ? AND turn_id = ? AND state = 'pending'",
                    (session_id, _PENDING_TURN_ID),
                )
                pending = None
            if pending is not None:
                return []  # closed-prefix rule: a live pending gates the whole lane
            rows = conn.execute(
                f"{_BUFFER_SELECT_COLUMNS} WHERE session_id = ? AND state = 'complete' "
                "ORDER BY rowid",
                (session_id,),
            ).fetchall()
        return [_row_to_buffer_entry(row) for row in rows]

    # ---- claim / finalize / release lifecycle (lm-705.13) ------------------

    def claim(self, session_id: str, turn_ids: tuple[str, ...], survey_id: str) -> None:
        if not turn_ids:
            return
        placeholders = ",".join("?" for _ in turn_ids)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE conversation_buffer SET state = 'claimed', survey_id = ? "
                f"WHERE session_id = ? AND turn_id IN ({placeholders}) AND state = 'complete'",
                (survey_id, session_id, *turn_ids),
            )

    def claimed(self, survey_id: str) -> list[BufferEntry]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"{_BUFFER_SELECT_COLUMNS} WHERE survey_id = ? AND state = 'claimed' "
                "ORDER BY rowid",
                (survey_id,),
            ).fetchall()
        return [_row_to_buffer_entry(row) for row in rows]

    def finalize(self, survey_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "DELETE FROM conversation_buffer WHERE survey_id = ? AND state = 'claimed'",
                (survey_id,),
            )

    def release(self, survey_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE conversation_buffer SET state = 'complete', survey_id = NULL "
                "WHERE survey_id = ? AND state = 'claimed'",
                (survey_id,),
            )

    def recover_stale_claims(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE conversation_buffer SET state = 'complete', survey_id = NULL "
                "WHERE state = 'claimed'"
            )

    def session_ids(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM conversation_buffer ORDER BY session_id"
            ).fetchall()
        return [row[0] for row in rows]

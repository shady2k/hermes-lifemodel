"""JSON file adapter for :class:`StatePort` — the safety-critical state store.

Writes ``<base_dir>/state.json``, where *base_dir* is injected via the
constructor (DI, HLA §13). The composition root (task 0.4) wires
``base_dir`` = the profile-home state dir from ``lifemodel.paths.state_dir``;
tests inject a ``tmp_path``. The adapter imports nothing from Hermes.

**Atomicity (crash safety).** ``commit`` writes to a uniquely-named temp file
in the *same directory* (so the final rename stays on one filesystem),
``fsync``s it, then ``os.replace``s it onto ``state.json`` — an atomic rename
on POSIX. A reader therefore never observes a half-written file, and a crash or
error mid-write leaves the previous good ``state.json`` untouched; the temp
file is cleaned up rather than left to linger.

*Crash-safety vs. durability (Phase 1).* The Phase-1 guarantee is **integrity**:
``state.json`` is never torn, partial, or lost to a half-completed write — a
reader always sees either the whole previous state or the whole new one. Full
**durability** of the very last commit (surviving power loss / OS crash in the
window right after ``os.replace``) is only *best-effort*: the file's contents
are ``fsync``ed before the rename, and the directory entry is ``fsync``ed
best-effort afterwards (:meth:`_fsync_dir`), but that dir ``fsync`` is not
portable everywhere and its failure is intentionally swallowed. Hard durability
guarantees are Phase 7 (HLA §9).

**Scope (deliberately Phase 1).** No lock, no merge/retry, no snapshot recovery,
no migrations — those are Phase 7 (HLA §9). ``load`` maps every "cannot read the
persisted state" case to a typed error: missing file → documented default;
unsupported ``schema_version`` → :class:`StateSchemaError`; unparseable or
malformed (bad JSON, invalid UTF-8, wrong shape/types, non-finite floats) →
:class:`StateCorruptError`. ``commit`` refuses to persist a ``State`` that would
not be valid JSON (a non-finite float) with :class:`StateSerializationError`,
before touching the filesystem.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..logging import EventLogger, get_logger
from .errors import StateCorruptError, StateSchemaError, StateSerializationError
from .model import SCHEMA_VERSION, State

_STATE_FILENAME = "state.json"
#: Temp files share a recognizable prefix so cleanup and tests can find them.
_TMP_PREFIX = ".state-"
_TMP_SUFFIX = ".tmp"


class JsonStateStore:
    """A :class:`StatePort` backed by a single human-readable JSON file."""

    def __init__(self, base_dir: Path, *, logger: EventLogger | None = None) -> None:
        self._base_dir = base_dir
        self._path = base_dir / _STATE_FILENAME
        self._log = logger or get_logger("lifemodel.state")

    def load(self) -> State:
        """Read and return the persisted state.

        Missing file (or missing base dir) → the documented default ``State``
        (read is non-mutating; it never creates the directory). An unsupported
        ``schema_version`` raises :class:`StateSchemaError`; anything that
        cannot be parsed/interpreted (invalid UTF-8, bad JSON, wrong shape or
        types, non-finite floats) raises :class:`StateCorruptError`.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return State()
        except UnicodeDecodeError as exc:
            # Invalid bytes are malformed state, not an I/O error — honor the
            # typed-error contract instead of letting UnicodeDecodeError escape.
            raise StateCorruptError(f"{self._path} is not valid UTF-8: {exc}") from exc

        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StateCorruptError(f"{self._path} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise StateCorruptError(
                f"{self._path} must contain a JSON object, got {type(data).__name__}"
            )

        # Gate the schema *before* interpreting any fields: a newer/unknown
        # version may reuse field names with different meanings, so we must not
        # trust the body. Migrations/back-compat are Phase 7 (HLA §9 / FR16).
        version = data.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise StateCorruptError(f"{self._path} is missing a valid integer 'schema_version'")
        if version != SCHEMA_VERSION:
            raise StateSchemaError(
                f"{self._path} schema_version={version} is not supported by this build "
                f"(expects {SCHEMA_VERSION}); state migration is Phase 7."
            )

        return State.from_dict(data)

    def commit(self, state: State) -> None:
        """Atomically persist *state* (tmp file + ``fsync`` + ``os.replace``).

        Creates the base directory on demand — ``paths.state_dir`` only computes
        the path, so the store owns first-write creation. On any failure the
        temp file is removed and the previous ``state.json`` is left intact;
        the error propagates to the caller.

        Fail-closed: the payload is serialized with ``allow_nan=False`` *before*
        the filesystem is touched, so a non-finite float (which ``json`` would
        emit as an invalid JSON token) raises :class:`StateSerializationError`
        with nothing written and no temp file created.
        """
        try:
            payload = (
                json.dumps(state.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
        except ValueError as exc:
            # Out-of-range float (NaN/Infinity): refuse to persist poison. Raised
            # before mkdir/mkstemp, so the previous good state.json is untouched.
            raise StateSerializationError(
                f"refusing to persist a State that is not valid JSON: {exc}"
            ) from exc

        self._base_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self._base_dir, prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())  # data hits disk before it becomes visible
            os.replace(tmp_path, self._path)  # atomic rename onto state.json
        except BaseException:
            # Write or rename failed: drop the temp file (never leave a partial
            # artifact) and re-raise so the previous good state.json stands.
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

        self._fsync_dir()
        self._log.info(
            "state_commit",
            schema_version=state.schema_version,
            path=str(self._path),
        )

    def _fsync_dir(self) -> None:
        """Best-effort ``fsync`` of the base dir so the rename itself is durable.

        The rename already gives crash-atomic *visibility*; this only
        strengthens durability and is not portable everywhere (e.g. Windows),
        so failures are swallowed.
        """
        try:
            dir_fd = os.open(self._base_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            with contextlib.suppress(OSError):
                os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

"""The memory-record domain model — entities the being remembers (HLA §4.1/D7).

`MemoryPort` (the read/write contract, :mod:`lifemodel.ports.memory`) and
`PressureSensorPort` (the pressure-index read, :mod:`lifemodel.ports.pressure`)
both depend only on the plain, JSON-native value types defined here — never on
a concrete store — so higher layers stay storage-agnostic (HLA §13) and the
same contract runs against an in-memory fake or the real
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (roadmap lm-fib.6.1).

Three dataclasses split write/read/patch so each carries only the fields that
make sense for its direction: :class:`MemoryDraft` (put input — no
store-stamped fields), :class:`MemoryRecord` (get/find output — the full
persisted row), :class:`MemoryPatch` (transition's partial update, where
``None`` means "leave unchanged" for every field except ``payload_merge``,
which shallow-merges rather than replacing). :class:`PressureIndex` is the
read model :meth:`~lifemodel.ports.pressure.PressureSensorPort.read_pressure_index`
returns.

This is purely additive (lm-fib.6.1): nothing here is wired into the live tick,
composition root, or ``core/proactive.py``. It imports nothing from Hermes and
stays unit-testable off-host.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias, TypeVar

#: Any value that round-trips through :mod:`json` with no custom encoder.
#: Recursive: a ``JsonValue`` is a JSON scalar, or a list/dict of ``JsonValue``.
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
#: A JSON *object* specifically (every ``MemoryDraft``/``MemoryRecord`` payload
#: is a top-level object, not an arbitrary scalar or array).
JsonObject: TypeAlias = dict[str, JsonValue]


class MemoryPortError(Exception):
    """Base class for every :class:`~lifemodel.ports.memory.MemoryPort` failure.

    Named to avoid shadowing the builtin :class:`MemoryError` (an unrelated
    out-of-memory condition), mirroring :class:`~lifemodel.state.errors.StateError`
    as the root of a small per-package taxonomy.
    """


class StaleTransition(MemoryPortError):
    """Raised by ``transition()`` when its guarded ``UPDATE`` matches no row.

    Covers both cases a caller must treat as "my view was stale" (HLA §4.1):
    the record does not exist at all, or it exists but is no longer in
    ``from_state`` (someone else already transitioned it). The message text
    distinguishes the two — mirroring what a follow-up :meth:`MemoryPort.get`
    would show — via :func:`describe_stale_transition`.
    """


class MemorySerializationError(MemoryPortError):
    """Raised when a draft/patch cannot be persisted as valid JSON.

    Fail-before-write (mirrors :class:`~lifemodel.state.errors.StateSerializationError`):
    a non-JSON-serializable payload (e.g. a non-finite float, or a value with
    no JSON encoding), a malformed/timezone-naive ``expires_at``, or a
    timezone-naive clock value is rejected by :func:`ensure_json_serializable` /
    :func:`parse_expires_at_epoch_ms` / :func:`stamp_iso_utc` *before* either
    store implementation touches its backing storage.
    """


@dataclass(frozen=True)
class MemoryDraft:
    """Write input to :meth:`MemoryPort.put` — no timestamps, no revision.

    ``put`` upserts keyed by ``(kind, id)``: the store stamps
    ``created_at``/``updated_at``/``revision`` itself (via the injected
    ``ClockPort``), so a draft never carries them — that would let a caller
    forge history.
    """

    kind: str
    id: str
    state: str
    payload: JsonObject
    source: str
    recipient_id: str = "owner"
    salience: float = 0.0
    confidence: float | None = None
    #: Caller-provided, timezone-aware ISO-8601 instant, or ``None`` for "never
    #: expires". The store derives the epoch-millisecond column from this; see
    #: :func:`parse_expires_at_epoch_ms`.
    expires_at: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    """Read output from :meth:`MemoryPort.get`/:meth:`MemoryPort.find`.

    The full persisted row, including the fields the store alone controls
    (``created_at``, ``updated_at``, ``revision``, ``schema_version``).
    ``payload`` is always a fresh dict — no implementation may hand back a
    reference into its own storage (a frozen dataclass holding a mutable dict
    is not truly immutable otherwise).
    """

    kind: str
    id: str
    state: str
    payload: JsonObject
    source: str
    recipient_id: str
    salience: float
    confidence: float | None
    expires_at: str | None
    #: Timezone-aware ISO-8601 UTC, stamped by the store from ``ClockPort`` at
    #: first ``put`` — never changes across updates.
    created_at: str
    #: Timezone-aware ISO-8601 UTC, stamped at every ``put``/``transition``.
    updated_at: str
    #: Starts at 0 on insert; incremented on every subsequent ``put`` or
    #: ``transition`` of the same ``(kind, id)``.
    revision: int
    schema_version: int


@dataclass(frozen=True)
class MemoryPatch:
    """Partial update applied by :meth:`MemoryPort.transition`.

    ``None`` means "leave this field unchanged" for every field *except*
    ``payload_merge``, which shallow-merges its keys into the existing payload
    (top-level keys only — nested dicts are replaced wholesale, not deep
    merged). One consequence of the "``None`` = unchanged" convention: a patch
    cannot use ``transition`` to reset ``confidence`` back to ``None`` once
    set — callers needing that must ``put`` a fresh draft instead.
    """

    payload_merge: JsonObject | None = None
    salience: float | None = None
    confidence: float | None = None
    expires_at: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class PressureIndex:
    """The read model :meth:`PressureSensorPort.read_pressure_index` returns.

    All-default (``PressureIndex()``) is the documented "nothing pressing"
    answer: returned both for a genuinely empty store and, fail-soft, for a
    transient sensor error (HLA §4.1) — see
    :class:`~lifemodel.ports.pressure.PressureSensorPort`.
    """

    active_desire_count: int = 0
    max_desire_salience: float = 0.0
    #: True iff at least one active, unexpired ``kind='desire'`` record exists.
    contact_frame_available: bool = False


def ensure_json_serializable(payload: JsonObject) -> None:
    """Raise :class:`MemorySerializationError` if *payload* is not valid JSON.

    Shared by every ``MemoryPort`` implementation so the fail-before-write
    guard (mirrors :meth:`~lifemodel.state.json_store.JsonStateStore.commit`)
    is identical everywhere: called *before* a draft/patch touches any backing
    storage. ``allow_nan=False`` also rejects the non-finite-float poison
    ``json.loads`` would otherwise have accepted on the way in.
    """
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MemorySerializationError(
            f"refusing to persist a payload that is not valid JSON: {exc}"
        ) from exc


def parse_expires_at_epoch_ms(expires_at: str | None) -> int | None:
    """Parse a caller-provided ``expires_at`` into epoch milliseconds UTC.

    Mirrors :func:`lifemodel.state.model._as_opt_iso`: ``None`` passes through
    as ``None`` ("never expires"); otherwise the string must be a
    timezone-*aware* ISO-8601 instant, or :class:`MemorySerializationError` is
    raised — a naive value would silently misorder against the epoch columns.
    Shared by every ``MemoryPort`` implementation so "caller gave us a bad
    timestamp" fails identically everywhere.
    """
    if expires_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(expires_at)
    except ValueError as exc:
        raise MemorySerializationError(
            f"'expires_at' must be an ISO-8601 timestamp, got {expires_at!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MemorySerializationError(
            f"'expires_at' must be a timezone-aware timestamp, got naive {expires_at!r}"
        )
    return epoch_ms(parsed)


def epoch_ms(instant: datetime) -> int:
    """Convert a timezone-aware ``datetime`` to whole epoch milliseconds UTC."""
    return int(instant.timestamp() * 1000)


def stamp_iso_utc(instant: datetime) -> str:
    """Canonical ISO-8601 UTC text for a store-stamped timestamp (``ClockPort``).

    ``ClockPort`` promises a timezone-aware UTC ``datetime``, but a misconfigured
    clock could hand back a naive value; rather than silently misinterpret it as
    local time (``.isoformat()`` would emit an offset-less string that later reads
    back naive and misorders against the epoch columns), reject it before write —
    mirroring :func:`parse_expires_at_epoch_ms`'s tz-aware requirement. The result
    is normalized through ``astimezone(UTC)`` so the text column is canonical UTC
    regardless of the clock's zone; :func:`epoch_ms` is tz-independent for an
    aware instant, so the two stay consistent. Shared by every ``MemoryPort``
    implementation so fake and real stamp identically.
    """
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise MemorySerializationError(
            f"clock returned a timezone-naive datetime {instant!r}; "
            "MemoryPort requires a timezone-aware UTC clock"
        )
    return instant.astimezone(UTC).isoformat()


def merge_payload(existing: JsonObject, payload_merge: JsonObject | None) -> JsonObject:
    """Apply ``MemoryPatch.payload_merge``'s shallow-merge semantics.

    ``None`` returns a fresh copy of *existing* unchanged; otherwise *existing*
    is shallow-merged with *payload_merge* (top-level keys replaced, nested
    dicts wholesale — not deep-merged). Shared by every ``MemoryPort``
    implementation's ``transition`` so the merge rule is identical everywhere.
    """
    if payload_merge is None:
        return dict(existing)
    return {**existing, **payload_merge}


_T = TypeVar("_T")


def coalesce_patch(patch_value: _T | None, existing_value: _T) -> _T:
    """Apply the ``MemoryPatch`` convention: ``None`` means "leave unchanged"."""
    return patch_value if patch_value is not None else existing_value


def describe_stale_transition(kind: str, id: str, from_state: str, actual_state: str | None) -> str:
    """The :class:`StaleTransition` message text, shared by every implementation.

    Distinguishes "no such record" (``actual_state is None``, as a follow-up
    :meth:`MemoryPort.get` would confirm) from "record exists but is not in
    ``from_state``" — the two cases the brief's contract calls out.
    """
    if actual_state is None:
        return f"no memory record kind={kind!r} id={id!r} exists"
    return (
        f"memory record kind={kind!r} id={id!r} is in state {actual_state!r}, "
        f"not the expected {from_state!r}"
    )


def summarize_pressure_index(records: Iterable[MemoryRecord], now: datetime) -> PressureIndex:
    """Compute :class:`PressureIndex` from ``kind='desire'`` records in Python.

    The pure, storage-agnostic version of
    :meth:`~lifemodel.ports.pressure.PressureSensorPort.read_pressure_index`'s
    logic: used by in-memory fakes (:mod:`lifemodel.testing.fakes`) so their
    answer matches :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`'s
    SQL aggregate exactly. A record counts as *active* iff ``kind == "desire"``,
    ``state == "active"``, and it is unexpired (``expires_at is None`` or its
    epoch is strictly after *now*'s).
    """
    now_epoch = epoch_ms(now)
    active = [
        record
        for record in records
        if record.kind == "desire"
        and record.state == "active"
        and (record.expires_at is None or _expires_after(record.expires_at, now_epoch))
    ]
    if not active:
        return PressureIndex()
    return PressureIndex(
        active_desire_count=len(active),
        max_desire_salience=max(record.salience for record in active),
        contact_frame_available=True,
    )


def _expires_after(expires_at: str, now_epoch: int) -> bool:
    expires_epoch = parse_expires_at_epoch_ms(expires_at)
    return expires_epoch is None or expires_epoch > now_epoch

"""``BaseObject`` — the shared envelope + the codec written *once* (lm-27n.1).

The ``memory_records`` columns are fixed (HLA §4.1): there is no column for
sensitivity, supersession, provenance, or tags. Those envelope fields live
*inside* the payload under **reserved, underscore-namespaced keys**
(``_sensitivity``/``_supersedes``/``_superseded_by``/``_tags``/``_provenance``);
each kind's *semantic* fields sit at the payload top level and must never start
with ``_`` (the registry enforces that at registration).

This module owns the envelope codec (:func:`pack_envelope` /
:func:`unpack_envelope`) so it is written exactly once. Each kind adds only a
:meth:`BaseObject._semantic_payload` and a :meth:`BaseObject._rebuild` — an
**explicit** per-kind codec (no reflection: it is less failure surface and
plays nicely with the reserved keys and strict mypy). The typed field helpers
here (``req_str``/``req_float``/...) are the strict decode boundary shared by
every kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self, TypedDict, TypeVar

from ..memory import JsonObject, JsonValue, MemoryDraft, MemoryRecord
from .errors import InvalidPayload
from .provenance import Provenance, Sensitivity

# --- Reserved payload keys (the envelope fields with no dedicated column). ----
RESERVED_SENSITIVITY = "_sensitivity"
RESERVED_SUPERSEDES = "_supersedes"
RESERVED_SUPERSEDED_BY = "_superseded_by"
RESERVED_TAGS = "_tags"
RESERVED_PROVENANCE = "_provenance"

#: Every reserved key. Semantic field names must be disjoint from this set (and,
#: more strictly, must not start with ``_`` at all).
RESERVED_KEYS = frozenset(
    {
        RESERVED_SENSITIVITY,
        RESERVED_SUPERSEDES,
        RESERVED_SUPERSEDED_BY,
        RESERVED_TAGS,
        RESERVED_PROVENANCE,
    }
)


# --- Deterministic id policy (never random; HLA §4.1). -----------------------
#: The singleton ``kind="desire"`` contact-frame id.
CONTACT_DESIRE_ID = "contact:owner"

#: The singleton ``kind="intention"`` contact-frame id — the committed decision
#: to reach out (the Bratman act-gate owner). Same bare id as the desire; the
#: store keys by ``(kind, id)`` so the ``intention`` and ``desire`` singletons
#: are distinct rows sharing the contact frame.
CONTACT_INTENTION_ID = "contact:owner"

#: The singleton ``kind="relationship"`` id for the owner relationship (lm-27n.5)
#: — the being's learned interaction norms about its owner (good/bad hours,
#: cadence, privacy boundaries, acceptable styles, explicit prefs). One row per
#: being; created/updated only through the intent bus (a ``PutRecord`` upsert),
#: never direct SQL/config. Read via :mod:`lifemodel.core.relationship_view`.
OWNER_RELATIONSHIP_ID = "owner"


def derive_id(*parts: str) -> str:
    """Compose a deterministic id from *parts* (``":".join`` — reproducible)."""
    return ":".join(parts)


def qualified_id(kind: str, id: str) -> str:
    """Qualify a bare id with its kind: ``qualified_id("desire", "x") -> "desire:x"``."""
    return f"{kind}:{id}"


@dataclass(frozen=True, kw_only=True)
class BaseObject:
    """The shared envelope every typed BDI kind subclasses (HLA §4.1).

    ``frozen`` + ``kw_only`` on the base *and* every kind: ``kw_only`` removes
    the "base defaults before subclass required fields" ordering trap, and
    all-frozen is what makes frozen inheritance legal. ``KIND`` and
    ``SCHEMA_VERSION`` are :class:`typing.ClassVar` class metadata (ignored by
    ``@dataclass`` — not instance fields); each concrete kind sets a literal.
    ``state`` is a plain ``str`` (a kind's :class:`~enum.StrEnum` state *is* a
    ``str``); runtime state-legality is the registry's job, not the type's.
    """

    id: str
    state: str
    source: str
    recipient_id: str = "owner"
    salience: float = 0.0
    confidence: float | None = None
    expires_at: str | None = None
    sensitivity: Sensitivity = Sensitivity.NORMAL
    supersedes: str | None = None
    superseded_by: str | None = None
    tags: tuple[str, ...] = ()
    provenance: Provenance | None = None

    KIND: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int]

    def _semantic_payload(self) -> JsonObject:
        """Return the kind's semantic (top-level, non-reserved) payload fields."""
        raise NotImplementedError

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        """Rebuild the kind from decoded envelope *base* + its semantic *payload*."""
        raise NotImplementedError


class BaseFields(TypedDict):
    """The decoded envelope, keyed exactly like ``BaseObject``'s own fields.

    Handed to :meth:`BaseObject._rebuild` so a kind can splat it (``cls(**base,
    ...semantic...)``) — fully typed, no reflection.
    """

    id: str
    state: str
    source: str
    recipient_id: str
    salience: float
    confidence: float | None
    expires_at: str | None
    sensitivity: Sensitivity
    supersedes: str | None
    superseded_by: str | None
    tags: tuple[str, ...]
    provenance: Provenance | None


def state_set(*states: str) -> frozenset[str]:
    """A ``frozenset[str]`` of state names (accepts ``StrEnum`` members as str)."""
    return frozenset(states)


# --- The envelope codec, written once. ---------------------------------------
def pack_envelope(obj: BaseObject, semantic: JsonObject) -> MemoryDraft:
    """Pack *obj*'s envelope + *semantic* fields into a :class:`MemoryDraft`.

    Columnar fields map to draft columns; the envelope fields with no column
    map to reserved payload keys. ``schema_version`` rides on the draft (from
    the kind's ``SCHEMA_VERSION`` class metadata) so the store stamps the row
    with the *kind's* version, not a hardcoded ``1``; :meth:`~KindRegistry.decode`
    checks it off the record.
    """
    payload: JsonObject = dict(semantic)
    payload[RESERVED_SENSITIVITY] = str(obj.sensitivity)
    payload[RESERVED_TAGS] = list(obj.tags)
    if obj.supersedes is not None:
        payload[RESERVED_SUPERSEDES] = obj.supersedes
    if obj.superseded_by is not None:
        payload[RESERVED_SUPERSEDED_BY] = obj.superseded_by
    if obj.provenance is not None:
        payload[RESERVED_PROVENANCE] = _encode_provenance(obj.provenance)
    return MemoryDraft(
        kind=obj.KIND,
        id=obj.id,
        state=str(obj.state),
        payload=payload,
        source=obj.source,
        recipient_id=obj.recipient_id,
        salience=obj.salience,
        confidence=obj.confidence,
        expires_at=obj.expires_at,
        schema_version=obj.SCHEMA_VERSION,
    )


def unpack_envelope(record: MemoryRecord) -> tuple[BaseFields, JsonObject]:
    """Split *record* into its decoded envelope and its leftover semantic payload.

    The reserved keys are popped off a copy of the payload and decoded (raising
    :class:`InvalidPayload` on malformed values); what remains is exactly the
    kind's semantic fields, handed on to :meth:`BaseObject._rebuild`.
    """
    if not isinstance(record.payload, dict):
        raise InvalidPayload(
            f"record payload must be a JSON object, got {_typename(record.payload)}"
        )
    payload: JsonObject = dict(record.payload)
    sensitivity = _decode_sensitivity(payload.pop(RESERVED_SENSITIVITY, None))
    supersedes = _decode_opt_id(payload.pop(RESERVED_SUPERSEDES, None), RESERVED_SUPERSEDES)
    superseded_by = _decode_opt_id(
        payload.pop(RESERVED_SUPERSEDED_BY, None), RESERVED_SUPERSEDED_BY
    )
    tags = _decode_tags(payload.pop(RESERVED_TAGS, None))
    provenance = _decode_provenance(payload.pop(RESERVED_PROVENANCE, None))
    base: BaseFields = {
        "id": record.id,
        "state": record.state,
        "source": record.source,
        "recipient_id": record.recipient_id,
        "salience": record.salience,
        "confidence": record.confidence,
        "expires_at": record.expires_at,
        "sensitivity": sensitivity,
        "supersedes": supersedes,
        "superseded_by": superseded_by,
        "tags": tags,
        "provenance": provenance,
    }
    return base, payload


def _encode_provenance(provenance: Provenance) -> JsonObject:
    return {
        "created_by": provenance.created_by,
        "component": provenance.component,
        "reason": provenance.reason,
        "turn_id": provenance.turn_id,
        "source_object_ids": list(provenance.source_object_ids),
        "source_signal_ids": list(provenance.source_signal_ids),
        "trace_id": provenance.trace_id,
        "creation_span_id": provenance.creation_span_id,
        "parent_span_id": provenance.parent_span_id,
        "trace_flags": provenance.trace_flags,
    }


def _decode_provenance(value: JsonValue | None) -> Provenance | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidPayload(f"{RESERVED_PROVENANCE!r} must be an object, got {_typename(value)}")
    # Provenance.__post_init__ validates/normalizes the trace-context fields.
    return Provenance(
        created_by=req_str(value, "created_by"),
        component=req_str(value, "component"),
        reason=req_str(value, "reason"),
        turn_id=opt_str(value, "turn_id"),
        source_object_ids=opt_str_tuple(value, "source_object_ids"),
        source_signal_ids=opt_str_tuple(value, "source_signal_ids"),
        trace_id=opt_str(value, "trace_id"),
        creation_span_id=opt_str(value, "creation_span_id"),
        parent_span_id=opt_str(value, "parent_span_id"),
        trace_flags=opt_str(value, "trace_flags"),
    )


def _decode_sensitivity(value: JsonValue | None) -> Sensitivity:
    if value is None:
        return Sensitivity.NORMAL
    if not isinstance(value, str):
        raise InvalidPayload(f"{RESERVED_SENSITIVITY!r} must be a str, got {_typename(value)}")
    try:
        return Sensitivity(value)
    except ValueError as exc:
        raise InvalidPayload(f"unknown sensitivity {value!r}") from exc


def _decode_opt_id(value: JsonValue | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidPayload(f"{name!r} must be a str or null, got {_typename(value)}")
    return value


def _decode_tags(value: JsonValue | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidPayload(f"{RESERVED_TAGS!r} must be a list, got {_typename(value)}")
    return tuple(_as_str_item(item, RESERVED_TAGS) for item in value)


# --- Strict typed field helpers (the shared decode boundary). ----------------
EnumT = TypeVar("EnumT", bound=StrEnum)


def _missing(key: str) -> InvalidPayload:
    return InvalidPayload(f"missing required field {key!r}")


def _typename(value: object) -> str:
    return type(value).__name__


def _as_str_item(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise InvalidPayload(f"every item of {key!r} must be a str, got {_typename(value)}")
    return value


def _as_int_item(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidPayload(f"every item of {key!r} must be an int, got {_typename(value)}")
    return value


def _as_finite_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidPayload(f"field {key!r} must be a number, got {_typename(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidPayload(f"field {key!r} must be finite, got {number}")
    return number


def req_str(payload: JsonObject, key: str) -> str:
    if key not in payload:
        raise _missing(key)
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidPayload(f"field {key!r} must be a str, got {_typename(value)}")
    return value


def opt_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidPayload(f"field {key!r} must be a str or null, got {_typename(value)}")
    return value


def req_float(payload: JsonObject, key: str) -> float:
    if key not in payload:
        raise _missing(key)
    return _as_finite_float(payload[key], key)


def opt_float(payload: JsonObject, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _as_finite_float(value, key)


def req_int(payload: JsonObject, key: str) -> int:
    if key not in payload:
        raise _missing(key)
    return _as_int_item(payload[key], key)


def req_str_tuple(payload: JsonObject, key: str) -> tuple[str, ...]:
    if key not in payload:
        raise _missing(key)
    value = payload[key]
    if not isinstance(value, list):
        raise InvalidPayload(f"field {key!r} must be a list, got {_typename(value)}")
    return tuple(_as_str_item(item, key) for item in value)


def opt_str_tuple(payload: JsonObject, key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InvalidPayload(f"field {key!r} must be a list, got {_typename(value)}")
    return tuple(_as_str_item(item, key) for item in value)


def req_int_tuple(payload: JsonObject, key: str) -> tuple[int, ...]:
    if key not in payload:
        raise _missing(key)
    value = payload[key]
    if not isinstance(value, list):
        raise InvalidPayload(f"field {key!r} must be a list, got {_typename(value)}")
    return tuple(_as_int_item(item, key) for item in value)


def req_enum(payload: JsonObject, key: str, enum_cls: type[EnumT]) -> EnumT:
    raw = req_str(payload, key)
    try:
        return enum_cls(raw)
    except ValueError as exc:
        raise InvalidPayload(f"field {key!r} is not a valid {enum_cls.__name__}: {raw!r}") from exc

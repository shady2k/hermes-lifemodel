"""Test builders for the typed contact-desire row (lm-27n.3).

The contact desire is a ``kind='desire'`` singleton record now, not a ``State``
flag, so tests seed it as a :class:`~lifemodel.domain.memory.MemoryRecord` in the
tick snapshot (``ctx.objects``) or in a store. These builders go through the same
registry door the runtime uses (:func:`~lifemodel.core.desire_view.build_contact_desire`
+ ``encode``), so a hand-built row can never drift from what aggregation writes.
"""

from __future__ import annotations

from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import DesireState

#: A fixed, timezone-aware stamp for the store-controlled columns of a hand-built
#: record (the builders never touch a real clock).
_STAMP = "2026-07-06T00:00:00+00:00"


def contact_desire_record(
    state: str = "active",
    *,
    salience: float = 0.0,
    source_drive: float | None = None,
    created_at: str = _STAMP,
    updated_at: str = _STAMP,
    revision: int = 0,
) -> MemoryRecord:
    """A persisted contact-desire :class:`MemoryRecord` in *state* (default active).

    Encodes a real :class:`~lifemodel.domain.objects.Desire` through the registry
    then stamps the store-controlled columns, so the payload/envelope shape is
    byte-identical to a row the store would hold."""
    draft = encode_contact_desire(
        build_contact_desire(state=DesireState(state), salience=salience, source_drive=source_drive)
    )
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at=created_at,
        updated_at=updated_at,
        revision=revision,
        schema_version=draft.schema_version,
    )


def contact_desire_objects(state: str = "active", **kw: object) -> tuple[MemoryRecord, ...]:
    """A one-record ``ctx.objects`` snapshot holding the live contact desire."""
    return (contact_desire_record(state, **kw),)  # type: ignore[arg-type]

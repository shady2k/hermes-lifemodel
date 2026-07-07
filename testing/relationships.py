"""Test builders for the typed owner-relationship row (lm-27n.5).

The owner relationship — the being's learned interaction norms about its owner —
is a ``kind='relationship'`` singleton record ``owner``, mirroring the contact
desire (:mod:`lifemodel.testing.desires`) and intention
(:mod:`lifemodel.testing.intentions`). Tests seed it as a
:class:`~lifemodel.domain.memory.MemoryRecord` in the tick snapshot
(``ctx.objects``) or in a store. These builders go through the same registry door
the runtime uses (:func:`~lifemodel.core.relationship_view.build_owner_relationship`
+ ``encode``), so a hand-built row can never drift from what the being writes.
"""

from __future__ import annotations

from lifemodel.core.relationship_view import (
    build_owner_relationship,
    encode_owner_relationship,
)
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import RelationshipState

#: A fixed, timezone-aware stamp for the store-controlled columns of a hand-built
#: record (the builders never touch a real clock).
_STAMP = "2026-07-06T00:00:00+00:00"


def owner_relationship_record(
    state: str = "active",
    *,
    created_at: str = _STAMP,
    updated_at: str = _STAMP,
    revision: int = 0,
    **prefs: object,
) -> MemoryRecord:
    """A persisted owner-relationship :class:`MemoryRecord` in *state* (default active).

    ``**prefs`` are forwarded to
    :func:`~lifemodel.core.relationship_view.build_owner_relationship` (e.g.
    ``bad_hours=(2, 3)``, ``cadence="2h"``, ``confidence=EXPLICIT_CONFIDENCE``).
    Encodes a real :class:`~lifemodel.domain.objects.Relationship` through the
    registry then stamps the store-controlled columns, so the payload/envelope
    shape is byte-identical to a row the store would hold."""
    draft = encode_owner_relationship(
        build_owner_relationship(state=RelationshipState(state), **prefs)  # type: ignore[arg-type]
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


def owner_relationship_objects(state: str = "active", **prefs: object) -> tuple[MemoryRecord, ...]:
    """A one-record ``ctx.objects`` snapshot holding the live owner relationship."""
    return (owner_relationship_record(state, **prefs),)  # type: ignore[arg-type]

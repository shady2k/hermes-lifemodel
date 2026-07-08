"""Test builders for the typed thought rows (lm-27n.6).

A thought is a ``kind='thought'`` record — but, unlike the desire/intention/
relationship singletons (:mod:`lifemodel.testing.desires` et al.), it is
**NON-singleton**: there can be many live at once. Tests seed them as
:class:`~lifemodel.domain.memory.MemoryRecord`s in the tick snapshot
(``ctx.objects``) or in a store. These builders go through the same registry
door the runtime uses (:func:`~lifemodel.core.thought_view.build_thought` +
``encode``), so a hand-built row can never drift from what the being writes.
"""

from __future__ import annotations

from lifemodel.core.thought_view import build_thought, encode_thought, seed_thought_id
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import ThoughtState

#: A fixed, timezone-aware stamp for the store-controlled columns of a hand-built
#: record (the builders never touch a real clock).
_STAMP = "2026-07-06T00:00:00+00:00"


def thought_record(
    content: str = "I wonder how the owner is doing",
    state: str = "active",
    *,
    id: str | None = None,
    salience: float = 0.0,
    trigger: str = "seed",
    parent_id: str | None = None,
    created_at: str = _STAMP,
    updated_at: str = _STAMP,
    revision: int = 0,
) -> MemoryRecord:
    """A persisted thought :class:`MemoryRecord` in *state* (default active).

    ``id`` defaults to the deterministic content-digest seed id, so two records
    with distinct content get distinct ids. Encodes a real
    :class:`~lifemodel.domain.objects.Thought` through the registry then stamps
    the store-controlled columns, so the payload/envelope shape is byte-identical
    to a row the store would hold."""
    draft = encode_thought(
        build_thought(
            id=id if id is not None else seed_thought_id(content),
            content=content,
            state=ThoughtState(state),
            salience=salience,
            trigger=trigger,
            parent_id=parent_id,
        )
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


def thought_objects(
    content: str = "I wonder how the owner is doing", state: str = "active", **kw: object
) -> tuple[MemoryRecord, ...]:
    """A one-record ``ctx.objects`` snapshot holding a single live thought."""
    return (thought_record(content, state, **kw),)  # type: ignore[arg-type]

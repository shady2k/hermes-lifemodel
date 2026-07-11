"""Test builders for the typed owner user-model row (spec §8).

The owner user-model — the being's derived model of its owner (learned
interaction norms + receptivity) — is a ``kind='user_model'`` singleton record
``owner``, mirroring the contact desire (:mod:`lifemodel.testing.desires`) and
intention (:mod:`lifemodel.testing.intentions`). Tests seed it as a
:class:`~lifemodel.domain.memory.MemoryRecord` in the tick snapshot
(``ctx.objects``) or in a store. These builders go through the same registry door
the runtime uses (:func:`~lifemodel.core.user_model_view.build_owner_user_model`
+ ``encode``), so a hand-built row can never drift from what the being writes.
"""

from __future__ import annotations

from lifemodel.core.user_model_view import (
    build_owner_user_model,
    encode_owner_user_model,
)
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import UserModelState

#: A fixed, timezone-aware stamp for the store-controlled columns of a hand-built
#: record (the builders never touch a real clock).
_STAMP = "2026-07-06T00:00:00+00:00"


def owner_user_model_record(
    state: str = "active",
    *,
    created_at: str = _STAMP,
    updated_at: str = _STAMP,
    revision: int = 0,
    **prefs: object,
) -> MemoryRecord:
    """A persisted owner user-model :class:`MemoryRecord` in *state* (default active).

    ``**prefs`` are forwarded to
    :func:`~lifemodel.core.user_model_view.build_owner_user_model` (e.g.
    ``bad_hours=(2, 3)``, ``cadence="2h"``, ``confidence=EXPLICIT_CONFIDENCE``).
    Encodes a real :class:`~lifemodel.domain.objects.UserModel` through the
    registry then stamps the store-controlled columns, so the payload/envelope
    shape is byte-identical to a row the store would hold."""
    draft = encode_owner_user_model(
        build_owner_user_model(state=UserModelState(state), **prefs)  # type: ignore[arg-type]
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


def owner_user_model_objects(state: str = "active", **prefs: object) -> tuple[MemoryRecord, ...]:
    """A one-record ``ctx.objects`` snapshot holding the live owner user-model."""
    return (owner_user_model_record(state, **prefs),)  # type: ignore[arg-type]

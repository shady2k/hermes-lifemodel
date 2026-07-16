"""``Commitment`` — a follow-up the being *owes*, having thought it over (§4.2, v3).

The first catalog EXTENSION type (D8): what the being decided to do after processing a
thought — "ask how their interview went", "come back to the moving-house topic". HLA §4.1:
"the strongest non-intrusive reason, serving the other". A crystallization target
(``core/thought_processing.py``): a processed thought becomes a ``Commitment`` via ``PutRecord``,
and the source thought resolves. NON-singleton (many coexist), unlike the contact ``Desire``.

Distinct from ``Intention``: a ``Commitment`` is an enduring *owed follow-up / source object*;
an ``Intention`` (Bratman) is an executable, send-gating plan. Turning a commitment into an
outreach (commitment → contact ``Desire`` → ``Intention``) is the deferred contact work, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import (
    BaseFields,
    BaseObject,
    opt_str,
    req_enum,
    req_float,
    req_str,
    req_str_tuple,
    state_set,
)


class CommitmentState(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    HONOURED = "honoured"
    DROPPED = "dropped"
    EXPIRED = "expired"


class CommitmentBasis(StrEnum):
    """WHY the being holds this — so an ordinary interesting thought cannot masquerade as a debt."""

    PROMISED = "promised"
    FOLLOW_UP = "follow_up"
    SELF_ASSUMED = "self_assumed"


class CommitmentTriggerKind(StrEnum):
    """WHEN to honour it (Gollwitzer if-then): a wall-clock time, an event, or a condition."""

    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"


#: The explicit transition table (held by the registry). Terminal states are keys with
#: empty out-sets. ``active``/``deferred`` are live; ``honoured``/``dropped``/``expired`` terminal.
COMMITMENT_TRANSITIONS: dict[str, frozenset[str]] = {
    CommitmentState.ACTIVE: state_set(
        CommitmentState.DEFERRED,
        CommitmentState.HONOURED,
        CommitmentState.DROPPED,
        CommitmentState.EXPIRED,
    ),
    CommitmentState.DEFERRED: state_set(
        CommitmentState.ACTIVE,
        CommitmentState.HONOURED,
        CommitmentState.DROPPED,
        CommitmentState.EXPIRED,
    ),
    CommitmentState.HONOURED: state_set(),
    CommitmentState.DROPPED: state_set(),
    CommitmentState.EXPIRED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Commitment(BaseObject):
    content: str
    basis: CommitmentBasis
    trigger_kind: CommitmentTriggerKind
    trigger_value: str
    due_at: str | None
    source_thought_ids: tuple[str, ...]
    other_regarding_value: float

    KIND: ClassVar[str] = "commitment"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "content": self.content,
            "basis": str(self.basis),
            "trigger_kind": str(self.trigger_kind),
            "trigger_value": self.trigger_value,
            "due_at": self.due_at,
            "source_thought_ids": list(self.source_thought_ids),
            "other_regarding_value": self.other_regarding_value,
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            content=req_str(payload, "content"),
            basis=req_enum(payload, "basis", CommitmentBasis),
            trigger_kind=req_enum(payload, "trigger_kind", CommitmentTriggerKind),
            trigger_value=req_str(payload, "trigger_value"),
            due_at=opt_str(payload, "due_at"),
            source_thought_ids=req_str_tuple(payload, "source_thought_ids"),
            other_regarding_value=req_float(payload, "other_regarding_value"),
        )

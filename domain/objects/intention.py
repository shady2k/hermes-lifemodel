"""``Intention`` — a committed plan of action (BDI, HLA §4.1).

Where a :class:`~lifemodel.domain.objects.desire.Desire` is a pull, an intention
is a *commitment*: a goal the being has resolved to pursue, with a plan, an
implementation trigger, constraints, an admissibility filter, and the triggers
that would make it reconsider. State machine: ``pending`` is the freshly formed
commitment; it may activate, defer, complete, drop, or expire; ``active`` and
``deferred`` move among themselves and to the terminal ``completed``/
``dropped``/``expired``.
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
    req_float,
    req_str,
    req_str_tuple,
    state_set,
)


class IntentionState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    DROPPED = "dropped"
    EXPIRED = "expired"


INTENTION_TRANSITIONS: dict[str, frozenset[str]] = {
    IntentionState.PENDING: state_set(
        IntentionState.ACTIVE,
        IntentionState.DEFERRED,
        IntentionState.COMPLETED,
        IntentionState.DROPPED,
        IntentionState.EXPIRED,
    ),
    IntentionState.ACTIVE: state_set(
        IntentionState.COMPLETED,
        IntentionState.DEFERRED,
        IntentionState.DROPPED,
        IntentionState.EXPIRED,
    ),
    IntentionState.DEFERRED: state_set(
        IntentionState.ACTIVE,
        IntentionState.COMPLETED,
        IntentionState.DROPPED,
        IntentionState.EXPIRED,
    ),
    IntentionState.COMPLETED: state_set(),
    IntentionState.DROPPED: state_set(),
    IntentionState.EXPIRED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Intention(BaseObject):
    goal: str
    commitment_strength: float
    plan: str
    implementation_trigger: str
    constraints: tuple[str, ...]
    admissibility_filter: str
    reconsideration_triggers: tuple[str, ...]
    expiry: str | None
    rationale: str

    KIND: ClassVar[str] = "intention"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "goal": self.goal,
            "commitment_strength": self.commitment_strength,
            "plan": self.plan,
            "implementation_trigger": self.implementation_trigger,
            "constraints": list(self.constraints),
            "admissibility_filter": self.admissibility_filter,
            "reconsideration_triggers": list(self.reconsideration_triggers),
            "expiry": self.expiry,
            "rationale": self.rationale,
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            goal=req_str(payload, "goal"),
            commitment_strength=req_float(payload, "commitment_strength"),
            plan=req_str(payload, "plan"),
            implementation_trigger=req_str(payload, "implementation_trigger"),
            constraints=req_str_tuple(payload, "constraints"),
            admissibility_filter=req_str(payload, "admissibility_filter"),
            reconsideration_triggers=req_str_tuple(payload, "reconsideration_triggers"),
            expiry=opt_str(payload, "expiry"),
            rationale=req_str(payload, "rationale"),
        )

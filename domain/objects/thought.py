"""``Thought`` — a first-person mental content the being is turning over (§4.1).

A thought is what the being is *thinking*: first-person content with a trigger
(idle, an event, a parent thought, a drive, an emotion), an attention score, a
loop signature and no-progress counter (for the thought engine's loop/park
logic), and appraisal hints (actionability, other-regarding value). State
machine: ``active`` and ``parked`` move between each other and to the terminal
``resolved``/``dropped``/``expired``/``merged``.
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
    req_int,
    req_str,
    state_set,
)


class ThoughtState(StrEnum):
    ACTIVE = "active"
    PARKED = "parked"
    RESOLVED = "resolved"
    DROPPED = "dropped"
    EXPIRED = "expired"
    MERGED = "merged"


THOUGHT_TRANSITIONS: dict[str, frozenset[str]] = {
    ThoughtState.ACTIVE: state_set(
        ThoughtState.PARKED,
        ThoughtState.RESOLVED,
        ThoughtState.DROPPED,
        ThoughtState.EXPIRED,
        ThoughtState.MERGED,
    ),
    ThoughtState.PARKED: state_set(
        ThoughtState.ACTIVE,
        ThoughtState.RESOLVED,
        ThoughtState.DROPPED,
        ThoughtState.EXPIRED,
        ThoughtState.MERGED,
    ),
    ThoughtState.RESOLVED: state_set(),
    ThoughtState.DROPPED: state_set(),
    ThoughtState.EXPIRED: state_set(),
    ThoughtState.MERGED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Thought(BaseObject):
    content: str
    trigger: str
    parent_id: str | None
    attention_score: float
    no_progress_count: int
    loop_signature: str
    parked_until: str | None
    #: How many times this thought has been parked — the exponential-backoff
    #: cycle counter (lm-27n.7). Bumped on each ``active→parked`` and used to widen
    #: the next park window (6h/24h/72h); once it exceeds the cap the thought
    #: expires rather than re-arming. A declared field (not a stray payload key) so
    #: the anti-rumination engine tracks the cycle through the typed registry door.
    park_count: int
    actionability: float
    other_regarding_value: float

    KIND: ClassVar[str] = "thought"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "content": self.content,
            "trigger": self.trigger,
            "parent_id": self.parent_id,
            "attention_score": self.attention_score,
            "no_progress_count": self.no_progress_count,
            "loop_signature": self.loop_signature,
            "parked_until": self.parked_until,
            "park_count": self.park_count,
            "actionability": self.actionability,
            "other_regarding_value": self.other_regarding_value,
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            content=req_str(payload, "content"),
            trigger=req_str(payload, "trigger"),
            parent_id=opt_str(payload, "parent_id"),
            attention_score=req_float(payload, "attention_score"),
            no_progress_count=req_int(payload, "no_progress_count"),
            loop_signature=req_str(payload, "loop_signature"),
            parked_until=opt_str(payload, "parked_until"),
            park_count=req_int(payload, "park_count"),
            actionability=req_float(payload, "actionability"),
            other_regarding_value=req_float(payload, "other_regarding_value"),
        )

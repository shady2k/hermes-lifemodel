"""``Belief`` — a defeasible proposition the being has inferred about the person
or world (spec ``docs/superpowers/specs/2026-07-17-fact-track-design.md``, v2).

The second catalog EXTENSION type (after :class:`~lifemodel.domain.objects.commitment.Commitment`):
a fallible understanding formed where the evidence lives (the noticing pass),
never authoritative merely because it is stored. It carries a mandatory
``confidence`` and its ``source_message_ids`` evidence (both ride the shared
:class:`~lifemodel.domain.objects.base.BaseObject` envelope, validated by the
view builder — the registry does not range-check numbers).

Distinct from ``Opinion`` (an evaluative stance, unbuilt), ``Prediction``
(future-oriented, unbuilt), ``UserModel`` (a closed receptivity/norms schema,
not an open proposition), and ``Thought`` (the bounded first-person reasoning
stream a belief may be born alongside, not from).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import BaseFields, BaseObject, req_str, req_str_tuple, state_set


class BeliefState(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DROPPED = "dropped"
    EXPIRED = "expired"


#: The explicit transition table (held by the registry). ``active`` is the
#: only live state; ``superseded``/``dropped``/``expired`` are terminal. v1
#: does not wire the supersession *operation* (spec §4) — only the state.
BELIEF_TRANSITIONS: dict[str, frozenset[str]] = {
    BeliefState.ACTIVE: state_set(BeliefState.SUPERSEDED, BeliefState.DROPPED, BeliefState.EXPIRED),
    BeliefState.SUPERSEDED: state_set(),
    BeliefState.DROPPED: state_set(),
    BeliefState.EXPIRED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Belief(BaseObject):
    content: str
    subject: str
    source_message_ids: tuple[str, ...]
    source_thought_ids: tuple[str, ...]

    KIND: ClassVar[str] = "belief"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "content": self.content,
            "subject": self.subject,
            "source_message_ids": list(self.source_message_ids),
            "source_thought_ids": list(self.source_thought_ids),
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            content=req_str(payload, "content"),
            subject=req_str(payload, "subject"),
            source_message_ids=req_str_tuple(payload, "source_message_ids"),
            source_thought_ids=req_str_tuple(payload, "source_thought_ids"),
        )

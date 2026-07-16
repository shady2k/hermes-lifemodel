"""The appraisal seam — decide whether a completed exchange is worth a thought.

Runs OUT-OF-BAND (in the ``post_llm`` hook, which holds the finished turn), never
inside the 0-LLM tick. Slice 1 ships a deterministic, no-LLM :class:`HeuristicAppraiser`
so the whole capture pipeline is testable and cost-free; the richer LLM/rides-the-tail
appraiser is a deferred bead (spec §8). The being's REPLY is its thinking; this only
drops a mental bookmark to return to later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Cheap, language-agnostic-ish markers that a turn left something open worth
#: returning to — a forward reference (a plan, a future event) or an unresolved
#: question. Intentionally crude: slice 1 proves the PLUMBING; appraisal QUALITY is
#: the next refinement, tuned against live traces.
_FORWARD_MARKERS = (
    "tomorrow",
    "next week",
    "friday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "saturday",
    "sunday",
    "later",
    "soon",
    "appointment",
    "deadline",
    "interview",
    "meeting",
    "trip",
    "plan to",
    "going to",
)
_MIN_CONTENT_CHARS = 24


@dataclass(frozen=True)
class ThoughtSeed:
    """An appraisal result: the content + salience of a thought worth capturing."""

    content: str
    salience: float
    actionability: float = 0.0
    other_regarding_value: float = 0.0


@runtime_checkable
class Appraiser(Protocol):
    """Judge a completed owner↔being exchange; return a seed, or ``None`` to decline."""

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None: ...


class HeuristicAppraiser:
    """A deterministic, no-LLM appraiser (slice 1). Seeds a thought when the user's
    message is substantive AND carries a forward-reference / open-loop marker."""

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None:
        text = user_message.strip()
        if len(text) < _MIN_CONTENT_CHARS:
            return None
        low = text.lower()
        if not any(marker in low for marker in _FORWARD_MARKERS):
            return None
        # First-person content: this is the being's own note to itself.
        content = (
            f"the owner said: {text}" if len(text) <= 200 else f"the owner said: {text[:200]}…"
        )
        return ThoughtSeed(
            content=content,
            salience=0.5,
            actionability=0.3,
            other_regarding_value=0.5,
        )

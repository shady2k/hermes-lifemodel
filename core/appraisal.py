"""The appraisal seam — the being's judgment of whether a completed exchange left
something worth returning to.

Runs OUT-OF-BAND (in the ``post_llm`` hook, which holds the finished turn), never
inside the 0-LLM tick. Deciding "is this worth a thought" is **judgment** — the being's
own cognition, never a keyword/pattern heuristic (owner principle, 2026-07-16). The
intended appraiser is **rides-the-tail**: the being notes what to return to *during its
own reply turn* (0 extra LLM, language-agnostic; the instruction lives in a tool
description = prose). No concrete :class:`Appraiser` is wired yet — this module holds only
the port and result type, so the capture pipeline (``ThoughtCapture``) is ready. Tests
inject a fake. The being's REPLY is its thinking; this only drops a mental bookmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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

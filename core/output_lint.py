"""Output-lint: a send-time safety filter on a proactive message (spec §13).

Catches the two anti-patterns of the old monolith — mechanical self-justification
by the clock/monitor, and contentless filler — without touching natural human
time references ("давно не виделись" is fine; "обнаружено 6ч тишины" is not).
Pure and language-agnostic: it matches a bilingual list of *mechanical* phrases,
so a warm message passes while a timer-narrated one is flagged. Applied at
send-time in Phase E; unit-tested here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Mechanical self-justification / filler markers (RU + EN). Deliberately narrow —
#: they target the *mechanism* narration, not any mention of time.
DEFAULT_MECHANICAL_PATTERNS: tuple[str, ...] = (
    r"\d+\s*(час|hour|минут|minute)\w*\s+тишин",
    r"\d+\s*(hours?|minutes?)\s+of\s+silence",
    r"тишин\w*\s+\d+",
    r"(шесть|восемь|двенадцать|несколько|полтора|два|три|четыре|пять)\s+час\w*\s+тишин",
    r"инициир\w*\s+проверк",
    r"检查|checking in\b",
    r"нечего\s+(сказать|добавить)",
    r"nothing\s+to\s+(say|add)",
    r"silence\s+detected",
    r"обнаружен\w*\s+\d",
)


@dataclass(frozen=True)
class LintResult:
    ok: bool
    reason: str = ""


def lint_proactive(
    text: str, *, patterns: Sequence[str] = DEFAULT_MECHANICAL_PATTERNS
) -> LintResult:
    """Flag mechanical timer-justification / contentless filler. Warm natural
    messages (incl. human time references) pass."""
    low = text.lower()
    for pat in patterns:
        if re.search(pat, low):
            return LintResult(ok=False, reason=f"mechanical_pattern:{pat}")
    return LintResult(ok=True)

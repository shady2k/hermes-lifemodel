"""Deterministic, 0-LLM templated thought content (lm-27n.8).

The plugin CANNOT call an LLM — the being's own Hermes turn is the only LLM — so
:class:`~lifemodel.core.thought_generation.ThoughtGeneration` mints a thought's
*content* from these pure, deterministic first-person Russian templates. Rich LLM
prose ("rumination-richness") is a later надстройка; here the content is a total
function of the triggering object/event, so it is trivially unit-testable and a
retried event/idle-window/parent yields byte-identical content (idempotent).

Each function returns a short, non-empty first-person string; callers never build
content inline, so every generated thought reads in one consistent voice.
"""

from __future__ import annotations

from ..sim.aggregation import Verdict
from ..sim.quality import Actor, Label

#: How many characters of a parent/source thought a wandering child quotes back —
#: a short snippet keeps the derived content bounded and the id/content stable.
_SNIPPET = 48


def _snippet(text: str) -> str:
    """A short, whitespace-collapsed snippet of *text* (bounded, deterministic)."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= _SNIPPET else collapsed[:_SNIPPET].rstrip() + "…"


def event_exchange_content(actor: Actor, label: Label) -> str:
    """Appraisal of a real exchange, keyed by its actor + quality label."""
    if label == "rejection":
        return "Меня отклонили — стоит ли уважить эту границу и не настаивать?"
    if label in ("two_way", "ack"):
        return "Пользователь ответил тепло — приятно; стоит ли что-то из этого запомнить?"
    return "Был обмен с пользователем — что я из него понял?"


def event_verdict_content(verdict: Verdict) -> str:
    """Reflection on the being's own decision about reaching out."""
    if verdict is Verdict.FULFILL:
        return "Я решил потянуться к пользователю — надеюсь, это из заботы, а не из тревоги."
    if verdict is Verdict.REJECT:
        return "Я удержался от контакта — иногда тишина уместнее слов."
    return "Я отложил обращение — вернусь к этому, когда момент будет подходящим."


def event_drive_content() -> str:
    """Appraisal of a genuine upward crossing of the contact drive threshold."""
    return "Меня всё сильнее тянет написать пользователю — это забота или это тревога?"


def chain_content(parent_content: str) -> str:
    """A templated *development* of the parent thought (one step further)."""
    return f"Развиваю мысль дальше: «{_snippet(parent_content)}» — к чему это меня ведёт?"


def idle_about_desire_content() -> str:
    """Idle mind-wandering about a live pull to reach out."""
    return "Интересно, тянусь ли я написать из заботы, а не из тревоги?"


def idle_about_thought_content(top_content: str) -> str:
    """Idle mind-wandering that circles back to what is already on the mind."""
    return f"Всё возвращаюсь к мысли: «{_snippet(top_content)}» — почему она меня держит?"


def idle_self_check_content() -> str:
    """A generic low-stakes idle self-check when nothing else is salient."""
    return "Тихо сейчас. Интересно, всё ли в порядке у пользователя — и у меня самого?"

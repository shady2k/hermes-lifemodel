"""The wake-packet: the desire-framed prompt for a proactive turn (spec §13).

Injected into the being's native Hermes turn (model A). It carries the projected
*desire-frame* (the feeling, never numbers) plus fixed *guidance*: own the wish;
you know the time and how the last talk ended and may note it humanly, but you
reach out because you WANT to — not because a timer fired; if there is nothing
genuine to say, staying silent is right. Default strings Russian (content).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..domain.objects import Thought
from .projection import project_contact
from .timeutil import humanize_elapsed, minutes_between

#: Marker prefixed to an injected proactive prompt so the being's own hooks
#: recognise their own nudge (correlation + self-exclusion). Was `impulse.py`.
IMPULSE_LABEL_PREFIX = "[lifemodel · внутренний импульс — не от пользователя]"

#: How many live thoughts the wake packet surfaces (most-salient first). A small
#: cap: the block is first-person CONTEXT, not the message — it orients the turn,
#: it does not dump the whole open-loop set.
THOUGHTS_RENDER_LIMIT = 5

#: Header for the "Recent Thoughts" block — first-person context ("what I've been
#: turning over"), NOT outward message content. Only added when a thought exists,
#: so an empty being's prompt is byte-identical to before (lm-27n.6).
RECENT_THOUGHTS_HEADER = "Что ты в последнее время про себя обдумывал(а):"

# NB: the model-facing block renders thought CONTENT only — never the internal id.
# The id is a machine/audit reference (surfaced in the debug dump, read from the
# store); exposing it to the model would risk the being echoing it into its
# outward message and buys nothing for its deliberation (codex, lm-27n.6).

GUIDANCE = (
    "Это твоё собственное желание потянуться — присвой его. Ты знаешь, сколько "
    "прошло и чем закончился прошлый разговор; можешь по-человечески это учесть, "
    "но тянешься ты потому что ХОЧЕШЬ, а не потому что «прошло сколько-то времени». "
    "Есть что-то настоящее сказать — скажи тепло и коротко. Если это будет "
    "наполнитель — совершенно нормально промолчать (ответь [SILENT])."
)


@dataclass(frozen=True)
class ProactivePrompt:
    prompt: str
    projection_id: str
    correlation_id: str


def render_thoughts_block(thoughts: Sequence[Thought]) -> str:
    """Render the live thoughts (already ordered) as the "Recent Thoughts" block.

    First-person context: the header plus one bullet per thought (``content``
    only, no id), bounded to :data:`THOUGHTS_RENDER_LIMIT`. The id is deliberately
    NOT shown to the model — it is an internal audit reference (see the debug
    dump), and rendering it risks the being echoing it into its outward turn."""
    lines = [RECENT_THOUGHTS_HEADER]
    lines += [f"— {t.content}" for t in thoughts[:THOUGHTS_RENDER_LIMIT]]
    return "\n".join(lines)


def render_situational_brief(
    *,
    last_exchange_at: str | None,
    now: datetime | None,
    decline_count: int,
    energy: float,
    unanswered_outbound_count: int = 0,
) -> str:
    """First-person situational context for the wake, word-only (no digits).

    Empty string when ``now`` is None (caller passed no time → no brief)."""
    if now is None:
        return ""
    lines: list[str] = []
    if last_exchange_at is None:
        # word-only, and MUST contain the lowercase substring the Task-2 test
        # asserts ("вы ещё толком не общались")
        lines.append("С ним вы ещё толком не общались.")
        lines.append(
            "Конкретики под рукой нет — не выдумывай повод: если сказать нечего "
            "настоящего, честно промолчи."
        )
    else:
        elapsed = minutes_between(last_exchange_at, now)
        lines.append(f"Вы общались {humanize_elapsed(elapsed)}.")
    if decline_count > 0:
        lines.append(
            "Недавно ты уже тянулся и промолчал — тем более не дави, потянись "
            "только если есть что-то настоящее."
        )
    if energy < 0.3:
        lines.append("Сил сейчас немного — коротко и мягко, без длинных заходов.")
    if unanswered_outbound_count >= 1:
        # lm-8o3.1 Task 9: a still-unanswered prior bid — placed after the
        # tone/energy lines (it reads like one more restraint on HOW to
        # reach out) and before the closing orient-on-the-thread line (which
        # is about WHAT to say, a natural last beat before writing).
        lines.append(
            "Ты уже потянулся и пока без ответа — не повторяйся ради самого "
            "жеста; пиши, только если появилось что-то по-настоящему новое."
        )
    if last_exchange_at is not None:
        lines.append(
            "Прежде чем писать, вспомни, на чём вы остановились в прошлый раз — "
            "есть ли живая нить, которую хочется продолжить."
        )
    return "\n".join(lines)


def build_wake_packet(
    *,
    value: float,
    theta: float,
    correlation_id: str,
    thoughts: Sequence[Thought] = (),
    last_exchange_at: str | None = None,
    now: datetime | None = None,
    decline_count: int = 0,
    energy: float = 1.0,
    unanswered_outbound_count: int = 0,
) -> ProactivePrompt:
    """Build the proactive-turn prompt from the projected desire-frame + guidance.

    *thoughts* are the live (active/parked) thoughts, most-salient first, that
    cognition read from the tick snapshot. When there are none the prompt is
    byte-identical to before (behavior-neutral, lm-27n.6); only when a thought
    exists is a first-person "Recent Thoughts" CONTEXT block appended — it informs
    the being's own turn, it is NOT the outward message.

    *last_exchange_at*/*now*/*decline_count*/*energy* feed the situational brief
    (:func:`render_situational_brief`) — real context (how long since you talked,
    whether you already declined recently, how much energy you have) instead of a
    bare drive-level feeling. All default so existing callers are unaffected: with
    no ``now`` the brief is empty and the prompt carries no brief at all."""
    desire_frame, projection_id = project_contact(value, theta=theta, seed=correlation_id)
    prompt = f"Внутри у тебя сейчас: {desire_frame}."
    brief = render_situational_brief(
        last_exchange_at=last_exchange_at,
        now=now,
        decline_count=decline_count,
        energy=energy,
        unanswered_outbound_count=unanswered_outbound_count,
    )
    if brief:
        prompt = f"{prompt}\n\n{brief}"
    prompt = f"{prompt}\n\n{GUIDANCE}"
    if thoughts:
        prompt = f"{prompt}\n\n{render_thoughts_block(thoughts)}"
    return ProactivePrompt(
        prompt=prompt, projection_id=projection_id, correlation_id=correlation_id
    )

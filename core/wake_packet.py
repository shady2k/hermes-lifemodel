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

from ..domain.objects import Thought
from .projection import project_contact

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


def build_wake_packet(
    *,
    value: float,
    theta: float,
    correlation_id: str,
    thoughts: Sequence[Thought] = (),
) -> ProactivePrompt:
    """Build the proactive-turn prompt from the projected desire-frame + guidance.

    *thoughts* are the live (active/parked) thoughts, most-salient first, that
    cognition read from the tick snapshot. When there are none the prompt is
    byte-identical to before (behavior-neutral, lm-27n.6); only when a thought
    exists is a first-person "Recent Thoughts" CONTEXT block appended — it informs
    the being's own turn, it is NOT the outward message."""
    desire_frame, projection_id = project_contact(value, theta=theta, seed=correlation_id)
    prompt = f"Внутри у тебя сейчас: {desire_frame}.\n\n{GUIDANCE}"
    if thoughts:
        prompt = f"{prompt}\n\n{render_thoughts_block(thoughts)}"
    return ProactivePrompt(
        prompt=prompt, projection_id=projection_id, correlation_id=correlation_id
    )

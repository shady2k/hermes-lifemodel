"""The wake-packet: the desire-framed prompt for a proactive turn (spec §13).

Injected into the being's native Hermes turn (model A). It carries the projected
*desire-frame* (the feeling, never numbers) plus fixed *guidance*: own the wish;
you know the time and how the last talk ended and may note it humanly, but you
reach out because you WANT to — not because a timer fired; if there is nothing
genuine to say, staying silent is right. Default strings Russian (content).
"""

from __future__ import annotations

from dataclasses import dataclass

from .projection import project_contact

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


def build_wake_packet(*, value: float, theta: float, correlation_id: str) -> ProactivePrompt:
    """Build the proactive-turn prompt from the projected desire-frame + guidance."""
    desire_frame, projection_id = project_contact(value, theta=theta, seed=correlation_id)
    prompt = f"Внутри у тебя сейчас: {desire_frame}.\n\n{GUIDANCE}"
    return ProactivePrompt(
        prompt=prompt, projection_id=projection_id, correlation_id=correlation_id
    )

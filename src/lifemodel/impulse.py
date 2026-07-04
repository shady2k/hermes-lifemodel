"""Compose the proactive impulse — a single internal-labeled ``role=user`` text.

Spec §4: this text serves as BOTH the model seed AND the honest transcript record.
It must (a) clearly mark itself as not-user-authored, (b) never start with '/'
(would enter Hermes command routing — spec §5 guard f), (c) elicit a warm,
human, self-owned message rather than machinery talk (soul lever, bead lm-pbm).
"""

from __future__ import annotations

from datetime import datetime

from .domain.wake import WakePacket

IMPULSE_LABEL_PREFIX = "[lifemodel · внутренний импульс — не от пользователя]"


def _hours_silent(now: datetime, last_contact_at: datetime | None) -> int | None:
    if last_contact_at is None:
        return None
    delta = now - last_contact_at
    return max(0, int(delta.total_seconds() // 3600))


def compose_impulse(
    packet: WakePacket, *, now: datetime, last_contact_at: datetime | None
) -> str:
    """Return the internal-labeled user text that seeds the proactive turn."""
    hours = _hours_silent(now, last_contact_at)
    if hours is None:
        window = "давно не общались"
    else:
        window = f"около {hours} ч тишины"
    body = (
        "Прошло время без разговора по-человечески "
        f"({window}). Если есть, чем поделиться от себя — "
        "напиши сейчас, коротко и живо, своими словами. Ничего не требуется, "
        "если сказать нечего."
    )
    return f"{IMPULSE_LABEL_PREFIX} {body}"

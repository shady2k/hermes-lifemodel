"""Global safety backstop — a hard rate limit on real proactive sends (spec §14).

A fail-closed guard *above* the desire model: even if the drive model and the LLM
both misbehave, the being cannot send more than ``max_per_day`` proactive messages
or send twice within ``min_interval_min``. This protects the user from a buggy /
hallucinating cognition; it is NOT the restraint mechanism (that is emergent).
Pure over a persisted log of ISO send timestamps; malformed entries are ignored
(they never count as "no recent send").
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta


def _parse(ts: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return value if value.tzinfo is not None else None


def allow_send(
    send_log: Sequence[str],
    now: datetime,
    *,
    max_per_day: int = 3,
    min_interval_min: float = 60.0,
) -> bool:
    """True only if under the daily cap AND past the minimum interval."""
    day_ago = now - timedelta(hours=24)
    recent = [t for ts in send_log if (t := _parse(ts)) is not None and t >= day_ago]
    if len(recent) >= max_per_day:
        return False
    if recent:
        last = max(recent)
        if (now - last).total_seconds() / 60.0 < min_interval_min:
            return False
    return True


def record_send(send_log: Sequence[str], now: datetime, *, keep: int = 20) -> list[str]:
    """Append this send and keep the most recent ``keep`` entries."""
    return [*send_log, now.isoformat()][-keep:]

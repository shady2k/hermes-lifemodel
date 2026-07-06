"""Async semantic invalidation of a proactive verdict (spec §7.3).

A proactive turn outlives the tick that launched it; when its verdict returns we
must decide whether it is still valid. Invalidation is **semantic, not
version-based** (Codex): a mere energy/mood tick must not drop a good verdict.
A verdict is stale only if the situation it was about has genuinely changed —
the desire was resolved, its correlation no longer matches, the user replied
after the launch (the reactive path already answered — applying would double-
message), the pressure was satisfied while thinking, or the deadline elapsed.
"""

from __future__ import annotations

from datetime import datetime


def _parse(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    try:
        value = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return value if value.tzinfo is not None else None


def is_verdict_stale(
    *,
    desire_status: str,
    pending_id: str | None,
    verdict_correlation_id: str,
    last_exchange_at: str | None,
    pending_since: str | None,
    effective: float,
    threshold: float,
    now: datetime,
    deadline_min: float = 30.0,
) -> tuple[bool, str]:
    """Return ``(stale, reason)`` for a returning proactive verdict."""
    if desire_status != "active":
        return True, "desire_resolved"
    if verdict_correlation_id != pending_id:
        return True, "stale_desire_id"

    launched = _parse(pending_since)
    exchanged = _parse(last_exchange_at)
    if launched is not None and exchanged is not None and exchanged > launched:
        return True, "user_replied"
    if effective < threshold:
        return True, "pressure_satisfied"
    if launched is not None and (now - launched).total_seconds() / 60.0 > deadline_min:
        return True, "deadline"
    return False, "fresh"

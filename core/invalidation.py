"""Async semantic invalidation of a proactive outcome (spec §7.3).

A proactive turn outlives the frame that launched it; when its outcome returns we
must decide whether it is still valid. Invalidation is **semantic, not
version-based** (Codex): a mere energy/mood frame must not drop a good outcome.
An outcome is stale only if the situation it was about has genuinely changed —
the desire was resolved, its correlation no longer matches, the user replied
after the launch (the reactive path already answered — applying would double-
message), the pressure was satisfied while thinking, or the deadline elapsed.
"""

from __future__ import annotations

from datetime import datetime

from .timeutil import from_iso


def _parse(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    try:
        return from_iso(ts)  # strict: malformed/naive both raise -> None
    except (ValueError, TypeError):
        return None


def is_proactive_outcome_stale(
    *,
    desire_state: str,
    pending_id: str | None,
    outcome_correlation_id: str,
    last_exchange_at: str | None,
    pending_since: str | None,
    effective: float,
    threshold: float,
    now: datetime,
    deadline_min: float = 30.0,
    pressure_sprung: bool = True,
) -> tuple[bool, str]:
    """Return ``(stale, reason)`` for a returning proactive outcome.

    ``desire_state`` is the live desire's lifecycle state (``active``/``deferred``/
    ``none``) read from the typed row — an outcome is only ever fresh for an
    ``active`` desire; any other state means it was already resolved.

    *pressure_sprung* is ``False`` for an outcome whose desire did NOT spring from the
    drive — today, a genesis first-waking (spec §6.2). "The pressure was satisfied while
    I was thinking" is then not a fact ABOUT it: its reason to reach was that the being
    is nobody yet, and a newborn's ``u`` is 0 < ``θ`` BY CONSTRUCTION. Applying the
    pressure rule to it would judge EVERY genesis outcome stale, which is not a lost
    message but a DEADLOCK: the desire would stay ``active`` and ``pending_proactive_id``
    would never clear, so the launcher would hold every future launch for the rest of the
    being's life. The other rules still bind — a resolved desire, a mismatched
    correlation, a user who replied while it composed, and the deadline are all just as
    true of a birth as of a longing."""
    if desire_state != "active":
        return True, "desire_resolved"
    if outcome_correlation_id != pending_id:
        return True, "stale_desire_id"

    launched = _parse(pending_since)
    exchanged = _parse(last_exchange_at)
    if launched is not None and exchanged is not None and exchanged > launched:
        return True, "user_replied"
    if pressure_sprung and effective < threshold:
        return True, "pressure_satisfied"
    if launched is not None and (now - launched).total_seconds() / 60.0 > deadline_min:
        return True, "deadline"
    return False, "fresh"

"""Latent vs effective pressure + ActionPending inhibition (spec §9.1, §9.2).

The certified drive ``u`` is the *latent* pressure (true deprivation; rises by
silence, satiated only by real contact). After the being reaches out we suppress
*effective* pressure — the permission to wake — with an ``inhibition`` that holds
at ``i0`` for a grace plateau (a guaranteed quiet window) then decays
exponentially. ``effective = latent · (1 − inhibition)``. Escalation/duration read
latent; only the wake threshold reads effective.
"""

from __future__ import annotations

import math
from datetime import datetime

from .timeutil import minutes_between


def inhibition_at(
    action_pending_since: str | None,
    now: datetime,
    *,
    i0: float,
    grace_min: float,
    halflife_min: float,
) -> float:
    """Inhibition ∈ [0,1] at ``now`` for an ActionPending started at
    ``action_pending_since`` (ISO). ``None`` → 0. Grace plateau then half-life
    decay."""
    if action_pending_since is None:
        return 0.0
    t = minutes_between(action_pending_since, now)
    if t <= grace_min:
        value = i0
    else:
        lam = math.log(2.0) / halflife_min
        value = i0 * math.exp(-lam * (t - grace_min))
    return max(0.0, min(1.0, value))


def effective_pressure(latent: float, inhibition: float) -> float:
    """Permission-to-wake pressure: ``max(0, latent·(1−inhibition))``."""
    return max(0.0, latent * (1.0 - inhibition))

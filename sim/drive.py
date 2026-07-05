"""The drive-component: the contact urge ``u`` and its dynamics (spec §5).

The drive holds the one continuous state variable and evolves it. It knows
nothing about gates, thresholds, or cognition — those live in the wake-decision
layer. Constants are normalised so ``θ_u = 1`` and ``β = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Drive:
    """The contact urge and its three moves: rise, satiate, drain.

    - ``rise(dt)``     — accumulate in genuine silence: ``u ← min(u_max, u + dt·α)``.
    - ``satiate(q)``   — a positive exchange drains ``β·q``: ``u ← max(0, u − β·q)``.
    - ``drain(f)``     — the wake-decision consumed an URGE: ``u ← (1 − f)·u``
      (``f = 1`` full drain to zero; ``0 < f < 1`` partial).
    """

    alpha: float
    beta: float = 1.0
    u_max: float = math.inf
    u: float = 0.0

    def rise(self, *, dt: float) -> None:
        """Accumulate the urge over ``dt`` of genuine silence, capped at ``u_max``."""
        self.u = min(self.u_max, self.u + dt * self.alpha)

    def satiate(self, *, q: float) -> None:
        """Reduce the urge on a positive exchange. Non-positive ``q`` does nothing."""
        if q <= 0.0:
            return
        self.u = max(0.0, self.u - self.beta * q)

    def drain(self, *, fraction: float = 1.0) -> None:
        """Consume the urge by ``fraction`` (default full drain to zero)."""
        self.u = max(0.0, self.u * (1.0 - fraction))

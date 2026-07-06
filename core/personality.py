"""Personality — the being's physiology component (spec §8, §9).

A cheap (0-energy) layer that runs every tick and evolves the battery ``E`` and
the fatigue debt ``S`` from elapsed rest, modulated by the circadian rhythm
``C``. Here the being only *recovers*: ``E`` climbs back toward ``E_MAX`` (faster
when ``C`` is low — night rest) and ``S`` decays toward 0. The *spend* that
drains ``E`` and raises ``S`` is cognition's, wired in Phase D (only cognition
pays energy — the coma-fix). No sleep state: the digital human simply rests
during idle.
"""

from __future__ import annotations

from collections.abc import Sequence

from .circadian import circadian
from .component import TickContext
from .intents import Intent, UpdateState
from .timeutil import minutes_between


class Personality:
    """Holds and evolves physiology (energy, fatigue) against the circadian clock."""

    def __init__(
        self,
        *,
        e_max: float,
        recovery_per_min: float,
        night_boost: float,
        fatigue_decay_per_min: float,
        peak_hour_utc: float,
        id: str = "personality",
    ) -> None:
        self.id = id
        self._e_max = e_max
        self._recovery_per_min = recovery_per_min
        self._night_boost = night_boost
        self._fatigue_decay_per_min = fatigue_decay_per_min
        self._peak_hour_utc = peak_hour_utc

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        dt = max(0.0, minutes_between(state.last_tick_at, ctx.now))
        c = circadian(ctx.now, peak_hour_utc=self._peak_hour_utc)

        recovery = self._recovery_per_min * (1.0 + self._night_boost * (1.0 - c))
        energy = min(self._e_max, state.energy + recovery * dt)
        fatigue = max(0.0, state.fatigue - self._fatigue_decay_per_min * dt)

        return [UpdateState({"energy": energy, "fatigue": fatigue})]

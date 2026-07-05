"""``Aggregator`` — the aggregation-layer extension point (HLA §2/§5/§13).

The aggregator is the thalamus: it consumes signals off the bus, weighs the
being's **accumulated** pressure, applies habituation/deferral ("don't wake for
trifles"), and returns a :class:`~lifemodel.domain.wake.WakeDecision` — wake
cognition, or stay quiet at zero LLM cost (HLA §1/§5). Most ticks stay quiet.

The wake call is made against an *accumulated* pressure value (the drive that
grows tick over tick), not just this tick's transient signals, so the tick
orchestrator threads that value in as a keyword argument. The threshold/wake
logic lives **entirely here** (HLA "aggregation = thalamus"); the tick only
orchestrates.

This module ships the contract plus :class:`SilentAggregator` — a pass-through
that never wakes, kept available for tests and any wiring that wants a
guaranteed-quiet graph. The Phase-1.3 ``ThresholdAggregator`` default was
removed by the wire-desire-model plan (Task 8): the certified desire model in
:mod:`lifemodel.sim` — reconstructed each tick by :mod:`lifemodel.core.decision`
— now makes the wake decision directly from ``State``, bypassing this
aggregator seam entirely in the live path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..domain.signal import Signal
from ..domain.wake import WakeDecision


class Aggregator(ABC):
    """Decide, from the tick's signals and accumulated pressure, whether to wake.

    ``decide`` receives the tick's fresh *signals* (for future habituation /
    per-kind weighing) and, as a keyword-only argument, the being's accumulated
    ``pressure`` — the value the wake threshold is weighed against (HLA §5/§11).
    """

    @abstractmethod
    def decide(self, signals: Sequence[Signal], *, pressure: float) -> WakeDecision:
        """Weigh the accumulated *pressure* and return a wake-or-quiet decision."""
        raise NotImplementedError


class SilentAggregator(Aggregator):
    """A pass-through aggregator that never wakes cognition, whatever the pressure.

    Kept available for tests and for any wiring that wants a guaranteed
    zero-LLM graph (HLA §1: below threshold → quiet); it is also the live
    default the composition root wires now that the cron path decides nothing
    (wire-desire-model Task 4).
    """

    def decide(self, signals: Sequence[Signal], *, pressure: float) -> WakeDecision:
        return WakeDecision.stay_asleep()

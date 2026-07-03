"""``Aggregator`` — the aggregation-layer extension point (HLA §2/§5/§13).

The aggregator is the thalamus: it consumes signals off the bus, accumulates
salience, applies habituation/deferral ("don't wake for trifles"), and returns a
:class:`~lifemodel.domain.wake.WakeDecision` — wake cognition, or stay quiet at
zero LLM cost (HLA §1/§5). Most ticks stay quiet.

This module ships the contract plus one trivial default,
:class:`SilentAggregator`, which never wakes. That default lets the composition
root build a complete, testable object graph now; the real thresholding
aggregator is Phase 1.3, which *implements* this interface rather than inventing
its shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..domain.signal import Signal
from ..domain.wake import WakeDecision


class Aggregator(ABC):
    """Decide, from the tick's signals, whether to wake cognition (HLA §5)."""

    @abstractmethod
    def decide(self, signals: Sequence[Signal]) -> WakeDecision:
        """Weigh *signals* and return a wake-or-stay-quiet decision (HLA §11)."""
        raise NotImplementedError


class SilentAggregator(Aggregator):
    """A pass-through default that never wakes cognition (Phase-0 stub).

    Real salience/threshold/habituation logic is Phase 1.3; until then this keeps
    the walking skeleton at zero LLM cost — the correct baseline behaviour, not a
    placeholder that would misbehave (HLA §1: below threshold → quiet).
    """

    def decide(self, signals: Sequence[Signal]) -> WakeDecision:
        return WakeDecision.stay_asleep()

"""``Aggregator`` — the aggregation-layer extension point (HLA §2/§5/§13).

The aggregator is the thalamus: it consumes signals off the bus, weighs the
being's **accumulated** pressure, applies habituation/deferral ("don't wake for
trifles"), and returns a :class:`~lifemodel.domain.wake.WakeDecision` — wake
cognition, or stay quiet at zero LLM cost (HLA §1/§5). Most ticks stay quiet.

The wake call is made against the *accumulated* ``State.pressure`` (the drive
that grows tick over tick), not just this tick's transient signals, so the tick
orchestrator threads that value in as a keyword argument. The threshold/wake
logic lives **entirely here** (HLA "aggregation = thalamus"); the tick only
orchestrates.

This module ships the contract plus two implementations:

* :class:`SilentAggregator` — a pass-through that never wakes. It stays
  available for tests and any wiring that wants a guaranteed-quiet graph.
* :class:`ThresholdAggregator` — the Phase-1.3 default: wake when the
  accumulated pressure has crossed a threshold, emitting the wake-packet that
  explains why. Both *implement* the ABC rather than inventing its shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..domain.signal import Signal
from ..domain.wake import WakeDecision, WakePacket

#: Default wake threshold the composition root wires with no argument. A sane
#: constant that keeps the seam clean; the HLA "thresholds from disk, hot-reload"
#: source plugs in here later by feeding the ``threshold`` constructor arg from a
#: config file — no reshape of this interface (do not over-build it now).
DEFAULT_WAKE_THRESHOLD = 10.0

#: What the threshold aggregator stamps on its wake-packet. The accumulated
#: ``State.pressure`` is a single aggregate scalar in Phase 1; a richer
#: aggregator later attributes the crossing to a specific neuron kind.
WAKE_REASON = "accumulated pressure crossed wake threshold"
WAKE_PRESSURE_KIND = "pressure"


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

    The real thresholding default is :class:`ThresholdAggregator`; this stays
    available for tests and for any wiring that wants a guaranteed zero-LLM graph
    (HLA §1: below threshold → quiet).
    """

    def decide(self, signals: Sequence[Signal], *, pressure: float) -> WakeDecision:
        return WakeDecision.stay_asleep()


class ThresholdAggregator(Aggregator):
    """Wake when the accumulated pressure crosses a threshold (roadmap 1.3).

    The being's drive to act (``State.pressure``) grows tick over tick while it
    stays below ``threshold`` — every such tick is quiet, zero LLM (HLA §1). The
    moment the accumulated pressure reaches the threshold (``pressure >=
    threshold``), cognition wakes and receives a :class:`WakePacket` explaining
    *why* it woke — the reason, the pressure that crossed, and the threshold it
    crossed (HLA §11).

    The threshold is the aggregator's own knob: injected at construction with a
    sane :data:`DEFAULT_WAKE_THRESHOLD`. That is the seam a disk-backed,
    hot-reloadable threshold source plugs into later (HLA "thresholds from disk")
    without reshaping this interface.

    Scope note (roadmap 1.3 vs 1.4): the aggregator stays a *pure thalamus* —
    pressure vs threshold — and never drains, cools down, or single-fires. So on
    its own, above threshold it keeps deciding "wake" every tick. The drain, the
    cooldown veto, and the one-message-per-cycle limit are task 1.4 and live in
    the orchestrator (:func:`lifemodel.tick.run_tick`), which threads ``now`` +
    the stored cooldown around this call — this interface is deliberately left
    untouched (no ``now``/cooldown args here).
    """

    def __init__(self, *, threshold: float = DEFAULT_WAKE_THRESHOLD) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The pressure level at (or above) which this aggregator wakes."""
        return self._threshold

    def decide(self, signals: Sequence[Signal], *, pressure: float) -> WakeDecision:
        """Wake iff the accumulated *pressure* has reached the threshold."""
        if pressure < self._threshold:
            return WakeDecision.stay_asleep()
        packet = WakePacket(
            reason=WAKE_REASON,
            pressure_kind=WAKE_PRESSURE_KIND,
            pressure=pressure,
            threshold=self._threshold,
        )
        return WakeDecision.wake_with(packet)

"""Tests for :class:`ThresholdAggregator` — the real wake decision (roadmap 1.3).

The aggregator is the thalamus (HLA §1/§2): it weighs the *accumulated* pressure
(``State.pressure``) against a threshold and returns a
:class:`~lifemodel.domain.wake.WakeDecision` — wake cognition when pressure has
crossed, stay quiet (zero LLM) below it. The threshold decision lives entirely
here, never in the tick orchestrator. Imports no Hermes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lifemodel.core.aggregator import (
    DEFAULT_WAKE_THRESHOLD,
    Aggregator,
    SilentAggregator,
    ThresholdAggregator,
)
from lifemodel.domain.signal import Signal
from lifemodel.domain.wake import WakePacket


def test_is_an_aggregator() -> None:
    assert isinstance(ThresholdAggregator(), Aggregator)


def test_below_threshold_stays_asleep_with_no_packet() -> None:
    agg = ThresholdAggregator(threshold=3.0)

    decision = agg.decide([], pressure=2.999)

    assert decision.wake is False
    assert decision.packet is None


def test_at_threshold_wakes() -> None:
    # ``>=`` — pressure exactly equal to the threshold crosses it.
    agg = ThresholdAggregator(threshold=3.0)

    decision = agg.decide([], pressure=3.0)

    assert decision.wake is True
    assert decision.packet is not None


def test_above_threshold_wakes_with_a_packet_carrying_reason_pressure_threshold() -> None:
    agg = ThresholdAggregator(threshold=3.0)

    decision = agg.decide([], pressure=5.0)

    assert decision.wake is True
    packet = decision.packet
    assert packet is not None
    assert packet.reason  # a non-empty human-readable reason
    assert packet.pressure == 5.0
    assert packet.threshold == 3.0


def test_wake_packet_round_trips_and_serializes_cleanly() -> None:
    # The packet is the neuron script's stdout schema (HLA §11): it must survive
    # the hardened to_dict/from_dict round-trip and serialize on the
    # allow_nan=False path with the threshold field intact.
    decision = ThresholdAggregator(threshold=2.0).decide([], pressure=4.0)
    assert decision.packet is not None

    restored = WakePacket.from_dict(decision.packet.to_dict())
    assert restored == decision.packet
    assert restored.threshold == 2.0
    assert restored.pressure == 4.0

    # allow_nan=False path (to_json) does not raise on a finite packet.
    assert WakePacket.from_json(decision.packet.to_json()) == decision.packet


def test_threshold_is_configurable_and_defaulted() -> None:
    # The threshold is the aggregator's own knob (the disk/hot-reload seam);
    # a sane default lets the composition root wire it with no argument.
    assert ThresholdAggregator().threshold == DEFAULT_WAKE_THRESHOLD
    assert ThresholdAggregator(threshold=7.5).threshold == 7.5

    # The default threshold actually gates: below it sleeps, at/above it wakes.
    default_agg = ThresholdAggregator()
    assert default_agg.decide([], pressure=DEFAULT_WAKE_THRESHOLD - 0.5).wake is False
    assert default_agg.decide([], pressure=DEFAULT_WAKE_THRESHOLD).wake is True


def test_decision_ignores_transient_signals_and_reads_accumulated_pressure() -> None:
    # The decision is against the ACCUMULATED pressure argument, not the salience
    # of this tick's transient signals — a big transient below-threshold pressure
    # stays asleep; a small transient above-threshold pressure wakes.
    agg = ThresholdAggregator(threshold=10.0)
    loud: Sequence[Signal] = [Signal(origin_id="s", kind="timer_pressure", salience=99.0)]

    assert agg.decide(loud, pressure=1.0).wake is False
    assert agg.decide([], pressure=10.0).wake is True


def test_silent_aggregator_still_satisfies_the_updated_interface() -> None:
    # SilentAggregator stays available (tests use it) and honours the new
    # keyword-only pressure arg — it never wakes, whatever the pressure.
    agg = SilentAggregator()
    assert isinstance(agg, Aggregator)
    assert agg.decide([], pressure=0.0).wake is False
    assert agg.decide([], pressure=10_000.0).wake is False


class _MinimalAggregator(Aggregator):
    """A hand-rolled subclass proving the ABC's decide signature is stable."""

    def decide(self, signals: Sequence[Signal], *, pressure: float) -> Any:
        from lifemodel.domain.wake import WakeDecision

        return WakeDecision.stay_asleep()


def test_abc_decide_signature_is_signals_plus_keyword_pressure() -> None:
    assert _MinimalAggregator().decide([], pressure=1.0).wake is False

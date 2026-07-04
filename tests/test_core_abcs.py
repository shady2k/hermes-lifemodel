"""Tests for the core extension-point ABCs (HLA §13).

The contract every extension point must honour:
* it cannot be instantiated without implementing its abstract method(s);
* a trivial concrete subclass instantiates and works.

Plus the small behaviour the base classes own (``Layer.meets_confidence``,
``SilentAggregator`` never waking). Imports no Hermes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from lifemodel.core.act_gate import ActGate
from lifemodel.core.aggregator import Aggregator, SilentAggregator
from lifemodel.core.layer import Layer, ProcessingLayer
from lifemodel.core.neuron import Neuron
from lifemodel.core.signal_bus import SignalBus
from lifemodel.domain.act import Decision
from lifemodel.domain.layer import LayerResult
from lifemodel.domain.signal import Signal
from lifemodel.domain.wake import WakeDecision, WakePacket
from lifemodel.state.model import State

ABSTRACT_CLASSES = [Neuron, Layer, Aggregator, ActGate, SignalBus]


@pytest.mark.parametrize("cls", ABSTRACT_CLASSES)
def test_abc_cannot_be_instantiated_directly(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]


def test_incomplete_subclass_still_cannot_instantiate() -> None:
    class HalfNeuron(Neuron):
        pass  # does not implement tick

    with pytest.raises(TypeError):
        HalfNeuron()  # type: ignore[abstract]


# --- trivial concrete impls: each extension point is subclassable ---


def test_trivial_neuron_impl_works() -> None:
    class QuietNeuron(Neuron):
        def tick(self, state: State) -> list[Signal]:
            return [Signal(origin_id="n1", kind="test")]

    assert QuietNeuron().tick(State()) == [Signal(origin_id="n1", kind="test")]


def test_trivial_aggregator_impl_works() -> None:
    packet = WakePacket(reason="r", pressure_kind="k", pressure=1.0)

    class EagerAggregator(Aggregator):
        def decide(self, signals: Sequence[Signal], *, pressure: float) -> WakeDecision:
            return WakeDecision.wake_with(packet)

    assert EagerAggregator().decide([], pressure=0.0).wake is True


def test_trivial_act_gate_impl_works() -> None:
    class OpenGate(ActGate):
        def allow(self, ctx: Any) -> Decision:
            return Decision.allowed("test")

    assert OpenGate().allow(None).allow is True


def test_trivial_layer_impl_works_and_alias_is_layer() -> None:
    assert ProcessingLayer is Layer

    class ConstLayer(Layer):
        confidence_threshold = 0.7

        def process(self, ctx: Any) -> LayerResult:
            return LayerResult(confidence=0.9)

    assert ConstLayer().process(None).confidence == 0.9


def test_layer_meets_confidence_threshold_and_escalation() -> None:
    class L(Layer):
        confidence_threshold = 0.7

        def process(self, ctx: Any) -> LayerResult:
            return LayerResult(confidence=0.5)

    layer = L()
    assert layer.meets_confidence(LayerResult(confidence=0.8)) is True
    assert layer.meets_confidence(LayerResult(confidence=0.6)) is False
    # An explicit escalate request overrides a high confidence.
    assert layer.meets_confidence(LayerResult(confidence=0.99, escalate=True)) is False


def test_silent_aggregator_never_wakes() -> None:
    agg = SilentAggregator()
    assert isinstance(agg, Aggregator)
    signals = [Signal(origin_id="a", kind="x", salience=99.0)]
    # Never wakes, whatever the accumulated pressure — even far above any threshold.
    assert agg.decide(signals, pressure=99.0) == WakeDecision.stay_asleep()
    assert agg.decide([], pressure=10_000.0).wake is False

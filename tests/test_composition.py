"""Tests for the composition root — the one DI wiring module (HLA §13).

Acceptance (roadmap 0.4):
* the full graph builds from injected fakes with **no Hermes import**;
* it also builds with the concrete JSON state store + durable bus over a tmp dir;
* every collaborator is overridable (so ``register(ctx)`` can inject real Hermes
  adapters and tests can inject fakes).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.adapters.clock import SystemClock
from lifemodel.adapters.delivery import NoopDelivery
from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.aggregator import SilentAggregator
from lifemodel.core.neuron import Neuron, StubTimerNeuron
from lifemodel.domain.signal import Signal
from lifemodel.state.json_store import JsonStateStore
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore


class _CountingNeuron(Neuron):
    def tick(self, state: State) -> list[Signal]:
        return [Signal(origin_id="n", kind="count")]


def _assert_no_hermes() -> None:
    assert "hermes_constants" not in sys.modules
    assert not any(m == "hermes" or m.startswith("hermes.") for m in sys.modules)


def test_builds_full_graph_from_injected_fakes_without_hermes(tmp_path: Path) -> None:
    fake_state = FakeStateStore()
    fake_bus = FakeSignalBus()
    fake_clock = FakeClock(datetime(2026, 7, 3, tzinfo=UTC))
    fake_delivery = FakeDelivery()
    neurons = [_CountingNeuron()]

    lm = build_lifemodel(
        base_dir=tmp_path / "unused",  # fakes injected, so base_dir is untouched
        state=fake_state,
        bus=fake_bus,
        clock=fake_clock,
        delivery=fake_delivery,
        neurons=neurons,
    )

    assert isinstance(lm, LifeModel)
    assert lm.state is fake_state
    assert lm.bus is fake_bus
    assert lm.clock is fake_clock
    assert lm.delivery is fake_delivery
    assert lm.neurons == (neurons[0],)
    _assert_no_hermes()


def test_builds_with_concrete_json_store_over_tmp_dir(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)

    assert isinstance(lm.state, JsonStateStore)
    assert isinstance(lm.bus, FileSignalBus)
    assert isinstance(lm.clock, SystemClock)
    assert isinstance(lm.delivery, NoopDelivery)
    assert isinstance(lm.aggregator, SilentAggregator)
    # Phase 1.2 replaced the empty default neuron list with the stub timer neuron.
    assert len(lm.neurons) == 1
    assert isinstance(lm.neurons[0], StubTimerNeuron)
    _assert_no_hermes()


def test_explicit_empty_neurons_overrides_the_stub_default(tmp_path: Path) -> None:
    # Passing neurons explicitly (even empty) opts out of the default — the seam
    # tests that need a bare graph, and later real wiring, stay in control.
    lm = build_lifemodel(base_dir=tmp_path, neurons=())

    assert lm.neurons == ()


def test_default_graph_is_exercisable_end_to_end(tmp_path: Path) -> None:
    # The concrete graph actually works: commit state, publish + consume signals,
    # and the pass-through aggregator stays quiet.
    lm = build_lifemodel(base_dir=tmp_path)

    lm.state.commit(State(pressure=1.5))
    assert lm.state.load().pressure == 1.5

    lm.bus.publish(Signal(origin_id="m1", kind="incoming"))
    consumed: Sequence[Signal] = lm.bus.consume_unprocessed()
    assert [s.origin_id for s in consumed] == ["m1"]

    assert lm.aggregator.decide(consumed).wake is False
    _assert_no_hermes()


def test_lifemodel_graph_is_frozen(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    try:
        lm.state = FakeStateStore()  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("LifeModel should be frozen")

"""Tests for the composition root — the one DI wiring module (HLA §13).

Acceptance (roadmap 0.4):
* the full graph builds from injected fakes with **no Hermes import**;
* it also builds with the concrete SQLite state store + durable bus over a tmp dir;
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
from lifemodel.core.contact_neuron import ContactNeuron
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.neuron import Neuron
from lifemodel.core.registry import ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.core.taxonomy import exchange_signal
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
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


def test_builds_with_concrete_sqlite_store_over_tmp_dir(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)

    assert isinstance(lm.state, SQLiteRuntimeStore)
    assert isinstance(lm.bus, FileSignalBus)
    assert isinstance(lm.clock, SystemClock)
    assert isinstance(lm.delivery, NoopDelivery)
    # Wire-desire-model plan (Task 4): the live path decides nothing here — the
    # in-process service is the sole brain via core/decision — so the default
    # aggregator is the guaranteed-quiet SilentAggregator and there are no
    # default neurons.
    assert isinstance(lm.aggregator, SilentAggregator)
    assert lm.neurons == ()
    _assert_no_hermes()


def test_explicit_neurons_and_aggregator_override_the_defaults(tmp_path: Path) -> None:
    # Passing neurons/aggregator explicitly (even empty/guaranteed-quiet) opts
    # out of the default — the seam tests, and later real wiring, stay in
    # control.
    neuron = _CountingNeuron()
    lm = build_lifemodel(base_dir=tmp_path, neurons=(neuron,), aggregator=SilentAggregator())

    assert lm.neurons == (neuron,)
    assert isinstance(lm.aggregator, SilentAggregator)


def test_default_graph_is_exercisable_end_to_end(tmp_path: Path) -> None:
    # The concrete graph actually works: commit state, publish + consume signals,
    # and the pass-through aggregator stays quiet.
    lm = build_lifemodel(base_dir=tmp_path)

    lm.state.commit(State(u=1.5))
    assert lm.state.load().u == 1.5

    lm.bus.publish(Signal(origin_id="m1", kind="incoming"))
    consumed: Sequence[Signal] = lm.bus.consume_unprocessed()
    assert [s.origin_id for s in consumed] == ["m1"]

    # The default SilentAggregator never wakes, whatever it is asked to decide.
    assert lm.aggregator.decide(consumed, pressure=0.0).wake is False
    _assert_no_hermes()


def test_lifemodel_graph_is_frozen(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    try:
        lm.state = FakeStateStore()  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("LifeModel should be frozen")


def test_build_wires_registry_state_actor_and_coreloop(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    assert isinstance(lm.registry, ComponentRegistry)
    assert isinstance(lm.state_actor, StateActor)
    assert isinstance(lm.coreloop, CoreLoop)


def test_default_registry_contains_contact_neuron(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    ids = tuple(c.id for c in lm.registry.enabled())
    assert "contact" in ids
    assert "contact-aggregation" in ids


def test_coreloop_tick_bookkeeps_and_runs_contact(tmp_path: Path) -> None:
    # Contact neuron + aggregation run (last_tick_at=None → dt=0, no signals → no satiate).
    # Tick still checkpoints the bookkeeping bump — proves the wired seam works.
    lm = build_lifemodel(base_dir=tmp_path)
    report = lm.coreloop.tick()
    assert "contact" in report.ran
    assert "contact-aggregation" in report.ran
    assert lm.state.load().tick_count == 1


# --- Phase B1: ContactNeuron wiring ---


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._m = moment

    def now(self) -> datetime:
        return self._m


def test_contact_neuron_is_registered_enabled(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert "contact" in ids
    assert any(isinstance(c, ContactNeuron) for c in lm.registry.enabled())


def test_pipeline_tick_rises_u_and_persists(tmp_path: Path) -> None:
    # Seed last_tick_at 240 min before the clock; one tick should rise u to ~1.0.
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    assert abs(store.load().u - 1.0) < 1e-9


def test_pipeline_tick_satiates_on_inbound_exchange(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.bus.publish(exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    assert store.load().u == 0.0  # satiated by the two_way exchange


# --- Phase B2: ContactAggregation wiring ---


def test_aggregation_registered_after_neuron(tmp_path: Path) -> None:
    from lifemodel.core.aggregation import ContactAggregation

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert ids.index("contact") < ids.index("contact-aggregation")
    assert any(isinstance(c, ContactAggregation) for c in lm.registry.enabled())


def test_pipeline_rises_then_wakes_desire(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # u already high; 1 min elapsed → neuron keeps it high, aggregation wakes a desire
    store.commit(State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    assert store.load().desire_status == "active"


def test_pipeline_exchange_satiates_and_clears(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=3.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.bus.publish(exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    final = store.load()
    # neuron rises by alpha*dt then satiates by beta*q (q=1.0 for two_way)
    assert final.u < 3.0  # neuron satiated (reduced)
    assert final.desire_status == "none"  # aggregation cleared the desire
    assert final.last_exchange_at is not None


# --- Phase C1: inhibition constants wired into composition ---


def test_pipeline_send_suppresses_then_recovers(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # high latent, a send 10 min ago -> within grace -> no wake this tick
    store.commit(
        State(
            u=3.0,
            desire_status="none",
            action_pending_since="2026-07-06T03:50:00+00:00",
            last_tick_at="2026-07-06T03:59:00+00:00",
        )
    )
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    assert store.load().desire_status == "none"  # grace suppresses the wake end-to-end


# --- Phase C2: Personality component wiring ---


def test_personality_is_registered(tmp_path: Path) -> None:
    from lifemodel.core.personality import Personality

    lm = build_lifemodel(base_dir=tmp_path)
    assert any(isinstance(c, Personality) for c in lm.registry.enabled())


def test_pipeline_tick_recovers_energy_and_decays_fatigue(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 12, 30, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(energy=0.5, fatigue=0.5, last_tick_at="2026-07-06T12:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    final = store.load()
    assert final.energy > 0.5  # recovered during the idle tick
    assert final.fatigue < 0.5  # decayed


# --- Phase D1: Cognition component wiring ---


def test_cognition_registered_after_aggregation(tmp_path: Path) -> None:
    from lifemodel.core.cognition import Cognition

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert ids.index("contact-aggregation") < ids.index("cognition")
    assert any(isinstance(c, Cognition) for c in lm.registry.enabled())

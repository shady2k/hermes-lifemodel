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
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.neuron import Neuron
from lifemodel.core.registry import ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.core.taxonomy import exchange_signal
from lifemodel.domain.objects import DesireState
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


def test_injected_fake_state_store_graph_can_tick(tmp_path: Path) -> None:
    # lm-27n.2: an injected StatePort-only fake leaves the memory/pressure slots
    # unwired (CoreLoop reads empty snapshots) and the StateActor falls back to
    # the fake's own commit_tick — a full tick must still run and persist.
    fake_state = FakeStateStore()
    lm = build_lifemodel(
        base_dir=tmp_path / "unused",
        state=fake_state,
        bus=FakeSignalBus(),
        clock=FakeClock(datetime(2026, 7, 3, tzinfo=UTC)),
        delivery=FakeDelivery(),
    )
    assert lm.coreloop is not None
    lm.coreloop.tick()  # must not raise
    assert lm.state.load().tick_count == 1


def test_default_graph_wires_the_store_as_the_atomic_committer(tmp_path: Path) -> None:
    # The one SQLite store backs state + memory + pressure + the tick committer,
    # so a tick's commit spans vitals and entities in one transaction (HLA §4.1).
    lm = build_lifemodel(base_dir=tmp_path)
    assert isinstance(lm.state, SQLiteRuntimeStore)
    lm.coreloop.tick()  # ticks through commit_tick over that same store
    assert lm.state.load().tick_count == 1


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


def _seed_active_desire(store: SQLiteRuntimeStore) -> None:
    """Persist a live active contact-desire row through the real store."""
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=3.0)))


def test_pipeline_rises_then_wakes_desire(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # u already high; 1 min elapsed → neuron keeps it high, aggregation wakes a desire
    store.commit(State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    desire = read_live_contact_desire(store)  # the desire is now a typed row, not a flag
    assert desire is not None and desire.state == "active"


def test_pipeline_dedups_desire_across_ticks(tmp_path: Path) -> None:
    # tick 1 births the desire; tick 2 sees it live in the snapshot and dedups —
    # the singleton row is not re-written (revision stays put), no double-wake.
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    born = store.get("desire", "contact:owner")
    assert born is not None and born.state == "active"
    lm.coreloop.tick()  # a second high-u tick must not re-birth the desire
    again = store.get("desire", "contact:owner")
    assert again is not None and again.revision == born.revision  # untouched -> deduped


def test_pipeline_exchange_satiates_and_clears(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    _seed_active_desire(store)  # a live desire the exchange must terminalize
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.bus.publish(exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    final = store.load()
    # neuron rises by alpha*dt then satiates by beta*q (q=1.0 for two_way)
    assert final.u < 3.0  # neuron satiated (reduced)
    assert read_live_contact_desire(store) is None  # aggregation terminalized the desire
    assert store.get("desire", "contact:owner").state == "satisfied"
    assert final.last_exchange_at is not None


# --- Phase C1: inhibition constants wired into composition ---


def test_pipeline_send_suppresses_then_recovers(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # high latent, a send 10 min ago -> within grace -> no wake this tick
    store.commit(
        State(
            u=3.0,
            action_pending_since="2026-07-06T03:50:00+00:00",
            last_tick_at="2026-07-06T03:59:00+00:00",
        )
    )
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    assert read_live_contact_desire(store) is None  # grace suppresses the wake end-to-end


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


# --- lm-27n.9: ThoughtCrystallization proposes BEFORE aggregation + attention ---


def test_thought_crystallization_registered_before_aggregation_and_attention(
    tmp_path: Path,
) -> None:
    from lifemodel.core.thought_crystallization import ThoughtCrystallization

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    # The pure proposer runs AFTER the neuron and BEFORE the two writers so its
    # in-tick EmitSignal reaches aggregation (desire writer) + attention (thought
    # writer) the SAME tick — the reordered pipeline's contract.
    assert ids.index("contact") < ids.index("thought-crystallization")
    assert ids.index("thought-crystallization") < ids.index("contact-aggregation")
    assert ids.index("thought-crystallization") < ids.index("thought-attention")
    assert any(isinstance(c, ThoughtCrystallization) for c in lm.registry.enabled())


def test_full_pipeline_order_is_asserted(tmp_path: Path) -> None:
    # The whole reordered spine (lm-27n.9): personality -> neuron ->
    # thought-crystallization -> contact-aggregation -> thought-attention ->
    # thought-generation -> cognition.
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    spine = [
        "personality",
        "contact",
        "thought-crystallization",
        "contact-aggregation",
        "thought-attention",
        "thought-generation",
        "cognition",
    ]
    positions = [ids.index(cid) for cid in spine]
    assert positions == sorted(positions), ids


# --- lm-27n.7: the ThoughtAttention brake sits between aggregation and cognition ---


def test_thought_attention_registered_between_aggregation_and_cognition(tmp_path: Path) -> None:
    from lifemodel.core.thought_attention import ThoughtAttention

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    # The 0-LLM brake reads the settled snapshot (after aggregation) and feeds the
    # attended set to cognition (before it) — the enabled() order is the contract.
    assert ids.index("contact-aggregation") < ids.index("thought-attention")
    assert ids.index("thought-attention") < ids.index("cognition")
    assert any(isinstance(c, ThoughtAttention) for c in lm.registry.enabled())


# --- lm-27n.8: ThoughtGeneration sits between the attention brake and cognition ---


def test_thought_generation_registered_between_attention_and_cognition(tmp_path: Path) -> None:
    from lifemodel.core.thought_generation import ThoughtGeneration

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    # The 0-LLM generative stream reads the settled snapshot (after the .7 brake so
    # it can chain the just-attended thought) and runs before cognition (a generated
    # thought is visible only NEXT tick, so it can never launch a turn it created).
    assert ids.index("thought-attention") < ids.index("thought-generation")
    assert ids.index("thought-generation") < ids.index("cognition")
    assert any(isinstance(c, ThoughtGeneration) for c in lm.registry.enabled())


# --- lm-27n.4: the Intention decision record, end-to-end through the store ---


def test_pipeline_crystallizes_intention_on_launch(tmp_path: Path) -> None:
    # A high-u tick that launches also births the singleton intention active, in
    # the SAME commit as pending being set (behavior-neutral on send timing).
    from lifemodel.core.intention_view import read_live_contact_intention

    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(State(u=3.0, energy=1.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    _seed_active_desire(store)  # a live active desire cognition can launch on
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    final = store.load()
    intention = read_live_contact_intention(store)
    assert intention is not None and intention.state == "active"  # decision recorded
    assert final.pending_proactive_id is not None  # ...in the same commit as pending


def test_pipeline_exchange_resolves_desire_and_intention_atomically(tmp_path: Path) -> None:
    # A real exchange terminalizes BOTH the desire and the intention in one tick,
    # through the one atomic store committer — never a split-brain (HLA §4.1).
    from lifemodel.core.intention_view import read_live_contact_intention

    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # a turn already in flight (pending set) gates cognition off, so this tick only
    # RESOLVES the launched pair — it does not re-launch/re-crystallize.
    store.commit(
        State(
            u=3.0,
            pending_proactive_id="proactive-inflight",
            last_tick_at="2026-07-06T03:59:00+00:00",
        )
    )
    _seed_active_desire(store)
    # seed a live intention too (the launched decision record)
    from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
    from lifemodel.domain.objects import IntentionState

    store.put(encode_contact_intention(build_contact_intention(state=IntentionState.ACTIVE)))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.bus.publish(exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    assert read_live_contact_desire(store) is None  # desire terminalized
    assert read_live_contact_intention(store) is None  # intention terminalized in lockstep
    assert store.get("desire", "contact:owner").state == "satisfied"
    assert store.get("intention", "contact:owner").state == "completed"

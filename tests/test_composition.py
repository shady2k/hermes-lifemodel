"""Tests for the composition root — the one DI wiring module (HLA §13).

Acceptance (roadmap 0.4):
* the full graph builds from injected fakes with **no Hermes import**;
* it also builds with the concrete SQLite state store over a tmp dir;
* every collaborator is overridable (so ``register(ctx)`` can inject real Hermes
  adapters and tests can inject fakes).

The nervous flow is ephemeral (spec §2/§3): there is no durable signal bus — a
frame is seeded with ``initial_signals`` and the pipeline folds them.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.adapters.clock import SystemClock
from lifemodel.adapters.delivery import NoopDelivery
from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.contact_sensor import ContactSensor
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.registry import ComponentRegistry
from lifemodel.core.solitude_drive import SolitudeDrive
from lifemodel.core.state_actor import StateActor
from lifemodel.core.taxonomy import contact_observed_signal
from lifemodel.domain.objects import DesireState
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock, FakeDelivery, FakeStateStore

#: These are DRIVE-path pipeline tests, so every being in them has already been born and
#: says so. A being with no ``genesis_completed_at`` and nothing on record is at its FIRST
#: WAKING (spec §6.2): it wakes to be BORN, without ``u`` ever crossing ``θ``, and would
#: make "below threshold ⇒ quiet" assertions here false for a reason that has nothing to
#: do with the drive. The genesis wake path is pinned in ``tests/test_genesis_wake.py``.
BORN_AT = "2026-07-01T10:00:00+00:00"


def _born(**kw: object) -> State:
    """A ``State`` for a being that has been born (see :data:`BORN_AT`)."""
    kw.setdefault("genesis_completed_at", BORN_AT)
    return State(**kw)  # type: ignore[arg-type]


def _assert_no_hermes() -> None:
    assert "hermes_constants" not in sys.modules
    assert not any(m == "hermes" or m.startswith("hermes.") for m in sys.modules)


def test_builds_full_graph_from_injected_fakes_without_hermes(tmp_path: Path) -> None:
    fake_state = FakeStateStore()
    fake_clock = FakeClock(datetime(2026, 7, 3, tzinfo=UTC))
    fake_delivery = FakeDelivery()

    lm = build_lifemodel(
        base_dir=tmp_path / "unused",  # fakes injected, so base_dir is untouched
        state=fake_state,
        clock=fake_clock,
        delivery=fake_delivery,
    )

    assert isinstance(lm, LifeModel)
    assert lm.state is fake_state
    assert lm.clock is fake_clock
    assert lm.delivery is fake_delivery
    _assert_no_hermes()


def test_injected_fake_state_store_graph_can_tick(tmp_path: Path) -> None:
    # lm-27n.2: an injected StatePort-only fake leaves the memory/pressure slots
    # unwired (CoreLoop reads empty snapshots) and the StateActor falls back to
    # the fake's own commit_tick — a full frame must still run and persist.
    # The being is BORN: an unborn one wakes to be born on its very first tick (§6.2)
    # and the frame would then carry a desire-row mutation this memory-less fake
    # (correctly) refuses — which is a genesis assertion, not a wiring one.
    fake_state = FakeStateStore(_born())
    lm = build_lifemodel(
        base_dir=tmp_path / "unused",
        state=fake_state,
        clock=FakeClock(datetime(2026, 7, 3, tzinfo=UTC)),
        delivery=FakeDelivery(),
    )
    assert lm.coreloop is not None
    lm.coreloop.tick()  # must not raise
    assert lm.state.load().tick_count == 1


def test_default_graph_wires_the_store_as_the_atomic_committer(tmp_path: Path) -> None:
    # The one SQLite store backs state + memory + pressure + the tick committer,
    # so a frame's commit spans vitals and entities in one transaction (HLA §4.1).
    lm = build_lifemodel(base_dir=tmp_path)
    assert isinstance(lm.state, SQLiteRuntimeStore)
    lm.coreloop.tick()  # ticks through commit_tick over that same store
    assert lm.state.load().tick_count == 1


def test_builds_with_concrete_sqlite_store_over_tmp_dir(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)

    assert isinstance(lm.state, SQLiteRuntimeStore)
    assert isinstance(lm.clock, SystemClock)
    assert isinstance(lm.delivery, NoopDelivery)
    _assert_no_hermes()


def test_default_graph_is_exercisable_end_to_end(tmp_path: Path) -> None:
    # The concrete graph actually works: commit state, then a heartbeat frame runs
    # the registered pipeline and checkpoints the bookkeeping bump.
    lm = build_lifemodel(base_dir=tmp_path)

    lm.state.commit(_born(u=1.5))
    assert lm.state.load().u == 1.5

    report = lm.coreloop.tick()
    assert "contact" in report.ran  # the registered pipeline ran
    assert lm.state.load().tick_count == 1
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


def test_default_registry_contains_contact_sensor(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    ids = tuple(c.id for c in lm.registry.enabled())
    assert "contact" in ids
    assert "contact-aggregation" in ids


def test_coreloop_tick_bookkeeps_and_runs_contact(tmp_path: Path) -> None:
    # ContactSensor + SolitudeDrive + aggregation run (last_tick_at=None → dt=0,
    # no signals → no satiate). Frame still checkpoints the bookkeeping bump.
    lm = build_lifemodel(base_dir=tmp_path)
    report = lm.coreloop.tick()
    assert "contact" in report.ran
    assert "solitude-drive" in report.ran
    assert "contact-aggregation" in report.ran
    assert lm.state.load().tick_count == 1


# --- Phase B1: contact sensor/drive wiring (T2 split) ---


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._m = moment

    def now(self) -> datetime:
        return self._m


def test_presence_and_solitude_drive_are_registered_enabled(tmp_path: Path) -> None:
    # T2 split: the instantaneous sensor (ContactSensor, "contact" slot) + the
    # AUTONOMIC u-integrator (SolitudeDrive) replace the old monolithic ContactNeuron.
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert "contact" in ids  # ContactSensor keeps the historical sensor slot id
    assert "solitude-drive" in ids
    assert any(isinstance(c, ContactSensor) for c in lm.registry.enabled())
    assert any(isinstance(c, SolitudeDrive) for c in lm.registry.enabled())


def test_pipeline_tick_rises_u_and_persists(tmp_path: Path) -> None:
    # Seed last_tick_at 240 min before the clock; one frame should rise u to ~1.0.
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(_born(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    assert abs(store.load().u - 1.0) < 1e-9


def test_pipeline_tick_satiates_on_inbound_exchange(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(_born(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    # A contact_observed reading seeded into the frame (spec §3) — no durable bus.
    lm.coreloop.tick(
        [contact_observed_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)]
    )
    assert store.load().u == 0.0  # satiated by the two_way contact


# --- Phase B2: ContactAggregation wiring ---


def test_aggregation_registered_after_neuron(tmp_path: Path) -> None:
    from lifemodel.core.aggregation import ContactAggregation

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    # The T2 spine prefix: sensor (ContactSensor) -> integrator (SolitudeDrive) ->
    # aggregation, so aggregation reads the drive's fresh-u contact signal same-frame.
    assert ids.index("contact") < ids.index("solitude-drive")
    assert ids.index("solitude-drive") < ids.index("contact-aggregation")
    assert any(isinstance(c, ContactAggregation) for c in lm.registry.enabled())


def _seed_active_desire(store: SQLiteRuntimeStore) -> None:
    """Persist a live active contact-desire row through the real store."""
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=3.0)))


def test_pipeline_rises_then_wakes_desire(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # u already high; 1 min elapsed → neuron keeps it high, aggregation wakes a desire
    store.commit(_born(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    desire = read_live_contact_desire(store)  # the desire is now a typed row, not a flag
    assert desire is not None and desire.state == "active"


def test_pipeline_dedups_desire_across_ticks(tmp_path: Path) -> None:
    # frame 1 births the desire; frame 2 sees it live in the snapshot and dedups —
    # the singleton row is not re-written (revision stays put), no double-wake.
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(_born(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    born = store.get("desire", "contact:owner")
    assert born is not None and born.state == "active"
    lm.coreloop.tick()  # a second high-u frame must not re-birth the desire
    again = store.get("desire", "contact:owner")
    assert again is not None and again.revision == born.revision  # untouched -> deduped


def test_pipeline_exchange_satiates_and_clears(tmp_path: Path) -> None:
    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(_born(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    _seed_active_desire(store)  # a live desire the contact must terminalize
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick(
        [contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)]
    )
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
    # high latent, a send 10 min ago -> within grace -> no wake this frame
    store.commit(
        _born(
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
    store.commit(_born(energy=0.5, fatigue=0.5, last_tick_at="2026-07-06T12:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    final = store.load()
    assert final.energy > 0.5  # recovered during the idle frame
    assert final.fatigue < 0.5  # decayed


# --- Phase D1: Cognition component wiring ---


def test_cognition_registered_after_aggregation(tmp_path: Path) -> None:
    from lifemodel.core.cognition import CognitionLauncher

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert ids.index("contact-aggregation") < ids.index("cognition-launcher")
    assert any(isinstance(c, CognitionLauncher) for c in lm.registry.enabled())


# --- T7: the single-spine tick order (thought machinery cut; lm-705.1 re-seeds
# --- ONE capture-only component, see the assertion below) -------------------


def test_full_pipeline_order_is_asserted(tmp_path: Path) -> None:
    # The spine is exactly the five links: personality -> contact (sensor) ->
    # solitude-drive -> contact-aggregation -> cognition-launcher. ContactAggregation
    # is the SOLE birth point of a durable desire in the frame.
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    spine = [
        "personality",
        "contact",
        "solitude-drive",
        "contact-aggregation",
        "cognition-launcher",
    ]
    positions = [ids.index(cid) for cid in spine]
    assert positions == sorted(positions), ids
    # T7 cut every thought component; lm-705.1 (waking mind slice 1) re-seeds
    # exactly ONE — ThoughtCapture, capture-only (no processing/rumination/desire/
    # arbiter, spec §4.1's boundary) — so the pin is no longer "never", it is
    # "no thought component OTHER than the capture-only one".
    from lifemodel.core.thought_capture import THOUGHT_CAPTURE_ID

    assert [cid for cid in ids if "thought" in cid] == [THOUGHT_CAPTURE_ID], ids


# --- lm-27n.4: the Intention decision record, end-to-end through the store ---


def test_pipeline_crystallizes_intention_on_launch(tmp_path: Path) -> None:
    # A high-u frame that launches also births the singleton intention active, in
    # the SAME commit as pending being set (behavior-neutral on send timing).
    from lifemodel.core.intention_view import read_live_contact_intention

    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    store.commit(_born(u=3.0, energy=1.0, last_tick_at="2026-07-06T03:59:00+00:00"))
    _seed_active_desire(store)  # a live active desire cognition can launch on
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.coreloop.tick()
    final = store.load()
    intention = read_live_contact_intention(store)
    assert intention is not None and intention.state == "active"  # decision recorded
    assert final.pending_proactive_id is not None  # ...in the same commit as pending


def test_pipeline_exchange_resolves_desire_and_intention_atomically(tmp_path: Path) -> None:
    # A real contact terminalizes BOTH the desire and the intention in one frame,
    # through the one atomic store committer — never a split-brain (HLA §4.1).
    from lifemodel.core.intention_view import read_live_contact_intention

    clock = _FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    # a turn already in flight (pending set) gates cognition off, so this frame only
    # RESOLVES the launched pair — it does not re-launch/re-crystallize.
    store.commit(
        _born(
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
    lm.coreloop.tick(
        [contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)]
    )
    assert read_live_contact_desire(store) is None  # desire terminalized
    assert read_live_contact_intention(store) is None  # intention terminalized in lockstep
    assert store.get("desire", "contact:owner").state == "satisfied"
    assert store.get("intention", "contact:owner").state == "completed"


# --- lm-705.1: ThoughtCapture wiring (waking mind slice 1) -------------------


def test_build_lifemodel_registers_thought_capture(tmp_path: Path) -> None:
    from lifemodel.core.thought_capture import THOUGHT_CAPTURE_ID

    lm = build_lifemodel(base_dir=tmp_path)
    ids = {m.id for m in lm.registry.manifests()}
    assert THOUGHT_CAPTURE_ID in ids

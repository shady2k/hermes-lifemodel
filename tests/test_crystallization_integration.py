"""lm-27n.9 cross-cutting integration: the two-springs Desire, end-to-end.

Proves the three properties the coordination rests on:
* ONE writer per object — in a single tick crystallization writes NOTHING,
  aggregation writes ONLY the desire, attention writes ONLY the thought, and no
  two mutations target the same ``(kind, id)``;
* the top-down spring flows through the real store — a crystallized thought becomes
  a ``spring=thought`` contact desire carrying ``source_thought_ids``, and its
  source thought resolves;
* anti-frivolity end-to-end — over many idle ticks the being never crystallizes
  contact from idle wandering alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.component import TickContext
from lifemodel.core.desire_view import read_live_contact_desire
from lifemodel.core.intents import EmitSignal, Intent, PutRecord, TransitionRecord
from lifemodel.core.thought_attention import ThoughtAttention
from lifemodel.core.thought_crystallization import ThoughtCrystallization
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, thought_record

_NOW = datetime(2026, 7, 6, 0, 1, tzinfo=UTC)
PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)


def _mutation_targets(intents: list[Intent]) -> list[tuple[str, str]]:
    """The ``(kind, id)`` each record mutation writes to (PutRecord/TransitionRecord)."""
    targets: list[tuple[str, str]] = []
    for i in intents:
        if isinstance(i, PutRecord):
            targets.append((i.op.draft.kind, i.op.draft.id))
        elif isinstance(i, TransitionRecord):
            targets.append((i.op.kind, i.op.id))
    return targets


def test_one_writer_per_object_no_same_tick_conflict(tmp_path) -> None:
    # A crystallizable thought, run through the pipeline order against the threaded
    # proposal signal. Assert the one-writer split holds with no (kind,id) collision.
    bus = FileSignalBus(tmp_path)
    thought = thought_record(
        "check in on the owner after a hard week",
        "active",
        id="t-serve",
        salience=0.8,
        other_regarding_value=0.6,
        sustained_attention_count=2,
    )
    objects = (thought,)
    state = State(last_tick_at="2026-07-06T00:00:00+00:00")

    # 1) crystallization — a PURE PROPOSER: emits a proposal, writes NO record.
    cryst = ThoughtCrystallization()
    cctx = TickContext(state=state, now=_NOW, bus=bus, objects=objects, signals=())
    c_intents = list(cryst.step(cctx))
    assert not _mutation_targets(c_intents)  # writes nothing
    proposal_signals = tuple(i.signal for i in c_intents if isinstance(i, EmitSignal))
    assert proposal_signals  # it did propose

    # 2) aggregation + attention see the SAME threaded proposal (as in-tick).
    downstream_ctx = TickContext(
        state=state, now=_NOW, bus=bus, objects=objects, signals=proposal_signals
    )
    agg = ContactAggregation(params=PARAMS, theta=1.0, beta=1.0, u_max=100.0)
    a_intents = list(agg.step(downstream_ctx))
    att = ThoughtAttention()
    t_intents = list(att.step(downstream_ctx))

    # aggregation writes ONLY the desire; attention writes ONLY the thought.
    assert _mutation_targets(a_intents) == [("desire", "contact:owner")]
    assert _mutation_targets(t_intents) == [("thought", "t-serve")]

    # the crux: across ALL three components, no two mutations hit the same object.
    all_targets = _mutation_targets(c_intents + a_intents + t_intents)
    assert len(all_targets) == len(set(all_targets)), all_targets
    assert set(all_targets) == {("desire", "contact:owner"), ("thought", "t-serve")}


def test_top_down_spring_flows_through_the_store(tmp_path) -> None:
    # End-to-end through the real graph: a crystallizable thought becomes a
    # spring=thought contact desire carrying source_thought_ids, and the source
    # thought resolves — all in ONE tick, committed atomically (no rollback).
    clock = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    store.commit(State(last_tick_at=clock.now().isoformat()))
    store.put(
        encode_thought(
            build_thought(
                id="t-serve",
                content="check in on the owner after a hard week",
                trigger="idle",
                salience=0.8,
                other_regarding_value=0.6,
                sustained_attention_count=2,
            )
        )
    )

    clock.advance(timedelta(minutes=1))
    lm.coreloop.tick()

    desire = read_live_contact_desire(store)
    assert desire is not None
    assert str(desire.spring) == "thought"  # top-down spring
    assert desire.source_thought_ids == ("t-serve",)  # the concrete reason carried
    # the source thought's job is done — resolved, not decayed/parked.
    assert store.get("thought", "t-serve").state == "resolved"


def test_suppressed_top_down_proposal_leaves_the_thought_live(tmp_path) -> None:
    # codex P1 fix: a crystallizable thought whose desire aggregation SUPPRESSES via
    # an appropriateness gate (here the silence window — the being just talked) must
    # NOT resolve the source thought. The genuine reason stays LIVE (it re-competes,
    # bounded by normal decay/parking) — it is not silently spent by timing.
    clock = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    # last exchange 4 min before the tick's start -> inside the silence window (w=15).
    store.commit(
        State(last_exchange_at="2026-07-05T23:56:00+00:00", last_tick_at=clock.now().isoformat())
    )
    store.put(
        encode_thought(
            build_thought(
                id="t-serve",
                content="check in on the owner after a hard week",
                trigger="idle",
                salience=0.8,
                other_regarding_value=0.6,
                sustained_attention_count=2,
            )
        )
    )

    clock.advance(timedelta(minutes=1))  # now 00:01 — 5 min since the last exchange
    lm.coreloop.tick()

    assert read_live_contact_desire(store) is None  # suppressed by the silence window
    # the reason is NOT spent — the thought stays live (active), not resolved.
    assert store.get("thought", "t-serve").state == "active"


def test_anti_frivolity_idle_wandering_never_crystallizes_contact(tmp_path) -> None:
    # Over many idle ticks — only idle mind-wandering generates thoughts — the being
    # NEVER crystallizes a contact desire. Idle thoughts are low-salience and
    # non-viable, so they can never accrue the persistence the Rubicon gate needs.
    clock = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    # last exchange 40 min ago -> quiet enough for idle wandering to fire; energy up.
    store.commit(
        State(
            energy=1.0,
            last_exchange_at="2026-07-05T23:20:00+00:00",
            last_tick_at=clock.now().isoformat(),
        )
    )

    for _ in range(15):  # drive stays sub-threshold across this window
        clock.advance(timedelta(minutes=1))
        lm.coreloop.tick()
        assert read_live_contact_desire(store) is None  # never sprung from idle wandering

    # some idle wandering DID happen (the mind was not merely inert)...
    thoughts = store.find(kind="thought")
    assert thoughts, "expected idle mind-wandering to have generated thoughts"
    # ...yet none accrued the persistence a contact reason needs (all idle-non-viable).
    assert all(r.payload["sustained_attention_count"] == 0 for r in thoughts)

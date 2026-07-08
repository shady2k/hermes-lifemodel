"""ThoughtGeneration — the 0-LLM generative stream (lm-27n.8).

The behaviour contract, proven load-bearing-test-first:

* **snapshot-per-tick / no same-tick recursion** — a generated thought's
  ``parent_id`` is always a thought that existed in ``ctx.objects`` at tick start;
  at most ONE thought is emitted per tick, so the tree grows ≤1 layer/tick and a
  thought's own child is never minted the same tick it is;
* **≤1/tick, priority event > chaining > idle** — the first warranted + affordable
  trigger wins;
* **deterministic idempotent ids** — a retried event / same idle window / same
  parent upserts one row (skipped when already live);
* **the anti-runaway caps** — at ``MAX_LIVE_THOUGHTS`` nothing generates;
  unaffordable energy ⇒ nothing + no debit; every chain gate blocks a child;
* **idle fires only quiet + sparse and is LOW-salience** — a fresh idle thought
  cannot itself be a strong contact reason (it never mints a desire — that is .9).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.composition import build_lifemodel
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent, PutRecord, UpdateState
from lifemodel.core.taxonomy import contact_signal, exchange_signal, verdict_signal
from lifemodel.core.thought_generation import (
    IDLE_QUIET_MIN,
    IDLE_SOFT_CAP,
    MAX_LIVE_THOUGHTS,
    THOUGHT_GEN_COST,
    ThoughtGeneration,
)
from lifemodel.core.thought_templates import (
    chain_content,
    event_exchange_content,
    event_verdict_content,
    idle_about_desire_content,
)
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import derive_id
from lifemodel.sim.aggregation import Verdict
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeSignalBus, contact_desire_record, thought_record

_NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
#: A last-exchange stamp old enough (> IDLE_QUIET_MIN) to count as "quiet".
_OLD_EXCHANGE = (_NOW - timedelta(minutes=IDLE_QUIET_MIN + 10)).isoformat()
_ALPHA = 2.0  # COST_ALPHA


def _component(**kw: object) -> ThoughtGeneration:
    return ThoughtGeneration(alpha=_ALPHA, **kw)  # type: ignore[arg-type]


def _quiet_state(**kw: object) -> State:
    base: dict[str, object] = {"energy": 1.0, "last_exchange_at": _OLD_EXCHANGE}
    base.update(kw)
    return State(**base)  # type: ignore[arg-type]


def _busy_state(**kw: object) -> State:
    # last_exchange_at=None ⇒ never "quiet" ⇒ idle is suppressed (isolates
    # event/chaining gates from the idle fallback).
    base: dict[str, object] = {"energy": 1.0}
    base.update(kw)
    return State(**base)  # type: ignore[arg-type]


def _ctx(
    objects: tuple[MemoryRecord, ...] = (),
    signals: tuple[object, ...] = (),
    state: State | None = None,
    now: datetime = _NOW,
) -> TickContext:
    return TickContext(
        state=state if state is not None else _busy_state(),
        now=now,
        bus=FakeSignalBus(),
        signals=tuple(signals),  # type: ignore[arg-type]
        objects=objects,
    )


def _puts(intents: list[Intent]) -> list[PutRecord]:
    return [i for i in intents if isinstance(i, PutRecord)]


def _only_put(intents: list[Intent]) -> PutRecord:
    puts = _puts(intents)
    assert len(puts) == 1, f"expected exactly one PutRecord, got {len(puts)}"
    return puts[0]


def _commit(objects: tuple[MemoryRecord, ...], put: PutRecord) -> tuple[MemoryRecord, ...]:
    """Turn a minted draft into a committed record (as the store would next tick)."""
    d = put.op.draft
    record = MemoryRecord(
        kind=d.kind,
        id=d.id,
        state=d.state,
        payload=d.payload,
        source=d.source,
        recipient_id=d.recipient_id,
        salience=d.salience,
        confidence=d.confidence,
        expires_at=d.expires_at,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
        revision=0,
        schema_version=d.schema_version,
    )
    return (*objects, record)


# --- snapshot-per-tick / no same-tick recursion (the load-bearing invariant) --


def test_chain_child_parent_id_is_in_the_snapshot() -> None:
    root = thought_record("a lone worry", "active", id="t-root", salience=0.9, trigger="idle")
    intents = list(_component().step(_ctx((root,), state=_busy_state())))
    put = _only_put(intents)  # ≤1 emit/tick
    draft = put.op.draft
    assert draft.payload["parent_id"] == "t-root"
    # the invariant: the child references a parent that existed at tick start.
    assert draft.payload["parent_id"] in {r.id for r in (root,)}


def test_no_same_tick_recursion_the_childs_own_child_is_not_minted() -> None:
    root = thought_record("worry", "active", id="t-root", salience=0.9, trigger="idle")
    intents = list(_component().step(_ctx((root,), state=_busy_state())))
    minted_ids = {p.op.draft.id for p in _puts(intents)}
    child_id = derive_id("thought", "chain", "t-root")
    grandchild_id = derive_id("thought", "chain", child_id)
    assert minted_ids == {child_id}  # exactly the child — never its child too
    assert grandchild_id not in minted_ids


def test_tree_grows_at_most_one_layer_per_tick() -> None:
    # Step 1: root → child (depth 1). Step 2 (child now committed + selected) →
    # grandchild (depth 2). Each step extends the lineage by exactly one.
    comp = _component()
    root = thought_record("root", "active", id="t-root", salience=0.9, trigger="idle")
    step1 = list(comp.step(_ctx((root,), state=_busy_state())))
    child_draft = _only_put(step1).op.draft
    assert child_draft.payload["parent_id"] == "t-root"

    # Commit the child; park the root so the child is the single selected thought.
    root_parked = thought_record(
        "root", "parked", id="t-root", salience=0.9, trigger="idle", park_count=1
    )
    child = thought_record(
        "child",
        "active",
        id=child_draft.id,
        salience=0.9,
        trigger=child_draft.payload["trigger"],  # type: ignore[arg-type]
        parent_id="t-root",
    )
    step2 = list(comp.step(_ctx((root_parked, child), state=_busy_state())))
    grand_draft = _only_put(step2).op.draft
    assert grand_draft.payload["parent_id"] == child.id  # depth grew by one
    assert grand_draft.id == derive_id("thought", "chain", child.id)


# --- ≤1/tick + priority order event > chaining > idle -------------------------


def test_event_beats_chaining_and_idle() -> None:
    chainable = thought_record("develop me", "active", id="t1", salience=0.9, trigger="idle")
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = list(_component().step(_ctx((chainable,), (sig,), state=_quiet_state())))
    draft = _only_put(intents).op.draft
    assert draft.payload["trigger"] == "event"  # event wins the single slot
    assert draft.payload["parent_id"] is None


def test_chaining_beats_idle() -> None:
    chainable = thought_record("develop me", "active", id="t1", salience=0.9, trigger="idle")
    intents = list(_component().step(_ctx((chainable,), state=_quiet_state())))
    draft = _only_put(intents).op.draft
    assert str(draft.payload["trigger"]).startswith("thought:")  # chain, not idle
    assert draft.payload["parent_id"] == "t1"


def test_idle_is_the_lowest_priority_fallback() -> None:
    intents = list(_component().step(_ctx((), state=_quiet_state())))
    draft = _only_put(intents).op.draft
    assert draft.payload["trigger"] == "idle"


# --- deterministic idempotent ids (upsert, no flood) --------------------------


def test_event_id_is_idempotent_no_second_row() -> None:
    origin = derive_id("thought", "event", "exchange", "e1")
    # the event thought is already live (np at the brake so it cannot be chained);
    # the same exchange re-delivered must NOT mint a second event row.
    existing = thought_record("prior", "active", id=origin, salience=0.4, no_progress_count=2)
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = list(_component().step(_ctx((existing,), (sig,), state=_busy_state())))
    assert _puts(intents) == []  # idempotent: no duplicate event


def test_idle_bucket_id_is_idempotent() -> None:
    bucket = int(_NOW.timestamp() // (60.0 * 60.0))
    idle_id = derive_id("thought", "idle", str(bucket))
    existing = thought_record("wandered", "active", id=idle_id, salience=0.15, no_progress_count=2)
    intents = list(_component().step(_ctx((existing,), state=_quiet_state())))
    assert _puts(intents) == []  # one idle thought per cooldown window


def test_chain_id_is_idempotent_one_child_per_parent() -> None:
    parent = thought_record("parent", "active", id="t-p", salience=0.9, trigger="idle")
    child = thought_record(
        "existing child",
        "active",
        id=derive_id("thought", "chain", "t-p"),
        salience=0.4,
        parent_id="t-p",
    )
    intents = list(_component().step(_ctx((parent, child), state=_busy_state())))
    assert _puts(intents) == []  # the parent already has its one child


# --- the anti-runaway caps ----------------------------------------------------


def test_at_max_live_thoughts_generates_nothing() -> None:
    objects = tuple(
        thought_record(f"t{i}", "active", id=f"t{i:03d}", salience=0.5)
        for i in range(MAX_LIVE_THOUGHTS)
    )
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    assert list(_component().step(_ctx(objects, (sig,), state=_quiet_state()))) == []


def test_unaffordable_energy_generates_nothing_and_no_debit() -> None:
    poor = _quiet_state(energy=THOUGHT_GEN_COST / 2.0)  # cannot afford the mint
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = list(_component().step(_ctx((), (sig,), state=poor)))
    assert intents == []  # no thought AND no UpdateState debit


def test_affordable_generation_debits_the_gen_cost() -> None:
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = list(_component().step(_ctx((), (sig,), state=_quiet_state(energy=1.0))))
    updates = [i for i in intents if isinstance(i, UpdateState)]
    assert len(updates) == 1
    energy_after = updates[0].changes["energy"]
    assert isinstance(energy_after, float)
    assert 1.0 - THOUGHT_GEN_COST - 1e-9 <= energy_after <= 1.0 - THOUGHT_GEN_COST + 1e-9


def test_no_warranted_trigger_means_no_debit() -> None:
    # Not quiet, no events, no chainable thought → nothing to mint → energy untouched.
    intents = list(_component().step(_ctx((), state=_busy_state(energy=1.0))))
    assert intents == []


# --- each chain gate blocks a child (with positive controls) ------------------


def test_chain_blocked_when_parent_near_the_brake() -> None:
    near = thought_record("looping", "active", id="t1", salience=0.9, no_progress_count=2)
    assert list(_component().step(_ctx((near,), state=_busy_state()))) == []
    # positive control: one below the brake → a child is minted.
    ok = thought_record("looping", "active", id="t1", salience=0.9, no_progress_count=1)
    assert len(_puts(list(_component().step(_ctx((ok,), state=_busy_state()))))) == 1


def test_chain_blocked_at_max_depth() -> None:
    # root(d0, parked) → child(d1, parked) → grandchild(d2, active + selected).
    root = thought_record("root", "parked", id="t-root", salience=0.1, trigger="idle")
    child = thought_record(
        "child", "parked", id="t-child", salience=0.1, parent_id="t-root", trigger="thought:t-root"
    )
    grand = thought_record(
        "grand",
        "active",
        id="t-grand",
        salience=0.9,
        parent_id="t-child",
        trigger="thought:t-child",
    )
    # grandchild sits at depth == MAX_DEPTH, so developing it is refused.
    assert list(_component().step(_ctx((root, child, grand), state=_busy_state()))) == []
    # positive control: a depth-1 leaf (parent has no other child) → chain fires.
    leaf = thought_record(
        "leaf", "active", id="t-leaf", salience=0.9, parent_id="t-root", trigger="thought:t-root"
    )
    root_only = thought_record("root", "parked", id="t-root", salience=0.1, trigger="idle")
    assert len(_puts(list(_component().step(_ctx((root_only, leaf), state=_busy_state()))))) == 1


def test_chain_blocked_when_parent_already_has_a_live_child() -> None:
    parent = thought_record("parent", "active", id="t-p", salience=0.9, trigger="idle")
    other_child = thought_record("kid", "active", id="t-otherkid", salience=0.3, parent_id="t-p")
    assert list(_component().step(_ctx((parent, other_child), state=_busy_state()))) == []
    # positive control: remove the child → the parent is developed.
    assert len(_puts(list(_component().step(_ctx((parent,), state=_busy_state()))))) == 1


def test_no_chaining_from_a_parked_thought() -> None:
    parked = thought_record("resting", "parked", id="t1", salience=0.9, trigger="idle")
    assert list(_component().step(_ctx((parked,), state=_busy_state()))) == []


# --- idle fires only when quiet + sparse, and is LOW salience ------------------


def test_idle_fires_when_quiet_and_is_low_salience() -> None:
    draft = _only_put(list(_component().step(_ctx((), state=_quiet_state())))).op.draft
    assert draft.payload["trigger"] == "idle"
    assert 0.10 <= draft.salience <= 0.20  # LOW salience band (weak anti-frivolity)
    assert float(draft.payload["actionability"]) <= 0.10  # low actionability


def test_idle_suppressed_without_exchange_history() -> None:
    # last_exchange_at=None ⇒ conservatively not "quiet" ⇒ no idle wandering.
    assert list(_component().step(_ctx((), state=_busy_state()))) == []


def test_idle_suppressed_too_soon_after_an_exchange() -> None:
    recent = (_NOW - timedelta(minutes=IDLE_QUIET_MIN - 5)).isoformat()
    assert list(_component().step(_ctx((), state=_quiet_state(last_exchange_at=recent)))) == []


def test_idle_suppressed_while_a_proactive_turn_is_pending() -> None:
    pending = _quiet_state(pending_proactive_id="proactive-x")
    assert list(_component().step(_ctx((), state=pending))) == []


def test_idle_suppressed_by_a_real_exchange_this_tick() -> None:
    # The exchange event id is already live (np at the brake → not chainable), so
    # the event is idempotently skipped — yet idle must ALSO stay suppressed
    # because a real exchange happened this tick.
    origin = derive_id("thought", "event", "exchange", "e1")
    prior = thought_record("prior", "active", id=origin, salience=0.4, no_progress_count=2)
    sig = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    assert list(_component().step(_ctx((prior,), (sig,), state=_quiet_state()))) == []


def test_idle_suppressed_when_the_mind_is_busy() -> None:
    # >= IDLE_SOFT_CAP parked thoughts (no chainable active, no event) → no idle.
    future = (_NOW + timedelta(hours=2)).isoformat()
    objects = tuple(
        thought_record(f"t{i}", "parked", id=f"t{i:03d}", salience=0.5, parked_until=future)
        for i in range(IDLE_SOFT_CAP)
    )
    assert list(_component().step(_ctx(objects, state=_quiet_state()))) == []


# --- event appraisal: one thought per exchange / verdict / drive origin --------


def test_exchange_event_mints_one_appraisal() -> None:
    sig = exchange_signal(origin_id="msg-7", actor="user", label="two_way", timestamp=None)
    draft = _only_put(list(_component().step(_ctx((), (sig,), state=_quiet_state())))).op.draft
    assert draft.id == derive_id("thought", "event", "exchange", "msg-7")
    assert draft.payload["trigger"] == "event"
    assert draft.payload["parent_id"] is None
    assert 0.3 <= draft.salience <= 0.5
    assert "тепло" in str(draft.payload["content"])


def test_verdict_event_reflects_the_outcome() -> None:
    sig = verdict_signal(origin_id="v-1", verdict=Verdict.REJECT, timestamp=None)
    draft = _only_put(list(_component().step(_ctx((), (sig,), state=_quiet_state())))).op.draft
    assert draft.id == derive_id("thought", "event", "verdict", "v-1")
    assert draft.payload["trigger"] == "event"


def test_exchange_event_beats_a_verdict_same_tick() -> None:
    exch = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    verd = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    draft = _only_put(
        list(_component().step(_ctx((), (exch, verd), state=_quiet_state())))
    ).op.draft
    assert draft.id == derive_id("thought", "event", "exchange", "e1")


def test_proactive_internal_exchange_is_not_an_event() -> None:
    # An internal (proactive) exchange is the being's own turn — not an external
    # event — so it mints no event thought.
    sig = exchange_signal(
        origin_id="p1", actor="proactive_internal", label="monologue", timestamp=None
    )
    intents = list(_component().step(_ctx((), (sig,), state=_busy_state())))
    assert _puts(intents) == []


def test_drive_crossing_mints_an_event_but_only_on_a_genuine_crossing() -> None:
    crossed = contact_signal(origin_id="c1", value=1.2, delta=0.5, timestamp=None)  # 0.7→1.2 up
    draft = _only_put(list(_component().step(_ctx((), (crossed,), state=_busy_state())))).op.draft
    assert draft.id == derive_id("thought", "event", "drive", "contact")
    # already above θ but no crossing this tick (1.15 → 1.2) → no event.
    steady = contact_signal(origin_id="c2", value=1.2, delta=0.05, timestamp=None)
    assert list(_component().step(_ctx((), (steady,), state=_busy_state()))) == []


def test_idle_wanders_about_a_live_desire_when_no_active_thought() -> None:
    # No active thought to circle back to, but a live desire → idle wonders about
    # the pull to reach out (still LOW salience — it does not mint the desire).
    objects = (contact_desire_record("active", salience=3.0),)
    draft = _only_put(list(_component().step(_ctx(objects, state=_quiet_state())))).op.draft
    assert draft.payload["trigger"] == "idle"
    assert draft.payload["content"] == idle_about_desire_content()
    assert 0.10 <= draft.salience <= 0.20


# --- templated content is deterministic + total (pure fns) --------------------


def test_templates_are_deterministic_and_branch_on_input() -> None:
    assert event_exchange_content("user", "rejection") != event_exchange_content("user", "two_way")
    # the "generic" exchange branch (neither warm nor a rejection).
    assert "обмен" in event_exchange_content("assistant", "monologue")
    # every verdict has its own reflection, DEFER included.
    reflections = {
        event_verdict_content(Verdict.FULFILL),
        event_verdict_content(Verdict.REJECT),
        event_verdict_content(Verdict.DEFER),
    }
    assert len(reflections) == 3
    # a long parent is quoted back as a bounded snippet (ellipsis), deterministically.
    long_parent = "почему меня так держит эта мысль " * 10
    snippet = chain_content(long_parent)
    assert snippet == chain_content(long_parent)  # pure
    assert "…" in snippet and len(snippet) < len(long_parent)


# --- boundedness: the stream cannot flood ------------------------------------


def test_component_never_exceeds_the_hard_cap_over_many_ticks() -> None:
    # Drive the component for many idle-eligible windows WITHOUT any draining
    # (.7 is not in this loop): it self-limits and never exceeds MAX_LIVE_THOUGHTS.
    comp = _component()
    objects: tuple[MemoryRecord, ...] = ()
    now = _NOW
    for _ in range(400):
        now += timedelta(minutes=61)  # advance past an idle cooldown bucket each step
        state = _quiet_state(last_exchange_at=_OLD_EXCHANGE)
        intents = list(comp.step(_ctx(objects, state=state, now=now)))
        for put in _puts(intents):
            objects = _commit(objects, put)
        assert len(objects) <= MAX_LIVE_THOUGHTS  # the cap holds every step
    assert len(objects) > 0  # ...and the stream did come alive


# --- end-to-end through the real store ---------------------------------------


def test_e2e_no_same_tick_recursion(tmp_path) -> None:
    clock = FakeClock(_NOW)
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    store.commit(State(u=0.0, last_tick_at=clock.now().isoformat()))
    store.put(encode_thought(build_thought(id="t-root", content="did I upset them?", salience=0.9)))

    clock.advance(timedelta(minutes=1))
    lm.coreloop.tick()

    ids = {r.id for r in store.find(kind="thought")}
    child_id = derive_id("thought", "chain", "t-root")
    grandchild_id = derive_id("thought", "chain", child_id)
    assert ids == {"t-root", child_id}  # exactly one new thought — no t2
    assert grandchild_id not in ids  # the child's own child is NOT minted this tick


def test_e2e_stream_lives_and_stays_bounded(tmp_path) -> None:
    clock = FakeClock(_NOW)
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    store.commit(State(last_exchange_at=_OLD_EXCHANGE, last_tick_at=clock.now().isoformat()))
    store.put(
        encode_thought(build_thought(id="t-seed", content="a warm exchange earlier", salience=0.8))
    )

    for _ in range(200):
        clock.advance(timedelta(minutes=5))
        lm.coreloop.tick()

    live = [r for r in store.find(kind="thought") if r.state in {"active", "parked"}]
    assert len(live) <= MAX_LIVE_THOUGHTS  # the .7 brake + caps keep it bounded
    assert len(store.find(kind="thought")) >= 1  # the generative stream is alive

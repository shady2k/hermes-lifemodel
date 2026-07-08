from __future__ import annotations

import re
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.cognition import Cognition
from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchProactive, PutRecord, UpdateState
from lifemodel.core.relationship_view import EXPLICIT_CONFIDENCE
from lifemodel.core.wake_packet import RECENT_THOUGHTS_HEADER, build_wake_packet
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import default_registry
from lifemodel.state.model import State
from lifemodel.testing import (
    FakeTracer,
    contact_desire_objects,
    contact_desire_record,
    owner_relationship_record,
    thought_record,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)

# a live active-desire snapshot (what the old ``desire_status="active"`` meant)
ACTIVE = contact_desire_objects("active")


def _intention_put(intents):
    """The PutRecord birthing the contact intention this tick, if any."""
    return next(
        (i for i in intents if isinstance(i, PutRecord) and i.op.draft.kind == "intention"),
        None,
    )


def _cog() -> Cognition:
    return Cognition(fast_cost=0.02, send_cost=0.03, alpha=2.0)


def _ctx(state: State, *, objects=(), tmp_path, now: datetime = NOW) -> TickContext:
    return TickContext(
        state=state, now=now, bus=FileSignalBus(tmp_path), signals=(), objects=tuple(objects)
    )


def _launch(intents):
    return next((i for i in intents if isinstance(i, LaunchProactive)), None)


def _update(intents):
    return next((i for i in intents if isinstance(i, UpdateState)), None)


def test_no_active_desire_does_nothing(tmp_path) -> None:
    # no live desire in the snapshot -> the old desire_status="none"
    intents = _cog().step(_ctx(State(u=2.0), tmp_path=tmp_path))
    assert list(intents) == []


def test_active_desire_launches_proactive_turn(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    launch = _launch(intents)
    assert launch is not None
    assert launch.correlation_id == f"proactive-{NOW.isoformat()}"
    assert launch.prompt  # carries the wake-packet prompt
    upd = _update(intents)
    assert upd.changes["pending_proactive_id"] == launch.correlation_id
    assert upd.changes["pending_proactive_since"] == NOW.isoformat()
    assert upd.changes["energy"] < 1.0  # reserved


def test_pending_turn_is_not_relaunched(tmp_path) -> None:
    state = State(u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is None  # idempotent — a turn is already in flight


def test_insufficient_energy_holds_no_launch(tmp_path) -> None:
    # estimate = (0.02+0.03)*(1+2*1.0)=0.15 at max fatigue; energy 0.05 can't afford
    state = State(u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is None  # emergent shutoff — hold
    assert _update(intents) is None  # energy untouched, desire stays active


def test_deferred_desire_does_not_launch(tmp_path) -> None:
    # only an ACTIVE desire launches; a deferred one is held (cognition never re-wakes it)
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(
        _ctx(state, objects=contact_desire_objects("deferred"), tmp_path=tmp_path)
    )
    assert _launch(intents) is None


def test_launch_carries_the_reserved_energy(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    # estimate = (0.02+0.03)*(1+2*0) = 0.05
    assert abs(launch.reserved_energy - 0.05) < 1e-9


def test_prompt_has_no_raw_numbers(tmp_path) -> None:
    import re

    state = State(u=3.2, energy=1.0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert not re.search(r"\d", launch.prompt)


# --- lm-27n.4: 0-LLM crystallization of the Bratman decision record ---


def test_launch_crystallizes_an_active_intention(tmp_path) -> None:
    # A launch now ALSO births the singleton intention, directly ``active`` so it
    # is visible in the next tick's snapshot and owns the gate.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5, source_drive=2.0)
    intents = _cog().step(_ctx(state, objects=desire, tmp_path=tmp_path))
    put = _intention_put(intents)
    assert put is not None
    draft = put.op.draft
    assert draft.kind == "intention"
    assert draft.id == "contact:owner"
    assert draft.state == "active"  # born committed, not pending
    # Rubicon fields recorded for auditability (0-LLM, deterministic).
    payload = draft.payload
    assert payload["commitment_strength"] == 2.5  # = the desire's effective pressure
    assert payload["goal"]
    assert payload["plan"]
    assert payload["implementation_trigger"]
    assert payload["reconsideration_triggers"]  # recorded, not yet acted on
    assert payload["rationale"]


def test_crystallize_and_launch_fire_together(tmp_path) -> None:
    # Behavior-neutral parity: the intention is created EXACTLY when a launch
    # happens — same tick, same gate. Never one without the other.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is not None
    assert _intention_put(intents) is not None


def test_no_active_desire_crystallizes_nothing(tmp_path) -> None:
    # Parity: no live desire -> no launch AND no intention (old none gate).
    intents = _cog().step(_ctx(State(u=2.0), tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_pending_turn_crystallizes_nothing(tmp_path) -> None:
    # Parity: a turn in flight -> no launch AND no intention (no double-crystallize).
    state = State(u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_insufficient_energy_crystallizes_nothing(tmp_path) -> None:
    # Parity: unaffordable -> hold; no launch AND no intention (emergent shutoff).
    state = State(u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_deferred_desire_crystallizes_nothing(tmp_path) -> None:
    # Parity: only an ACTIVE desire launches; a deferred one crystallizes nothing.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(
        _ctx(state, objects=contact_desire_objects("deferred"), tmp_path=tmp_path)
    )
    assert _intention_put(intents) is None


# --- lm-27n.5: receptivity re-check before launch ---


def test_default_relationship_launches_identically(tmp_path) -> None:
    # Parity: a permissive relationship in the snapshot launches EXACTLY as the
    # no-relationship path.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    objects = (contact_desire_record("active"), owner_relationship_record())
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is not None
    assert _intention_put(intents) is not None


def test_explicit_quiet_hours_holds_the_launch(tmp_path) -> None:
    # NOW is hour 12 UTC; an explicit bad-hours=(12,) that started AFTER the desire
    # was born -> cognition re-checks and HOLDS (no launch, no intention). The
    # live desire persists for a later admissible tick.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    rel = owner_relationship_record(bad_hours=(12,), confidence=EXPLICIT_CONFIDENCE)
    objects = (contact_desire_record("active"), rel)
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is None  # held
    assert _intention_put(intents) is None
    assert list(intents) == []  # nothing committed -> the desire survives


# --- lm-27n.6: live thoughts render into the wake packet ---


def test_launch_prompt_carries_situational_brief(tmp_path) -> None:
    state = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        last_exchange_at="2026-07-06T09:00:00+00:00",
        decline_count=0,
    )
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert launch is not None
    assert "несколько часов назад" in launch.prompt  # NOW is 2026-07-06 12:00, 180 min
    assert "вспомни, на чём вы остановились" in launch.prompt
    assert re.search(r"\d", launch.prompt) is None


def test_launch_prompt_has_no_thoughts_block_without_thoughts(tmp_path) -> None:
    # Behavior-neutral: a desire but no live thought -> the launch prompt is
    # byte-identical to the no-thoughts wake packet (no Recent Thoughts block),
    # now built WITH the same situational context the cognition path passes.
    state = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        last_exchange_at="2026-07-06T09:00:00+00:00",
        decline_count=0,
    )
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert RECENT_THOUGHTS_HEADER not in launch.prompt
    expected = build_wake_packet(
        value=2.0,
        theta=1.0,
        correlation_id=launch.correlation_id,
        last_exchange_at="2026-07-06T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
    ).prompt
    assert launch.prompt == expected


# --- lm-8o3.1 Task 9: unanswered-bid line threaded through cognition -------


def test_launch_prompt_carries_unanswered_bid_line_when_pending(tmp_path) -> None:
    state = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        last_exchange_at="2026-07-06T09:00:00+00:00",
        decline_count=0,
        unanswered_outbound_count=1,
    )
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert launch is not None
    assert "пока без ответа" in launch.prompt
    assert re.search(r"\d", launch.prompt) is None


def test_launch_prompt_omits_unanswered_bid_line_when_zero(tmp_path) -> None:
    state = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        last_exchange_at="2026-07-06T09:00:00+00:00",
        decline_count=0,
        unanswered_outbound_count=0,
    )
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert launch is not None
    assert "пока без ответа" not in launch.prompt


def test_launch_prompt_renders_live_thoughts_from_the_snapshot(tmp_path) -> None:
    # A live thought in the start-of-tick snapshot surfaces (CONTENT only, no id)
    # in the launch prompt, most-salient first.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    objects = (
        contact_desire_record("active"),
        thought_record("the owner sounded tired last week", "active", id="t-hi", salience=0.9),
        thought_record("also that book they mentioned", "parked", id="t-lo", salience=0.2),
    )
    launch = _launch(_cog().step(_ctx(state, objects=objects, tmp_path=tmp_path)))
    assert RECENT_THOUGHTS_HEADER in launch.prompt
    assert "the owner sounded tired last week" in launch.prompt
    assert "also that book they mentioned" in launch.prompt  # parked is live too
    # salience order (most-salient first); ids are never rendered to the model
    hi = launch.prompt.index("the owner sounded tired last week")
    lo = launch.prompt.index("also that book they mentioned")
    assert hi < lo
    assert "t-hi" not in launch.prompt and "t-lo" not in launch.prompt


def test_terminal_thought_does_not_render(tmp_path) -> None:
    # A resolved/dropped thought is gone — never surfaces in the prompt.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    objects = (
        contact_desire_record("active"),
        thought_record("already dealt with this", "resolved", id="t-dead", salience=0.9),
    )
    launch = _launch(_cog().step(_ctx(state, objects=objects, tmp_path=tmp_path)))
    assert RECENT_THOUGHTS_HEADER not in launch.prompt


def test_launch_records_appraisal_constraints_on_the_intention(tmp_path) -> None:
    # An allowed launch with style/topic norms records them on the intention (audit).
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    rel = owner_relationship_record(
        acceptable_styles=("playful", "concise"),
        topic_sensitivity=("work",),
        confidence=EXPLICIT_CONFIDENCE,
    )
    objects = (contact_desire_record("active"), rel)
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is not None  # styles/topics are constraints, not vetoes
    put = _intention_put(intents)
    assert put is not None
    constraints = put.op.draft.payload["constraints"]
    assert any(c == "style: playful|concise" for c in constraints)
    assert any(c == "avoid topic: work" for c in constraints)


# --- lm-27n.11: creation-provenance is IMMUTABLE per episode (preserve-on-retry) ---

_STAMP = "2026-07-06T00:00:00+00:00"


def _record_from_draft(draft, *, state: str | None = None) -> MemoryRecord:
    """A persisted MemoryRecord from a just-emitted draft (so tick-1's intention can
    become tick-2's ctx.objects snapshot)."""
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=state if state is not None else draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at=_STAMP,
        updated_at=_STAMP,
        revision=0,
        schema_version=draft.schema_version,
    )


def _intention_provenance(draft):
    """Decode a just-emitted intention draft back to its typed provenance."""
    return default_registry().decode(_record_from_draft(draft)).provenance


def _traced_ctx(state: State, *, objects, trace, tmp_path) -> TickContext:
    return TickContext(
        state=state,
        now=NOW,
        bus=FileSignalBus(tmp_path),
        signals=(),
        objects=tuple(objects),
        trace=trace,
    )


def test_first_crystallize_stamps_a_fresh_trace(tmp_path) -> None:
    trace = FakeTracer().start_root()
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5, source_drive=2.0)
    put = _intention_put(
        _cog().step(_traced_ctx(state, objects=desire, trace=trace, tmp_path=tmp_path))
    )
    assert put is not None
    prov = _intention_provenance(put.op.draft)
    assert prov is not None
    assert prov.trace_id == trace.trace_id  # the tick's trace stamped as the birth
    assert prov.creation_span_id == trace.span_id
    assert prov.component == "cognition"


def test_intention_retry_preserves_birth_trace_not_the_retry_tick(tmp_path) -> None:
    # THE test (codex's highest risk): crystallize on tick 1 (trace A); delivery fails,
    # the intention stays LIVE; tick 2 (trace B) re-emits PutRecord(intention active) ->
    # the provenance must still carry the BIRTH trace A, NOT the retry tick's trace B.
    tracer = FakeTracer()
    trace_a = tracer.start_root()  # tick 1 — birth
    trace_b = tracer.start_root()  # tick 2 — retry
    assert trace_a.trace_id != trace_b.trace_id

    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5, source_drive=2.0)

    # Tick 1: no live intention -> FRESH trace A stamped.
    put1 = _intention_put(
        _cog().step(_traced_ctx(state, objects=desire, trace=trace_a, tmp_path=tmp_path))
    )
    assert put1 is not None
    assert _intention_provenance(put1.op.draft).trace_id == trace_a.trace_id

    # Delivery fails -> pending cleared, desire still active, the intention is STILL
    # LIVE (active) in the next tick's snapshot.
    live_intention = _record_from_draft(put1.op.draft, state="active")
    objects2 = (*desire, live_intention)

    # Tick 2 (trace B): the retry re-emits PutRecord(intention active) while it is live
    # -> PRESERVE the birth trace A.
    put2 = _intention_put(
        _cog().step(_traced_ctx(state, objects=objects2, trace=trace_b, tmp_path=tmp_path))
    )
    assert put2 is not None
    prov2 = _intention_provenance(put2.op.draft)
    assert prov2 is not None
    assert prov2.trace_id == trace_a.trace_id  # BIRTH trace preserved
    assert prov2.trace_id != trace_b.trace_id  # NOT the retry tick's trace
    assert prov2.creation_span_id == trace_a.span_id


def test_new_episode_after_resolution_gets_a_fresh_trace(tmp_path) -> None:
    # Once the prior intention resolved (terminal -> absent from the snapshot), the next
    # crystallize is a NEW episode -> a fresh trace, legitimately.
    trace = FakeTracer().start_root()
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5)  # no live intention present
    put = _intention_put(
        _cog().step(_traced_ctx(state, objects=desire, trace=trace, tmp_path=tmp_path))
    )
    assert put is not None
    assert _intention_provenance(put.op.draft).trace_id == trace.trace_id


def test_untraced_intention_carries_lineage_without_trace_fields(tmp_path) -> None:
    # No tracer wired (trace defaults None): the intention still carries lineage, but
    # NO W3C trace fields — behaviour-neutral for ids/timing.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    put = _intention_put(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert put is not None
    prov = _intention_provenance(put.op.draft)
    assert prov is not None
    assert prov.component == "cognition"
    assert prov.trace_id is None


# --- lm-27n.10: the ONE new causal stamp — the Intention→Desire edge ---


def test_fresh_intention_stamps_the_desire_source_edge(tmp_path) -> None:
    # A freshly-crystallized intention carries exactly the intention->desire link in
    # source_object_ids (the only edge the domain has no typed field for).
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    put = _intention_put(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert put is not None
    prov = _intention_provenance(put.op.draft)
    assert prov is not None
    assert prov.source_object_ids == ("desire:contact:owner",)


def test_intention_source_edge_does_not_duplicate_typed_thought_links(tmp_path) -> None:
    # Domain links stay the truth: even a THOUGHT-sprung desire's source_thought_ids are
    # NOT mirrored into the intention's source_object_ids (no drift). Only the desire
    # edge is stamped.
    from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
    from lifemodel.domain.objects import DesireSpring, DesireState

    desire_record = _record_from_draft(
        encode_contact_desire(
            build_contact_desire(
                state=DesireState.ACTIVE,
                salience=2.0,
                spring=DesireSpring.THOUGHT,
                source_thought_ids=("thought:seed:abc",),
            )
        )
    )
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    put = _intention_put(_cog().step(_ctx(state, objects=(desire_record,), tmp_path=tmp_path)))
    assert put is not None
    prov = _intention_provenance(put.op.draft)
    assert prov is not None
    assert prov.source_object_ids == ("desire:contact:owner",)  # NOT the thought id


# --- lm-8o3: deterministic launch jitter (human unpredictability, not a timer) ---

# sha256(f"proactive-{NOW.isoformat()}").digest()[0] % 5: module NOW (12:00) hashes
# to bucket 3 (not held) -- confirmed clear of every existing launch test above.
JITTER_HOLD_NOW = datetime(2026, 7, 6, 12, 5, tzinfo=UTC)  # digest[0] % 5 == 0 -> HOLD
JITTER_LAUNCH_NOW = datetime(2026, 7, 6, 12, 1, tzinfo=UTC)  # digest[0] % 5 != 0 -> launches


def test_seeded_tick_holds_the_launch_without_resolving_the_desire(tmp_path) -> None:
    # A correlation-id that hashes into the hold bucket defers an otherwise-
    # launchable ACTIVE desire by one tick: no LaunchProactive, no intention, no
    # state update -- the desire is NOT resolved, so the next admissible tick fires.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path, now=JITTER_HOLD_NOW))
    assert list(intents) == []


def test_non_seeded_tick_launches_normally(tmp_path) -> None:
    # A correlation-id outside the hold bucket launches exactly as before.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path, now=JITTER_LAUNCH_NOW))
    assert _launch(intents) is not None
    assert _intention_put(intents) is not None


def test_intention_retry_preserves_the_source_edge_unchanged(tmp_path) -> None:
    # The preserve-on-retry branch keeps the birth provenance — including its
    # source_object_ids — unchanged; a retry never re-stamps or duplicates the edge.
    tracer = FakeTracer()
    trace_a = tracer.start_root()
    trace_b = tracer.start_root()
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5)

    put1 = _intention_put(
        _cog().step(_traced_ctx(state, objects=desire, trace=trace_a, tmp_path=tmp_path))
    )
    assert put1 is not None
    assert _intention_provenance(put1.op.draft).source_object_ids == ("desire:contact:owner",)

    live_intention = _record_from_draft(put1.op.draft, state="active")
    put2 = _intention_put(
        _cog().step(
            _traced_ctx(state, objects=(*desire, live_intention), trace=trace_b, tmp_path=tmp_path)
        )
    )
    assert put2 is not None
    prov2 = _intention_provenance(put2.op.draft)
    assert prov2 is not None
    assert prov2.source_object_ids == ("desire:contact:owner",)  # unchanged, not doubled

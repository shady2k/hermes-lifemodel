"""ThoughtAttention — the 0-LLM anti-rumination brake (lm-27n.7).

The behaviour contract: behavior-neutral with no thoughts; the two width caps
(scan ≤ W, attend ≤ K); the typed mutation shapes (PutRecord for a field-only
update, TransitionRecord for park/unpark/expire); and the convergence guarantee —
a never-resolved attended thought PARKS within PARK_AFTER ticks with a
monotone-non-increasing salience, then unparks after its window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.composition import build_lifemodel
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent, PutRecord, TransitionRecord
from lifemodel.core.thought_attention import ThoughtAttention
from lifemodel.core.thought_score import ATTEND_K, SCAN_WIDTH
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.memory import MemoryDraft, MemoryRecord
from lifemodel.domain.objects import Thought, default_registry
from lifemodel.state.model import State
from lifemodel.testing import (
    FakeClock,
    FakeSignalBus,
    contact_desire_record,
    thought_record,
)

_NOW = datetime(2026, 7, 6, 0, 1, tzinfo=UTC)  # one minute after the builder stamp


def _component() -> ThoughtAttention:
    return ThoughtAttention()


def _ctx(
    objects: tuple[MemoryRecord, ...], now: datetime = _NOW, signals: tuple = ()
) -> TickContext:
    return TickContext(
        state=State(), now=now, bus=FakeSignalBus(), objects=objects, signals=signals
    )


def _puts(intents: list[Intent]) -> list[PutRecord]:
    return [i for i in intents if isinstance(i, PutRecord)]


def _transitions(intents: list[Intent]) -> list[TransitionRecord]:
    return [i for i in intents if isinstance(i, TransitionRecord)]


def _record_from_draft(draft: MemoryDraft) -> MemoryRecord:
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
        revision=1,
        schema_version=draft.schema_version,
    )


# --- behavior-neutral with no live thoughts ---------------------------------


def test_no_thoughts_returns_empty_no_mutation() -> None:
    assert list(_component().step(_ctx(()))) == []


def test_non_thought_records_are_ignored() -> None:
    assert list(_component().step(_ctx((contact_desire_record("active"),)))) == []


def test_all_parked_within_window_is_a_noop() -> None:
    future = (_NOW + timedelta(hours=2)).isoformat()
    objects = (
        thought_record("resting a", "parked", id="t-a", salience=0.5, parked_until=future),
        thought_record("resting b", "parked", id="t-b", salience=0.5, parked_until=future),
    )
    assert list(_component().step(_ctx(objects))) == []  # suspended → no churn


# --- the single field-only update (typed PutRecord door) --------------------


def test_single_active_thought_emits_one_typed_put() -> None:
    objects = (thought_record("a lone worry", "active", id="t1", salience=0.5),)
    intents = list(_component().step(_ctx(objects)))
    assert len(intents) == 1
    put = intents[0]
    assert isinstance(put, PutRecord)  # a field update, NOT a same-state transition
    draft = put.op.draft
    assert draft.state == "active"  # no lifecycle change
    assert draft.payload["no_progress_count"] == 1  # attended → bumped
    assert isinstance(draft.payload["attention_score"], float)
    assert draft.payload["attention_score"] > 0.0  # the score was recorded
    assert draft.salience < 0.5  # decayed
    # goes through the registry door: the emitted draft decodes back to a Thought.
    decoded = default_registry().decode(_record_from_draft(draft))
    assert isinstance(decoded, Thought)


def test_unattended_active_thought_decays_but_does_not_bump_no_progress() -> None:
    # Two active thoughts; only the top-scoring one is attended (K=1). The other
    # still decays (else an unattended loop would be immortal) but keeps np=0.
    objects = (
        thought_record("high", "active", id="t-hi", salience=0.9),
        thought_record("low", "active", id="t-lo", salience=0.6),
    )
    puts = _puts(list(_component().step(_ctx(objects))))
    by_id = {p.op.draft.id: p.op.draft for p in puts}
    assert by_id["t-hi"].payload["no_progress_count"] == 1  # attended
    assert by_id["t-lo"].payload["no_progress_count"] == 0  # not attended
    assert by_id["t-lo"].salience < 0.6  # but still decayed


# --- width caps (codex's two bounds) ----------------------------------------


def test_scan_and_attend_width_caps() -> None:
    # 40 live active thoughts, distinct salience → only the top SCAN_WIDTH are
    # touched, and exactly ATTEND_K is attended (no_progress bumped).
    objects = tuple(
        thought_record(f"thought number {i}", "active", id=f"t{i:03d}", salience=0.5 + i * 0.001)
        for i in range(40)
    )
    intents = list(_component().step(_ctx(objects)))
    assert not _transitions(intents)  # none park (np<3, salience>floor)
    puts = _puts(intents)
    assert len(puts) == SCAN_WIDTH  # at most W thoughts updated
    bumped = [p for p in puts if p.op.draft.payload["no_progress_count"] == 1]
    assert len(bumped) == ATTEND_K  # at most K attended
    assert bumped[0].op.draft.id == "t039"  # the most salient wins the single slot
    touched = {p.op.draft.id for p in puts}
    assert "t039" in touched and "t000" not in touched  # top-W scanned, tail dropped


# --- parking (state change: active → parked) --------------------------------


def test_loop_parks_after_park_after_via_transition_with_patch() -> None:
    # A thought already at no_progress=2: attending it this tick reaches PARK_AFTER.
    objects = (thought_record("stuck loop", "active", id="t1", salience=0.8, no_progress_count=2),)
    intents = list(_component().step(_ctx(objects)))
    assert len(intents) == 1
    tr = intents[0]
    assert isinstance(tr, TransitionRecord)
    assert tr.op.from_state == "active" and tr.op.to_state == "parked"
    assert tr.op.patch is not None
    assert tr.op.patch.salience is not None and tr.op.patch.salience < 0.8  # decayed
    merge = tr.op.patch.payload_merge
    assert merge is not None
    assert merge["no_progress_count"] == 3
    assert merge["park_count"] == 1  # first park cycle
    assert isinstance(merge["parked_until"], str)


def test_below_floor_parks_before_expiring() -> None:
    # Salience already at/under the floor → parks even though it is not looping yet
    # (park-before-expire: recoverable, not destroyed).
    objects = (thought_record("faded", "active", id="t1", salience=0.02),)
    intents = list(_component().step(_ctx(objects)))
    tr = intents[0]
    assert isinstance(tr, TransitionRecord)
    assert tr.op.to_state == "parked"


# --- unpark / expire (state change out of parked) ---------------------------


def test_elapsed_park_window_unparks_and_resets_no_progress() -> None:
    past = (_NOW - timedelta(hours=1)).isoformat()
    objects = (
        thought_record(
            "re-entrant",
            "parked",
            id="t1",
            salience=0.5,
            parked_until=past,
            park_count=1,
            no_progress_count=3,
        ),
    )
    intents = list(_component().step(_ctx(objects)))
    tr = intents[0]
    assert isinstance(tr, TransitionRecord)
    assert tr.op.from_state == "parked" and tr.op.to_state == "active"
    assert tr.op.patch is not None and tr.op.patch.payload_merge is not None
    assert tr.op.patch.payload_merge["no_progress_count"] == 0  # fresh chance


def test_chronic_looper_expires_past_the_backoff_cap() -> None:
    past = (_NOW - timedelta(hours=1)).isoformat()
    objects = (
        thought_record(
            "chronic",
            "parked",
            id="t1",
            salience=0.5,
            parked_until=past,
            park_count=3,  # already at the cap
        ),
    )
    intents = list(_component().step(_ctx(objects)))
    tr = intents[0]
    assert isinstance(tr, TransitionRecord)
    assert tr.op.to_state == "expired"  # bounded rumination: the loop terminates


# --- the anti-rumination convergence guarantee (end-to-end, real store) -----


def test_convergence_a_never_resolved_thought_parks_then_unparks(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    store.commit(State(last_tick_at=clock.now().isoformat()))
    store.put(
        encode_thought(
            build_thought(id="t-loop", content="did I upset them?", salience=0.9, trigger="idle")
        )
    )

    saliences: list[float] = []
    for _ in range(3):  # PARK_AFTER attended ticks with no progress
        clock.advance(timedelta(minutes=1))
        lm.coreloop.tick()
        saliences.append(store.get("thought", "t-loop").salience)

    parked = store.get("thought", "t-loop")
    assert parked.state == "parked"  # converged: the brake pulled it out of active
    assert parked.payload["no_progress_count"] == 3
    assert parked.payload["park_count"] == 1
    assert parked.payload["parked_until"] is not None
    # salience monotone non-increasing under no progress (decay never grows it).
    assert all(saliences[i] >= saliences[i + 1] for i in range(len(saliences) - 1))

    # ... and after the park window it unparks for a fresh chance.
    clock.advance(timedelta(hours=6, minutes=2))  # past parked_until (parked at +3min, +6h window)
    lm.coreloop.tick()
    revived = store.get("thought", "t-loop")
    assert revived.state == "active"
    assert revived.payload["no_progress_count"] == 0


# --- lm-27n.9: sustained_attention_count (the top-down Rubicon persistence) -----


def test_viable_attended_thought_accrues_sustained_attention() -> None:
    # A salient, other-serving, attended thought accrues one tick of persistence.
    objects = (
        thought_record(
            "check in on the owner",
            "active",
            id="t1",
            salience=0.8,
            other_regarding_value=0.6,
            sustained_attention_count=1,
        ),
    )
    put = _puts(list(_component().step(_ctx(objects))))[0]
    assert put.op.draft.payload["sustained_attention_count"] == 2  # bumped
    assert put.op.draft.payload["no_progress_count"] == 1  # attended


def test_idle_wandering_thought_never_accrues_sustained_attention() -> None:
    # A low-salience idle thought is attended (it is the only one) but is NOT a
    # viable contact candidate -> sustained_attention_count stays put (anti-frivolity
    # at the counter: idle wandering can never persist into contact).
    objects = (
        thought_record(
            "just wandering",
            "active",
            id="t1",
            salience=0.15,
            trigger="idle",
            other_regarding_value=0.10,
            actionability=0.05,
            sustained_attention_count=0,
        ),
    )
    put = _puts(list(_component().step(_ctx(objects))))[0]
    assert put.op.draft.payload["sustained_attention_count"] == 0  # not viable -> no accrual
    assert put.op.draft.payload["no_progress_count"] == 1  # ...but still attended/decayed


def test_unattended_viable_thought_does_not_accrue_sustained_attention() -> None:
    # Two viable thoughts; only the top-scoring is attended (K=1). The other stays at
    # its sustained count (persistence needs sustained ATTENTION, not mere existence).
    objects = (
        thought_record(
            "high",
            "active",
            id="t-hi",
            salience=0.9,
            other_regarding_value=0.6,
            sustained_attention_count=1,
        ),
        thought_record(
            "low",
            "active",
            id="t-lo",
            salience=0.6,
            other_regarding_value=0.6,
            sustained_attention_count=1,
        ),
    )
    by_id = {p.op.draft.id: p.op.draft for p in _puts(list(_component().step(_ctx(objects))))}
    assert by_id["t-hi"].payload["sustained_attention_count"] == 2  # attended -> accrues
    assert by_id["t-lo"].payload["sustained_attention_count"] == 1  # not attended -> held


def test_sustained_attention_is_not_coupled_to_no_progress() -> None:
    # A viable thought accrues persistence while its no_progress climbs independently:
    # the two counters are distinct (coupling would park good thoughts faster).
    objects = (
        thought_record(
            "check in on the owner",
            "active",
            id="t1",
            salience=0.8,
            other_regarding_value=0.6,
            sustained_attention_count=0,
            no_progress_count=0,
        ),
    )
    put = _puts(list(_component().step(_ctx(objects))))[0]
    # both bumped this tick, but from independent counters — no shared arithmetic.
    assert put.op.draft.payload["sustained_attention_count"] == 1
    assert put.op.draft.payload["no_progress_count"] == 1


# --- lm-27n.9: resolve the crystallized thought on the CREATED signal -------------


def _created(thought_id: str):
    # Aggregation emits this ONLY when it actually creates a top-down desire — attention
    # resolves the source thought on genuine creation, never on a mere proposal.
    from lifemodel.core.taxonomy import thought_contact_created_signal

    return thought_contact_created_signal(
        origin_id="contact-aggregation", thought_id=thought_id, timestamp=None
    )


def test_created_signal_resolves_the_source_thought_instead_of_decaying_it() -> None:
    # On the "contact desire CREATED from this thought" signal, attention (the SOLE
    # thought writer) resolves the source thought (active->resolved) INSTEAD of the
    # decay PutRecord — its reason became a desire; it must not also re-crystallize.
    objects = (
        thought_record(
            "check in on the owner",
            "active",
            id="t-serve",
            salience=0.8,
            other_regarding_value=0.6,
            sustained_attention_count=2,
        ),
    )
    intents = list(_component().step(_ctx(objects, signals=(_created("t-serve"),))))
    assert len(intents) == 1
    tr = intents[0]
    assert isinstance(tr, TransitionRecord)
    assert tr.op.from_state == "active" and tr.op.to_state == "resolved"
    assert not _puts(intents)  # NOT decayed via a PutRecord


def test_created_signal_for_an_absent_thought_is_a_noop_for_others() -> None:
    # A created-signal naming a thought not in the snapshot resolves nothing; the live
    # thought is attended/decayed as usual.
    objects = (thought_record("some worry", "active", id="t1", salience=0.5),)
    intents = list(_component().step(_ctx(objects, signals=(_created("t-ghost"),))))
    assert not _transitions(intents)
    assert len(_puts(intents)) == 1  # the live thought still just decays


def test_bare_proposal_without_creation_does_not_resolve_the_thought() -> None:
    # The P1 fix (codex): a PROPOSAL that aggregation did NOT turn into a desire
    # (suppressed by silence window / backoff / in-flight) must NOT resolve the source
    # thought — the reason stays live (decays/parks normally), not silently spent.
    from lifemodel.core.taxonomy import thought_contact_proposal_signal

    proposal = thought_contact_proposal_signal(
        origin_id="thought-crystallization",
        thought_id="t-serve",
        score=0.7,
        reason="other-serving",
        other_regarding=0.6,
        actionability=0.3,
        salience=0.8,
        timestamp=None,
    )
    objects = (
        thought_record(
            "check in on the owner", "active", id="t-serve", salience=0.8, other_regarding_value=0.6
        ),
    )
    intents = list(_component().step(_ctx(objects, signals=(proposal,))))
    assert not _transitions(intents)  # NOT resolved on a bare proposal
    assert len(_puts(intents)) == 1  # the thought stays live and just decays


def test_convergence_never_exceeds_the_width_caps(tmp_path) -> None:
    # Even with a crowd of live thoughts, one tick attends at most K and the brake
    # still parks the loop within PARK_AFTER attended ticks for the top thought.
    clock = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    store = lm.state
    store.commit(State(last_tick_at=clock.now().isoformat()))
    for i in range(50):  # more than SCAN_WIDTH
        store.put(
            encode_thought(
                build_thought(id=f"t{i:03d}", content=f"worry {i}", salience=0.5, trigger="idle")
            )
        )
    clock.advance(timedelta(minutes=1))
    lm.coreloop.tick()
    # exactly one thought was attended (its no_progress bumped to 1); the rest that
    # were scanned only decayed.
    attended = [r for r in store.find(kind="thought") if r.payload["no_progress_count"] == 1]
    assert len(attended) == ATTEND_K

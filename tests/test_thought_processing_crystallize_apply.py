from datetime import UTC, datetime

from lifemodel.core.intents import PutRecord, TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
_GOOD = {
    "content": "ask how their interview went",
    "basis": "follow_up",
    "trigger_kind": "event",
    "trigger_value": "next time we talk",
}


def _ctx(parsed):
    thought = build_thought(
        id="thought:seed:x", content="interview Friday", state=ThoughtState.ACTIVE
    )
    sig = internal_result_signal(
        origin_id="r1",
        correlation_id="c1",
        raw="{...}",
        parsed=parsed,
        timestamp="2026-07-17T12:00:00+00:00",
    )
    return make_tick_context(
        state=State(pending_internal_id="c1", pending_internal_subject_id="thought:seed:x"),
        now=NOW,
        objects=[draft_to_record(encode_thought(thought), now=NOW)],
        signals=[sig],
    )


def test_crystallize_emits_resolve_transition_and_a_commitment_put():
    ctx = _ctx({"outcome": "crystallize_commitment", "commitment": _GOOD})
    intents = list(ThoughtProcessingApply().step(ctx))
    trs = [i for i in intents if isinstance(i, TransitionRecord)]
    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(trs) == 1 and trs[0].op.to_state == ThoughtState.RESOLVED.value
    assert len(puts) == 1 and puts[0].op.draft.kind == "commitment"
    assert puts[0].op.draft.payload["content"] == "ask how their interview went"
    # provenance target→source: the commitment points back at the thought
    assert "thought:seed:x" in puts[0].op.draft.payload["source_thought_ids"]


def test_crystallize_with_bad_enum_falls_back_to_no_progress_no_put():
    # schema-shaped but domain-invalid (bad basis) → registry.encode rejects → no-progress
    bad = {**_GOOD, "basis": "not_a_basis"}
    ctx = _ctx({"outcome": "crystallize_commitment", "commitment": bad})
    intents = list(ThoughtProcessingApply().step(ctx))
    assert [i for i in intents if isinstance(i, PutRecord)] == []  # no commitment persisted
    trs = [i for i in intents if isinstance(i, TransitionRecord)]
    assert trs and trs[0].op.to_state == ThoughtState.PARKED.value  # bounded no-progress
    assert trs[0].op.patch.payload_merge["no_progress_count"] == 1

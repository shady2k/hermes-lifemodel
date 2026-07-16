from datetime import UTC, datetime

import pytest

from lifemodel.core.component import TickContext
from lifemodel.core.intents import PutRecord, TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ProcessingReason, ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)
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


def _ctx_with_logger(parsed):
    """Like :func:`_ctx` but with a real (fake) ``logger`` so a test can assert on the
    ``processing_reason`` span field the malformed-crystallize path stamps."""
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
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(
        state=State(pending_internal_id="c1", pending_internal_subject_id="thought:seed:x"),
        now=NOW,
        trace=_TRACE,
        objects=(draft_to_record(encode_thought(thought), now=NOW),),
        signals=(sig,),
        logger=logger,
    )
    return ctx, logger


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


# I1 (lm-705.3 review, both reviewers) — strict field parsing: a coercing
# str(...)/float(...) build used to mint a garbage commitment (an int `content`)
# or escape the bounded except entirely (an overflowing `other_regarding_value`
# raised uncaught `OverflowError`, stranding the thought). Every one of these
# bad-model-data shapes must now land on the SAME bounded no-progress outcome —
# no PutRecord, a park transition, no_progress_count bumped by exactly one.
@pytest.mark.parametrize(
    "bad_fields",
    [
        {**_GOOD, "content": 123},
        {**_GOOD, "other_regarding_value": 10**400},
        {**_GOOD, "basis": "BAD"},
        {**_GOOD, "trigger_value": ["not", "a", "string"]},
    ],
    ids=[
        "content_is_an_int",
        "other_regarding_value_overflows_float",
        "basis_is_not_a_valid_enum",
        "trigger_value_is_not_a_string",
    ],
)
def test_crystallize_bad_model_data_is_bounded_no_progress_not_a_strand(bad_fields):
    ctx = _ctx({"outcome": "crystallize_commitment", "commitment": bad_fields})
    intents = list(ThoughtProcessingApply().step(ctx))
    assert [i for i in intents if isinstance(i, PutRecord)] == []  # no garbage commitment
    trs = [i for i in intents if isinstance(i, TransitionRecord)]
    assert len(trs) == 1
    assert trs[0].op.to_state == ThoughtState.PARKED.value  # bounded, never stranded
    assert trs[0].op.patch.payload_merge["no_progress_count"] == 1


def test_crystallize_bad_model_data_logs_the_distinguishing_reason():
    # I1 (opus) — a crystallize validation failure must be telemetrically distinct
    # from the generic PARKED_NO_PROGRESS, not indistinguishable from an infra bug.
    bad = {**_GOOD, "content": 123}
    ctx, logger = _ctx_with_logger({"outcome": "crystallize_commitment", "commitment": bad})
    list(ThoughtProcessingApply().step(ctx))
    assert logger.span.attrs["processing_reason"] == ProcessingReason.CRYSTALLIZE_MALFORMED.value

from datetime import UTC, datetime

from lifemodel.core.component import TickContext
from lifemodel.core.intents import TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ProcessingReason, ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)

# ctx.trace is non-optional (spec §4.1) — a literal span's ids for span-field fixtures.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


def _ctx(subject_id, *, parsed, raw, thought_state=ThoughtState.ACTIVE):
    thought = build_thought(id="thought:seed:a", content="ca", state=thought_state)
    sig = internal_result_signal(
        origin_id="r1",
        correlation_id="process-x",
        raw=raw,
        parsed=parsed,
        timestamp="2026-07-16T12:00:00+00:00",
    )
    return make_tick_context(
        state=State(pending_internal_id="process-x", pending_internal_subject_id=subject_id),
        now=NOW,
        objects=[draft_to_record(encode_thought(thought), now=NOW)],
        signals=[sig],
    )


def _ctx_with_logger(subject_id, *, parsed, raw, thought_state=ThoughtState.ACTIVE):
    """Like :func:`_ctx` but with a real (fake) ``logger`` so a test can assert on
    the span fields ``step`` stamps — ``make_tick_context`` deliberately leaves
    ``logger`` ``None`` (bare unit-test default), so this builds the ``TickContext``
    directly, mirroring the pattern in ``tests/test_suppression.py``/``test_cognition.py``."""
    thought = build_thought(id="thought:seed:a", content="ca", state=thought_state)
    sig = internal_result_signal(
        origin_id="r1",
        correlation_id="process-x",
        raw=raw,
        parsed=parsed,
        timestamp="2026-07-16T12:00:00+00:00",
    )
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(
        state=State(pending_internal_id="process-x", pending_internal_subject_id=subject_id),
        now=NOW,
        trace=_TRACE,
        objects=(draft_to_record(encode_thought(thought), now=NOW),),
        signals=(sig,),
        logger=logger,
    )
    return ctx, logger


def test_resolve_emits_terminal_transition():
    ctx = _ctx("thought:seed:a", parsed={"outcome": "resolve"}, raw="{...}")
    trs = [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)]
    assert len(trs) == 1
    assert trs[0].op.id == "thought:seed:a"
    assert trs[0].op.to_state == ThoughtState.RESOLVED.value


def test_malformed_bumps_no_progress():
    ctx = _ctx("thought:seed:a", parsed=None, raw="junk")
    trs = [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)]
    assert trs[0].op.to_state == ThoughtState.PARKED.value
    assert trs[0].op.patch.payload_merge["no_progress_count"] == 1


def test_transient_failure_emits_no_transition():
    ctx = _ctx("thought:seed:a", parsed=None, raw="   ")
    assert [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)] == []


def test_no_subject_is_a_noop():  # a subjectless (noticing) pass, or cleared subject
    ctx = _ctx(None, parsed={"outcome": "resolve"}, raw="{...}")
    assert list(ThoughtProcessingApply().step(ctx)) == []


def test_no_internal_result_signal_is_a_noop():  # runs on every completion frame; guards
    ctx = make_tick_context(state=State(), now=NOW, objects=[], signals=[])
    assert list(ThoughtProcessingApply().step(ctx)) == []


def test_subject_no_longer_live_is_a_noop():  # thought already terminal — nothing to do
    ctx = _ctx("thought:seed:gone", parsed={"outcome": "resolve"}, raw="{...}")
    assert list(ThoughtProcessingApply().step(ctx)) == []


def test_reflection_is_stamped_on_the_apply_span():  # I1 — the reflection rides the span
    ctx, logger = _ctx_with_logger(
        "thought:seed:a",
        parsed={"outcome": "resolve", "reflection": "a quiet resolve"},
        raw="{...}",
    )
    list(ThoughtProcessingApply().step(ctx))
    assert logger.span.attrs["reflection"] == "a quiet resolve"
    assert logger.span.attrs["processing_reason"] == ProcessingReason.RESOLVED.value


def test_reflection_is_capped_at_500_chars():
    ctx, logger = _ctx_with_logger(
        "thought:seed:a",
        parsed={"outcome": "resolve", "reflection": "x" * 600},
        raw="{...}",
    )
    list(ThoughtProcessingApply().step(ctx))
    assert logger.span.attrs["reflection"] == "x" * 500


def test_missing_reflection_key_stamps_nothing():  # no field when the model omits it
    ctx, logger = _ctx_with_logger("thought:seed:a", parsed={"outcome": "resolve"}, raw="{...}")
    list(ThoughtProcessingApply().step(ctx))
    assert "reflection" not in logger.span.attrs

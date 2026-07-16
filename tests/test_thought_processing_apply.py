from datetime import UTC, datetime

from lifemodel.core.intents import TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)


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

from datetime import UTC, datetime

from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchInternalCognition, TransitionRecord
from lifemodel.core.thought_processing import ThoughtProcessingSelector
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)

# ctx.trace is non-optional (spec §4.1) — a literal span's ids for span-field fixtures.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


def _rec(thought):
    # a MemoryRecord as the start-of-tick snapshot would carry it
    from lifemodel.testing.harness import draft_to_record  # helper added in THIS task's Step 3

    return draft_to_record(encode_thought(thought), now=NOW)


def _active(id_, salience):
    return build_thought(id=id_, content=f"c{id_}", state=ThoughtState.ACTIVE, salience=salience)


def test_selects_top_salience_active_thought():
    ctx = make_tick_context(
        state=State(),
        now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.3)), _rec(_active("thought:seed:b", 0.9))],
    )
    intents = list(ThoughtProcessingSelector().step(ctx))
    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(launches) == 1
    assert launches[0].subject_id == "thought:seed:b"  # top salience
    assert launches[0].json_schema is not None  # a structured pass
    assert launches[0].instructions  # processing framing


def test_empty_backlog_emits_nothing():
    ctx = make_tick_context(state=State(), now=NOW, objects=[])
    assert list(ThoughtProcessingSelector().step(ctx)) == []


def test_single_flight_blocks_when_a_pass_is_in_flight():
    ctx = make_tick_context(
        state=State(pending_internal_id="process-x"),
        now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [
        i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_interval_gate_blocks_a_recent_pass():
    ctx = make_tick_context(
        state=State(last_internal_call_at="2026-07-16T11:50:00+00:00"),
        now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [
        i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_budget_gate_blocks_at_ceiling():
    ctx = make_tick_context(
        state=State(internal_calls_today=50, internal_calls_day="2026-07-16"),
        now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [
        i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_rearms_expired_parked_thought_and_does_not_launch_it():
    parked = build_thought(
        id="thought:seed:p",
        content="cp",
        state=ThoughtState.PARKED,
        salience=0.9,
        park_count=1,
        parked_until="2026-07-16T06:00:00+00:00",  # past → re-eligible
    )
    ctx = make_tick_context(state=State(), now=NOW, objects=[_rec(parked)])
    intents = list(ThoughtProcessingSelector().step(ctx))
    transitions = [i for i in intents if isinstance(i, TransitionRecord)]
    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(transitions) == 1
    assert transitions[0].op.from_state == ThoughtState.PARKED.value
    assert transitions[0].op.to_state == ThoughtState.ACTIVE.value
    assert launches == []  # re-armed this tick, processed a later tick


def test_still_parked_thought_is_not_rearmed():
    parked = build_thought(
        id="thought:seed:p",
        content="cp",
        state=ThoughtState.PARKED,
        salience=0.9,
        parked_until="2026-07-17T00:00:00+00:00",  # future → still parked
    )
    ctx = make_tick_context(state=State(), now=NOW, objects=[_rec(parked)])
    assert list(ThoughtProcessingSelector().step(ctx)) == []


def test_rearm_is_recorded_as_an_observable_unparked_count_on_the_span():  # M1
    # make_tick_context leaves logger=None (bare unit-test default) — build the
    # TickContext directly with a FakeSpanLogger so the span field is assertable,
    # mirroring the pattern in tests/test_suppression.py / tests/test_cognition.py.
    parked = build_thought(
        id="thought:seed:p",
        content="cp",
        state=ThoughtState.PARKED,
        salience=0.9,
        park_count=1,
        parked_until="2026-07-16T06:00:00+00:00",  # past → re-eligible
    )
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(state=State(), now=NOW, trace=_TRACE, objects=(_rec(parked),), logger=logger)
    list(ThoughtProcessingSelector().step(ctx))
    assert logger.span.attrs["unparked"] == 1


def test_no_rearm_stamps_no_unparked_field():
    active = _active("thought:seed:a", 0.9)
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(state=State(), now=NOW, trace=_TRACE, objects=(_rec(active),), logger=logger)
    list(ThoughtProcessingSelector().step(ctx))
    assert "unparked" not in logger.span.attrs

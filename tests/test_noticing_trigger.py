"""Unit tests for :class:`NoticingTrigger` (lm-705.5 combined task E4, Task 4).

Mirrors ``tests/test_thought_processing_selector.py``'s style: a bare
``TickContext`` (``make_tick_context``) drives the component in isolation, a
real :class:`NoticingBuffer` (Task 2) stands in for the live conversation
buffer, and each gate is exercised in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchInternalCognition
from lifemodel.core.noticing import (
    DEFAULT_NOTICING_IDLE,
    DEFAULT_NOTICING_SIZE_CAP,
    NOTICING_INSTRUCTIONS,
    NOTICING_JSON_SCHEMA,
    NoticingReason,
    NoticingTrigger,
)
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)

# ctx.trace is non-optional (spec §4.1) — a literal span's ids for span-field fixtures.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


def _complete_turn(
    buffer: NoticingBuffer,
    session_id: str,
    turn_id: str,
    *,
    user_text: str = "hi",
    assistant_text: str = "hello",
    ts: datetime,
) -> None:
    """Drive the buffer's real public API to land one ``complete`` entry."""
    buffer.open_pending(session_id, user_text=user_text, now=ts)
    buffer.complete(session_id, turn_id, assistant_text=assistant_text, now=ts)


def test_closed_segment_past_idle_emits_one_subjectless_launch():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", user_text="tell me about it", ts=old_ts)

    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE)
    intents = list(NoticingTrigger(buffer).step(ctx))

    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(launches) == 1
    launch = launches[0]
    assert launch.subject_id is None  # subjectless (noticing, not processing)
    assert launch.json_schema == NOTICING_JSON_SCHEMA
    assert launch.instructions == NOTICING_INSTRUCTIONS
    assert "t1" in launch.prompt
    assert "tell me about it" in launch.prompt
    assert launch.correlation_id.startswith("notice-s1@t1@")
    assert launch.origin_traceparent  # mandatory async-correlation anchor


def test_open_pending_with_no_completion_emits_nothing():
    buffer = NoticingBuffer()
    buffer.open_pending("s1", user_text="mid-turn", now=NOW - timedelta(hours=1))

    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE)
    assert list(NoticingTrigger(buffer).step(ctx)) == []


def test_below_size_cap_and_within_idle_emits_nothing():
    buffer = NoticingBuffer()
    _complete_turn(buffer, "s1", "t1", ts=NOW - timedelta(minutes=1))

    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE)
    assert list(NoticingTrigger(buffer).step(ctx)) == []


def test_size_cap_reached_launches_even_though_idle_has_not_elapsed():
    buffer = NoticingBuffer()
    for i in range(DEFAULT_NOTICING_SIZE_CAP):
        _complete_turn(
            buffer, "s1", f"t{i}", ts=NOW - timedelta(seconds=DEFAULT_NOTICING_SIZE_CAP - i)
        )

    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE)
    intents = list(NoticingTrigger(buffer).step(ctx))
    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(launches) == 1
    # the anchor is the LAST (most recent) entry's turn_id
    assert launches[0].correlation_id.startswith(f"notice-s1@t{DEFAULT_NOTICING_SIZE_CAP - 1}@")


def test_single_flight_blocks_when_a_pass_is_in_flight():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    ctx = make_tick_context(state=State(pending_internal_id="process-x"), now=NOW, trace=_TRACE)
    assert [
        i for i in NoticingTrigger(buffer).step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_interval_gate_blocks_a_recent_pass():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    ctx = make_tick_context(
        state=State(last_internal_call_at="2026-07-17T11:50:00+00:00"), now=NOW, trace=_TRACE
    )
    assert [
        i for i in NoticingTrigger(buffer).step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_budget_gate_blocks_at_ceiling():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    ctx = make_tick_context(
        state=State(internal_calls_today=50, internal_calls_day="2026-07-17"),
        now=NOW,
        trace=_TRACE,
    )
    assert [
        i for i in NoticingTrigger(buffer).step(ctx) if isinstance(i, LaunchInternalCognition)
    ] == []


def test_empty_buffer_emits_nothing():
    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE)
    assert list(NoticingTrigger(NoticingBuffer()).step(ctx)) == []


def test_span_logs_idle_launch_reason():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(state=State(), now=NOW, trace=_TRACE, logger=logger)

    list(NoticingTrigger(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.IDLE_LAUNCH.value


def test_span_logs_size_cap_launch_reason():
    buffer = NoticingBuffer()
    for i in range(DEFAULT_NOTICING_SIZE_CAP):
        _complete_turn(
            buffer, "s1", f"t{i}", ts=NOW - timedelta(seconds=DEFAULT_NOTICING_SIZE_CAP - i)
        )
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(state=State(), now=NOW, trace=_TRACE, logger=logger)

    list(NoticingTrigger(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.SIZE_CAP_LAUNCH.value


def test_span_logs_nothing_lingered_when_no_segment_is_due():
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(state=State(), now=NOW, trace=_TRACE, logger=logger)

    list(NoticingTrigger(NoticingBuffer()).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTHING_LINGERED.value


def test_span_logs_skipped_in_flight():
    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    ctx = TickContext(
        state=State(pending_internal_id="process-x"), now=NOW, trace=_TRACE, logger=logger
    )

    list(NoticingTrigger(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.SKIPPED_IN_FLIGHT.value


def test_backlog_thought_gists_are_folded_into_the_prompt():
    from lifemodel.core.thought_view import build_thought, encode_thought
    from lifemodel.testing.harness import draft_to_record

    buffer = NoticingBuffer()
    old_ts = NOW - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)
    thought = build_thought(id="thought:seed:a", content="a lingering worry", salience=0.9)
    record = draft_to_record(encode_thought(thought), now=NOW)

    ctx = make_tick_context(state=State(), now=NOW, trace=_TRACE, objects=[record])
    intents = list(NoticingTrigger(buffer).step(ctx))

    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(launches) == 1
    assert "a lingering worry" in launches[0].prompt

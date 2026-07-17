"""Unit tests for :class:`NoticingApply` (lm-705.5 combined task E4, Task 5).

Mirrors ``tests/test_thought_processing_apply.py``'s style: a real
:class:`NoticingBuffer` (Task 2) stands in for the live conversation buffer,
seeded through its OWN public API (``open_pending``/``stamp_source``/
``complete``) rather than hand-built ``BufferEntry`` values, and an
``internal_result`` signal carries the (fake) aux call's typed result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.component import TickContext
from lifemodel.core.intents import PutRecord, UpdateState
from lifemodel.core.noticing import NoticingApply, NoticingReason
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.timeutil import to_iso
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)

# ctx.trace is non-optional (spec §4.1) — a literal span's ids for span-field fixtures.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


def _seeded_buffer() -> NoticingBuffer:
    """A buffer with two closed turns on lane ``s1``: ``t1`` (stamped ``m1``)
    then ``t2`` (no stamped source id — the deferred inbound seam, E3)."""
    buffer = NoticingBuffer()
    old = NOW - timedelta(hours=1)
    buffer.open_pending("s1", user_text="first", now=old)
    buffer.stamp_source("s1", "m1")
    buffer.complete("s1", "t1", assistant_text="reply1", now=old + timedelta(seconds=1))
    buffer.open_pending("s1", user_text="second", now=old + timedelta(seconds=2))
    buffer.complete("s1", "t2", assistant_text="reply2", now=old + timedelta(seconds=3))
    return buffer


def _ctx(
    *,
    correlation_id: str,
    subject_id: str | None,
    parsed: dict | None,
    raw: str = "{...}",
    noticed_source_ids: tuple[str, ...] = (),
):
    sig = internal_result_signal(
        origin_id="r1",
        correlation_id=correlation_id,
        raw=raw,
        parsed=parsed,
        timestamp=to_iso(NOW),
    )
    return make_tick_context(
        state=State(
            pending_internal_id=correlation_id,
            pending_internal_subject_id=subject_id,
            noticed_source_ids=noticed_source_ids,
        ),
        now=NOW,
        signals=[sig],
        trace=_TRACE,
    )


def _ctx_with_logger(**kwargs):
    """Like :func:`_ctx` but with a real (fake) ``logger`` for span assertions —
    ``make_tick_context`` deliberately leaves ``logger`` ``None``."""
    ctx = _ctx(**kwargs)
    logger = FakeSpanLogger(FakeActiveSpan(_TRACE, component="cognition", tick=1))
    return (
        TickContext(
            state=ctx.state,
            now=ctx.now,
            trace=ctx.trace,
            objects=ctx.objects,
            signals=ctx.signals,
            logger=logger,
        ),
        logger,
    )


def test_two_valid_seeds_produce_two_thoughts_and_append_ring():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {
        "seeds": [
            {
                "gist": "carry this one",
                "source_message_ids": ["m1"],
                "turn_id": "t1",
                "salience": 0.4,
            },
            {"gist": "and this one", "source_message_ids": ["t2"], "salience": 0.6},
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 2
    contents = {p.op.draft.payload["content"] for p in puts}
    assert contents == {"carry this one", "and this one"}
    # provenance carries the source lineage
    by_content = {p.op.draft.payload["content"]: p.op.draft for p in puts}
    assert by_content["carry this one"].payload["_provenance"]["source_object_ids"] == ["m1"]
    assert by_content["carry this one"].payload["_provenance"]["turn_id"] == "t1"

    updates = [i for i in intents if isinstance(i, UpdateState)]
    assert len(updates) == 1
    assert set(updates[0].changes["noticed_source_ids"]) == {"m1", "t2"}

    # the cursor advanced through the anchor — nothing left to survey
    assert buffer.closed_segment("s1", now=NOW) == []


def test_hallucinated_source_id_is_dropped():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {
        "seeds": [
            {"gist": "grounded", "source_message_ids": ["t1"]},
            {"gist": "hallucinated", "source_message_ids": ["ghost-id-never-shown"]},
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["content"] == "grounded"


def test_already_consumed_source_id_is_dropped():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {
        "seeds": [
            {"gist": "already seen", "source_message_ids": ["t1"]},
            {"gist": "fresh", "source_message_ids": ["t2"]},
        ]
    }
    ctx = _ctx(
        correlation_id=correlation, subject_id=None, parsed=parsed, noticed_source_ids=("t1",)
    )

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["content"] == "fresh"


def test_subject_set_completion_is_a_noop():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id="thought:seed:a", parsed=parsed)

    assert list(NoticingApply(buffer).step(ctx)) == []
    # a processing-owned completion must not touch the noticing buffer at all
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]


def test_no_matching_internal_result_signal_is_a_noop():
    buffer = _seeded_buffer()
    ctx = make_tick_context(
        state=State(pending_internal_id="notice-s1@t2@x", pending_internal_subject_id=None),
        now=NOW,
        signals=[],
        trace=_TRACE,
    )
    assert list(NoticingApply(buffer).step(ctx)) == []


def test_malformed_correlation_id_is_a_noop_and_does_not_clear():
    buffer = _seeded_buffer()
    correlation = "not-a-noticing-correlation"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    assert list(NoticingApply(buffer).step(ctx)) == []
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]


def test_no_seeds_survive_still_clears_the_surveyed_cursor():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "hallucinated", "source_message_ids": ["ghost"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []
    assert [i for i in intents if isinstance(i, UpdateState)] == []
    assert buffer.closed_segment("s1", now=NOW) == []  # surveyed prefix cleared regardless
    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTHING_LINGERED.value
    assert logger.span.attrs["noticed_count"] == 0


def test_a_turn_arriving_during_the_async_gap_is_not_swept_away():
    """The anchor-scoped clear: a NEW turn (t3) completing on the SAME lane
    between the trigger's launch and this apply must survive — only the
    anchor's prefix (t1, t2) is surveyed/cleared, never a fresher recompute."""
    buffer = _seeded_buffer()
    buffer.open_pending("s1", user_text="third", now=NOW - timedelta(seconds=1))
    buffer.complete("s1", "t3", assistant_text="reply3", now=NOW)
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    list(NoticingApply(buffer).step(ctx))

    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t3"]


def test_noticed_span_fields():
    buffer = _seeded_buffer()
    correlation = f"notice-s1@t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "carried", "source_message_ids": ["t1"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    list(NoticingApply(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTICED.value
    assert logger.span.attrs["noticed_count"] == 1
    assert logger.span.attrs["source_ids"] == ["t1"]

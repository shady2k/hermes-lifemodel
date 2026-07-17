"""Unit tests for :class:`NoticingApply` (lm-705.5 combined task E4, Task 5;
claim/finalize rewire lm-705.13).

Mirrors ``tests/test_thought_processing_apply.py``'s style: a real
:class:`NoticingBuffer` (Task 2) stands in for the live conversation buffer,
seeded through its OWN public API (``open_pending``/``stamp_source``/
``complete``) rather than hand-built ``BufferEntry`` values, and an
``internal_result`` signal carries the (fake) aux call's typed result.

Since lm-705.13 the apply reads the surveyed segment via
:meth:`NoticingBuffer.claimed` (keyed by the correlation's ``survey_id``) and
advances the cursor by EMITTING :class:`FinalizeBuffer` (applied atomically with
the thought commit) rather than a direct buffer clear — so each scenario first
``claim``s the prefix a launched pass would have claimed, and asserts the emitted
``FinalizeBuffer`` (not an in-place clear).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.component import TickContext
from lifemodel.core.intents import FinalizeBuffer, PutRecord, UpdateState
from lifemodel.core.noticing import NOTICING_TOP_K, NoticingApply, NoticingReason
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_view import build_thought, encode_thought, seed_thought_id
from lifemodel.core.timeutil import to_iso
from lifemodel.domain.objects import Sensitivity, ThoughtState
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeActiveSpan, FakeClock, FakeMemoryStore, FakeSpanLogger
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


def _buffer_with_n_turns(n: int) -> NoticingBuffer:
    """A buffer with *n* closed turns on lane ``s1``, ``t0``..``t{n-1}``, each
    its own citable turn_id — for tests that need more source ids than
    :func:`_seeded_buffer`'s two."""
    buffer = NoticingBuffer()
    old = NOW - timedelta(hours=1)
    for i in range(n):
        ts = old + timedelta(seconds=2 * i)
        buffer.open_pending("s1", user_text=f"msg{i}", now=ts)
        buffer.complete("s1", f"t{i}", assistant_text=f"reply{i}", now=ts + timedelta(seconds=1))
    return buffer


def _claim(buffer: NoticingBuffer, *, anchor: str, turn_ids: tuple[str, ...]) -> tuple[str, str]:
    """Claim *turn_ids* under a deterministic survey_id anchored at *anchor* — the
    immutable snapshot the trigger leaves behind at launch time. Returns
    ``(survey_id, correlation_id)``."""
    survey_id = f"{anchor}@{to_iso(NOW)}"
    buffer.claim("s1", turn_ids, survey_id)
    return survey_id, f"notice-s1#{survey_id}"


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
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
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

    # the cursor advance is an EMITTED FinalizeBuffer (atomic with the thoughts),
    # keyed by the launch's survey_id — never an in-place clear.
    assert FinalizeBuffer(survey_id) in intents


def test_hallucinated_source_id_is_dropped():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
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
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
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
    correlation = f"notice-s1#t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id="thought:seed:a", parsed=parsed)

    assert list(NoticingApply(buffer).step(ctx)) == []
    # a processing-owned completion must not touch the noticing buffer at all
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]


def test_no_matching_internal_result_signal_is_a_noop():
    buffer = _seeded_buffer()
    ctx = make_tick_context(
        state=State(pending_internal_id="notice-s1#t2@x", pending_internal_subject_id=None),
        now=NOW,
        signals=[],
        trace=_TRACE,
    )
    assert list(NoticingApply(buffer).step(ctx)) == []


def test_malformed_correlation_id_is_a_noop_and_does_not_finalize():
    buffer = _seeded_buffer()
    correlation = "not-a-noticing-correlation"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert intents == []  # no FinalizeBuffer for an unparseable correlation
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]


def test_empty_claimed_snapshot_is_a_noop_and_does_not_finalize():
    """A parseable correlation whose survey_id names NOTHING claimed (an already
    finalized/released claim, or a duplicate completion) does no work and — crucially
    — emits no FinalizeBuffer (there is nothing claimed to finalize)."""
    buffer = _seeded_buffer()  # t1, t2 completed but NOT claimed under this survey_id
    correlation = f"notice-s1#t2@{to_iso(NOW)}"
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert intents == []
    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTHING_LINGERED.value
    # the completed turns are untouched — nothing was claimed under this survey_id.
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]


def test_no_seeds_survive_still_finalizes_the_surveyed_window():
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "hallucinated", "source_message_ids": ["ghost"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []
    assert [i for i in intents if isinstance(i, UpdateState)] == []
    # surveyed-but-fruitless still advances the cursor: a FinalizeBuffer is emitted.
    assert FinalizeBuffer(survey_id) in intents
    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTHING_LINGERED.value
    assert logger.span.attrs["noticed_count"] == 0


def test_a_turn_arriving_during_the_async_gap_is_not_swept_away():
    """The claim-scoped finalize: a NEW turn (t3) completing on the SAME lane
    between the trigger's launch (which claimed t1, t2) and this apply is NEVER in
    the claimed snapshot — only the claimed prefix is surveyed/finalized, and t3
    stays ``complete`` for a later pass."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    # t3 arrives on the same lane during the async gap — AFTER the claim.
    buffer.open_pending("s1", user_text="third", now=NOW - timedelta(seconds=1))
    buffer.complete("s1", "t3", assistant_text="reply3", now=NOW)
    parsed = {"seeds": [{"gist": "x", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    # the emitted FinalizeBuffer covers ONLY the claimed prefix (t1, t2)...
    assert FinalizeBuffer(survey_id) in intents
    assert [e.turn_id for e in buffer.claimed(survey_id)] == ["t1", "t2"]
    # ...and t3 (never claimed) is still available for a later pass.
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t3"]


def test_noticed_span_fields():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "carried", "source_message_ids": ["t1"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    list(NoticingApply(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTICED.value
    assert logger.span.attrs["noticed_count"] == 1
    assert logger.span.attrs["source_ids"] == ["t1"]


# ---- D10: the raw aux result and the model's reflection must ride the span ----


def test_seeds_completion_logs_aux_raw_and_reflection():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [{"gist": "carried", "source_message_ids": ["t1"]}],
        "reflection": "a quiet realization about t1",
    }
    ctx, logger = _ctx_with_logger(
        correlation_id=correlation, subject_id=None, parsed=parsed, raw='{"seeds": [...]}'
    )

    list(NoticingApply(buffer).step(ctx))

    assert logger.span.attrs["aux_raw"] == '{"seeds": [...]}'
    assert logger.span.attrs["reflection"] == "a quiet realization about t1"
    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTICED.value


def test_nothing_lingered_still_logs_aux_raw_and_reflection():
    """The whole point of capturing ``reflection`` (spec D10) is the empty-seeds
    case — WHY nothing lingered, not just THAT nothing did. Both ``aux_raw`` and
    ``reflection`` must land on the span even when ``seeds`` is empty."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    raw = '{"seeds": [], "reflection": "nothing worth carrying from this stretch"}'
    parsed = {"seeds": [], "reflection": "nothing worth carrying from this stretch"}
    ctx, logger = _ctx_with_logger(
        correlation_id=correlation, subject_id=None, parsed=parsed, raw=raw
    )

    list(NoticingApply(buffer).step(ctx))

    assert logger.span.attrs["noticing_reason"] == NoticingReason.NOTHING_LINGERED.value
    assert logger.span.attrs["noticed_count"] == 0
    assert logger.span.attrs["aux_raw"] == raw
    assert logger.span.attrs["reflection"] == "nothing worth carrying from this stretch"


def test_missing_reflection_key_stamps_nothing():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "carried", "source_message_ids": ["t1"]}]}
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    list(NoticingApply(buffer).step(ctx))

    assert "reflection" not in logger.span.attrs


def test_aux_raw_is_capped_at_2000_chars():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    long_raw = "x" * 2500
    parsed = {"seeds": [{"gist": "carried", "source_message_ids": ["t1"]}]}
    ctx, logger = _ctx_with_logger(
        correlation_id=correlation, subject_id=None, parsed=parsed, raw=long_raw
    )

    list(NoticingApply(buffer).step(ctx))

    assert logger.span.attrs["aux_raw"] == "x" * 2000


# ---- F1: a transient (transport/provider) failure must NOT finalize the claim ----


def test_transient_failure_releases_the_claim_for_a_retry():
    """``raw=""``/``parsed=None`` is the runner's shape for a failed/timed-out aux
    call (``adapters/internal_runner.py``) — refunded like ``ThoughtProcessingApply``'s
    transient guard (no PutRecord, no consumed-ring update, and NO FinalizeBuffer),
    but C2: the claim is now RELEASED (un-claimed back to ``complete``) so the next
    eligible tick re-surveys the segment, rather than leaving it stranded
    ``claimed`` until a global recovery."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=None, raw="")

    intents = list(NoticingApply(buffer).step(ctx))

    assert intents == []  # nothing — crucially, no FinalizeBuffer
    assert buffer.claimed(survey_id) == []  # claim RELEASED (C2), not stranded
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == [
        "t1",
        "t2",
    ]  # re-surveyable
    assert logger.span.attrs["noticing_reason"] == NoticingReason.TRANSIENT_FAILURE.value


def test_malformed_non_empty_result_releases_the_claim():
    """review-2 G1 + C2: ``raw`` is non-empty (the model DID respond) but ``parsed``
    never took the ``{"seeds": [...]}`` shape (e.g. plain prose, or truncated JSON).
    A malformed response is treated like the transient case: no FinalizeBuffer, no
    consumed-ring update, no PutRecord — and the claim is RELEASED so a later pass
    re-surveys the segment rather than it being swept away by a parse failure."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    ctx, logger = _ctx_with_logger(
        correlation_id=correlation,
        subject_id=None,
        parsed=None,
        raw="garbage that didn't parse",
    )

    intents = list(NoticingApply(buffer).step(ctx))

    assert intents == []
    assert buffer.claimed(survey_id) == []  # RELEASED (C2)
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"]
    assert logger.span.attrs["noticing_reason"] == NoticingReason.TRANSIENT_FAILURE.value
    assert logger.span.attrs["noticed_count"] == 0


def test_malformed_parsed_shapes_release_the_claim():
    """Same G1 guard, exercised over every malformed (but still ``dict | None``
    -typed, per :class:`~lifemodel.core.taxonomy.InternalResultRead`'s own
    contract) ``parsed`` shape short of a valid ``{"seeds": [...]}`` dict: a
    dict with a non-list ``seeds``, and a dict missing the ``seeds`` key
    entirely. Each RELEASES the claim (C2), never finalizes it."""
    for bad_parsed in ({"seeds": "not a list"}, {"no_seeds_key": []}):
        buffer = _seeded_buffer()
        survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
        ctx = _ctx(
            correlation_id=correlation,
            subject_id=None,
            parsed=bad_parsed,
            raw="some non-empty raw text",
        )

        intents = list(NoticingApply(buffer).step(ctx))

        assert intents == [], bad_parsed
        assert buffer.claimed(survey_id) == [], bad_parsed  # RELEASED (C2)
        assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1", "t2"], bad_parsed


def test_valid_empty_seeds_shape_still_finalizes_distinct_from_malformed():
    """The counterpoint to the malformed case above (G1): ``{"seeds": []}`` IS
    a well-formed shape (a dict with a ``seeds`` list, even if empty) -- "the
    model looked and found nothing" -- so it still finalizes, unlike a truly
    malformed response."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    ctx = _ctx(
        correlation_id=correlation,
        subject_id=None,
        parsed={"seeds": []},
        raw="{...}",
    )

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []
    assert FinalizeBuffer(survey_id) in intents


def test_genuine_empty_seeds_result_still_finalizes():
    """A REAL result (non-empty ``raw``, a parsed ``{"seeds": [...]}`` shape)
    whose ``seeds`` list is genuinely empty is "the model looked and found
    nothing" — NOT transient — so the cursor still advances (FinalizeBuffer)."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    ctx = _ctx(
        correlation_id=correlation,
        subject_id=None,
        parsed={"seeds": []},
        raw='{"seeds": []}',
    )

    intents = list(NoticingApply(buffer).step(ctx))

    assert FinalizeBuffer(survey_id) in intents


# ---- F2a: at most NOTICING_TOP_K validated seeds survive one pass ----


def test_more_than_top_k_valid_seeds_are_capped():
    buffer = _buffer_with_n_turns(10)
    _survey_id, correlation = _claim(
        buffer, anchor="t9", turn_ids=tuple(f"t{i}" for i in range(10))
    )
    parsed = {"seeds": [{"gist": f"seed {i}", "source_message_ids": [f"t{i}"]} for i in range(10)]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == NOTICING_TOP_K == 3


# ---- F3: within-batch dedup + turn_id must be in the surveyed segment ----


def test_two_seeds_citing_the_same_source_id_only_the_first_is_accepted():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {"gist": "first claim", "source_message_ids": ["t1"]},
            {"gist": "second claim, same source", "source_message_ids": ["t1"]},
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["content"] == "first claim"


def test_seed_with_turn_id_outside_the_segment_is_dropped():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {
                "gist": "ghost anchor",
                "source_message_ids": ["t1"],
                "turn_id": "t-never-in-segment",
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []


def test_seed_with_turn_id_inside_the_segment_is_kept():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "grounded anchor", "source_message_ids": ["t1"], "turn_id": "t1"}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1


# ---- G3 (review-2): per-call dedup on the scheduled thought id ----


def test_same_gist_disjoint_source_seeds_produce_exactly_one_put():
    """Two seeds with the IDENTICAL gist (-> the same content-digest thought
    id) but DISJOINT source ids both pass source-validation independently --
    neither cites an id the other already claimed, so the existing within-
    batch consumed-id dedup (F3a) does not catch this. Without G3's per-call
    ``seen_thought_ids`` guard, both would become a PutRecord for the SAME id,
    the later silently winning with different provenance."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {"gist": "same gist both times", "source_message_ids": ["t1"]},
            {"gist": "same gist both times", "source_message_ids": ["t2"]},
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["content"] == "same gist both times"
    # both source ids are still marked consumed, even though only one thought
    # was created -- neither is left re-eligible for a later pass to re-notice.
    updates = [i for i in intents if isinstance(i, UpdateState)]
    assert len(updates) == 1
    assert set(updates[0].changes["noticed_source_ids"]) == {"t1", "t2"}


# ---- F4: no terminal-thought resurrection / provenance overwrite ----


def test_seed_matching_an_existing_terminal_thought_is_not_resurrected():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    content = "already handled, long since resolved"
    parsed = {"seeds": [{"gist": content, "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    memory = FakeMemoryStore(clock=FakeClock(NOW))
    terminal = build_thought(
        id=seed_thought_id(content), content=content, state=ThoughtState.RESOLVED
    )
    memory.put(encode_thought(terminal))

    intents = list(NoticingApply(buffer, memory=memory).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []


def test_seed_with_genuinely_new_content_is_still_seeded_when_memory_is_wired():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "brand new content", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)
    memory = FakeMemoryStore(clock=FakeClock(NOW))

    intents = list(NoticingApply(buffer, memory=memory).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["content"] == "brand new content"


# ---- salience clamp ----


def test_salience_is_clamped_to_the_unit_range():
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"gist": "over the top", "source_message_ids": ["t1"], "salience": 5.0}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.salience == 1.0


# ---- belief-track v1 (lm-705.19 Task 3): a seed may be a grounded belief ----


def test_belief_seed_becomes_a_belief_with_evidence_and_confidence():
    """A ``kind:"belief"`` seed grounded in the surveyed segment becomes a
    ``Belief`` PutRecord carrying its evidence (``source_message_ids``),
    ``confidence``, and — a proposition about the person — a SENSITIVE floor;
    the surveyed window still finalizes and the source id is consumed."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "they get anxious before status loss",
                "content": "They get anxious before a loss of status.",
                "source_message_ids": ["t1"],
                "confidence": 0.75,
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    draft = puts[0].op.draft
    assert draft.kind == "belief"
    assert draft.payload["content"] == "They get anxious before a loss of status."
    assert draft.payload["source_message_ids"] == ["t1"]
    assert draft.confidence == 0.75
    assert draft.payload["_sensitivity"] == Sensitivity.SENSITIVE.value
    # provenance records the belief lineage + the "believed" reason (not "noticed")
    assert draft.payload["_provenance"]["reason"] == "believed"
    assert draft.payload["_provenance"]["source_object_ids"] == ["t1"]
    # the surveyed window still finalizes and the source is consumed for dedup
    assert FinalizeBuffer(survey_id) in intents
    updates = [i for i in intents if isinstance(i, UpdateState)]
    assert len(updates) == 1
    assert set(updates[0].changes["noticed_source_ids"]) == {"t1"}


def test_belief_seed_id_is_content_scoped_not_survey_scoped():
    """The belief id is derived from a CONSTANT source anchor + content (the
    partial dedup we can afford), so it is stable across surveys — an exact-
    duplicate belief upserts ONE row rather than a fresh duplicate per re-notice."""
    from lifemodel.core.belief_view import belief_id

    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    content = "They get anxious before a loss of status."
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "anxious",
                "content": content,
                "source_message_ids": ["t1"],
                "confidence": 0.5,
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert puts[0].op.draft.id == belief_id("noticing", content)


def test_belief_seed_with_ungrounded_source_id_is_dropped():
    """Anti-hallucination applies to belief seeds too: a belief citing a source
    id never shown to the model is dropped, no Belief created."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "ungrounded",
                "content": "A claim grounded in a turn never shown.",
                "source_message_ids": ["ghost-never-shown"],
                "confidence": 0.6,
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []
    # surveyed-but-fruitless still advances the cursor
    assert FinalizeBuffer(survey_id) in intents
    # nothing consumed (the seed never validated)
    assert [i for i in intents if isinstance(i, UpdateState)] == []


def test_belief_seed_with_bad_confidence_is_dropped_not_crashed():
    """A belief needs a numeric confidence in [0, 1]; a missing, non-numeric, or
    out-of-range confidence DROPS the seed (never a crash) — no Belief, no put."""
    bad_confidences: list[dict] = [
        {
            "kind": "belief",
            "gist": "g",
            "content": "missing confidence.",
            "source_message_ids": ["t1"],
        },
        {
            "kind": "belief",
            "gist": "g",
            "content": "out of range high.",
            "source_message_ids": ["t1"],
            "confidence": 1.5,
        },
        {
            "kind": "belief",
            "gist": "g",
            "content": "out of range low.",
            "source_message_ids": ["t1"],
            "confidence": -0.1,
        },
        {
            "kind": "belief",
            "gist": "g",
            "content": "not a number.",
            "source_message_ids": ["t1"],
            "confidence": "high",
        },
        {
            "kind": "belief",
            "gist": "g",
            "content": "a bool is not a number.",
            "source_message_ids": ["t1"],
            "confidence": True,
        },
    ]
    for bad in bad_confidences:
        buffer = _seeded_buffer()
        _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
        ctx = _ctx(correlation_id=correlation, subject_id=None, parsed={"seeds": [bad]})

        intents = list(NoticingApply(buffer).step(ctx))

        assert [i for i in intents if isinstance(i, PutRecord)] == [], bad


def test_belief_seed_missing_content_is_dropped():
    """A belief needs a proposition to hold: a ``kind:"belief"`` seed with no
    ``content`` is dropped (JSON-schema can't express conditional-required, so
    it is enforced here)."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "no content",
                "source_message_ids": ["t1"],
                "confidence": 0.7,
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    assert [i for i in intents if isinstance(i, PutRecord)] == []


def test_thought_seed_kind_still_produces_a_thought():
    """An explicit ``kind:"thought"`` seed (and, by every other test, a seed
    with NO kind) builds a Thought exactly as before — the thought path is
    untouched."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {"seeds": [{"kind": "thought", "gist": "carry this", "source_message_ids": ["t1"]}]}
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.kind == "thought"
    assert puts[0].op.draft.payload["content"] == "carry this"


def test_belief_seed_with_private_sensitivity_is_private():
    """The model may escalate a belief to PRIVATE; anything else floors to
    SENSITIVE (never below)."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "private matter",
                "content": "A private worry they confided.",
                "source_message_ids": ["t1"],
                "confidence": 0.9,
                "sensitivity": "private",
            }
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    assert puts[0].op.draft.payload["_sensitivity"] == Sensitivity.PRIVATE.value


def test_belief_and_thought_seeds_coexist_in_one_pass():
    """A single pass may carry both a thought and a belief — each routed to its
    own kind, both grounded, both consumed."""
    buffer = _seeded_buffer()
    survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    parsed = {
        "seeds": [
            {"kind": "thought", "gist": "keep chewing on this", "source_message_ids": ["t1"]},
            {
                "kind": "belief",
                "gist": "steady preference",
                "content": "They prefer async updates.",
                "source_message_ids": ["t2"],
                "confidence": 0.8,
            },
        ]
    }
    ctx = _ctx(correlation_id=correlation, subject_id=None, parsed=parsed)

    intents = list(NoticingApply(buffer).step(ctx))

    puts = [i for i in intents if isinstance(i, PutRecord)]
    kinds = {p.op.draft.kind for p in puts}
    assert kinds == {"thought", "belief"}
    assert FinalizeBuffer(survey_id) in intents
    updates = [i for i in intents if isinstance(i, UpdateState)]
    assert set(updates[0].changes["noticed_source_ids"]) == {"t1", "t2"}


def test_belief_creation_logs_redacted_metadata_not_content():
    """D10 (tightened): a created belief logs id/subject/confidence/sensitivity
    on the span — NEVER the full content string in any span field."""
    buffer = _seeded_buffer()
    _survey_id, correlation = _claim(buffer, anchor="t2", turn_ids=("t1", "t2"))
    secret = "A secret worry about their job security."
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "worry",
                "content": secret,
                "source_message_ids": ["t1"],
                "confidence": 0.8,
                "sensitivity": "private",
            }
        ]
    }
    ctx, logger = _ctx_with_logger(correlation_id=correlation, subject_id=None, parsed=parsed)

    list(NoticingApply(buffer).step(ctx))

    beliefs = logger.span.attrs["beliefs"]
    assert len(beliefs) == 1
    assert beliefs[0]["subject"] == "owner"
    assert beliefs[0]["confidence"] == 0.8
    assert beliefs[0]["sensitivity"] == Sensitivity.PRIVATE.value
    assert "id" in beliefs[0]
    # the raw content string must never ride ANY span field (redaction)
    assert all(secret not in str(v) for v in logger.span.attrs.values())

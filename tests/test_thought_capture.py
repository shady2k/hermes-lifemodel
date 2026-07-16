"""``ThoughtCapture`` — seed signal to ``PutRecord(thought)``, idempotent (lm-705.1 Task 3)."""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.intents import PutRecord
from lifemodel.core.taxonomy import thought_seed_signal
from lifemodel.core.thought_capture import ThoughtCapture
from lifemodel.core.thought_view import live_thoughts, seed_thought_id
from lifemodel.domain.objects import ThoughtState
from lifemodel.ports.tracer import TraceContext
from lifemodel.testing.fakes import FakeClock, FakeMemoryStore
from lifemodel.testing.tick import make_tick_context


def test_capture_emits_one_put_for_a_seed():
    content = "the owner said: dentist on Friday"
    ctx = make_tick_context(
        signals=[
            thought_seed_signal(
                origin_id="s1",
                content=content,
                salience=0.5,
                timestamp="2026-07-16T00:00:00+00:00",
            )
        ],
    )
    intents = list(ThoughtCapture().step(ctx))
    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    draft = puts[0].op.draft
    assert draft.kind == "thought"
    assert draft.id == seed_thought_id(content)  # deterministic / idempotent
    assert draft.state == ThoughtState.ACTIVE.value
    assert draft.payload["content"] == content
    assert draft.payload["trigger"] == "event"


def test_capture_is_idempotent_on_identical_content():
    content = "the owner said: interview next week"
    sig = thought_seed_signal(
        origin_id="s2", content=content, salience=0.5, timestamp="2026-07-16T00:00:00+00:00"
    )
    ctx = make_tick_context(signals=[sig, sig])  # same content twice this frame
    puts = [i for i in ThoughtCapture().step(ctx) if isinstance(i, PutRecord)]
    assert {p.op.draft.id for p in puts} == {seed_thought_id(content)}  # one id, not two


def test_no_seed_no_put():
    ctx = make_tick_context(signals=[])
    assert list(ThoughtCapture().step(ctx)) == []


def test_reseed_of_same_content_preserves_original_provenance():
    """A same-episode upsert (host post_llm retry, or the same content recurring)
    must PRESERVE the first row's birth lineage (core/trace.py:8-16), never mint a
    fresh ``creation_provenance`` over it — mirrors how ``CognitionLauncher``
    preserves ``live_contact_intention(...).provenance`` (core/cognition.py)."""
    content = "the owner said: same content, twice"
    birth_trace = TraceContext(trace_id="1" * 32, span_id="2" * 16)
    retry_trace = TraceContext(trace_id="3" * 32, span_id="4" * 16)
    sig = thought_seed_signal(
        origin_id="s1", content=content, salience=0.5, timestamp="2026-07-16T00:00:00+00:00"
    )
    store = FakeMemoryStore(clock=FakeClock(datetime(2026, 1, 1, tzinfo=UTC)))

    # First capture: no live row yet → mints fresh provenance under birth_trace.
    ctx1 = make_tick_context(signals=[sig], trace=birth_trace)
    puts1 = [i for i in ThoughtCapture().step(ctx1) if isinstance(i, PutRecord)]
    assert len(puts1) == 1
    store.put(puts1[0].op.draft)
    thought_id = seed_thought_id(content)
    first_record = store.get("thought", thought_id)
    assert first_record is not None

    # Re-seed of the SAME content, snapshotted into ctx.objects, under a DIFFERENT
    # trace (standing in for a host retry / a recurring exchange in a later tick).
    ctx2 = make_tick_context(signals=[sig], objects=[first_record], trace=retry_trace)
    puts2 = [i for i in ThoughtCapture().step(ctx2) if isinstance(i, PutRecord)]
    assert len(puts2) == 1
    store.put(puts2[0].op.draft)
    second_record = store.get("thought", thought_id)
    assert second_record is not None

    first_provenance = live_thoughts([first_record])[0].provenance
    second_provenance = live_thoughts([second_record])[0].provenance
    assert first_provenance is not None
    assert second_provenance == first_provenance  # birth lineage untouched
    assert second_provenance.trace_id == birth_trace.trace_id  # NOT retry_trace's

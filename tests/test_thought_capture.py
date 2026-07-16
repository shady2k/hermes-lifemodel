"""``ThoughtCapture`` — seed signal to ``PutRecord(thought)``, idempotent (lm-705.1 Task 3)."""

from __future__ import annotations

from lifemodel.core.intents import PutRecord
from lifemodel.core.taxonomy import thought_seed_signal
from lifemodel.core.thought_capture import ThoughtCapture
from lifemodel.core.thought_view import seed_thought_id
from lifemodel.domain.objects import ThoughtState
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

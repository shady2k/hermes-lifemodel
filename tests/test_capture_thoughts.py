"""CoreLoop.capture_thoughts — the restricted capture entrypoint (lm-705.11 Task 3)."""

from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import thought_seed_signal
from lifemodel.testing.harness import build_capture_harness


def _seed(content: str, producer: str = "create-thought-tool"):
    return thought_seed_signal(
        origin_id=f"o-{content}",
        content=content,
        salience=0.5,
        producer=producer,
        timestamp="2026-07-18T00:00:00+00:00",
    )


def test_capture_creates_thoughts_and_touches_nothing_else() -> None:
    h = build_capture_harness()
    before = h.state_store.load()
    result = h.coreloop.capture_thoughts([_seed("alpha"), _seed("beta")])
    after = h.state_store.load()
    # thoughts created …
    assert result.accepted == 2 and result.deduped == 0
    assert {r.id for r in h.memory.find(state="active", limit=50) if r.kind == "thought"} == set(
        result.thought_ids
    )
    # … and NOTHING else moved: no tick advance, no state field changed, no launch produced.
    assert after.tick_count == before.tick_count
    assert after == before
    assert h.egress.sent == []


def test_capture_dedups_and_does_not_resurrect_terminal() -> None:
    h = build_capture_harness()
    h.coreloop.capture_thoughts([_seed("alpha")])
    (tid,) = [r.id for r in h.memory.find(state="active", limit=50) if r.kind == "thought"]
    h.memory.transition("thought", tid, "active", "resolved")  # terminate it
    result = h.coreloop.capture_thoughts([_seed("alpha")])  # same content
    assert result.deduped == 1 and result.accepted == 1
    assert h.memory.get("thought", tid).state == "resolved"  # NOT resurrected


def test_capture_fails_closed_on_non_put_intent() -> None:
    h = build_capture_harness()

    class _EmitsTransition:
        id = "thought-capture"

        def step(self, ctx):
            from lifemodel.core.intents import TransitionRecord
            from lifemodel.domain.memory import TransitionOp

            yield TransitionRecord(
                op=TransitionOp(kind="thought", id="t", from_state="active", to_state="resolved")
            )
            return

    h.coreloop._capture_component = _EmitsTransition()
    with pytest.raises(AssertionError):
        h.coreloop.capture_thoughts([_seed("alpha")])

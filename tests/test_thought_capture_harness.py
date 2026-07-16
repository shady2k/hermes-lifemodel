"""Real-code sim: exchange to thought row, idempotent, idle stays zero (lm-705.1 Task 6)."""

from __future__ import annotations

from lifemodel.core.appraisal import ThoughtSeed
from lifemodel.core.thought_view import read_live_thoughts, seed_thought_id
from lifemodel.hooks import make_post_llm_observer
from lifemodel.testing.appraisal import FakeAppraiser
from lifemodel.testing.harness import build_capture_lifemodel


def test_exchange_to_thought_row_end_to_end():
    lm = build_capture_lifemodel()
    content = "the owner said: interview on Monday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.7))
    )
    observer(user_message="big interview on Monday, nervous", assistant_response="You'll do great.")
    live = read_live_thoughts(lm.state)
    assert [t.content for t in live] == [content]
    assert live[0].salience == 0.7


def test_retry_of_same_exchange_upserts_one_row():
    lm = build_capture_lifemodel()
    content = "the owner said: interview on Monday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.7))
    )
    observer(user_message="big interview on Monday", assistant_response="You'll do great.")
    observer(user_message="big interview on Monday", assistant_response="You'll do great.")  # retry
    live = read_live_thoughts(lm.state)
    assert len(live) == 1
    assert live[0].id == seed_thought_id(content)


def test_idle_heartbeat_is_still_zero_capture():
    from lifemodel.core.frame import FrameTrigger, run_frame

    lm = build_capture_lifemodel()
    run_frame(lm.coreloop, [], trigger=FrameTrigger.HEARTBEAT)  # empty world
    assert read_live_thoughts(lm.state) == ()

"""The ``post_llm`` appraisal seam — a completed exchange seeds a thought (lm-705.1 Task 4)."""

from __future__ import annotations

from lifemodel.core.appraisal import ThoughtSeed
from lifemodel.core.thought_view import read_live_thoughts, seed_thought_id
from lifemodel.hooks import make_post_llm_observer
from lifemodel.testing.appraisal import FakeAppraiser
from lifemodel.testing.harness import build_capture_lifemodel


def test_reactive_exchange_captures_a_thought():
    lm = build_capture_lifemodel()  # real CoreLoop + ThoughtCapture registered
    content = "the owner said: trip on Friday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.5))
    )
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    thoughts = read_live_thoughts(lm.state)  # lm.state is also a MemoryPort
    assert [t.id for t in thoughts] == [seed_thought_id(content)]


def test_declining_appraiser_captures_nothing():
    lm = build_capture_lifemodel()
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(None))
    observer(user_message="ok thanks", assistant_response="anytime")
    assert read_live_thoughts(lm.state) == ()


def test_no_appraiser_is_a_noop():  # back-compat: existing wiring unaffected
    lm = build_capture_lifemodel()
    observer = make_post_llm_observer(lambda: lm)  # appraiser omitted
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    assert read_live_thoughts(lm.state) == ()

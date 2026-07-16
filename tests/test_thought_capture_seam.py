"""The ``post_llm`` appraisal seam — a completed exchange seeds a thought (lm-705.1 Task 4)."""

from __future__ import annotations

from pathlib import Path

from lifemodel.core.appraisal import ThoughtSeed
from lifemodel.core.thought_view import read_live_thoughts, seed_thought_id
from lifemodel.core.wake_packet import DECLINE_MARKER, IMPULSE_LABEL_PREFIX
from lifemodel.hooks import make_post_llm_observer
from lifemodel.testing.appraisal import FakeAppraiser
from lifemodel.testing.harness import build_capture_lifemodel


def test_reactive_exchange_captures_a_thought(tmp_path: Path):
    lm = build_capture_lifemodel(base_dir=tmp_path)  # real CoreLoop + ThoughtCapture registered
    content = "the owner said: trip on Friday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.5))
    )
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    thoughts = read_live_thoughts(lm.state)  # lm.state is also a MemoryPort
    assert [t.id for t in thoughts] == [seed_thought_id(content)]


def test_declining_appraiser_captures_nothing(tmp_path: Path):
    lm = build_capture_lifemodel(base_dir=tmp_path)
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(None))
    observer(user_message="ok thanks", assistant_response="anytime")
    assert read_live_thoughts(lm.state) == ()


def test_no_appraiser_is_a_noop(tmp_path: Path):  # back-compat: existing wiring unaffected
    lm = build_capture_lifemodel(base_dir=tmp_path)
    observer = make_post_llm_observer(lambda: lm)  # appraiser omitted
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    assert read_live_thoughts(lm.state) == ()


#: A seed that WOULD be captured if the guard under test did not fire — every
#: guard test below proves the *guard*, not the appraiser (it is fed a seed that
#: is trivially eligible).
_WOULD_CAPTURE = ThoughtSeed(content="the owner said: this would be captured", salience=0.9)


def test_control_command_message_captures_nothing(tmp_path: Path):
    """A slash/control command is not dialogue (sensor band-pass, spec §4) — it
    must never reach the appraiser as a genuine exchange."""
    lm = build_capture_lifemodel(base_dir=tmp_path)
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(_WOULD_CAPTURE))
    observer(user_message="/lifemodel force-wake", assistant_response="Woke.")
    assert read_live_thoughts(lm.state) == ()


def test_own_impulse_message_captures_nothing(tmp_path: Path):
    """Our own composed proactive impulse (``_is_own_impulse``, keyed on
    ``IMPULSE_LABEL_PREFIX``) is not a genuine owner exchange — capturing it would
    be the being appraising its own voice as if the owner had said it."""
    lm = build_capture_lifemodel(base_dir=tmp_path)
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(_WOULD_CAPTURE))
    observer(
        user_message=f"{IMPULSE_LABEL_PREFIX}I've been thinking about reaching out.",
        assistant_response="Hey, just checking in.",
    )
    assert read_live_thoughts(lm.state) == ()


def test_no_reply_response_captures_nothing(tmp_path: Path):
    """A ``[SILENT]``/``NO_REPLY`` assistant response is a declined turn, not a
    genuine exchange — it must not be appraised into a thought."""
    lm = build_capture_lifemodel(base_dir=tmp_path)
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(_WOULD_CAPTURE))
    observer(user_message="how's it going?", assistant_response=DECLINE_MARKER)
    assert read_live_thoughts(lm.state) == ()

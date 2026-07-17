"""The noticing-buffer seam — pre_llm opens, post_llm completes (lm-705.5 Task 3/E3).

Wires :class:`~lifemodel.core.noticing_buffer.NoticingBuffer` (Task 2) into the two
live-turn hooks: ``make_felt_state_injector`` (``pre_llm_call``, the OPEN side) and
``make_post_llm_observer`` (``post_llm_call``, the CLOSE side). Both take ``buffer``
as an OPTIONAL kwarg (``None`` default) so every existing caller/test is unaffected.

This slice's source pointer is ``turn_id`` (spec §8's own lean), not the platform
message id — so ``NoticingBuffer.stamp_source``/the inbound observer stay
deliberately unwired for now (deferred to a later slice, if a deep-link ever needs
it). Only the two seams below are exercised here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.objects import DesireState
from lifemodel.hooks import make_felt_state_injector, make_post_llm_observer
from lifemodel.ports.memory import MemoryPort
from lifemodel.state.model import State
from lifemodel.testing import FakeClock
from lifemodel.testing.harness import build_capture_lifemodel

_NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

#: A being that has been BORN, with a live pending PROACTIVE desire — the precondition
#: for a genuine async read-back (mirrors ``tests/test_hooks.py``'s ``_lm_with_pending``).
_BORN = "2026-07-01T10:00:00+00:00"


def _lm_with_pending_proactive(tmp_path: Path, clock: FakeClock) -> object:
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.state.commit(
        State(
            genesis_completed_at=_BORN,
            pending_proactive_id="p-1",
            last_tick_at=clock.now().isoformat(),
        )
    )
    memory = lm.state
    assert isinstance(memory, MemoryPort)
    memory.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))
    return lm


def test_pre_llm_open_then_post_llm_complete_closes_a_segment(tmp_path: Path) -> None:
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    injector(session_id="s1", user_message="hello there", conversation_history=[])
    observer(
        session_id="s1",
        turn_id="t1",
        user_message="hello there",
        assistant_response="hi! good to hear from you",
    )

    segment = buffer.closed_segment("s1", now=_NOW)
    assert [e.turn_id for e in segment] == ["t1"]
    assert segment[0].session_id == "s1"
    assert segment[0].user_text == "hello there"
    assert segment[0].assistant_text == "hi! good to hear from you"


def test_pending_proactive_readback_does_not_complete_the_reactive_entry(tmp_path: Path) -> None:
    clock = FakeClock(_NOW)
    lm = _lm_with_pending_proactive(tmp_path, clock)
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    # A normal inbound opened this session's pending slot...
    injector(session_id="s1", user_message="are you there?", conversation_history=[])
    # ...but the turn that actually completes on post_llm is OUR OWN async read-back
    # (the pending proactive desire resolving), not a reactive exchange.
    observer(
        session_id="s1",
        turn_id="ignored",
        user_message=f"{IMPULSE_LABEL_PREFIX} reaching out",
        assistant_response="Hey, just checking in!",
    )
    # The closed-prefix rule blocks the segment: the ORIGINAL pending is still open —
    # proof the read-back never completed it.
    assert buffer.closed_segment("s1", now=_NOW) == []

    # A genuine follow-up reactive turn completes the untouched original pending.
    observer(
        session_id="s1",
        turn_id="t-real",
        user_message="are you there?",
        assistant_response="yes, I'm here!",
    )
    segment = buffer.closed_segment("s1", now=_NOW)
    assert [e.turn_id for e in segment] == ["t-real"]
    assert segment[0].user_text == "are you there?"


def test_control_command_and_own_impulse_are_not_captured(tmp_path: Path) -> None:
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    injector(session_id="s1", user_message="/lifemodel force-wake", conversation_history=[])
    observer(
        session_id="s1",
        turn_id="t1",
        user_message="/lifemodel force-wake",
        assistant_response="Woke.",
    )
    assert buffer.closed_segment("s1", now=_NOW) == []

    injector(
        session_id="s1", user_message=f"{IMPULSE_LABEL_PREFIX} musing", conversation_history=[]
    )
    observer(
        session_id="s1",
        turn_id="t2",
        user_message=f"{IMPULSE_LABEL_PREFIX} musing",
        assistant_response="Hey!",
    )
    assert buffer.closed_segment("s1", now=_NOW) == []


def test_control_command_turn_does_not_block_an_already_closed_segment(tmp_path: Path) -> None:
    # M1: a genuine turn closes the lane normally...
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    injector(session_id="s1", user_message="hello there", conversation_history=[])
    observer(session_id="s1", turn_id="t1", user_message="hello there", assistant_response="hi!")
    assert [e.turn_id for e in buffer.closed_segment("s1", now=_NOW)] == ["t1"]

    # ...then a control command arrives on the SAME lane. Its post_llm never
    # completes it (control commands are filtered at the CLOSE side already),
    # so if the OPEN side is left ungated, this control-command turn re-opens
    # (and re-blocks) the lane until pending_ttl — the already-closed t1
    # segment would wrongly read as [] until then.
    injector(session_id="s1", user_message="/lifemodel force-wake", conversation_history=[])

    assert [e.turn_id for e in buffer.closed_segment("s1", now=_NOW)] == ["t1"]


def test_own_impulse_turn_does_not_block_an_already_closed_segment(tmp_path: Path) -> None:
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    injector(session_id="s1", user_message="hello there", conversation_history=[])
    observer(session_id="s1", turn_id="t1", user_message="hello there", assistant_response="hi!")
    assert [e.turn_id for e in buffer.closed_segment("s1", now=_NOW)] == ["t1"]

    injector(
        session_id="s1", user_message=f"{IMPULSE_LABEL_PREFIX} musing", conversation_history=[]
    )

    assert [e.turn_id for e in buffer.closed_segment("s1", now=_NOW)] == ["t1"]


def test_empty_final_response_does_not_complete_the_buffer_entry(tmp_path: Path) -> None:
    # M2: an empty assistant_response has nothing for a later noticing pass to
    # read; _is_genuine_reactive_exchange alone does not reject it (only the
    # explicit [SILENT]/NO_REPLY markers count), so this needs its own guard.
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    buffer = NoticingBuffer()
    injector = make_felt_state_injector(lambda: lm, buffer=buffer)
    observer = make_post_llm_observer(lambda: lm, buffer=buffer)

    injector(session_id="s1", user_message="hello there", conversation_history=[])
    observer(session_id="s1", turn_id="t1", user_message="hello there", assistant_response="")

    assert buffer.closed_segment("s1", now=_NOW) == []


def test_buffer_none_is_a_noop_back_compat(tmp_path: Path) -> None:  # existing wiring unaffected
    lm = build_capture_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    injector = make_felt_state_injector(lambda: lm)  # buffer omitted
    observer = make_post_llm_observer(lambda: lm)  # buffer omitted

    # Neither raises, and both accept the new session_id/turn_id kwargs harmlessly —
    # exactly like every pre-E3 caller that never passed them at all.
    injector(session_id="s1", user_message="hello", conversation_history=[])
    observer(session_id="s1", turn_id="t1", user_message="hello", assistant_response="hi!")

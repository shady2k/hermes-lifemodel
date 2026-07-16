"""Proves :func:`lifemodel.core.internal_cognition.run_internal_completion`
FORWARDS its ``voice`` param to :func:`~lifemodel.core.proactive.dispatch_launches`
(lm-705.2, Task 6 — "first live emitter" prereq #1, birth-voice).

An unborn being's INCIDENTAL proactive launch on a completion frame (the
strand fix, codex #2 — some OTHER already-registered component, not this
seam, surfaces the ``LaunchProactive``) must go through the SAME birth
pre-flight a native ``proactive_tick`` launch does — otherwise the
internal-cognition seam becomes a side door around ADR-0002/lm-4fv.4's voice
gate. The behavioral EFFECT of the gate (suppressing delivery for an unborn
being mid-conversation) is already covered by
``tests/test_core_proactive.py``/``tests/test_birth_voice.py``; this test only
proves the WIRING — that whatever ``voice`` is handed to
``run_internal_completion`` is the exact object ``dispatch_launches`` receives.
"""

from __future__ import annotations

from collections.abc import Mapping

from lifemodel.composition import build_lifemodel
from lifemodel.core.internal_cognition import NullInternalApply, run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.session import BirthVoice

TARGET: Mapping[str, str | None] = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class FakeEgress:
    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        raise AssertionError("must not be reached — no launch on this frame")


def _sentinel_voice() -> BirthVoice:
    return BirthVoice.READY


def test_run_internal_completion_forwards_voice_to_dispatch_launches(tmp_path, monkeypatch) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    seen: dict[str, object] = {}

    def _recorder(lm_arg, report, egress, target, *, voice=None):
        seen["voice"] = voice
        return None

    monkeypatch.setattr("lifemodel.core.internal_cognition.dispatch_launches", _recorder)

    outcome = run_internal_completion(
        lm,
        FakeEgress(),
        TARGET,
        correlation_id="c-1",
        result=InternalCognitionResult(raw="", parsed=None),
        apply=NullInternalApply(),
        voice=_sentinel_voice,
    )

    assert outcome is None
    assert seen["voice"] is _sentinel_voice


def test_run_internal_completion_defaults_voice_to_none(tmp_path, monkeypatch) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    seen: dict[str, object] = {}

    def _recorder(lm_arg, report, egress, target, *, voice=None):
        seen["voice"] = voice
        return None

    monkeypatch.setattr("lifemodel.core.internal_cognition.dispatch_launches", _recorder)

    run_internal_completion(
        lm,
        FakeEgress(),
        TARGET,
        correlation_id="c-2",
        result=InternalCognitionResult(raw="", parsed=None),
        apply=NullInternalApply(),
    )

    assert seen["voice"] is None

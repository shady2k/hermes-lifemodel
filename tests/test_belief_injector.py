"""Tests for the belief-surfacing ``pre_llm_call`` injector (lm-705.19 Task 4).

``make_belief_injector`` is the third ``pre_llm_call`` hook (beside the felt-state and
genesis injectors): once per turn it reads the being's live, high-confidence, non-private
held beliefs (:func:`~lifemodel.core.belief_view.read_active_beliefs`), drops any it has
surfaced recently (the ``surfaced_belief_ids`` cooldown ring), composes a compact
first-person, FALLIBLE-framed block and returns it as ``{"context": …}`` — ephemeral, glued
onto a copy of the user message for one call, never persisted. It stamps the surfaced ids
atomically (never a stale full-State commit) and is fail-soft (a throw → recorded + ``None``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.belief_view import belief_id, build_belief, encode_belief
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import OBSERVER_ERRORS, register_universal_metrics
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.hooks import DEFAULT_BELIEF_INJECT_PARAMS, make_belief_injector
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State
from lifemodel.testing import FakeClock

_NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _registry() -> MetricRegistry:
    reg = MetricRegistry()
    register_universal_metrics(reg)
    return reg


def _lm(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _put_belief(
    store,
    thought: str,
    content: str,
    *,
    confidence: float,
    sensitivity: Sensitivity = Sensitivity.SENSITIVE,
) -> str:
    bid = belief_id(thought, content)
    belief = build_belief(
        id=bid,
        content=content,
        confidence=confidence,
        sensitivity=sensitivity,
        source_thought_ids=(thought,),
    )
    store.put(encode_belief(belief))
    return bid


_CONFIDENT = "They brace for status loss before big meetings."
_TENTATIVE = "They probably like tea."


def test_defaults_are_a_small_frozen_dataclass() -> None:
    assert DEFAULT_BELIEF_INJECT_PARAMS.min_confidence == 0.6
    assert DEFAULT_BELIEF_INJECT_PARAMS.top_n == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_BELIEF_INJECT_PARAMS.top_n = 5  # type: ignore[misc]  # frozen


def test_surfaces_confident_belief_with_fallible_framing(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put_belief(lm.state, "t1", _CONFIDENT, confidence=0.8)
    _put_belief(lm.state, "t2", _TENTATIVE, confidence=0.4)  # below θ=0.6 → filtered
    injector = make_belief_injector(lambda: _lm(tmp_path))

    result = injector(session_id="s", user_message="hi")

    assert isinstance(result, dict)
    ctx = result["context"]
    assert "I could be wrong" in ctx  # the fallibility marker (D framing)
    assert _CONFIDENT in ctx
    assert _TENTATIVE not in ctx  # the 0.4 belief never surfaces


def test_private_belief_is_never_surfaced(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put_belief(lm.state, "t1", "A private read.", confidence=0.95, sensitivity=Sensitivity.PRIVATE)
    injector = make_belief_injector(lambda: _lm(tmp_path))

    assert injector(session_id="s", user_message="hi") is None


def test_cooldown_prevents_resurfacing_on_a_second_immediate_call(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    bid = _put_belief(lm.state, "t1", _CONFIDENT, confidence=0.8)
    injector = make_belief_injector(lambda: _lm(tmp_path))

    first = injector(session_id="s", user_message="hi")
    assert first is not None and _CONFIDENT in first["context"]
    # the surfaced id is stamped atomically into the cooldown ring
    assert bid in lm.state.load().surfaced_belief_ids
    # a second immediate call does NOT re-surface it; nothing else qualifies → None
    assert injector(session_id="s", user_message="hi") is None


def test_no_active_beliefs_returns_none(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    injector = make_belief_injector(lambda: _lm(tmp_path))

    assert injector(session_id="s", user_message="hi") is None


def test_block_is_ephemeral_not_persisted(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put_belief(lm.state, "t1", _CONFIDENT, confidence=0.8)
    injector = make_belief_injector(lambda: _lm(tmp_path))

    result = injector(session_id="s", user_message="hi")
    assert result is not None and _CONFIDENT in result["context"]
    # ONLY the belief id lands (in the cooldown ring); the composed block/content is not persisted
    persisted = json.dumps(lm.state.load().to_dict())
    assert _CONFIDENT not in persisted
    assert "I could be wrong" not in persisted


def test_raising_read_is_fail_soft_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import lifemodel.hooks as hooks_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("read blew up")

    monkeypatch.setattr(hooks_mod, "read_active_beliefs", _boom)
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put_belief(lm.state, "t1", _CONFIDENT, confidence=0.8)
    health = BrainHealth(tmp_path)
    reg = _registry()
    injector = make_belief_injector(lambda: _lm(tmp_path), health=health, metrics=reg)

    with caplog.at_level(logging.DEBUG):
        assert injector(session_id="s", user_message="hi") is None  # never raises

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert health.last_observer_error.get("pre_llm_call") is not None
    assert reg.get(OBSERVER_ERRORS).value(component="pre_llm_call") == 1.0

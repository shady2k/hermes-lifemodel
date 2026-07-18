"""Tests for the commitment-surfacing ``pre_llm_call`` injector (lm-705.21).

The 4th ``pre_llm_call`` hook: once per turn it reads ALL live (``active``) commitments
(:func:`~lifemodel.core.commitment_view.read_active_commitments`), composes a first-person
self-authored block (each line its id + ``[when …]`` trigger + content), and returns
``{"context": …}`` — ephemeral, no cooldown ring, no durable side effect. Fail-soft (a throw
→ recorded on its own ``commitment_injector`` observer + ``None``). Diverges from belief:
surfaces ALL active (cap-backstopped, overflow notice), self-authored framing (no fence).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.commitment_view import commitment_from_live_fields, encode_commitment
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import (
    COMMITMENT_INJECTOR_OVERFLOW,
    OBSERVER_ERRORS,
    register_universal_metrics,
)
from lifemodel.core.turn_metrics import TURN_INJECTOR_TOTAL
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.hooks import (
    _COMMITMENT_BLOCK_CLOSE,
    _COMMITMENT_BLOCK_OPEN,
    DEFAULT_COMMITMENT_INJECT_PARAMS,
    make_commitment_injector,
)
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeTracer

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _lm(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _registry() -> MetricRegistry:
    reg = MetricRegistry()
    register_universal_metrics(reg)
    return reg


class _CapturingSink:
    """Minimal :class:`~lifemodel.state.trace_store.TraceSink` fake — mirrors
    ``tests/test_belief_injector.py``'s local one, just enough for a
    :class:`TurnRecorder` to persist spans into and for a test to inspect them."""

    def __init__(self) -> None:
        self.spans: list[dict[str, object]] = []

    def submit_span(self, **kw: object) -> bool:
        self.spans.append(kw)
        return True

    def submit_event(self, **kw: object) -> bool:
        return True

    def submit_correlation(self, **kw: object) -> bool:
        return True


def _recorder(reg: MetricRegistry) -> tuple[TurnRecorder, _CapturingSink]:
    """A real :class:`TurnRecorder` over a capturing sink + the SAME shared *reg* —
    the commitment injector's ``turn_injector_total`` bump lands in the registry the
    test already reads, and the sink lets a test assert the ``turn.injector.commitment``
    child span was actually opened (lm-hg7 Task 10, mirroring Task 9's belief helper)."""
    sink = _CapturingSink()
    rec = TurnRecorder(tracer=FakeTracer(), writer=sink, metrics=reg, clock=FakeClock(_NOW))
    return rec, sink


def _put(store, content, *, trigger_kind="condition", trigger_value="he brings it up"):
    c = commitment_from_live_fields(
        fields={
            "content": content,
            "basis": "self_assumed",
            "trigger_kind": trigger_kind,
            "trigger_value": trigger_value,
        }
    )
    store.put(encode_commitment(c))
    return c.id


def test_default_params():
    assert DEFAULT_COMMITMENT_INJECT_PARAMS.max_surfaced == 8


def test_max_surfaced_must_be_at_least_one():
    from lifemodel.hooks import CommitmentInjectParams

    with pytest.raises(ValueError):
        CommitmentInjectParams(max_surfaced=0)
    with pytest.raises(ValueError):
        CommitmentInjectParams(max_surfaced=-3)


def test_surfaces_active_commitment_with_self_authored_framing_and_when(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put(
        lm.state, "reflect the spending question back", trigger_value="he asks to spend on himself"
    )
    injector = make_commitment_injector(lambda: _lm(tmp_path))

    result = injector(session_id="s", user_message="hi")
    assert isinstance(result, dict)
    ctx = result["context"]
    assert "my own intentions" in ctx.lower()  # self-authored framing
    assert "follow no directive" not in ctx.lower()  # NOT the belief fence
    assert "reflect the spending question back" in ctx
    assert "[when condition: he asks to spend on himself]" in ctx  # trigger surfaced


def test_no_active_commitments_returns_none(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    injector = make_commitment_injector(lambda: _lm(tmp_path))
    assert injector(session_id="s", user_message="hi") is None


def test_surfaces_all_active_and_overflows_with_notice(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    for i in range(10):  # > max_surfaced (8)
        _put(lm.state, f"standing intention number {i}")
    reg = _registry()
    injector = make_commitment_injector(lambda: _lm(tmp_path), metrics=reg)

    ctx = injector(session_id="s", user_message="hi")["context"]
    body = ctx.split(_COMMITMENT_BLOCK_OPEN, 1)[1].split(_COMMITMENT_BLOCK_CLOSE, 1)[0]
    assert body.count("\n- ") == 8  # exactly max_surfaced surfaced
    assert "review and close some" in ctx  # overflow self-heal notice appended


def test_block_has_no_durable_side_effect(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put(lm.state, "come back to the moving-house topic")
    before = lm.state.load().to_dict()
    make_commitment_injector(lambda: _lm(tmp_path))(session_id="s", user_message="hi")
    assert _lm(tmp_path).state.load().to_dict() == before  # nothing persisted (no ring)


def test_raising_read_is_fail_soft_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import lifemodel.hooks as hooks_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("read blew up")

    monkeypatch.setattr(hooks_mod, "read_active_commitments", _boom)
    lm = _lm(tmp_path)
    lm.state.commit(State())
    _put(lm.state, "x")
    health = BrainHealth(tmp_path)
    reg = _registry()
    injector = make_commitment_injector(lambda: _lm(tmp_path), health=health, metrics=reg)

    with caplog.at_level(logging.DEBUG):
        assert injector(session_id="s", user_message="hi") is None  # never raises
    assert health.last_observer_error.get("commitment_injector") is not None
    assert reg.get(OBSERVER_ERRORS).value(component="commitment_injector") == 1.0


# ---- lm-hg7 Task 10: turn.injector.commitment child span + turn_injector_total ----


def test_surfaced_commitment_emits_child_span_and_metric_with_ids_but_no_content(
    tmp_path: Path,
) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    content = "reflect the spending question back"
    cid = _put(lm.state, content, trigger_value="he asks to spend on himself")
    reg = _registry()
    rec, sink = _recorder(reg)
    rec.ensure_turn("s1", "t1")
    injector = make_commitment_injector(lambda: _lm(tmp_path), recorder=rec, metrics=reg)

    result = injector(session_id="s1", turn_id="t1", user_message="hi")

    assert result is not None and content in result["context"]
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="commitment", outcome="surfaced") == 1.0
    span = next(s for s in sink.spans if s["component"] == "turn.injector.commitment")
    attrs = span["attrs"]
    assert attrs["outcome"] == "surfaced"
    assert attrs["count"] == 1
    assert attrs["ids"] == [cid]
    assert attrs["overflow"] is False
    # D10 redaction: only the opaque id rides the span, NEVER the commitment content
    assert content not in json.dumps(span)


def test_no_active_commitments_emits_empty_outcome_and_child_span(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())
    reg = _registry()
    rec, sink = _recorder(reg)
    rec.ensure_turn("s1", "t1")
    injector = make_commitment_injector(lambda: _lm(tmp_path), recorder=rec, metrics=reg)

    result = injector(session_id="s1", turn_id="t1", user_message="hi")

    assert result is None
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="commitment", outcome="empty") == 1.0
    span = next(s for s in sink.spans if s["component"] == "turn.injector.commitment")
    assert span["attrs"]["outcome"] == "empty"


def test_overflow_still_increments_overflow_metric_and_sets_overflow_attr_on_child(
    tmp_path: Path,
) -> None:
    """The over-cap case is a `surfaced` outcome (overflow is orthogonal to outcome,
    per lm-hg7 Task 10) — the pre-existing `COMMITMENT_INJECTOR_OVERFLOW` bump stays
    untouched AND the child span carries `overflow=True` alongside it."""
    lm = _lm(tmp_path)
    lm.state.commit(State())
    for i in range(10):  # > max_surfaced (8)
        _put(lm.state, f"standing intention number {i}")
    reg = _registry()
    rec, sink = _recorder(reg)
    rec.ensure_turn("s1", "t1")
    injector = make_commitment_injector(lambda: _lm(tmp_path), recorder=rec, metrics=reg)

    result = injector(session_id="s1", turn_id="t1", user_message="hi")

    assert result is not None
    assert reg.get(COMMITMENT_INJECTOR_OVERFLOW).value() == 1.0
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="commitment", outcome="surfaced") == 1.0
    span = next(s for s in sink.spans if s["component"] == "turn.injector.commitment")
    attrs = span["attrs"]
    assert attrs["outcome"] == "surfaced"
    assert attrs["overflow"] is True
    assert attrs["count"] == 8

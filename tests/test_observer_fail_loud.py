"""Observer bodies are plugin-owned fail-loud (spec §4.3/MAJOR-4).

The afferent hooks (``post_llm_call`` / ``pre_gateway_dispatch``) build a fresh
LifeModel and run a frame. A throw inside that body must NOT rely on Hermes' hook
wrapper for observability: the plugin itself logs ERROR + traceback, records the
per-observer error on :class:`BrainHealth`, bumps the failure metric, and never
crashes the caller. Deterministic — a ``build_lm`` that raises, real BrainHealth +
MetricRegistry, no Hermes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import OBSERVER_ERRORS, register_universal_metrics
from lifemodel.hooks import make_inbound_observer, make_post_llm_observer
from lifemodel.state.brain_health import BrainHealth


def _registry() -> MetricRegistry:
    reg = MetricRegistry()
    register_universal_metrics(reg)  # declares OBSERVER_ERRORS
    return reg


class _FakeEvent:
    def __init__(self, text: str) -> None:
        self.text = text
        self.internal = False
        self.id = "evt-1"


def _boom() -> object:
    raise RuntimeError("build blew up")


# --------------------------------------------------------------------------- #
# post_llm_call observer
# --------------------------------------------------------------------------- #


def test_post_llm_observer_body_throw_is_loud_and_does_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = BrainHealth(tmp_path)
    reg = _registry()
    observer = make_post_llm_observer(_boom, health=h, metrics=reg)

    with caplog.at_level(logging.DEBUG):
        # Must NOT raise even though the body throws.
        observer(user_message="hi", assistant_response="yo")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert h.last_observer_error.get("post_llm_call") is not None
    assert "build blew up" in h.last_observer_error["post_llm_call"]
    assert reg.get(OBSERVER_ERRORS).value(component="post_llm_call") == 1.0


def test_post_llm_observer_happy_path_records_no_error(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    reg = _registry()
    # A real (fresh) LifeModel with a default State → no pending proactive turn, the
    # body early-returns cleanly; no observer error recorded, metric untouched.
    observer = make_post_llm_observer(
        lambda: build_lifemodel(base_dir=tmp_path), health=h, metrics=reg
    )
    observer(user_message="hi", assistant_response="yo")
    assert h.last_observer_error == {}
    assert reg.get(OBSERVER_ERRORS).value(component="post_llm_call") == 0.0


# --------------------------------------------------------------------------- #
# pre_gateway_dispatch observer
# --------------------------------------------------------------------------- #


def test_inbound_observer_body_throw_is_loud_and_does_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = BrainHealth(tmp_path)
    reg = _registry()
    observer = make_inbound_observer(_boom, health=h, metrics=reg)

    with caplog.at_level(logging.DEBUG):
        observer(event=_FakeEvent("a genuine inbound message"))

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors)
    assert h.last_observer_error.get("pre_gateway_dispatch") is not None
    assert reg.get(OBSERVER_ERRORS).value(component="pre_gateway_dispatch") == 1.0


def test_inbound_observer_filtered_event_does_not_build_or_error(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    reg = _registry()
    # A control command is filtered at the band-pass BEFORE build_lm; a raising
    # builder is never called, so no error is recorded.
    observer = make_inbound_observer(_boom, health=h, metrics=reg)
    observer(event=_FakeEvent("/lifemodel status"))
    assert h.last_observer_error == {}
    assert reg.get(OBSERVER_ERRORS).value(component="pre_gateway_dispatch") == 0.0

"""Tests for the logging backend, including the structlog-optional fallback.

The plugin loads inside Hermes' interpreter, which may lack structlog. These
tests prove get_logger()/configure() degrade to a stdlib shim (with the same
``.info(event, **fields)`` surface) instead of raising at import/use time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import lifemodel.log as lm_logging
from lifemodel.events import EVENTS_FILENAME, EventSink
from lifemodel.log import EventTee, _StdlibEventLogger, configure, get_logger


class _RecordingLogger:
    """A minimal :class:`EventLogger` that records the events it is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


def test_get_logger_returns_shim_when_structlog_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lm_logging, "_HAVE_STRUCTLOG", False)
    logger = get_logger("lifemodel.fallback")
    assert isinstance(logger, _StdlibEventLogger)


def test_shim_info_emits_over_stdlib(caplog: pytest.LogCaptureFixture) -> None:
    logger = _StdlibEventLogger("lifemodel.shim", base="ctx")
    with caplog.at_level(logging.INFO, logger="lifemodel.shim"):
        logger.info("plugin_registered", profile="nika")
    assert "plugin_registered" in caplog.text
    assert "nika" in caplog.text


def test_configure_is_safe_without_structlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # configure() must not raise when structlog is unavailable.
    monkeypatch.setattr(lm_logging, "_HAVE_STRUCTLOG", False)
    configure()


def test_configure_is_idempotent_with_structlog() -> None:
    configure()
    configure()
    assert get_logger("lifemodel.idempotent") is not None


def test_event_tee_fans_to_base_logger_and_sink(tmp_path: Path) -> None:
    base = _RecordingLogger()
    sink = EventSink(tmp_path / EVENTS_FILENAME)
    tee = EventTee(base, sink)

    tee.info("tick", pressure=1.0)

    # The wrapped logger still gets the event unchanged...
    assert base.calls == [("tick", {"pressure": 1.0})]
    # ...and it is also queryable from the sink.
    assert sink.read() == [{"event": "tick", "pressure": 1.0}]


def test_event_tee_survives_an_unwritable_sink(tmp_path: Path) -> None:
    # A sink whose writes fail (parent is a file) must not take the caller down,
    # and the base logger must still receive the event (best-effort, NFR9).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    base = _RecordingLogger()
    tee = EventTee(base, EventSink(blocker / EVENTS_FILENAME))

    tee.info("wake_decision", wake=False)  # must not raise

    assert base.calls == [("wake_decision", {"wake": False})]

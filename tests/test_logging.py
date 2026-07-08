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
    """A minimal :class:`EventLogger` that records the events it is handed, per level."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.debug_calls: list[tuple[str, dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, dict[str, Any]]] = []
        self.error_calls: list[tuple[str, dict[str, Any]]] = []
        self.critical_calls: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **fields: Any) -> None:
        self.debug_calls.append((event, dict(fields)))

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))

    def warning(self, event: str, **fields: Any) -> None:
        self.warning_calls.append((event, dict(fields)))

    def error(self, event: str, **fields: Any) -> None:
        self.error_calls.append((event, dict(fields)))

    def critical(self, event: str, **fields: Any) -> None:
        self.critical_calls.append((event, dict(fields)))


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


# --- B1 (lm-j2w): full standard log-level surface + effective-level gating ---


def test_event_tee_gates_debug_below_info_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    base = _RecordingLogger()
    sink = EventSink(tmp_path / EVENTS_FILENAME)
    tee = EventTee(base, sink)

    tee.debug("should_not_record", x=1)

    assert base.debug_calls == []
    assert sink.read() == []


def test_event_tee_records_info_warning_error_critical_at_info_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    base = _RecordingLogger()
    sink = EventSink(tmp_path / EVENTS_FILENAME)
    tee = EventTee(base, sink)

    tee.info("e_info", a=1)
    tee.warning("e_warning", a=2)
    tee.error("e_error", a=3)
    tee.critical("e_critical", a=4)

    assert base.calls == [("e_info", {"a": 1})]
    assert base.warning_calls == [("e_warning", {"a": 2})]
    assert base.error_calls == [("e_error", {"a": 3})]
    assert base.critical_calls == [("e_critical", {"a": 4})]
    assert sink.read() == [
        {"event": "e_info", "a": 1},
        {"event": "e_warning", "a": 2},
        {"event": "e_error", "a": 3},
        {"event": "e_critical", "a": 4},
    ]


def test_event_tee_records_debug_when_effective_level_is_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    sink = EventSink(tmp_path / EVENTS_FILENAME)
    tee = EventTee(base, sink)

    tee.debug("e_debug", a=1)

    assert base.debug_calls == [("e_debug", {"a": 1})]
    assert sink.read() == [{"event": "e_debug", "a": 1}]


def test_event_tee_gates_info_below_warning_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.WARNING)
    base = _RecordingLogger()
    sink = EventSink(tmp_path / EVENTS_FILENAME)
    tee = EventTee(base, sink)

    tee.info("should_not_record")
    tee.warning("e_warning")
    tee.error("e_error")
    tee.critical("e_critical")

    assert base.calls == []
    assert base.warning_calls == [("e_warning", {})]
    assert base.error_calls == [("e_error", {})]
    assert base.critical_calls == [("e_critical", {})]
    assert [record["event"] for record in sink.read()] == [
        "e_warning",
        "e_error",
        "e_critical",
    ]


def test_stdlib_event_logger_gates_debug_below_info(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    logger = _StdlibEventLogger("lifemodel.shim.gate_debug")
    with caplog.at_level(logging.DEBUG, logger="lifemodel.shim.gate_debug"):
        logger.debug("should_not_emit")
    assert "should_not_emit" not in caplog.text


def test_stdlib_event_logger_emits_debug_at_debug_level(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    logger = _StdlibEventLogger("lifemodel.shim.emit_debug")
    with caplog.at_level(logging.DEBUG, logger="lifemodel.shim.emit_debug"):
        logger.debug("should_emit")
    assert "should_emit" in caplog.text


def test_stdlib_event_logger_all_levels_emit_at_debug(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    logger = _StdlibEventLogger("lifemodel.shim.all_levels")
    with caplog.at_level(logging.DEBUG, logger="lifemodel.shim.all_levels"):
        logger.debug("d_event")
        logger.info("i_event")
        logger.warning("w_event")
        logger.error("e_event")
        logger.critical("c_event")
    for expected in ("d_event", "i_event", "w_event", "e_event", "c_event"):
        assert expected in caplog.text


def test_configure_sets_module_effective_level() -> None:
    configure(logging.WARNING)
    try:
        assert lm_logging._effective_level == logging.WARNING
    finally:
        configure(logging.INFO)  # restore default so later tests are unaffected
    assert lm_logging._effective_level == logging.INFO


@pytest.mark.parametrize(
    "name,expected",
    [
        ("debug", logging.DEBUG),
        ("DEBUG", logging.DEBUG),
        ("Debug", logging.DEBUG),
        ("info", logging.INFO),
        ("INFO", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_parse_log_level_accepts_standard_names(name: str, expected: int) -> None:
    assert lm_logging.parse_log_level(name) == expected


@pytest.mark.parametrize(
    "level,name",
    [
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "critical"),
    ],
)
def test_log_level_name_round_trips_with_parse_log_level(level: int, name: str) -> None:
    assert lm_logging.log_level_name(level) == name
    assert lm_logging.parse_log_level(name) == level


def test_parse_log_level_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="debug"):
        lm_logging.parse_log_level("verbose")


def test_log_level_names_constant_has_the_five_standard_names() -> None:
    assert set(lm_logging.LOG_LEVEL_NAMES) == {
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    }

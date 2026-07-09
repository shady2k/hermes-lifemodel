"""Tests for the stdlib log-level helpers (spec §4.2, codex fix #3).

The old structlog-optional pipeline (``configure``/``EventTee``/``get_logger``/
``_StdlibEventLogger``) is gone (spec §v1.2/§6.8): runtime logging is stdlib only.
What remains here is the level-name surface the ``/lifemodel loglevel`` command
uses plus :func:`apply_log_level`, which ``setLevel``s the ``lifemodel`` logger.
"""

from __future__ import annotations

import logging

import pytest

import lifemodel.log as lm_logging
from lifemodel.log import apply_log_level


def test_apply_log_level_sets_the_lifemodel_logger_level() -> None:
    try:
        apply_log_level(logging.WARNING)
        assert logging.getLogger("lifemodel").level == logging.WARNING
        apply_log_level(logging.DEBUG)
        assert logging.getLogger("lifemodel").level == logging.DEBUG
    finally:
        apply_log_level(logging.INFO)  # restore default so later tests are unaffected


def test_apply_log_level_gates_a_submodule_logger(caplog: pytest.LogCaptureFixture) -> None:
    # A child logger (lifemodel.<sub>) inherits the level set on the parent, so
    # a WARNING level drops an INFO line and keeps a WARNING one — no handler is
    # added (Hermes owns setup); caplog captures via propagation.
    sub = logging.getLogger("lifemodel.gate_check")
    sub.setLevel(logging.NOTSET)  # inherit from the parent
    try:
        apply_log_level(logging.WARNING)
        with caplog.at_level(logging.DEBUG):  # let caplog's handler see everything
            sub.info("should_be_gated")
            sub.warning("should_pass")
        assert "should_be_gated" not in caplog.text
        assert "should_pass" in caplog.text
    finally:
        apply_log_level(logging.INFO)


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

"""Tests for the logging backend, including the structlog-optional fallback.

The plugin loads inside Hermes' interpreter, which may lack structlog. These
tests prove get_logger()/configure() degrade to a stdlib shim (with the same
``.info(event, **fields)`` surface) instead of raising at import/use time.
"""

from __future__ import annotations

import logging

import pytest

import lifemodel.logging as lm_logging
from lifemodel.logging import _StdlibEventLogger, configure, get_logger


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

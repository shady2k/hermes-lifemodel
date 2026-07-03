"""Smoke test: the package imports and a structlog logger can be obtained.

Proves the toolchain (pytest + coverage) runs green on the scaffold.
"""

from __future__ import annotations

import lifemodel
from lifemodel.logging import configure, get_logger


def test_version_is_importable() -> None:
    assert lifemodel.__version__ == "0.0.0"


def test_logger_can_be_obtained() -> None:
    configure()
    logger = get_logger("lifemodel.smoke")
    assert logger is not None
    # A configured logger accepts a structured event without raising.
    logger.info("smoke", check="ok")

"""Smoke test: the package imports and the stdlib log-level surface works.

Proves the toolchain (pytest + coverage) runs green on the scaffold.
"""

from __future__ import annotations

import logging

import lifemodel
from lifemodel.log import apply_log_level


def test_version_is_importable() -> None:
    assert lifemodel.__version__ == "0.0.0"


def test_apply_log_level_sets_the_lifemodel_logger() -> None:
    apply_log_level(logging.WARNING)
    try:
        assert logging.getLogger("lifemodel").level == logging.WARNING
    finally:
        apply_log_level(logging.INFO)  # restore so later tests are unaffected

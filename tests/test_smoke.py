"""Smoke tests: package import surface + the adapter-shell smoke probe (bead lm-dte).

Proves the toolchain (pytest + coverage) runs green on the scaffold, and unit-tests
the pure ``run_smoke`` logic with fakes (no ``gateway`` import).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pytest

import lifemodel
from lifemodel.log import apply_log_level
from lifemodel.smoke import SmokeFailure, run_smoke


def test_version_is_importable() -> None:
    assert lifemodel.__version__ == "0.0.0"


def test_apply_log_level_sets_the_lifemodel_logger() -> None:
    apply_log_level(logging.WARNING)
    try:
        assert logging.getLogger("lifemodel").level == logging.WARNING
    finally:
        apply_log_level(logging.INFO)  # restore so later tests are unaffected


class _Incomplete(ABC):
    @abstractmethod
    def must_impl(self) -> None: ...


class _Complete(_Incomplete):
    def must_impl(self) -> None:
        return None


def test_run_smoke_raises_on_unimplemented_abstractmethods() -> None:
    with pytest.raises(SmokeFailure, match="must_impl"):
        run_smoke(_Incomplete, lambda: None)


def test_run_smoke_passes_on_concrete_class_and_calls_construct() -> None:
    called: list[int] = []
    run_smoke(_Complete, lambda: called.append(1) or object())
    assert called == [1]


def test_run_smoke_passes_with_no_construct() -> None:
    # The abstract-method guard is load-bearing and needs no construction; the
    # __main__ entry calls run_smoke without a construct thunk (see smoke.py).
    run_smoke(_Complete)  # must not raise


def test_run_smoke_no_construct_still_flags_abstractmethods() -> None:
    with pytest.raises(SmokeFailure, match="must_impl"):
        run_smoke(_Incomplete)


def test_run_smoke_wraps_construction_failure() -> None:
    def boom() -> object:
        raise RuntimeError("bad config shape")

    with pytest.raises(SmokeFailure, match="bad config shape"):
        run_smoke(_Complete, boom)

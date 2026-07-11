"""``wire(step, *, required, health, logger)`` — the fail-loud wiring boundary (§4.3).

The ONE context manager used at every wiring boundary in ``register()`` and
``connect()``. Its whole job is to invert the old default (``except → INFO
"…_skipped"; never break load``): a load-bearing failure is now LOUD (ERROR +
traceback, always) and observable (``BrainHealth`` + a persisted boot record) and
RE-RAISED; a genuinely-optional capability degrades to WARNING + traceback and
continues. These are deterministic caplog-asserted tests — no Hermes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lifemodel.state.brain_health import BrainHealth, brain_boot_path
from lifemodel.state.wiring import wire

_LOG = logging.getLogger("lifemodel.test.wire")


def test_success_logs_debug_and_leaves_state(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = BrainHealth(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="lifemodel.test.wire"):
        with wire("ok_step", required=True, health=h, logger=_LOG):
            pass
    assert h.state == "never_started"  # untouched on success
    # start + success both DEBUG (no ERROR/WARNING on the happy path).
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ok_step" in r.getMessage() and r.levelno == logging.DEBUG for r in caplog.records)


def test_required_failure_reraises_and_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = BrainHealth(tmp_path)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError, match="boom"):
            with wire("brain_step", required=True, health=h, logger=_LOG):
                raise RuntimeError("boom")
    # LOUD: ERROR with a full traceback (exc_info), independent of debug env.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "required failure must log at ERROR"
    assert any(r.exc_info is not None for r in errors), "ERROR must carry a traceback"
    assert any("brain_step" in r.getMessage() for r in errors)
    # Observable: BrainHealth flipped to boot_failed with the reason + durable record.
    assert h.state == "boot_failed"
    assert h.boot_error is not None and "brain_step" in h.boot_error and "boom" in h.boot_error
    assert brain_boot_path(tmp_path).exists()


def test_optional_failure_warns_with_traceback_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = BrainHealth(tmp_path)
    reached_after = False
    with caplog.at_level(logging.DEBUG):
        with wire("sampler", required=False, health=h, logger=_LOG):
            raise ValueError("degraded")
        reached_after = True  # wire swallowed → control continues here
    assert reached_after is True
    # WARNING with a traceback — never a bare INFO-without-traceback.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "optional failure must log at WARNING"
    assert any(r.exc_info is not None for r in warnings), "WARNING must carry a traceback"
    assert any("sampler" in r.getMessage() for r in warnings)
    # An OPTIONAL failure must NOT poison the health state or persist a boot record.
    assert h.state == "never_started"
    assert not brain_boot_path(tmp_path).exists()

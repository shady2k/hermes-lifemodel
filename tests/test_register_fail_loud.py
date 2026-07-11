"""``register()`` fail-loud wiring + the "both" strategy (spec §4.3, item 3).

The 2026-07-11 incident: a load-bearing wiring failure was caught as
``_LOG.info("…_skipped")`` (INFO, no traceback) and load continued — a brain-dead
plugin reported "enabled". This asserts the inversion: the ``/lifemodel`` command
is wired FIRST (the diagnostic lever survives even a brain-wiring failure), and a
REQUIRED wiring failure makes ``register()`` RAISE loudly (ERROR + traceback) with
``BrainHealth`` flipped to ``boot_failed`` and the durable record persisted.

Gateway-stubbed (like the Slice-1 smoke test): ``register()``'s lazy
``being_platform`` import needs ``gateway.*``, and ``_hermes_home()`` needs
``hermes_constants``.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from pathlib import Path

import pytest


def _install_stubs(home: Path) -> None:
    gateway = types.ModuleType("gateway")
    gateway.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("gateway.config")

    class Platform:
        def __init__(self, name: str) -> None:
            self.name = name

    config.Platform = Platform  # type: ignore[attr-defined]
    platforms = types.ModuleType("gateway.platforms")
    platforms.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("gateway.platforms.base")

    class BasePlatformAdapter:
        def __init__(self, *a: object, **k: object) -> None:
            pass

    class SendResult:
        def __init__(self, *a: object, **k: object) -> None:
            pass

    base.BasePlatformAdapter = BasePlatformAdapter  # type: ignore[attr-defined]
    base.SendResult = SendResult  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "gateway": gateway,
            "gateway.config": config,
            "gateway.platforms": platforms,
            "gateway.platforms.base": base,
        }
    )
    hc = types.ModuleType("hermes_constants")
    hc.get_hermes_home = lambda: home  # type: ignore[attr-defined]
    sys.modules["hermes_constants"] = hc


class _FakeCtx:
    """Duck-typed ctx recording the ORDER of every registration."""

    def __init__(self, *, platform_raises: bool = False) -> None:
        self.profile_name = "test-profile"
        self.calls: list[tuple[str, str]] = []
        self._platform_raises = platform_raises

    def register_command(self, name: str, fn: object, **k: object) -> None:
        self.calls.append(("command", name))

    def register_hook(self, name: str, cb: object) -> None:
        self.calls.append(("hook", name))

    def register_platform(self, name: str, **k: object) -> None:
        if self._platform_raises:
            raise RuntimeError("platform registration boom")
        self.calls.append(("platform", name))


def _sdir(home: Path) -> Path:
    d = home / "workspace" / "lifemodel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _release_trace_writer(sdir: Path) -> None:
    import lifemodel.state.trace_store as ts  # noqa: PLC0415

    with contextlib.suppress(Exception):
        ts.release_trace_writer(ts.observability_db_path(sdir))


def test_command_wired_first_and_all_required_wiring_succeeds(tmp_path: Path) -> None:
    import lifemodel  # noqa: PLC0415
    from lifemodel.state.brain_health import brain_boot_path, get_brain_health  # noqa: PLC0415

    home = tmp_path / "home"
    sdir = _sdir(home)
    _install_stubs(home)
    # A stale boot-failure record from a prior broken deploy must be wiped on a
    # clean boot this process (mark_boot_ok).
    brain_boot_path(sdir).write_text('{"state": "boot_failed"}', encoding="utf-8")

    ctx = _FakeCtx()
    try:
        lifemodel.register(ctx)
        kinds = [c for c, _ in ctx.calls]
        assert ctx.calls[0] == ("command", "lifemodel"), "the /lifemodel lever must be FIRST"
        assert ("hook", "post_llm_call") in ctx.calls
        assert ("hook", "pre_gateway_dispatch") in ctx.calls
        assert ("platform", "lifemodel") in ctx.calls
        # command before platform (both-strategy ordering).
        assert kinds.index("command") < kinds.index("platform")
        # Healthy boot: no boot_failed, stale durable record wiped.
        assert get_brain_health(sdir).state == "never_started"
        assert not brain_boot_path(sdir).exists()
    finally:
        _release_trace_writer(sdir)


def test_required_platform_failure_raises_loud_but_keeps_command(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import lifemodel  # noqa: PLC0415
    from lifemodel.state.brain_health import brain_boot_path, get_brain_health  # noqa: PLC0415

    home = tmp_path / "home"
    sdir = _sdir(home)
    _install_stubs(home)

    ctx = _FakeCtx(platform_raises=True)
    try:
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(RuntimeError, match="platform registration boom"),
        ):
            lifemodel.register(ctx)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
        # Loud + observable: not-enabled channel is the re-raise; state + durable record.
        health = get_brain_health(sdir)
        assert health.state == "boot_failed"
        assert health.boot_error is not None and "register_being_platform" in health.boot_error
        assert brain_boot_path(sdir).exists()
        # BOTH strategy: the diagnostic command was already wired before the failure.
        assert ("command", "lifemodel") in ctx.calls
    finally:
        _release_trace_writer(sdir)

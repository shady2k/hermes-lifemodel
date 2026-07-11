"""``BeingAdapter`` fail-loud: connect()/loop-death/check_fn (spec §4.3, items 4/5/7).

``adapters/being_platform.py`` imports ``gateway.*`` at module load, so — like the
Slice-1 register smoke test — these tests install minimal ``gateway.*`` stubs in
``sys.modules`` and import the adapter fresh under them. The stubs give
``BasePlatformAdapter`` the lifecycle hooks the adapter calls
(``_mark_connected`` / ``_set_fatal_error`` / ``_notify_fatal_error`` …) so the
real ``connect()`` / ``_on_loop_death`` code runs off-host. Deterministic: the
tick is stubbed to a no-op (the loop ticks immediately on run), and the acquire
seams are monkeypatched to force required/optional failures.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

from lifemodel.state.brain_health import brain_boot_path, get_brain_health


def _install_gateway_stubs() -> None:
    """Rich ``gateway.*`` so ``BeingAdapter`` runs off-host — the base carries the
    lifecycle hooks connect()/_on_loop_death call."""
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
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.connected = False
            self.fatal: tuple[object, ...] | None = None

        def _mark_connected(self) -> None:
            self.connected = True

        def _mark_disconnected(self) -> None:
            self.connected = False

        def _set_fatal_error(self, *args: object, **kwargs: object) -> None:
            self.fatal = args

        async def _notify_fatal_error(self) -> None:
            return None

    class SendResult:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    base.BasePlatformAdapter = BasePlatformAdapter  # type: ignore[attr-defined]
    base.SendResult = SendResult  # type: ignore[attr-defined]

    sys.modules["gateway"] = gateway
    sys.modules["gateway.config"] = config
    sys.modules["gateway.platforms"] = platforms
    sys.modules["gateway.platforms.base"] = base


def _fresh_being_platform() -> types.ModuleType:
    _install_gateway_stubs()
    for name in [n for n in sys.modules if n.endswith("adapters.being_platform")]:
        del sys.modules[name]
    import lifemodel.adapters.being_platform as bp  # noqa: PLC0415

    return bp


class _FakeCtx:
    def __init__(self) -> None:
        self.platforms: list[tuple[str, dict]] = []

    def register_platform(self, name: str, **kwargs: object) -> None:
        self.platforms.append((name, dict(kwargs)))


def _make_adapter(bp: types.ModuleType, base_dir: Path):
    adapter = bp.BeingAdapter(config=None, base_dir=base_dir, target=None, interval_sec=60.0)
    adapter._tick = lambda: None  # the loop ticks immediately on run — keep it inert
    return adapter


# --------------------------------------------------------------------------- #
# check_fn (item 7) — no longer lambda: True; reflects BrainHealth
# --------------------------------------------------------------------------- #


def test_register_being_platform_wires_health_derived_check_fn(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    ctx = _FakeCtx()
    bp.register_being_platform(ctx, base_dir=tmp_path, target=None)
    assert ctx.platforms and ctx.platforms[0][0] == "lifemodel"
    check_fn = ctx.platforms[0][1]["check_fn"]

    health = get_brain_health(tmp_path)
    # Enablement-safe: a fresh (never_started) brain is healthy so the gateway will
    # instantiate the adapter and connect() can run (a False here bricks the being).
    assert check_fn() is True
    # A boot failure → unhealthy.
    health.mark_boot_failed("register_being_platform: ImportError: x")
    assert check_fn() is False
    # A clean connect → healthy again.
    health.mark_connected(at=None)
    assert check_fn() is True
    # A loop death → unhealthy.
    health.record_loop_death("died", "tb")
    assert check_fn() is False


# --------------------------------------------------------------------------- #
# connect() coverage (item 4)
# --------------------------------------------------------------------------- #


def test_connect_happy_path_marks_connected(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())
    assert health.state == "connected"


def test_connect_required_trace_writer_failure_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("trace writer down")

    bp.acquire_trace_writer = _boom  # required-for-observability

    async def _run() -> None:
        await adapter.connect()

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="trace writer down"):
        asyncio.run(_run())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors)
    assert health.state == "boot_failed"
    assert brain_boot_path(tmp_path).exists()


def test_connect_optional_sampler_failure_degrades(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("sampler down")

    bp.acquire_metrics_sampler = _boom  # optional / degraded

    async def _run() -> None:
        assert await adapter.connect() is True  # brain stays alive
        await adapter.disconnect()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and any(r.exc_info is not None for r in warnings)
    assert health.state == "connected"
    assert adapter.metrics_degraded is True


# --------------------------------------------------------------------------- #
# _on_loop_death (item 5)
# --------------------------------------------------------------------------- #


def test_on_loop_death_records_and_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)
    health.mark_connected(at=None)

    with caplog.at_level(logging.DEBUG):
        adapter._on_loop_death(RuntimeError("loop boom"))

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert health.state == "loop_dead"
    assert health.death_count == 1
    assert health.last_loop_death is not None and "loop boom" in health.last_loop_death


def test_clean_reconnect_clears_loop_dead(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)
    adapter._on_loop_death(RuntimeError("loop boom"))
    assert health.state == "loop_dead"

    async def _run() -> None:
        assert await adapter.connect(is_reconnect=True) is True
        await adapter.disconnect()

    asyncio.run(_run())
    assert health.state == "connected"

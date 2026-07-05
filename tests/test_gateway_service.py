from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from lifemodel.gateway_core import install_core_shim, register_gateway_service


class _Runner:
    def __init__(self) -> None:
        self._gateway_loop = asyncio.new_event_loop()
        self._background_tasks: set[Any] = set()


def test_register_gateway_service_tracks_task_in_background_tasks() -> None:
    runner = _Runner()

    async def _svc() -> None:
        await asyncio.sleep(0)

    async def _run() -> None:
        ok = register_gateway_service(runner, "lifemodel-egress", lambda: _svc())
        assert ok is True
        assert len(runner._background_tasks) == 1
        await asyncio.sleep(0.01)  # let it finish

    try:
        asyncio.run(_run())
    finally:
        runner._gateway_loop.close()


def test_register_gateway_service_fail_closed_without_loop() -> None:
    class Bad:
        pass

    assert register_gateway_service(Bad(), "k", lambda: None) is False


def test_install_core_shim_adds_methods_best_effort() -> None:
    class Ctx:
        pass

    ctx = Ctx()
    install_core_shim(ctx)  # must not raise
    assert hasattr(ctx, "inject_proactive_turn")
    assert hasattr(ctx, "register_gateway_service")


def test_loop_yields_to_cron_when_reachin_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from lifemodel import egress_service
    from lifemodel.log import get_logger

    ticks: list[int] = []
    monkeypatch.setattr(egress_service, "run_proactive_tick", lambda *a, **k: ticks.append(1))

    class _Unavailable:
        # "running" but reach-in NOT available (no ``adapters`` attr) -> yield to cron.
        _running = True
        _draining = False
        _gateway_loop = object()
        _running_agents: set[Any] = set()

        def _build_process_event_source(self, evt: Any) -> Any:
            return object()

    class _Stop(Exception):
        pass

    calls = {"n": 0}

    async def _fake_sleep(_secs: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise _Stop()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    async def _run() -> None:
        with contextlib.suppress(_Stop):
            await egress_service.proactive_service_loop(
                build_lm=lambda: object(),
                egress=object(),
                target={},
                runner_accessor=lambda: _Unavailable(),
                logger=get_logger("t"),
                interval_seconds=0.0,
            )

    asyncio.run(_run())
    assert ticks == []  # never ticked — yielded to the cron fallback

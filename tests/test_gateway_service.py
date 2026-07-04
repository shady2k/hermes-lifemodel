from __future__ import annotations

import asyncio
from typing import Any

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

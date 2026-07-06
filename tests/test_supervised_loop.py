"""Tests for :class:`lifemodel.core.supervised_loop.SupervisedLoop`.

The load-bearing new unit: a periodic loop that DETECTS its own death and reports
it via an injected callback. Its absence (a self-spawned task that died unnoticed)
was the cause of the being's silent proactivity outage. Hermes-free: driven with a
fake ``sleep`` and plain callables, run under ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio

import pytest

from lifemodel.core.supervised_loop import SupervisedLoop


async def _noop_sleep(_seconds: float) -> None:
    return None


def test_tick_called_each_interval() -> None:
    calls: list[int] = []
    holder: dict[str, SupervisedLoop] = {}

    def tick() -> None:
        calls.append(1)
        if len(calls) >= 3:
            holder["loop"].stop()

    async def scenario() -> None:
        loop = SupervisedLoop(
            tick=tick, interval_sec=60.0, on_death=lambda _e: None, sleep=_noop_sleep
        )
        holder["loop"] = loop
        await loop.run()

    asyncio.run(scenario())
    assert len(calls) == 3


def test_tick_exception_calls_on_death_once_and_stops() -> None:
    calls: list[int] = []
    deaths: list[BaseException | None] = []

    def tick() -> None:
        calls.append(1)
        raise ValueError("boom")

    async def scenario() -> None:
        loop = SupervisedLoop(
            tick=tick, interval_sec=60.0, on_death=deaths.append, sleep=_noop_sleep
        )
        await loop.run()

    asyncio.run(scenario())
    assert len(calls) == 1  # stopped after the first raise
    assert len(deaths) == 1
    assert isinstance(deaths[0], ValueError)


def test_stop_exits_without_on_death() -> None:
    calls: list[int] = []
    deaths: list[BaseException | None] = []
    holder: dict[str, SupervisedLoop] = {}

    def tick() -> None:
        calls.append(1)
        holder["loop"].stop()

    async def scenario() -> None:
        loop = SupervisedLoop(
            tick=tick, interval_sec=60.0, on_death=deaths.append, sleep=_noop_sleep
        )
        holder["loop"] = loop
        await loop.run()

    asyncio.run(scenario())
    assert calls == [1]
    assert deaths == []  # a clean stop is not a death


def test_cancel_is_clean_no_on_death() -> None:
    deaths: list[BaseException | None] = []

    async def slow_sleep(_seconds: float) -> None:
        await asyncio.sleep(3600)

    async def scenario() -> None:
        loop = SupervisedLoop(
            tick=lambda: None, interval_sec=60.0, on_death=deaths.append, sleep=slow_sleep
        )
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)  # let it tick once and suspend in slow_sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert deaths == []  # cancellation is a clean shutdown, not a death

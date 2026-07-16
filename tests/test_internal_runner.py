"""Tests for :class:`lifemodel.adapters.internal_runner.InternalCognitionRunner` (lm-705.6).

The adapter-owned, gateway-loop task lifecycle: ``launch`` reserves FR20 and sets
``pending_internal_id`` synchronously (one frame, under the lock), then creates and
retains an asyncio task that awaits the injected :class:`FakeLlmPort` OFF the lock;
on completion (success, failure, or a raised exception) it runs
:func:`~lifemodel.core.internal_cognition.run_internal_completion` and the pending
correlation clears — never stranded. ``recover_stale``/``cancel_all`` cover boot
recovery and clean shutdown.

Uses plain ``async def test_...`` (pytest-asyncio ``asyncio_mode = "auto"``,
pyproject.toml) — the test's own running loop stands in for the gateway loop, since
``InternalCognitionRunner`` takes it as an explicit constructor argument rather than
discovering it ambiently (this seam's own connect() always calls it from within an
``async def connect()`` already running on the real gateway loop — see
``adapters/being_platform.py`` Task 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.adapters.internal_runner import InternalCognitionRunner
from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent
from lifemodel.core.llm_port import InternalCognitionRequest, InternalCognitionResult
from lifemodel.core.taxonomy import KIND_INTERNAL_RESULT, read_internal_result
from lifemodel.domain.egress import ReachOutcome
from lifemodel.state.model import State
from lifemodel.testing.llm import FakeLlmPort

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}
BORN_AT = "2026-07-01T10:00:00+00:00"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
TODAY = "2026-07-16"


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class FakeEgress:
    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


class RecordingApply:
    id = "recording-apply"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        for sig in ctx.signals:
            if sig.kind == KIND_INTERNAL_RESULT:
                self.seen.append(read_internal_result(sig).raw)
        return []


def _build_lm_factory(base_dir: Path):
    def _build() -> LifeModel:
        return build_lifemodel(base_dir=base_dir, clock=FixedClock(NOW))

    return _build


def _commit(build_lm, **fields: object) -> None:
    base = dict(genesis_completed_at=BORN_AT, last_tick_at="2026-07-16T11:59:00+00:00")
    base.update(fields)
    build_lm().state.commit(State(**base))  # type: ignore[arg-type]


async def test_launch_denied_over_budget_creates_no_task(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=1, internal_calls_day=TODAY)
    apply = RecordingApply()
    llm = FakeLlmPort(InternalCognitionResult(raw="unused", parsed=None))
    runner = InternalCognitionRunner(
        build_lm,
        llm,
        FakeEgress(),
        TARGET,
        daily_ceiling=1,
        gateway_loop=asyncio.get_running_loop(),
        apply=apply,
    )

    ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-1")

    assert ok is False
    assert len(runner._tasks) == 0  # no task created
    assert build_lm().state.load().pending_internal_id is None
    assert build_lm().state.load().internal_calls_today == 1  # unchanged — not consumed
    assert llm.requests == []  # the aux port was never even called


async def test_launch_within_budget_sets_pending_synchronously(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    llm = FakeLlmPort(InternalCognitionResult(raw="hi", parsed=None))
    runner = InternalCognitionRunner(
        build_lm,
        llm,
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=RecordingApply(),
    )

    ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-2")

    assert ok is True
    assert len(runner._tasks) == 1
    # Set SYNCHRONOUSLY inside launch() — before the task has had a chance to run
    # (no await has happened yet), proving the reservation+pending-set is one frame,
    # not deferred to the async body.
    state = build_lm().state.load()
    assert state.pending_internal_id == "c-2"
    assert state.internal_calls_today == 1
    assert state.internal_calls_day == TODAY

    # Let the task run to completion so the loop doesn't warn about a pending task.
    task = next(iter(runner._tasks))
    await task


async def test_launch_completes_applies_result_and_clears_pending(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    llm = FakeLlmPort(InternalCognitionResult(raw="the aux said this", parsed={"a": 1}))
    apply = RecordingApply()
    runner = InternalCognitionRunner(
        build_lm,
        llm,
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=apply,
    )

    ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-3")
    assert ok is True
    task = next(iter(runner._tasks))
    await task

    assert apply.seen == ["the aux said this"]
    assert build_lm().state.load().pending_internal_id is None
    assert llm.requests[0].instructions == "i"
    assert llm.requests[0].input_text == "t"
    assert task not in runner._tasks  # removed from the tracked set on completion


async def test_a_raising_llm_port_still_clears_pending_no_strand(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    llm = FakeLlmPort(RuntimeError("boom"))
    apply = RecordingApply()
    runner = InternalCognitionRunner(
        build_lm,
        llm,
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=apply,
    )

    ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-4")
    assert ok is True
    task = next(iter(runner._tasks))
    await task  # must not propagate — the runner swallows/logs, never crashes the loop

    assert build_lm().state.load().pending_internal_id is None  # cleared, no strand
    assert apply.seen == [""]  # applied over the empty failure-result


async def test_cancel_all_cancels_a_pending_task(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")

    class HangingLlmPort:
        def __init__(self) -> None:
            self.requests: list[InternalCognitionRequest] = []

        async def complete_structured(
            self, req: InternalCognitionRequest
        ) -> InternalCognitionResult:
            self.requests.append(req)
            await asyncio.sleep(3600)
            raise AssertionError("should have been cancelled first")

    runner = InternalCognitionRunner(
        build_lm,
        HangingLlmPort(),
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=RecordingApply(),
    )
    ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-5")
    assert ok is True
    await asyncio.sleep(0)  # let the task actually start and reach the sleep
    assert len(runner._tasks) == 1

    await runner.cancel_all()

    assert runner._tasks == set()  # no leaked task


async def test_recover_stale_clears_a_leftover_pending_id(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, pending_internal_id="stale-1")
    runner = InternalCognitionRunner(
        build_lm,
        FakeLlmPort(InternalCognitionResult(raw="", parsed=None)),
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=RecordingApply(),
    )

    runner.recover_stale(build_lm())

    assert build_lm().state.load().pending_internal_id is None


async def test_recover_stale_is_a_noop_when_nothing_pending(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, pending_internal_id=None)
    before = build_lm().state.load()
    runner = InternalCognitionRunner(
        build_lm,
        FakeLlmPort(InternalCognitionResult(raw="", parsed=None)),
        FakeEgress(),
        TARGET,
        daily_ceiling=3,
        gateway_loop=asyncio.get_running_loop(),
        apply=RecordingApply(),
    )

    runner.recover_stale(build_lm())

    assert build_lm().state.load() == before

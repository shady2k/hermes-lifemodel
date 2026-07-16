"""Tests for the single-flight gate + subject/interval stamping on
:class:`lifemodel.adapters.internal_runner.InternalCognitionRunner` (lm-705.2,
Task 6 — "first live emitter" prereqs #1/#2).

Mirrors ``tests/test_internal_runner.py``'s fixtures (plain ``async def
test_...`` under pytest-asyncio ``asyncio_mode = "auto"``, ``build_lifemodel``
+ ``FixedClock`` + ``FakeEgress`` + ``FakeLlmPort``, the running loop standing
in for the gateway loop).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.adapters.internal_runner import InternalCognitionRunner
from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.llm_port import InternalCognitionRequest, InternalCognitionResult
from lifemodel.domain.egress import ReachOutcome
from lifemodel.state.model import State
from lifemodel.testing.llm import FakeLlmPort

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}
BORN_AT = "2026-07-01T10:00:00+00:00"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
REQ = InternalCognitionRequest(instructions="i", input_text="t")


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


class _NoopApply:
    id = "noop-apply"

    def step(self, ctx):  # pragma: no cover - trivial
        return []


def _build_lm_factory(base_dir: Path):
    def _build() -> LifeModel:
        return build_lifemodel(base_dir=base_dir, clock=FixedClock(NOW))

    return _build


def _commit(build_lm, **fields: object) -> None:
    base = dict(genesis_completed_at=BORN_AT, last_tick_at="2026-07-16T11:59:00+00:00")
    base.update(fields)
    build_lm().state.commit(State(**base))  # type: ignore[arg-type]


def _runner(
    build_lm, *, daily_ceiling: int = 3, llm: FakeLlmPort | None = None
) -> InternalCognitionRunner:
    return InternalCognitionRunner(
        build_lm,
        llm if llm is not None else FakeLlmPort(InternalCognitionResult(raw="hi", parsed=None)),
        FakeEgress(),
        TARGET,
        daily_ceiling=daily_ceiling,
        gateway_loop=asyncio.get_running_loop(),
        apply=_NoopApply(),
    )


async def test_single_flight_denies_a_second_launch(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    runner = _runner(build_lm)

    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True

    calls_before = build_lm().state.load().internal_calls_today
    assert runner.launch(REQ, "c-2", subject_id="thought:seed:b") is False
    assert build_lm().state.load().internal_calls_today == calls_before  # no second reserve
    # the pending marker is still the FIRST launch's — the second never touched it
    assert build_lm().state.load().pending_internal_id == "c-1"

    # let the first task run to completion so the loop doesn't warn about a
    # pending task at shutdown
    for task in list(runner._tasks):
        await task


async def test_launch_stamps_subject_and_interval(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    runner = _runner(build_lm)

    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True

    state = build_lm().state.load()
    assert state.pending_internal_subject_id == "thought:seed:a"
    assert state.last_internal_call_at is not None

    for task in list(runner._tasks):
        await task


async def test_launch_without_subject_id_defaults_to_none(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    runner = _runner(build_lm)

    assert runner.launch(REQ, "c-1") is True

    state = build_lm().state.load()
    assert state.pending_internal_subject_id is None

    for task in list(runner._tasks):
        await task


async def test_recover_stale_clears_subject_too(tmp_path) -> None:
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, pending_internal_id="x", pending_internal_subject_id="y")
    runner = _runner(build_lm)

    runner.recover_stale(build_lm())

    state = build_lm().state.load()
    assert state.pending_internal_id is None
    assert state.pending_internal_subject_id is None


async def test_clear_pending_fail_loud_clears_subject_too(tmp_path, monkeypatch) -> None:
    # Exercises the OUTER guard in `_run`: a raise from the completion frame
    # ITSELF (not a mere LLM-call failure, which never reaches this path) must
    # still clear both `pending_internal_id` AND `pending_internal_subject_id`
    # so a bug in `apply`/the completion path can never strand a future launch.
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    runner = _runner(build_lm)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("completion frame exploded")

    monkeypatch.setattr("lifemodel.adapters.internal_runner.run_internal_completion", _boom)

    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True
    task = next(iter(runner._tasks))
    await task  # must not propagate — the runner's outer guard swallows/logs it

    state = build_lm().state.load()
    assert state.pending_internal_id is None
    assert state.pending_internal_subject_id is None


async def test_launch_denied_over_budget_still_requires_pending_clear_first(tmp_path) -> None:
    # Single-flight is checked BEFORE the budget reserve (prereq #2's ordering
    # guarantee): a pending launch denies a second one even when the daily
    # ceiling would otherwise still have room.
    build_lm = _build_lm_factory(tmp_path)
    _commit(build_lm, internal_calls_today=0, internal_calls_day="")
    runner = _runner(build_lm, daily_ceiling=50)

    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True
    assert runner.launch(REQ, "c-2", subject_id="thought:seed:b") is False

    for task in list(runner._tasks):
        await task

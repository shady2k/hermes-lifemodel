"""``BeingAdapter``'s internal-cognition seam wiring (lm-705.6, Task 7).

``adapters/being_platform.py`` imports ``gateway.*`` at module load — mirrors
``test_being_platform_fail_loud.py``'s gateway-stub harness so ``connect()``/
``disconnect()``/``_tick()`` run off-host, exercising the SAME wiring the live
gateway does: :class:`~lifemodel.adapters.internal_runner.InternalCognitionRunner`
is built (and ``recover_stale``'d) only when an ``LlmPort`` was injected, torn
down cleanly at ``disconnect``, and driven from the SAME frame ``_tick()``
already runs for the proactive path — never a second ``run_frame`` (that would
double-tick: a second bookkeeping bump, a second energy/fatigue recovery pass).
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

from lifemodel.core.llm_port import InternalCognitionRequest, InternalCognitionResult
from lifemodel.state.brain_health import get_brain_health
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing.llm import FakeLlmPort


def _install_gateway_stubs() -> None:
    """Rich ``gateway.*`` so ``BeingAdapter`` runs off-host (mirrors
    ``test_being_platform_fail_loud.py``'s stub — duplicated locally per that
    file's own convention of a stub tailored to what IT needs)."""
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


BORN_AT = "2026-07-01T10:00:00+00:00"


def _make_adapter(
    bp: types.ModuleType,
    base_dir: Path,
    *,
    llm=None,
    noticing_buffer=None,
    inert_tick: bool = True,
):
    adapter = bp.BeingAdapter(
        config=None,
        base_dir=base_dir,
        target=None,
        interval_sec=60.0,
        llm=llm,
        noticing_buffer=noticing_buffer,
    )
    if inert_tick:
        adapter._tick = lambda: None  # the loop ticks immediately on run — keep it inert
    return adapter


def test_connect_leaves_internal_runner_none_without_llm(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())
    assert adapter._internal_runner is None


def test_connect_builds_internal_runner_when_llm_injected(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    llm = FakeLlmPort(InternalCognitionResult(raw="", parsed=None))
    adapter = _make_adapter(bp, tmp_path, llm=llm)

    async def _run() -> None:
        assert await adapter.connect() is True
        assert adapter._internal_runner is not None
        await adapter.disconnect()

    asyncio.run(_run())


def test_connect_recovers_a_stale_pending_internal_id(tmp_path: Path) -> None:
    from lifemodel.adapters.clock import SystemClock

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(genesis_completed_at=BORN_AT, pending_internal_id="stale-from-a-dead-task"))

    bp = _fresh_being_platform()
    llm = FakeLlmPort(InternalCognitionResult(raw="", parsed=None))
    adapter = _make_adapter(bp, tmp_path, llm=llm)

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    assert store.load().pending_internal_id is None


def test_connect_recovers_a_stale_claimed_survey(tmp_path: Path) -> None:
    """A ``claimed`` survey left behind by a noticing pass that died mid-flight
    with the process (lm-705.14 Task 5) must be released back to ``complete``
    by the NEXT real ``connect()`` — regardless of whether an ``LlmPort`` was
    injected (buffer recovery needs no LLM, unlike the internal-runner
    recovery above)."""
    from datetime import timedelta

    from lifemodel.adapters.clock import SystemClock
    from lifemodel.core.noticing_buffer import NoticingBuffer
    from lifemodel.state.sqlite_store import SqliteBufferStore

    clock = SystemClock()
    now = clock.now()
    store = SqliteBufferStore(tmp_path, clock=clock)
    store.open_pending("session-1", user_text="hi", now=now)
    store.complete("session-1", "turn-1", assistant_text="yo", now=now)
    store.claim("session-1", ("turn-1",), "survey-dead")
    assert store.claimed("survey-dead"), "sanity: the seeded row is actually claimed"

    bp = _fresh_being_platform()
    # A SEPARATE NoticingBuffer/SqliteBufferStore instance over the SAME base_dir —
    # exactly what `register()` builds in production: one physical file, reached
    # through independent connections (D7).
    buffer = NoticingBuffer(store=SqliteBufferStore(tmp_path, clock=clock))
    adapter = _make_adapter(bp, tmp_path, noticing_buffer=buffer)

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    assert store.claimed("survey-dead") == []
    recovered = store.completed("session-1", now=now, ttl=timedelta(minutes=30))
    assert [entry.turn_id for entry in recovered] == ["turn-1"]


def test_disconnect_cancels_an_in_flight_internal_launch(tmp_path: Path) -> None:
    class HangingLlmPort:
        async def complete_structured(
            self, req: InternalCognitionRequest
        ) -> InternalCognitionResult:
            await asyncio.sleep(3600)
            raise AssertionError("should have been cancelled")

    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path, llm=HangingLlmPort())
    health = get_brain_health(tmp_path)
    health.mark_connecting()  # avoid touching the real store during this narrow test

    async def _run() -> None:
        assert await adapter.connect() is True
        runner = adapter._internal_runner
        assert runner is not None
        ok = runner.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-1")
        assert ok is True
        await asyncio.sleep(0)  # let the task actually start
        assert len(runner._tasks) == 1

        # disconnect() must complete promptly — cancel_all() cancels the hanging task
        # rather than waiting an hour for it.
        await asyncio.wait_for(adapter.disconnect(), timeout=5.0)
        assert runner._tasks == set()

    asyncio.run(_run())


def test_tick_drives_report_internal_launches_into_the_runner(tmp_path: Path, monkeypatch) -> None:
    # Prove the wiring end-to-end: a fake component that emits LaunchInternalCognition
    # (standing in for noticing/processing, lm-705.5/.2 — not built yet) makes it all
    # the way from CoreLoop's report, through _tick()'s SAME frame (never a second
    # run_frame), into runner.launch() -> pending_internal_id set. Monkeypatches
    # bp.build_lifemodel (the name being_platform.py imports into its own namespace)
    # to inject the fake component into an otherwise-real graph.
    from lifemodel.composition import build_lifemodel as real_build_lifemodel
    from lifemodel.core.component import ComponentLayer, TickContext
    from lifemodel.core.intents import LaunchInternalCognition
    from lifemodel.core.registry import ComponentManifest, UnknownComponent

    class FakeInternalLauncher:
        id = "fake-internal-launcher"

        def step(self, ctx: TickContext):
            return [
                LaunchInternalCognition(
                    prompt="notice this",
                    correlation_id="internal-from-tick",
                    origin_traceparent=("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
                )
            ]

    def _patched_build_lifemodel(*args, **kwargs):
        lm = real_build_lifemodel(*args, **kwargs)
        try:
            lm.registry.manifest(FakeInternalLauncher.id)
        except UnknownComponent:
            lm.registry.register(
                FakeInternalLauncher(),
                ComponentManifest(
                    id=FakeInternalLauncher.id,
                    type="cognition",
                    layer=ComponentLayer.COGNITION,
                    metric_surface=(),
                    accepts_signals=False,
                ),
            )
        return lm

    bp = _fresh_being_platform()
    monkeypatch.setattr(bp, "build_lifemodel", _patched_build_lifemodel)

    from lifemodel.adapters.clock import SystemClock

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(genesis_completed_at=BORN_AT, last_tick_at=None))

    llm = FakeLlmPort(InternalCognitionResult(raw="", parsed=None))
    adapter = _make_adapter(bp, tmp_path, llm=llm, inert_tick=False)

    async def _run() -> None:
        assert await adapter.connect() is True
        adapter._tick()  # one manual tick — proves the wiring without waiting on the loop
        # Synchronously, right after _tick(): the reservation + pending-set already
        # committed (launch() sets it under the lock before creating the task).
        assert store.load().pending_internal_id == "internal-from-tick"
        runner = adapter._internal_runner
        assert runner is not None
        for task in list(runner._tasks):
            await task
        await adapter.disconnect()

    asyncio.run(_run())

    assert store.load().pending_internal_id is None  # completion frame cleared it
    assert llm.requests[0].input_text == "notice this"

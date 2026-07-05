from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.coreloop import CoreLoop, TickReport
from lifemodel.core.intents import EmitSignal, Intent, UpdateState
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class RecordingStore:
    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)


class Healthy:
    id = "healthy"

    def __init__(self) -> None:
        self.calls = 0

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.calls += 1
        return [UpdateState({"u": 0.42})]


class Emitter:
    id = "emitter"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [EmitSignal(Signal(origin_id="emitter-1", kind="contact"))]


class Broken:
    id = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.calls += 1
        raise RuntimeError("boom")


def _loop(
    registry: ComponentRegistry,
    store: RecordingStore,
    bus: FileSignalBus,
    *,
    breaker_threshold: int = 3,
) -> CoreLoop:
    return CoreLoop(
        registry=registry,
        state_actor=StateActor(store),
        bus=bus,
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        breaker_threshold=breaker_threshold,
    )


def test_healthy_component_intents_reach_state_and_tick_bumps(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(Healthy(), ComponentManifest(id="healthy", type="neuron"))
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    report = loop.tick()
    assert isinstance(report, TickReport)
    assert store.commits[-1].u == 0.42
    assert store.commits[-1].tick_count == 1
    assert store.commits[-1].last_tick_at is not None
    assert report.ran == ("healthy",)


def test_emit_signal_is_published_to_bus(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(Emitter(), ComponentManifest(id="emitter", type="neuron"))
    bus = FileSignalBus(tmp_path)
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    published = bus.peek_unprocessed()
    assert [s.origin_id for s in published] == ["emitter-1"]


def test_failing_component_is_isolated_and_others_still_run(tmp_path) -> None:
    reg = ComponentRegistry()
    healthy = Healthy()
    reg.register(Broken(), ComponentManifest(id="broken", type="neuron"))
    reg.register(healthy, ComponentManifest(id="healthy", type="neuron"))
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    report = loop.tick()  # must not raise
    assert healthy.calls == 1
    assert store.commits[-1].u == 0.42  # tick still checkpointed
    assert store.commits[-1].tick_count == 1
    assert "broken" in report.failed


def test_repeated_failures_open_breaker_and_skip_component(tmp_path) -> None:
    reg = ComponentRegistry()
    broken = Broken()
    reg.register(broken, ComponentManifest(id="broken", type="neuron"))
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path), breaker_threshold=3)
    for _ in range(3):
        loop.tick()
    assert broken.calls == 3  # tripped after the 3rd failure
    report = loop.tick()
    assert broken.calls == 3  # not called again — breaker open
    assert "broken" in report.skipped_broken


def test_tick_count_increments_each_tick(tmp_path) -> None:
    reg = ComponentRegistry()
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    loop.tick()
    loop.tick()
    assert store.commits[-1].tick_count == 2

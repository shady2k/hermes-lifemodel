"""Heartbeat metric on the tick path (spec §4.4, bead lm-fib.9.3).

Every CoreLoop tick emits a **heartbeat** into the shared
:class:`~lifemodel.core.metrics.MetricRegistry` — SUPPORTING evidence only
(codex MAJOR-8): the PRIMARY liveness stays the durable ``last_tick_at`` /
``tick_count`` advanced into ``AgentState`` every tick, so a dead metrics
sampler can never re-introduce the silent-death ambiguity. These tests assert
BOTH halves after a single tick on fake ports, and that a metrics hiccup never
breaks the tick (fail-open).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.core.component import TickContext, layer_for_type
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.intents import Intent
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.core.tick_metrics import BRAIN_HEARTBEAT, BRAIN_LAST_TICK_EPOCH
from lifemodel.core.timeutil import to_iso
from lifemodel.state.model import State
from lifemodel.testing import FakeTracer

_NOW = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


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

    def commit_tick(
        self,
        state: State | None,
        mutations: Sequence[object],
        *,
        finalize_survey_id: str | None = None,
    ) -> None:
        if state is not None:
            self.commit(state)


class Ticking:
    def __init__(self, step: float = 0.01) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class OkComp:
    id = "ok"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return []


def _manifest(cid: str, ctype: str) -> ComponentManifest:
    return ComponentManifest(
        id=cid, type=ctype, layer=layer_for_type(ctype), metric_surface=(), accepts_signals=False
    )


def _loop(store: RecordingStore, metrics: MetricRegistry) -> CoreLoop:
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    return CoreLoop(
        registry=reg,
        state_actor=StateActor(store),
        clock=FixedClock(_NOW),
        tracer=FakeTracer(),
        metrics=metrics,
        monotonic=Ticking(),
    )


def test_tick_advances_durable_liveness_AND_emits_heartbeat() -> None:
    # The PRIMARY signal is durable AgentState (spec §4.2): the tick bumps
    # tick_count and stamps last_tick_at, persisted through the store.
    store = RecordingStore()
    metrics = MetricRegistry()
    report = _loop(store, metrics).tick()

    # --- durable, primary ---
    assert report.tick == 1
    assert store.load().tick_count == 1
    assert store.load().last_tick_at == to_iso(_NOW)

    # --- heartbeat, supporting ---
    heartbeat = metrics.get(BRAIN_HEARTBEAT)
    assert heartbeat is not None
    assert heartbeat.value() == 1.0  # type: ignore[attr-defined]

    epoch = metrics.get(BRAIN_LAST_TICK_EPOCH)
    assert epoch is not None
    assert epoch.value() == _NOW.timestamp()  # type: ignore[attr-defined]


def test_heartbeat_advances_once_per_tick() -> None:
    store = RecordingStore()
    metrics = MetricRegistry()
    loop = _loop(store, metrics)
    loop.tick()
    loop.tick()
    loop.tick()

    heartbeat = metrics.get(BRAIN_HEARTBEAT)
    assert heartbeat is not None
    assert heartbeat.value() == 3.0  # type: ignore[attr-defined]
    assert store.load().tick_count == 3


def test_heartbeat_metrics_are_declared_and_exported_to_metrics_sqlite() -> None:
    # The heartbeat is SUPPORTING evidence that must actually reach ``metrics.sqlite``
    # (spec §4.4): both series are declared universal specs and ``export=1`` so the
    # sampler snapshots them. A bare tick registers them via CoreLoop.__init__.
    from lifemodel.core.tick_metrics import UNIVERSAL_SPECS

    by_name = {spec.name: spec for spec in UNIVERSAL_SPECS}
    assert by_name[BRAIN_HEARTBEAT].kind == "counter"
    assert by_name[BRAIN_LAST_TICK_EPOCH].kind == "gauge"
    assert by_name[BRAIN_HEARTBEAT].export is True
    assert by_name[BRAIN_LAST_TICK_EPOCH].export is True

    # And a bare CoreLoop (no injected registry) declares them fail-fast in __init__,
    # so the emission on the hot path always lands on a real metric.
    store = RecordingStore()
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    report = CoreLoop(
        registry=reg,
        state_actor=StateActor(store),
        clock=FixedClock(_NOW),
        tracer=FakeTracer(),
        monotonic=Ticking(),
    ).tick()
    assert report.tick == 1
    assert store.load().last_tick_at == to_iso(_NOW)

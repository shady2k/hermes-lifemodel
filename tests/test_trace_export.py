"""Optional OTel tick-export (lm-27n.10) — no-op without the dep, best-effort.

The exporter is a SEPARATE, tick-end capability (NOT a TracerPort decorator): a
:class:`TraceExportPort` whose default :class:`NoopTraceExporter` does nothing,
and an :class:`OtelTraceExporter` behind an ``importlib`` try. The factory
``make_trace_exporter`` returns the OTel one ONLY when ``opentelemetry`` is
importable, else the Noop — so in the Hermes venv (no OTel) it is a true no-op,
and a failing exporter never breaks a tick (CoreLoop swallows it).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

from lifemodel.adapters.trace_export import (
    NoopTraceExporter,
    OtelTraceExporter,
    make_trace_exporter,
)
from lifemodel.core.component import TickContext, layer_for_type
from lifemodel.core.coreloop import CoreLoop, TickReport
from lifemodel.core.intents import Intent, UpdateState
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import FakeTracer


def _report() -> TickReport:
    return TickReport(
        tick=7,
        ran=("a", "b"),
        skipped_broken=(),
        failed=("c",),
        committed=True,
    )


def _trace() -> TraceContext:
    return TraceContext(
        trace_id="0" * 31 + "1", span_id="0" * 15 + "2", parent_span_id="0" * 15 + "3"
    )


# --- the factory: import-gated ----------------------------------------------


def _raise_import(_name: str) -> object:
    raise ImportError("opentelemetry is not installed")


def test_factory_returns_noop_when_opentelemetry_is_unimportable() -> None:
    exporter = make_trace_exporter(import_module=_raise_import)
    assert isinstance(exporter, NoopTraceExporter)


def test_factory_default_is_noop_in_this_venv() -> None:
    # The dev/Hermes venv has no opentelemetry, so the zero-arg default is Noop.
    assert isinstance(make_trace_exporter(), NoopTraceExporter)


def test_factory_returns_otel_when_importable() -> None:
    captured: list[tuple[str, object]] = []

    class _FakeSpan:
        def set_attribute(self, key: str, value: object) -> None:
            captured.append((key, value))

        def end(self) -> None:
            captured.append(("__end__", None))

    class _FakeTracer:
        def start_span(self, name: str) -> _FakeSpan:
            captured.append(("__start__", name))
            return _FakeSpan()

    fake_otel = SimpleNamespace(get_tracer=lambda name: _FakeTracer())

    exporter = make_trace_exporter(import_module=lambda name: fake_otel)
    assert isinstance(exporter, OtelTraceExporter)

    exporter.export_tick(_report(), _trace())
    keys = dict((k, v) for k, v in captured if k not in ("__start__", "__end__"))
    assert keys["trace_id"] == _trace().trace_id
    assert keys["span_id"] == _trace().span_id
    assert keys["parent_span_id"] == _trace().parent_span_id
    assert keys["tick"] == 7
    assert keys["ran"] == 2
    assert keys["failed"] == 1
    assert keys["committed"] is True
    assert keys["launch_count"] == 0
    assert ("__end__", None) in captured  # span always ended


# --- the Noop: does nothing, never raises -----------------------------------


def test_noop_export_does_nothing_and_never_raises() -> None:
    exporter = NoopTraceExporter()
    assert exporter.export_tick(_report(), _trace()) is None
    assert exporter.export_tick(_report(), None) is None  # untraced tick is fine too


# --- CoreLoop swallows a failing exporter -----------------------------------


class _Healthy:
    id = "healthy"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [UpdateState({"u": 0.5})]


class _ExplodingExporter:
    def export_tick(self, report: TickReport, trace: TraceContext | None) -> None:
        raise RuntimeError("exporter blew up")


class _RecordingStore:
    def __init__(self) -> None:
        self._state = State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)

    def reset(self) -> State:
        self._state = State()
        return self._state

    def commit_tick(self, state: State | None, mutations: Sequence[object]) -> None:
        if state is not None:
            self.commit(state)


class _FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def test_coreloop_swallows_a_failing_exporter(tmp_path) -> None:

    reg = ComponentRegistry()
    reg.register(
        _Healthy(),
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    store = _RecordingStore()
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(store),
        clock=_FixedClock(),
        trace_exporter=_ExplodingExporter(),
        tracer=FakeTracer(),
    )
    report = loop.tick()  # must NOT raise — the exporter failure is swallowed
    assert isinstance(report, TickReport)
    assert store.commits[-1].u == 0.5  # the tick committed normally
    assert report.ran == ("healthy",)

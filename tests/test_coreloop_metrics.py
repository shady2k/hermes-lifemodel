"""CoreLoop universal-metric instrumentation (telemetry-core §4.2, bead lm-fib.7.4).

Every tick the harness auto-emits universal metrics into the shared
:class:`~lifemodel.core.metrics.MetricRegistry` WITHOUT the component cooperating:
tick + component duration/lag histograms, run counts by derived status, intake
counts, the per-layer accepts-signals gauge, and the writer drop/error snapshot.
Suppressions are counted at the choke-point (``emit_suppression_span``), so a
failed component's suppression shows up as ``lifemodel_suppressions_total``.
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
from lifemodel.core.taxonomy import contact_observed_signal, contact_signal
from lifemodel.core.tick_metrics import (
    COMPONENT_DURATION,
    COMPONENT_RUNS,
    LAYER_ACCEPTS_SIGNALS,
    SIGNALS_INTAKE,
    SUPPRESSIONS_TOTAL,
    TICK_DURATION,
    TICK_LAG,
    TRACE_WRITER_DROPPED,
    TRACE_WRITER_WRITE_ERRORS,
)
from lifemodel.state.model import State
from lifemodel.testing import FakeTracer


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

    def commit_tick(self, state: State | None, mutations: Sequence[object]) -> None:
        if state is not None:
            self.commit(state)


class Ticking:
    """A monotonic source that advances a fixed step on every read (dt > 0)."""

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


class FailComp:
    id = "fail"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        raise RuntimeError("boom")


class SuppressComp:
    id = "supp"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        # A deliberate no-act: close this component's span "suppressed" (what a
        # real gate does through ``emit_suppression_span``).
        assert ctx.logger is not None
        ctx.logger.span.end(status="suppressed")
        return []


class FakeCountingWriter:
    """A ``TraceSink`` that also exposes absolute drop/error counters."""

    dropped_count = 3
    write_errors = 2

    def submit_span(self, **_kw: object) -> bool:
        return True

    def submit_event(self, **_kw: object) -> bool:
        return True

    def submit_correlation(self, **_kw: object) -> bool:
        return True


def _manifest(cid: str, ctype: str, *, accepts_signals: bool = False) -> ComponentManifest:
    return ComponentManifest(
        id=cid,
        type=ctype,
        layer=layer_for_type(ctype),
        metric_surface=(),
        accepts_signals=accepts_signals,
    )


def _loop(
    registry: ComponentRegistry,
    store: RecordingStore,
    *,
    metrics: MetricRegistry,
    trace_writer: object | None = None,
    breaker_threshold: int = 3,
) -> CoreLoop:
    return CoreLoop(
        registry=registry,
        state_actor=StateActor(store),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        tracer=FakeTracer(),
        metrics=metrics,
        monotonic=Ticking(),
        trace_writer=trace_writer,  # type: ignore[arg-type]
        breaker_threshold=breaker_threshold,
    )


def test_ok_component_records_run_and_positive_duration(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    metrics = MetricRegistry()
    _loop(reg, RecordingStore(), metrics=metrics).tick()

    runs = metrics.get(COMPONENT_RUNS)
    assert runs is not None
    assert runs.value(component="ok", layer="autonomic", outcome="ok") == 1.0  # type: ignore[attr-defined]

    dur = metrics.get(COMPONENT_DURATION)
    assert dur is not None
    snap = dur.snapshot(component="ok", layer="autonomic")  # type: ignore[attr-defined]
    assert snap.count == 1
    assert snap.sum > 0.0


def test_failed_component_is_failed_and_counts_a_suppression(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(FailComp(), _manifest("fail", "neuron"))
    metrics = MetricRegistry()
    _loop(reg, RecordingStore(), metrics=metrics).tick()

    runs = metrics.get(COMPONENT_RUNS)
    assert runs is not None
    assert runs.value(component="fail", layer="autonomic", outcome="failed") == 1.0  # type: ignore[attr-defined]
    assert runs.value(component="fail", layer="autonomic", outcome="ok") == 0.0  # type: ignore[attr-defined]

    # The choke-point (`emit_suppression_span`) counted the component-fault.
    supp = metrics.get(SUPPRESSIONS_TOTAL)
    assert supp is not None
    assert supp.value(component="fail", reason="component_failed") == 1.0  # type: ignore[attr-defined]


def test_suppressed_span_is_recorded_as_suppressed(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(SuppressComp(), _manifest("supp", "neuron"))
    metrics = MetricRegistry()
    _loop(reg, RecordingStore(), metrics=metrics).tick()

    runs = metrics.get(COMPONENT_RUNS)
    assert runs is not None
    assert runs.value(component="supp", layer="autonomic", outcome="suppressed") == 1.0  # type: ignore[attr-defined]
    assert runs.value(component="supp", layer="autonomic", outcome="ok") == 0.0  # type: ignore[attr-defined]


def test_intake_counts_the_frames_seeded_signals(tmp_path) -> None:
    # The ephemeral frame carries every seeded signal through (spec §3): priority-class
    # backpressure is a later slice, so all seeds count as "kept" — no coalescing.
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    metrics = MetricRegistry()
    seeds = [
        contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None),
        contact_observed_signal(origin_id="e2", actor="user", label="two_way", timestamp=None),
        contact_signal(origin_id="c1", value=1.0, delta=0.0, timestamp=None),
        contact_signal(origin_id="c2", value=2.0, delta=0.0, timestamp=None),
    ]
    _loop(reg, RecordingStore(), metrics=metrics).tick(seeds)

    intake = metrics.get(SIGNALS_INTAKE)
    assert intake is not None
    assert (
        intake.value(outcome="kept") == 4.0
    )  # every seeded signal counted  # type: ignore[attr-defined]


def test_tick_duration_and_lag(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    metrics = MetricRegistry()
    # last_tick_at is 60s before the fixed clock → lag == 60s.
    store = RecordingStore(State(last_tick_at="2026-07-06T11:59:00+00:00"))
    _loop(reg, store, metrics=metrics).tick()

    tdur = metrics.get(TICK_DURATION)
    assert tdur is not None
    assert tdur.snapshot().count == 1  # type: ignore[attr-defined]
    assert tdur.snapshot().sum > 0.0  # type: ignore[attr-defined]

    lag = metrics.get(TICK_LAG)
    assert lag is not None
    assert lag.value() == 60.0  # type: ignore[attr-defined]


def test_layer_accepts_signals_gauge(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron", accepts_signals=True))
    reg.register(
        type("Launcher", (), {"id": "lnch", "step": lambda self, ctx: []})(),
        _manifest("lnch", "launcher", accepts_signals=False),
    )
    metrics = MetricRegistry()
    _loop(reg, RecordingStore(), metrics=metrics).tick()

    gauge = metrics.get(LAYER_ACCEPTS_SIGNALS)
    assert gauge is not None
    assert gauge.value(layer="autonomic") == 1.0  # type: ignore[attr-defined]
    assert gauge.value(layer="cognition") == 0.0  # type: ignore[attr-defined]


def test_writer_snapshot_gauges_absolute(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    metrics = MetricRegistry()
    _loop(
        reg,
        RecordingStore(),
        metrics=metrics,
        trace_writer=FakeCountingWriter(),
    ).tick()

    dropped = metrics.get(TRACE_WRITER_DROPPED)
    errors = metrics.get(TRACE_WRITER_WRITE_ERRORS)
    assert dropped is not None and errors is not None
    assert dropped.value() == 3.0  # type: ignore[attr-defined]
    assert errors.value() == 2.0  # type: ignore[attr-defined]


def test_emit_suppression_span_counts_at_the_choke_point() -> None:
    # The choke-point counts EVERY suppression (in-tick + out-of-tick) so the metric
    # can never diverge from the trace — here exercised directly, off any loop.
    from lifemodel.core.suppression import SuppressionReason, emit_suppression_span
    from lifemodel.core.tick_metrics import register_universal_metrics
    from lifemodel.ports.tracer import TraceContext
    from lifemodel.testing import FakeActiveSpan, FakeSpanLogger

    metrics = MetricRegistry()
    register_universal_metrics(metrics)
    span = FakeActiveSpan(TraceContext(trace_id="t", span_id="s"), tick=1)
    logger = FakeSpanLogger(span)

    emit_suppression_span(
        logger,
        reason=SuppressionReason.SILENCE_WINDOW,
        component="contact-aggregation",
        metrics=metrics,
    )

    supp = metrics.get(SUPPRESSIONS_TOTAL)
    assert supp is not None
    assert supp.value(component="contact-aggregation", reason="silence_window") == 1.0  # type: ignore[attr-defined]
    # The span still ends suppressed — counting is additive, not a behaviour change.
    assert span.status == "suppressed"


def test_emit_suppression_span_without_metrics_is_unchanged() -> None:
    # No registry (bare unit test / hand-built graph) → span still records, no count.
    from lifemodel.core.suppression import SuppressionReason, emit_suppression_span
    from lifemodel.ports.tracer import TraceContext
    from lifemodel.testing import FakeActiveSpan, FakeSpanLogger

    span = FakeActiveSpan(TraceContext(trace_id="t", span_id="s"), tick=1)
    logger = FakeSpanLogger(span)
    emit_suppression_span(logger, reason=SuppressionReason.IN_FLIGHT, component="proactive")
    assert span.status == "suppressed"
    assert any(e["event"] == "suppression" for e in logger.events)


def test_tick_never_dies_without_a_metrics_registry(tmp_path) -> None:
    # A bare CoreLoop (no injected registry) still ticks — instrumentation degrades
    # to a private registry, never a crash.
    reg = ComponentRegistry()
    reg.register(OkComp(), _manifest("ok", "neuron"))
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        tracer=FakeTracer(),
        monotonic=Ticking(),
    )
    report = loop.tick()
    assert report.ran == ("ok",)

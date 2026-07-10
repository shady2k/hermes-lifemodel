from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext, layer_for_type
from lifemodel.core.coreloop import CoreLoop, TickReport
from lifemodel.core.intents import EmitSignal, Intent, LaunchProactive, UpdateState
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.core.taxonomy import contact_signal
from lifemodel.domain.memory import MemoryDraft, MemoryMutation, PressureIndex
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State
from lifemodel.testing import FakeMemoryStore, FakeTracer


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

    def reset(self) -> State:
        self._state = State()
        self.commits.append(self._state)
        return self._state

    def commit_tick(self, state: State | None, mutations: Sequence[MemoryMutation]) -> None:
        # State-only in the live loop (no component emits a mutation yet); a
        # state change routes through the same commit-recording path.
        if state is not None:
            self.commit(state)


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


# --- Phase B1 helpers ---


class SeenRecorder:
    """Records what signals it saw in ctx.signals."""

    id = "seen"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.seen = [s.origin_id for s in ctx.signals]
        return []


class ContactEmitter:
    id = "emitter"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [
            EmitSignal(contact_signal(origin_id="c-tick", value=1.0, delta=0.0, timestamp=None))
        ]


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
        tracer=FakeTracer(),
    )


def test_healthy_component_intents_reach_state_and_tick_bumps(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(
        Healthy(),
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    report = loop.tick()
    assert isinstance(report, TickReport)
    assert store.commits[-1].u == 0.42
    assert store.commits[-1].tick_count == 1
    assert store.commits[-1].last_tick_at is not None
    assert report.ran == ("healthy",)


def test_emit_signal_is_transient_not_durable(tmp_path) -> None:
    # EmitSignal threads to later components in-tick; it is NOT written to the bus.
    reg = ComponentRegistry()
    reg.register(
        ContactEmitter(),
        ComponentManifest(
            id="emitter", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    bus = FileSignalBus(tmp_path)
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    assert bus.peek_unprocessed() == []  # transient — nothing persisted


def test_later_component_sees_earlier_components_emitted_signal(tmp_path) -> None:
    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(
        ContactEmitter(),
        ComponentManifest(
            id="emitter", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    reg.register(
        seen,
        ComponentManifest(
            id="seen", type="aggregation", layer=layer_for_type("aggregation"), metric_surface=()
        ),
    )
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path))
    loop.tick()
    assert "c-tick" in seen.seen  # aggregation saw the neuron's transient contact signal


def test_durable_inbound_signal_is_consumed_once_and_threaded(tmp_path) -> None:
    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(
        seen,
        ComponentManifest(
            id="seen", type="aggregation", layer=layer_for_type("aggregation"), metric_surface=()
        ),
    )
    bus = FileSignalBus(tmp_path)
    bus.publish(
        Signal(origin_id="ext-1", kind="exchange", payload={"actor": "user", "label": "two_way"})
    )
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    assert seen.seen == ["ext-1"]  # inbound external input threaded in
    seen.seen = []
    loop.tick()
    assert seen.seen == []  # consumed once — not re-served next tick


def test_failing_component_is_isolated_and_others_still_run(tmp_path) -> None:
    reg = ComponentRegistry()
    healthy = Healthy()
    reg.register(
        Broken(),
        ComponentManifest(
            id="broken", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    reg.register(
        healthy,
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
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
    reg.register(
        broken,
        ComponentManifest(
            id="broken", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
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


def test_coreloop_bounds_a_flood_and_survives(tmp_path) -> None:
    # publish a flood of exchange signals to the bus; one tick must complete,
    # bounded, without raising, and the aggregation-facing ctx must be capped.
    from lifemodel.core.intake import IntakeLimits

    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(
        seen,
        ComponentManifest(
            id="seen", type="aggregation", layer=layer_for_type("aggregation"), metric_surface=()
        ),
    )
    bus = FileSignalBus(tmp_path)
    for i in range(1000):
        bus.publish(
            Signal(
                origin_id=f"e{i}", kind="exchange", payload={"actor": "user", "label": "two_way"}
            )
        )
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=bus,
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        intake_limits=IntakeLimits(max_control=256, max_sensor=64),
        tracer=FakeTracer(),
    )
    report = loop.tick()  # must not raise
    assert report.committed
    assert len(seen.seen) <= 256 + 64  # aggregation saw a bounded batch


def test_coreloop_default_intake_limits_present(tmp_path) -> None:
    loop = _loop(ComponentRegistry(), RecordingStore(), FileSignalBus(tmp_path))
    loop.tick()  # smoke: default IntakeLimits, no flood, still ticks


class Launcher:
    id = "launcher"

    def step(self, ctx) -> list:
        return [
            LaunchProactive(
                prompt="hi",
                correlation_id="c-1",
                origin_traceparent="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            )
        ]


def test_launch_proactive_is_surfaced_in_report(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(
        Launcher(),
        ComponentManifest(
            id="launcher", type="cognition", layer=layer_for_type("cognition"), metric_surface=()
        ),
    )
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path))
    report = loop.tick()
    assert len(report.launches) == 1
    assert report.launches[0].correlation_id == "c-1"
    assert report.launches[0].prompt == "hi"


def test_no_launch_means_empty_tuple(tmp_path) -> None:
    loop = _loop(ComponentRegistry(), RecordingStore(), FileSignalBus(tmp_path))
    assert loop.tick().launches == ()


# --- lm-27n.2: start-of-tick snapshots on TickContext ---


class SnapshotRecorder:
    def __init__(self, id: str) -> None:
        self.id = id
        self.pressure: PressureIndex | None = None
        self.objects: tuple = ()  # type: ignore[type-arg]

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.pressure = ctx.pressure
        self.objects = ctx.objects
        return []


class CountingMemory:
    """Wraps a FakeMemoryStore, counting find() calls to prove read-once."""

    def __init__(self, inner: FakeMemoryStore) -> None:
        self._inner = inner
        self.find_calls = 0

    def find(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.find_calls += 1
        return self._inner.find(*args, **kwargs)  # type: ignore[arg-type]


def _seeded_memory() -> FakeMemoryStore:
    mem = FakeMemoryStore(clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)))
    mem.put(
        MemoryDraft(kind="desire", id="d1", state="active", payload={}, source="t", salience=0.7)
    )
    mem.put(MemoryDraft(kind="desire", id="d2", state="archived", payload={}, source="t"))
    return mem


def test_tick_context_snapshot_populated_once_and_shared(tmp_path) -> None:
    mem = _seeded_memory()
    reg = ComponentRegistry()
    a, b = SnapshotRecorder("a"), SnapshotRecorder("b")
    reg.register(
        a,
        ComponentManifest(id="a", type="neuron", layer=layer_for_type("neuron"), metric_surface=()),
    )
    reg.register(
        b,
        ComponentManifest(
            id="b", type="aggregation", layer=layer_for_type("aggregation"), metric_surface=()
        ),
    )
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        pressure_sensor=mem,
        memory=mem,
        tracer=FakeTracer(),
    )
    loop.tick()

    assert a.pressure is not None and a.pressure.active_desire_count == 1
    assert a.objects == b.objects  # same snapshot handed to every component
    assert a.objects is b.objects  # read ONCE, shared by reference
    assert tuple(o.id for o in a.objects) == ("d1",)  # only the active record


def test_snapshot_includes_deferred_not_only_active(tmp_path) -> None:
    # A deferred desire is LIVE (held), so it must appear in the snapshot — else
    # dedup would miss it. Seed an active + a deferred + a terminal row; the
    # snapshot has the first two, never the terminal one.
    mem = FakeMemoryStore(clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)))
    mem.put(MemoryDraft(kind="desire", id="a", state="active", payload={}, source="t"))
    mem.put(MemoryDraft(kind="desire", id="d", state="deferred", payload={}, source="t"))
    mem.put(MemoryDraft(kind="desire", id="x", state="satisfied", payload={}, source="t"))
    rec = SnapshotRecorder("only")
    reg = ComponentRegistry()
    reg.register(
        rec,
        ComponentManifest(
            id="only", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        memory=mem,
        tracer=FakeTracer(),
    ).tick()
    ids = {o.id for o in rec.objects}
    assert ids == {"a", "d"}  # active + deferred, never the terminal 'satisfied'


def test_snapshot_includes_parked_thought_and_pending_intention(tmp_path) -> None:
    # lm-27n.6: the registry-aware live-state snapshot surfaces EVERY non-terminal
    # row — active + deferred + pending + parked — fixing the earlier active+deferred
    # gap that hid parked thoughts and pending intentions. Terminal rows stay absent.
    from lifemodel.domain.objects import default_registry

    clock = FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))
    mem = FakeMemoryStore(clock=clock)
    mem.put(MemoryDraft(kind="thought", id="t-parked", state="parked", payload={}, source="t"))
    mem.put(MemoryDraft(kind="intention", id="i-pending", state="pending", payload={}, source="t"))
    mem.put(MemoryDraft(kind="desire", id="d-active", state="active", payload={}, source="t"))
    mem.put(MemoryDraft(kind="desire", id="d-deferred", state="deferred", payload={}, source="t"))
    mem.put(MemoryDraft(kind="thought", id="t-dropped", state="dropped", payload={}, source="t"))
    rec = SnapshotRecorder("only")
    reg = ComponentRegistry()
    reg.register(
        rec,
        ComponentManifest(
            id="only", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=clock,
        memory=mem,
        live_states=default_registry().live_states(),
        tracer=FakeTracer(),
    ).tick()
    ids = {o.id for o in rec.objects}
    assert ids == {"t-parked", "i-pending", "d-active", "d-deferred"}  # never the dropped thought


def test_default_snapshot_stays_active_and_deferred_only(tmp_path) -> None:
    # Without an injected live-state set the coreloop keeps the legacy pair, so a
    # parked thought stays invisible — behavior-neutral for callers that don't wire it.
    clock = FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))
    mem = FakeMemoryStore(clock=clock)
    mem.put(MemoryDraft(kind="thought", id="t-parked", state="parked", payload={}, source="t"))
    mem.put(MemoryDraft(kind="desire", id="d-active", state="active", payload={}, source="t"))
    rec = SnapshotRecorder("only")
    reg = ComponentRegistry()
    reg.register(
        rec,
        ComponentManifest(
            id="only", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=clock,
        memory=mem,
        tracer=FakeTracer(),
    ).tick()
    assert {o.id for o in rec.objects} == {"d-active"}  # parked hidden without wiring


def test_tick_context_snapshot_read_exactly_once_per_tick(tmp_path) -> None:
    counting = CountingMemory(_seeded_memory())
    reg = ComponentRegistry()
    for cid in ("a", "b", "c"):
        reg.register(
            SnapshotRecorder(cid),
            ComponentManifest(
                id=cid, type="neuron", layer=layer_for_type("neuron"), metric_surface=()
            ),
        )
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        memory=counting,  # type: ignore[arg-type]
        tracer=FakeTracer(),
    )
    loop.tick()
    # The live snapshot is built ONCE per tick — two bounded finds (active +
    # deferred = the non-terminal states), NOT once per component (which, with 3
    # components, would be 6).
    assert counting.find_calls == 2


def test_snapshot_reads_do_not_change_tick_output(tmp_path) -> None:
    # A tick with snapshot ports commits exactly the same bookkeeping bump as one
    # without — no component consumes the snapshot yet (behavior-neutral).
    reg_with = ComponentRegistry()
    reg_with.register(
        Healthy(),
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    store_with = RecordingStore()
    mem = _seeded_memory()
    CoreLoop(
        registry=reg_with,
        state_actor=StateActor(store_with),
        bus=FileSignalBus(tmp_path / "with"),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        pressure_sensor=mem,
        memory=mem,
        tracer=FakeTracer(),
    ).tick()

    reg_without = ComponentRegistry()
    reg_without.register(
        Healthy(),
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    store_without = RecordingStore()
    _loop(reg_without, store_without, FileSignalBus(tmp_path / "without")).tick()

    assert store_with.commits[-1] == store_without.commits[-1]


def test_tick_context_snapshot_defaults_empty_without_ports(tmp_path) -> None:
    reg = ComponentRegistry()
    rec = SnapshotRecorder("only")
    reg.register(
        rec,
        ComponentManifest(
            id="only", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    _loop(reg, RecordingStore(), FileSignalBus(tmp_path)).tick()
    assert rec.pressure == PressureIndex()
    assert rec.objects == ()


class RaisingMemory:
    """A memory port whose find() always raises — to prove the read is fail-soft."""

    def find(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("transient DB error")


def test_objects_snapshot_read_is_fail_soft(tmp_path) -> None:
    # No component consumes the objects snapshot yet (.3), so a transient DB error
    # on that read must NOT fail the tick before component isolation — behavior-
    # neutral: the tick proceeds, degrading to an empty snapshot.
    reg = ComponentRegistry()
    rec = SnapshotRecorder("only")
    reg.register(
        rec,
        ComponentManifest(
            id="only", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    store = RecordingStore()
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(store),
        bus=FileSignalBus(tmp_path),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        memory=RaisingMemory(),  # type: ignore[arg-type]
        tracer=FakeTracer(),
    )
    report = loop.tick()
    assert report.ran == ("only",)  # the component still ran
    assert report.committed  # the tick still committed its bookkeeping
    assert rec.objects == ()  # degraded to an empty snapshot, no crash


# --- lm-27n.11: the tick mints a trace, threads it, and log-binds it per tick ---


class TraceRecorder:
    """Records the ``TickContext.trace`` it was handed."""

    id = "trace-rec"

    def __init__(self) -> None:
        self.trace = None  # type: ignore[var-annotated]

    def step(self, ctx: TickContext):  # type: ignore[no-untyped-def]
        self.trace = ctx.trace
        return []


class _CapturingSink:
    """A :class:`~lifemodel.state.trace_store.TraceSink` that records every submit.

    The tick fans SpanLoggers onto this, so a test reads back the durable events
    (``submit_event``) and span rows (``submit_span``) — the span tree the CoreLoop
    now persists — without standing up ``observability.sqlite``.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []  # type: ignore[type-arg]
        self.spans: list[dict] = []  # type: ignore[type-arg]

    def submit_event(self, *, record_id, trace_id, span_id, tick, event, ts, fields=None):  # type: ignore[no-untyped-def]
        self.events.append(
            {
                "event": event,
                "trace_id": trace_id,
                "span_id": span_id,
                "tick": tick,
                "fields": dict(fields) if fields else {},
            }
        )
        return True

    def submit_span(  # type: ignore[no-untyped-def]
        self,
        *,
        trace_id,
        span_id,
        parent_span_id=None,
        component=None,
        tick=None,
        started_at=None,
        ended_at=None,
        status=None,
        attrs=None,
    ):
        self.spans.append(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "component": component,
                "tick": tick,
                "status": status,
                "attrs": dict(attrs) if attrs else {},
            }
        )
        return True


def _clock() -> FixedClock:
    return FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))


def test_component_runs_in_a_child_span_of_the_tick_root(tmp_path) -> None:
    # spec §4.2/§5: each component runs in its OWN child span of the tick root — the
    # span tree that makes a tick fully observable. ctx.trace is that CHILD span
    # (not the root), so a creation site stamps the component's span and the
    # component's logs bind to it.
    reg = ComponentRegistry()
    rec = TraceRecorder()
    reg.register(
        rec,
        ComponentManifest(
            id="trace-rec", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        tracer=FakeTracer(),
    )
    loop.tick()
    # A fresh FakeTracer mirrors the loop's own deterministic sequence: start_root
    # consumes trace#1 + span#1; child_of consumes span#2 parented on span#1.
    mirror = FakeTracer()
    root = mirror.start_root()
    child = mirror.child_of(root)
    assert rec.trace is not None
    assert rec.trace.trace_id == root.trace_id == child.trace_id  # same trace
    assert rec.trace.span_id == child.span_id  # the component's own span
    assert rec.trace.span_id != root.span_id  # distinct from the root span
    assert rec.trace.parent_span_id == root.span_id  # parented on the tick root


def test_every_component_gets_a_distinct_child_span(tmp_path) -> None:
    # The span tree has one child per component, all parented on the root, all in the
    # same trace — so "which component did what" is distinguishable from the spans.
    reg = ComponentRegistry()
    rec_a = TraceRecorder()
    rec_b = TraceRecorder()
    rec_a.id = "trace-a"
    rec_b.id = "trace-b"
    reg.register(
        rec_a,
        ComponentManifest(
            id="trace-a", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    reg.register(
        rec_b,
        ComponentManifest(
            id="trace-b", type="aggregation", layer=layer_for_type("aggregation"), metric_surface=()
        ),
    )
    CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        tracer=FakeTracer(),
    ).tick()
    assert rec_a.trace is not None and rec_b.trace is not None
    assert rec_a.trace.trace_id == rec_b.trace.trace_id  # same tick trace
    assert rec_a.trace.span_id != rec_b.trace.span_id  # distinct child spans
    # Both parented on the root span (FakeTracer: root span#1, children span#2/##3).
    assert rec_a.trace.parent_span_id == rec_b.trace.parent_span_id
    assert rec_a.trace.parent_span_id is not None


def test_coreloop_requires_a_tracer_untraced_tick_is_impossible(tmp_path) -> None:
    # spec §5: a log/decision without an active trace+span is STRUCTURALLY impossible
    # — not by discipline. The tracer is a required CoreLoop dependency, so a CoreLoop
    # cannot even be assembled without one: an untraced tick cannot exist.
    with pytest.raises(TypeError):
        CoreLoop(  # type: ignore[call-arg]
            registry=ComponentRegistry(),
            state_actor=StateActor(RecordingStore()),
            bus=FileSignalBus(tmp_path),
            clock=_clock(),
        )


def test_each_tick_gets_a_fresh_trace(tmp_path) -> None:
    from lifemodel.testing import FakeTracer

    reg = ComponentRegistry()
    rec = TraceRecorder()
    reg.register(
        rec,
        ComponentManifest(
            id="trace-rec", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        tracer=FakeTracer(),
    )
    loop.tick()
    first = rec.trace.trace_id  # type: ignore[union-attr]
    loop.tick()
    second = rec.trace.trace_id  # type: ignore[union-attr]
    assert first != second  # a new execution unit per tick


def test_component_failure_is_a_suppression_span_under_the_child_span(tmp_path) -> None:
    # spec §4.1/§5: a component fault is a first-class SUPPRESSION span (reason
    # ``component_failed``), self-stamped with the FAILING component's child span
    # ids (not the tick root) so the span tree records WHICH component faulted. A
    # log without an active span never appears — the SpanLogger stamps it.
    reg = ComponentRegistry()
    reg.register(
        Broken(),
        ComponentManifest(
            id="broken", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    sink = _CapturingSink()
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        trace_writer=sink,
        tracer=FakeTracer(),
    )
    loop.tick()  # Broken raises -> component_failed suppression under the child span

    mirror = FakeTracer()  # mirror the loop's deterministic id sequence
    root = mirror.start_root()  # trace#1 / span#1
    child = mirror.child_of(root)  # span#2 — the component's child

    supp = next(e for e in sink.events if e["event"] == "suppression")
    assert supp["fields"]["reason"] == "component_failed"
    assert supp["fields"]["component"] == "broken"
    assert supp["fields"]["consecutive"] == 1
    assert supp["trace_id"] == root.trace_id
    assert supp["span_id"] == child.span_id  # bound to the failing child span
    assert supp["span_id"] != root.span_id  # not the tick root
    assert supp["tick"] == 1


# --- lm-fib.7.1: spans close at their REAL end instant (non-zero duration) ---


class _TimingSink:
    """A :class:`~lifemodel.state.trace_store.TraceSink` that keeps span timing.

    Unlike :class:`_CapturingSink` it retains ``started_at``/``ended_at`` so a test
    can assert each persisted span has a real, positive duration (spec §4.2:
    latency histograms read empty when ``started_at == ended_at``).
    """

    def __init__(self) -> None:
        self.spans: list[dict] = []  # type: ignore[type-arg]

    def submit_event(self, *, record_id, trace_id, span_id, tick, event, ts, fields=None):  # type: ignore[no-untyped-def]
        return True

    def submit_span(  # type: ignore[no-untyped-def]
        self,
        *,
        trace_id,
        span_id,
        parent_span_id=None,
        component=None,
        tick=None,
        started_at=None,
        ended_at=None,
        status=None,
        attrs=None,
    ):
        self.spans.append(
            {
                "component": component,
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )
        return True


class _FakeMonotonic:
    """A controllable ``time.monotonic`` stand-in: each call advances by ``step``.

    Injecting it makes the elapsed-time proof HONEST and deterministic — the wall
    clock is pinned by ``FixedClock`` (so a real ``time.monotonic`` would give a
    tiny, flaky delta), but this returns a known, strictly increasing sequence, so
    the span's ``ended_at`` is provably a real later instant than its ``started_at``.
    """

    def __init__(self, step: float = 1.0) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def test_every_persisted_span_closes_after_it_started(tmp_path) -> None:
    # lm-fib.7.1: a span must close at its REAL end instant, so ended_at is strictly
    # after started_at (positive duration). Register a healthy AND a failing
    # component so all three persist sites — an ok child span, a failed child span,
    # and the tick root — are exercised. Before the fix every span was closed with
    # started_at (ended_at == started_at) and latency histograms read empty.
    reg = ComponentRegistry()
    reg.register(
        Healthy(),
        ComponentManifest(
            id="healthy", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    reg.register(
        Broken(),
        ComponentManifest(
            id="broken", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    sink = _TimingSink()
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        trace_writer=sink,
        tracer=FakeTracer(),
        monotonic=_FakeMonotonic(),
    )
    loop.tick()

    # The healthy child span, the failed child span, and the tick root all persisted.
    assert {s["component"] for s in sink.spans} == {"healthy", "broken", None}
    for row in sink.spans:
        assert row["started_at"] is not None and row["ended_at"] is not None
        started = datetime.fromisoformat(row["started_at"])
        ended = datetime.fromisoformat(row["ended_at"])
        assert ended > started, f"{row['component']!r} span has non-positive duration"


def test_component_failure_span_row_is_failed_and_parented_on_root(tmp_path) -> None:
    # The persisted span ROW for the fault closes ``failed`` and is parented on the
    # tick root — the durable span tree makes "which component faulted, under which
    # tick" answerable from ``trace_spans`` alone.
    reg = ComponentRegistry()
    reg.register(
        Broken(),
        ComponentManifest(
            id="broken", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    sink = _CapturingSink()
    CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=FileSignalBus(tmp_path),
        clock=_clock(),
        trace_writer=sink,
        tracer=FakeTracer(),
    ).tick()

    mirror = FakeTracer()
    root = mirror.start_root()  # span#1
    child = mirror.child_of(root)  # span#2

    child_row = next(s for s in sink.spans if s["span_id"] == child.span_id)
    assert child_row["status"] == "failed"  # the component raised
    assert child_row["parent_span_id"] == root.span_id  # parented on the tick root
    assert child_row["component"] == "broken"
    assert child_row["attrs"]["reason"] == "component_failed"
    assert child_row["tick"] == 1

"""CoreLoop — the heart that runs one ExecutionFrame (spec §3/§7).

Runs the enabled components for one frame, isolated so no component fault can
crash the heart: every ``step`` call is wrapped; an exception skips that component
and counts toward a per-component circuit-breaker ("living without an organ").

Signal dataflow (spec §2/§3): a frame is seeded with its trigger's
``initial_signals`` into an in-memory :class:`~lifemodel.core.frame.SignalFrame`
(the whole "bus" — ephemeral, never persisted). Each component then sees those
seeds plus every signal emitted by earlier components THIS frame (``EmitSignal``
is threaded in-frame). A signal lives ``<=`` one frame. State intents are
collected and handed — with the frame's own bookkeeping — to the single
:class:`StateActor` for one atomic end-of-frame commit.

A frame is triggered by a heartbeat, an incoming event, an async-cognition
completion, or an admin mutation (:class:`~lifemodel.core.frame.FrameTrigger`);
:func:`~lifemodel.core.frame.run_frame` serializes them through one state-actor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..domain.memory import MemoryRecord, PressureIndex
from ..domain.signal import Signal
from ..events import EventRing
from ..log import SpanLogger
from ..ports.clock import ClockPort
from ..ports.memory import MemoryPort
from ..ports.pressure import PressureSensorPort
from ..ports.trace_export import TraceExportPort
from ..ports.tracer import ActiveSpan, TraceContext, TracerPort, start_span
from ..state.model import State
from ..state.trace_store import NULL_TRACE_SINK, TraceSink
from .component import TickContext
from .frame import FrameTrigger, SignalFrame
from .intents import (
    EmitSignal,
    Intent,
    LaunchProactive,
    PutRecord,
    TransitionRecord,
    UpdateState,
)
from .metrics import MetricRegistry, MetricSpec
from .observer import ComponentObserver
from .registry import ComponentManifest, ComponentRegistry, UnknownComponent
from .state_actor import StateActor
from .suppression import SuppressionReason, emit_suppression_span
from .tick_metrics import (
    COMPONENT_DURATION,
    COMPONENT_RUNS,
    INTAKE_KEPT,
    LAYER_ACCEPTS_SIGNALS,
    RUN_FAILED,
    RUN_OK,
    RUN_SUPPRESSED,
    SIGNALS_INTAKE,
    TICK_DURATION,
    TICK_LAG,
    TRACE_WRITER_DROPPED,
    TRACE_WRITER_WRITE_ERRORS,
    register_universal_metrics,
)
from .timeutil import minutes_between

#: How many records the start-of-tick snapshot pulls *per live state*. A per-tick
#: scan is fine at current scale (lm-fib.6.5 tracks scaling); the cap keeps it
#: bounded. Read by aggregation/cognition since .3.
OBJECTS_SNAPSHOT_LIMIT = 256

#: The fallback live (non-terminal) state-set used when no registry-derived set is
#: injected — the historical ``active`` + ``deferred`` pair. The composition root
#: injects the registry's full :meth:`~lifemodel.domain.objects.KindRegistry.live_states`
#: (``{active, deferred, pending, parked}``) so parked thoughts and pending
#: intentions — both non-terminal, previously invisible — appear in the snapshot.
_DEFAULT_LIVE_STATES: frozenset[str] = frozenset({"active", "deferred"})


@dataclass(frozen=True)
class TickReport:
    """What happened on one tick — for observability/tests."""

    tick: int
    ran: tuple[str, ...]
    skipped_broken: tuple[str, ...]
    failed: tuple[str, ...]
    committed: bool
    launches: tuple[LaunchProactive, ...] = ()
    trigger: FrameTrigger = FrameTrigger.HEARTBEAT


class CoreLoop:
    def __init__(
        self,
        *,
        registry: ComponentRegistry,
        state_actor: StateActor,
        clock: ClockPort,
        trace_writer: TraceSink | None = None,
        event_ring: EventRing | None = None,
        breaker_threshold: int = 3,
        pressure_sensor: PressureSensorPort | None = None,
        memory: MemoryPort | None = None,
        live_states: frozenset[str] | None = None,
        tracer: TracerPort,
        trace_exporter: TraceExportPort | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        metrics: MetricRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._state_actor = state_actor
        self._clock = clock
        # Elapsed is measured with a MONOTONIC source (never jumps when the system
        # clock is stepped), injected so a test can control it and prove a span's
        # duration is non-zero deterministically. The wall-clock start comes from the
        # clock; a span's ``ended_at`` is that start plus the monotonic elapsed.
        self._monotonic = monotonic
        # The tick's durable + freshness sinks (spec §4.2): every component gets a
        # SpanLogger over these, so all tick-path logging is span-bound. Off the
        # live being (a bare test / CLI tick with no BeingAdapter to acquire the
        # writer) both default to a no-op durable sink + a throwaway ring — the
        # pipeline stays uniform, it just persists nothing.
        self._writer: TraceSink = trace_writer if trace_writer is not None else NULL_TRACE_SINK
        self._ring: EventRing = event_ring if event_ring is not None else EventRing()
        self._tracer = tracer
        self._trace_exporter = trace_exporter
        self._breaker_threshold = breaker_threshold
        self._pressure_sensor = pressure_sensor
        self._memory = memory
        self._live_states = live_states if live_states is not None else _DEFAULT_LIVE_STATES
        self._failures: dict[str, int] = {}
        self._broken: set[str] = set()
        # The shared source of CURRENT metric state (telemetry-core §4.1/§4.2): the
        # composition root injects the singleton-per-base_dir registry so tick, hooks
        # and ``/lifemodel stats`` all read one registry; a bare test/CLI loop with no
        # graph falls back to a private registry so instrumentation never crashes for
        # want of one. The universal specs are declared fail-fast here (idempotent —
        # a fresh graph is built every tick), then emitted fail-open on the hot path.
        self._metrics = metrics if metrics is not None else MetricRegistry()
        register_universal_metrics(self._metrics)
        # Declare every component's DOMAIN metric surface into the same registry
        # (telemetry-core §4.3): a spec a component emits through ``ctx.observe`` must
        # exist here first, else the emission fails open as an unknown metric. Fail-fast
        # on a malformed spec, idempotent for an identical one — a fresh graph is built
        # every tick, so this re-runs harmlessly.
        self._register_surface_metrics()

    def _span_logger(self, span: ActiveSpan) -> SpanLogger:
        """A :class:`SpanLogger` bound to *span* over this loop's writer + ring."""
        return SpanLogger(span, writer=self._writer, ring=self._ring)

    def _span_ended_at(self, tick_started_at: datetime, tick_started_mono: float) -> str:
        """The REAL wall-clock instant a span closing *now* ended (spec §4.2).

        Elapsed since the tick began is read from the monotonic source (immune to a
        system-clock step), then added to the tick's wall-clock start — so a span's
        ``ended_at`` is a genuine later instant than its ``started_at`` and its
        persisted duration is the true elapsed time, not zero. Called once per span
        close, so each span records when *it* finished (root last, longest).
        """
        elapsed = self._monotonic() - tick_started_mono
        return (tick_started_at + timedelta(seconds=elapsed)).isoformat()

    def _persist_span(self, span: ActiveSpan, *, ended_at: str) -> None:
        """Upsert one finished span row (spec §4.3) — the durable span tree.

        Fail-open through the writer (a full queue drops it); the attrs bag carries
        the decision values components stamped, so the span is self-explaining.
        """
        ctx = span.context
        self._writer.submit_span(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            component=span.component,
            tick=span.tick,
            started_at=span.started_at,
            ended_at=ended_at,
            status=span.status,
            attrs=dict(span.attrs) or None,
        )

    def tick(
        self,
        initial_signals: Sequence[Signal] = (),
        *,
        trigger: FrameTrigger = FrameTrigger.HEARTBEAT,
    ) -> TickReport:
        """Run one ExecutionFrame, seeded with *initial_signals* (spec §3).

        A heartbeat frame passes no signals; an event / async-completion / admin
        frame passes its trigger signal(s). The signals live in an in-memory
        :class:`~lifemodel.core.frame.SignalFrame` for this one frame only.
        """
        now = self._clock.now()
        state = self._state_actor.state
        # Mint THE frame's root span (continue-or-mint; a frame has no upstream).
        # Tracing is MANDATORY (spec §5): the tracer is a required dependency, so
        # every frame has a root span and a log without an active span is structurally
        # impossible — there is no untraced branch. Components run in CHILD spans of
        # this root (minted per component in ``_run_tick``); each gets a SpanLogger
        # bound to its span that SELF-stamps trace/span/tick (no ambient contextvar
        # bind to leak across frames).
        root = self._tracer.start_root()
        tick_no = state.tick_count + 1
        return self._run_tick(
            now=now,
            state=state,
            root=root,
            tick_no=tick_no,
            initial_signals=initial_signals,
            trigger=trigger,
        )

    def _run_tick(
        self,
        *,
        now: datetime,
        state: State,
        root: TraceContext,
        tick_no: int,
        initial_signals: Sequence[Signal] = (),
        trigger: FrameTrigger = FrameTrigger.HEARTBEAT,
    ) -> TickReport:
        started = now.isoformat()
        # Monotonic origin for this tick: every span's ``ended_at`` is ``started``
        # plus the monotonic elapsed at its close (see ``_span_ended_at``), so span
        # durations are real, not zero.
        started_mono = self._monotonic()
        # The tick's ROOT span + its logger: tick-level bookkeeping (snapshot/intake
        # failures, trace export) binds here; each component rebinds to its own child.
        root_span = start_span(root, tick=tick_no, started_at=started)
        root_logger = self._span_logger(root_span)
        # Start-of-tick snapshot: read once, before any component runs, so every
        # component this tick sees one consistent view (HLA §4.1). Pure reads —
        # they never change tick output; no component consumes them yet (.3).
        pressure = (
            self._pressure_sensor.read_pressure_index(now)
            if self._pressure_sensor is not None
            else PressureIndex()
        )
        # The live (non-terminal) objects snapshot — one bounded find per state in
        # the registry's live-state set (``{active, deferred, pending, parked}``),
        # unioned. A row in any of these is still LIVE: a deferred/held desire, a
        # pending intention awaiting its trigger, a parked thought — all must be
        # visible, else next tick's dedup/render would miss them (the earlier
        # active+deferred-only snapshot silently dropped parked thoughts and pending
        # intentions). Each state is one state of exactly one row, so the finds are
        # disjoint; iterating sorted keeps the union order deterministic. Terminal
        # rows (satisfied/dropped/expired/archived/...) are absence, never fetched.
        # Fail-soft like the pressure read: a transient DB error degrades to an
        # empty snapshot rather than failing the tick before component isolation.
        objects: tuple[MemoryRecord, ...] = ()
        if self._memory is not None:
            try:
                found: list[MemoryRecord] = []
                for live_state in sorted(self._live_states):
                    found.extend(self._memory.find(state=live_state, limit=OBJECTS_SNAPSHOT_LIMIT))
                objects = tuple(found)
            except Exception as exc:  # noqa: BLE001 - fail-soft snapshot read
                root_logger.info("objects_snapshot_failed", error=repr(exc))
        # The frame's in-memory SignalFrame (spec §2/§3): seed it with the trigger's
        # signals. No durable bus, no cursor — an impulse lives only for this frame.
        # Backpressure (must_process vs best_effort shedding) is the AGGREGATION
        # layer's job now, not the bus's (spec §7) — the frame carries every seed
        # through; a component sheds what it must.
        frame = SignalFrame(initial_signals)
        # Universal metrics (telemetry-core §4.2): the seeded-signal count and the
        # per-layer accepts-signals gauge. Emission is fail-open (never raises).
        self._metrics.inc(SIGNALS_INTAKE, len(frame), outcome=INTAKE_KEPT)
        self._emit_accepts_signals()

        intents: list[Intent] = []
        launches: list[LaunchProactive] = []
        ran: list[str] = []
        failed: list[str] = []

        for component in self._registry.enabled():
            if component.id in self._broken:
                continue
            # Each component runs in its OWN child span of the tick root (spec §4.2):
            # the span tree makes "which component did what / why did it stay silent"
            # observable. ``ctx.trace`` carries the child's W3C ids (so a creation site
            # stamps the component's span, not the root); ``ctx.logger`` is a SpanLogger
            # bound to that span, self-stamping the component's logs — plus its decision
            # attrs and a failure record — onto it. Every span is persisted after the
            # step (ok / suppressed / failed) so the durable span tree is complete.
            child_ctx = self._tracer.child_of(root)
            span = start_span(child_ctx, component=component.id, tick=tick_no, started_at=started)
            logger = self._span_logger(span)
            manifest = self._manifest_or_none(component.id)
            layer = (
                manifest.layer.value
                if manifest is not None and manifest.layer is not None
                else "other"
            )
            ctx = TickContext(
                state=state,
                now=now,
                signals=frame.snapshot(),
                pressure=pressure,
                objects=objects,
                trace=child_ctx,
                logger=logger,
                # The shared metric registry (telemetry-core §4.2): a component gate
                # emits its suppression through the choke-point with this, so an
                # in-tick gate's suppression is counted like any other.
                metrics=self._metrics,
                # THIS component's domain-metric channel (telemetry-core §4.3): bound
                # to its DECLARED metric_surface, so ``ctx.observe`` can only emit what
                # the component said it would — anything else is a counted no-op.
                observe=ComponentObserver.bind(
                    self._metrics, manifest.metric_surface if manifest is not None else ()
                ),
                # The async-bridge emit trio (spec §4.4): a component can weave an
                # out-of-band span onto a FOREIGN origin trace (aggregation resolving
                # a proactive attempt under its launch trace) through the SAME sinks.
                tracer=self._tracer,
                trace_writer=self._writer,
                event_ring=self._ring,
            )
            # Real per-component latency (telemetry-core §4.2, needs the span-timing
            # fix, §5): measured off the MONOTONIC source so it never jumps on a
            # system-clock step and a test can prove dt > 0 deterministically.
            component_started_mono = self._monotonic()
            try:
                produced = component.step(ctx)
            except Exception as exc:  # isolation: the heart never dies
                component_dt = self._monotonic() - component_started_mono
                self._record_failure(component.id, exc, logger)
                self._persist_span(span, ended_at=self._span_ended_at(now, started_mono))
                self._emit_component_run(component.id, layer, RUN_FAILED, component_dt)
                failed.append(component.id)
                continue
            component_dt = self._monotonic() - component_started_mono
            self._failures[component.id] = 0
            for intent in produced:
                if isinstance(intent, EmitSignal):
                    # In-memory only (spec §3): visible to LATER components this frame,
                    # never persisted — the SignalFrame dies at end of frame.
                    frame.emit(intent.signal)
                elif isinstance(intent, LaunchProactive):
                    launches.append(intent)
                else:
                    intents.append(intent)
            ran.append(component.id)
            self._persist_span(span, ended_at=self._span_ended_at(now, started_mono))
            # Status derivation (§4.2): the exception path is RUN_FAILED above; here a
            # gate that closed its span "suppressed" is RUN_SUPPRESSED, else RUN_OK.
            status = RUN_SUPPRESSED if span.status == "suppressed" else RUN_OK
            self._emit_component_run(component.id, layer, status, component_dt)

        intents.append(
            UpdateState({"tick_count": state.tick_count + 1, "last_tick_at": now.isoformat()})
        )
        # A memory mutation commits durable state even when the State row itself is
        # unchanged, so ``committed`` must reflect it too (today the tick always
        # bumps ``tick_count`` so the State always changes, but this keeps the
        # report honest once mutation-only paths are live — lm-27n.3).
        had_mutation = any(isinstance(i, PutRecord | TransitionRecord) for i in intents)
        new_state = self._state_actor.apply(intents)

        report = TickReport(
            tick=new_state.tick_count,
            ran=tuple(ran),
            skipped_broken=tuple(sorted(self._broken)),
            failed=tuple(failed),
            committed=new_state is not state or had_mutation,
            launches=tuple(launches),
            trigger=trigger,
        )
        # Ship the finished tick to the (optional) trace backend AFTER the commit —
        # BEST-EFFORT: an exporter that raises must never affect the tick outcome
        # (fail-soft, like the snapshot read). Noop default → behaviour-neutral. The
        # root span is always present (tracing is mandatory), so the exporter always
        # has a root span to ship.
        self._export_tick(report, root, root_logger)
        # Persist the root span row LAST — the tick tree's parent (spec §4.3). Its
        # ``ended_at`` snapshots elapsed at the very end of the tick, so the root
        # encloses every child span.
        self._persist_span(root_span, ended_at=self._span_ended_at(now, started_mono))
        # Tick-level universal metrics LAST, so the duration encloses the whole tick
        # (telemetry-core §4.2). ``state`` is the start-of-tick snapshot: its
        # ``last_tick_at`` is the PREVIOUS tick's stamp, so the lag is genuine.
        self._emit_tick_summary(state=state, now=now, started_mono=started_mono)
        return report

    def _export_tick(self, report: TickReport, trace: TraceContext, logger: SpanLogger) -> None:
        if self._trace_exporter is None:
            return
        try:
            self._trace_exporter.export_tick(report, trace)
        except Exception as exc:  # noqa: BLE001 - best-effort; never break the tick
            logger.info("trace_export_failed", error=repr(exc))

    def _record_failure(self, component_id: str, exc: Exception, logger: SpanLogger) -> None:
        count = self._failures.get(component_id, 0) + 1
        self._failures[component_id] = count
        # A component fault suppresses the tick's outcome — record it as a proper
        # suppression span (spec §5): reason COMPONENT_FAILED, the span closed
        # ``failed`` with the error + consecutive count on its attrs bag. The
        # choke-point counts it into ``lifemodel_suppressions_total`` (§4.2).
        emit_suppression_span(
            logger,
            reason=SuppressionReason.COMPONENT_FAILED,
            component=component_id,
            status="failed",
            metrics=self._metrics,
            error=repr(exc),
            consecutive=count,
        )
        if count >= self._breaker_threshold and component_id not in self._broken:
            self._broken.add(component_id)
            logger.warning("circuit_breaker_open", component=component_id, after=count)

    # ---- universal-metric emission (telemetry-core §4.2) ---------------- #

    def _manifest_or_none(self, component_id: str) -> ComponentManifest | None:
        """This component's manifest, or ``None`` if somehow unregistered.

        The one lookup the per-component loop needs for both the low-cardinality
        ``layer`` label and the observer's declared ``metric_surface`` (register()
        guarantees a registered component has a non-``None`` layer)."""
        try:
            return self._registry.manifest(component_id)
        except UnknownComponent:
            return None

    def _register_surface_metrics(self) -> None:
        """Declare every registered component's domain ``metric_surface`` specs into
        the shared registry (telemetry-core §4.3).

        Only full :class:`~lifemodel.core.metrics.MetricSpec` entries can be declared;
        a bare-name entry references a metric declared elsewhere. Registration is
        fail-fast on a malformed spec and idempotent for an identical one."""
        for manifest in self._registry.manifests():
            for entry in manifest.metric_surface or ():
                if isinstance(entry, MetricSpec):
                    self._metrics.register(entry)

    def _emit_component_run(self, component_id: str, layer: str, status: str, dt: float) -> None:
        """Count one component run by derived status + record its real duration."""
        self._metrics.inc(COMPONENT_RUNS, component=component_id, layer=layer, outcome=status)
        self._metrics.observe(COMPONENT_DURATION, dt, component=component_id, layer=layer)

    def _emit_accepts_signals(self) -> None:
        """Set the per-layer accepts-signals gauge from the MANIFEST (§4.2 — registry
        knowledge, not a component's; the harness emits it, never ``ctx.observe``).

        A layer's gauge is 1 if ANY registered component in it consumes signals."""
        by_layer: dict[str, bool] = {}
        for manifest in self._registry.manifests():
            layer = manifest.layer.value if manifest.layer is not None else "other"
            by_layer[layer] = by_layer.get(layer, False) or manifest.accepts_signals
        for layer, accepts in by_layer.items():
            self._metrics.set(LAYER_ACCEPTS_SIGNALS, 1.0 if accepts else 0.0, layer=layer)

    def _emit_tick_summary(self, *, state: State, now: datetime, started_mono: float) -> None:
        """Emit the tick-level metrics: duration, lag, and the writer snapshot (§4.2)."""
        self._metrics.observe(TICK_DURATION, self._monotonic() - started_mono)
        # ``last_tick_at`` is the PREVIOUS tick's stamp; minutes_between is the
        # defensive parser (None / unparseable / naive → 0.0), converted to seconds.
        self._metrics.set(TICK_LAG, minutes_between(state.last_tick_at, now) * 60.0)
        # Writer drop/error counters are ABSOLUTE process-local values — SNAPSHOT them
        # as a gauge (§4.2 codex #7), never ``inc()`` (that would double-count). The
        # injected sink is the same instance the live being acquired; a bare
        # NULL_TRACE_SINK has no counters and is simply skipped.
        dropped = getattr(self._writer, "dropped_count", None)
        errors = getattr(self._writer, "write_errors", None)
        if dropped is not None:
            self._metrics.set(TRACE_WRITER_DROPPED, float(dropped))
        if errors is not None:
            self._metrics.set(TRACE_WRITER_WRITE_ERRORS, float(errors))

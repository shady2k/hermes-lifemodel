"""CoreLoop — the heart/scheduler (spec §7).

Runs the enabled components each tick, isolated so no component fault can crash
the heart: every ``step`` call is wrapped; an exception skips that component and
counts toward a per-component circuit-breaker ("living without an organ").

Signal dataflow (spec §7.4): durable external inputs are consumed from the bus
**once** at tick start; each component then sees those inputs plus every
transient signal emitted by earlier components this tick (``EmitSignal`` is
threaded in-tick, **not** re-published — a signal recomputed every tick must not
be re-consumed and double-counted). State intents are collected and handed —
together with the tick's own bookkeeping — to the single :class:`StateActor` for
one atomic checkpoint.

Phase B1 runs *every* enabled component each tick. Energy budgeting (which gates
the expensive cognition layer) slots into the per-component loop in Phase C.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
from .intake import IntakeLimits, apply_intake
from .intents import (
    EmitSignal,
    Intent,
    LaunchProactive,
    PutRecord,
    TransitionRecord,
    UpdateState,
)
from .registry import ComponentRegistry
from .signal_bus import SignalBus
from .state_actor import StateActor
from .suppression import SuppressionReason, emit_suppression_span
from .taxonomy import lane_of

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


class CoreLoop:
    def __init__(
        self,
        *,
        registry: ComponentRegistry,
        state_actor: StateActor,
        bus: SignalBus,
        clock: ClockPort,
        trace_writer: TraceSink | None = None,
        event_ring: EventRing | None = None,
        breaker_threshold: int = 3,
        intake_limits: IntakeLimits | None = None,
        pressure_sensor: PressureSensorPort | None = None,
        memory: MemoryPort | None = None,
        live_states: frozenset[str] | None = None,
        tracer: TracerPort,
        trace_exporter: TraceExportPort | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._state_actor = state_actor
        self._bus = bus
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
        self._intake_limits = intake_limits or IntakeLimits()
        self._pressure_sensor = pressure_sensor
        self._memory = memory
        self._live_states = live_states if live_states is not None else _DEFAULT_LIVE_STATES
        self._failures: dict[str, int] = {}
        self._broken: set[str] = set()

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

    def tick(self) -> TickReport:
        now = self._clock.now()
        state = self._state_actor.state
        # Mint THE tick's root span (continue-or-mint; a cron tick has no upstream).
        # Tracing is MANDATORY (spec §5): the tracer is a required dependency, so
        # every tick has a root span and a log without an active span is structurally
        # impossible — there is no untraced branch. Components run in CHILD spans of
        # this root (minted per component in ``_run_tick``); each gets a SpanLogger
        # bound to its span that SELF-stamps trace/span/tick (no ambient contextvar
        # bind to leak across ticks).
        root = self._tracer.start_root()
        tick_no = state.tick_count + 1
        return self._run_tick(now=now, state=state, root=root, tick_no=tick_no)

    def _run_tick(
        self, *, now: datetime, state: State, root: TraceContext, tick_no: int
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
        intake = apply_intake(
            self._bus.consume_unprocessed(), limits=self._intake_limits, lane_of=lane_of
        )
        if intake.shed_control or intake.shed_sensor or intake.coalesced_sensor:
            root_logger.info(
                "signals_shed",
                shed_control=intake.shed_control,
                shed_sensor=intake.shed_sensor,
                coalesced_sensor=intake.coalesced_sensor,
            )
        available: list[Signal] = list(intake.kept)

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
            ctx = TickContext(
                state=state,
                now=now,
                bus=self._bus,
                signals=tuple(available),
                pressure=pressure,
                objects=objects,
                trace=child_ctx,
                logger=logger,
                # The async-bridge emit trio (spec §4.4): a component can weave an
                # out-of-band span onto a FOREIGN origin trace (aggregation resolving
                # a proactive attempt under its launch trace) through the SAME sinks.
                tracer=self._tracer,
                trace_writer=self._writer,
                event_ring=self._ring,
            )
            try:
                produced = component.step(ctx)
            except Exception as exc:  # isolation: the heart never dies
                self._record_failure(component.id, exc, logger)
                self._persist_span(span, ended_at=self._span_ended_at(now, started_mono))
                failed.append(component.id)
                continue
            self._failures[component.id] = 0
            for intent in produced:
                if isinstance(intent, EmitSignal):
                    available.append(
                        intent.signal
                    )  # transient — visible to later components this tick
                elif isinstance(intent, LaunchProactive):
                    launches.append(intent)
                else:
                    intents.append(intent)
            ran.append(component.id)
            self._persist_span(span, ended_at=self._span_ended_at(now, started_mono))

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
        # ``failed`` with the error + consecutive count on its attrs bag.
        emit_suppression_span(
            logger,
            reason=SuppressionReason.COMPONENT_FAILED,
            component=component_id,
            status="failed",
            error=repr(exc),
            consecutive=count,
        )
        if count >= self._breaker_threshold and component_id not in self._broken:
            self._broken.add(component_id)
            logger.warning("circuit_breaker_open", component=component_id, after=count)

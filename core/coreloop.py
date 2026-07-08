"""CoreLoop — the heart/scheduler (spec §7).

Runs the enabled components each tick, isolated so no component fault can crash
the heart: every ``step`` call is wrapped; an exception skips that component and
counts toward a per-component circuit-breaker ("живёт без органа").

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

from dataclasses import dataclass
from datetime import datetime

from ..domain.memory import MemoryRecord, PressureIndex
from ..domain.signal import Signal
from ..log import EventLogger, bound_log_context
from ..ports.clock import ClockPort
from ..ports.memory import MemoryPort
from ..ports.pressure import PressureSensorPort
from ..ports.tracer import TraceContext, TracerPort
from ..state.model import State
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


def _tick_log_fields(trace: TraceContext | None, tick: int) -> dict[str, object]:
    """The fields bound onto every log line for one tick (lm-27n.11).

    Untraced (no tracer wired → ``trace is None``) → empty, so the bind is a no-op and
    the tick's logs are byte-identical to before .11. Traced → the W3C correlation ids
    + the tick number, so logs and durable provenance JOIN on ``trace_id``.
    """
    if trace is None:
        return {}
    return {
        "trace_id": trace.trace_id,
        "span_id": trace.span_id,
        "parent_span_id": trace.parent_span_id,
        "tick": tick,
    }


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
        logger: EventLogger | None = None,
        breaker_threshold: int = 3,
        intake_limits: IntakeLimits | None = None,
        pressure_sensor: PressureSensorPort | None = None,
        memory: MemoryPort | None = None,
        live_states: frozenset[str] | None = None,
        tracer: TracerPort | None = None,
    ) -> None:
        self._registry = registry
        self._state_actor = state_actor
        self._bus = bus
        self._clock = clock
        self._log = logger
        self._tracer = tracer
        self._breaker_threshold = breaker_threshold
        self._intake_limits = intake_limits or IntakeLimits()
        self._pressure_sensor = pressure_sensor
        self._memory = memory
        self._live_states = live_states if live_states is not None else _DEFAULT_LIVE_STATES
        self._failures: dict[str, int] = {}
        self._broken: set[str] = set()

    def tick(self) -> TickReport:
        now = self._clock.now()
        state = self._state_actor.state
        # Mint THE tick's root trace (continue-or-mint; a cron tick has no upstream).
        # ONE root span per tick — components read it via ``TickContext.trace``, never
        # the tracer itself (the tracer is an injected capability; the active trace is
        # per-tick state, not ambient). No tracer wired → ``None`` → behaviour-neutral.
        trace = self._tracer.start_root() if self._tracer is not None else None
        # Bind the trace onto every log line emitted THIS tick, reset at tick end (no
        # stale bind leaking across ticks). Wrapped in log.py so the CoreLoop never
        # imports structlog; empty fields (untraced) makes it a no-op.
        with bound_log_context(**_tick_log_fields(trace, state.tick_count + 1)):
            return self._run_tick(now=now, state=state, trace=trace)

    def _run_tick(self, *, now: datetime, state: State, trace: TraceContext | None) -> TickReport:
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
                if self._log is not None:
                    self._log.info("objects_snapshot_failed", error=repr(exc))
        intake = apply_intake(
            self._bus.consume_unprocessed(), limits=self._intake_limits, lane_of=lane_of
        )
        if self._log is not None and (
            intake.shed_control or intake.shed_sensor or intake.coalesced_sensor
        ):
            self._log.info(
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
            ctx = TickContext(
                state=state,
                now=now,
                bus=self._bus,
                signals=tuple(available),
                pressure=pressure,
                objects=objects,
                trace=trace,
            )
            try:
                produced = component.step(ctx)
            except Exception as exc:  # isolation: the heart never dies
                self._record_failure(component.id, exc)
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

        intents.append(
            UpdateState({"tick_count": state.tick_count + 1, "last_tick_at": now.isoformat()})
        )
        # A memory mutation commits durable state even when the State row itself is
        # unchanged, so ``committed`` must reflect it too (today the tick always
        # bumps ``tick_count`` so the State always changes, but this keeps the
        # report honest once mutation-only paths are live — lm-27n.3).
        had_mutation = any(isinstance(i, PutRecord | TransitionRecord) for i in intents)
        new_state = self._state_actor.apply(intents)

        return TickReport(
            tick=new_state.tick_count,
            ran=tuple(ran),
            skipped_broken=tuple(sorted(self._broken)),
            failed=tuple(failed),
            committed=new_state is not state or had_mutation,
            launches=tuple(launches),
        )

    def _record_failure(self, component_id: str, exc: Exception) -> None:
        count = self._failures.get(component_id, 0) + 1
        self._failures[component_id] = count
        if self._log is not None:
            self._log.info(
                "component_failed", component=component_id, error=repr(exc), consecutive=count
            )
        if count >= self._breaker_threshold and component_id not in self._broken:
            self._broken.add(component_id)
            if self._log is not None:
                self._log.info("circuit_breaker_open", component=component_id, after=count)

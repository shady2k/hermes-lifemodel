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

from ..domain.signal import Signal
from ..log import EventLogger
from ..ports.clock import ClockPort
from .component import TickContext
from .intake import IntakeLimits, apply_intake
from .intents import EmitSignal, Intent, LaunchProactive, UpdateState
from .registry import ComponentRegistry
from .signal_bus import SignalBus
from .state_actor import StateActor
from .taxonomy import lane_of


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
    ) -> None:
        self._registry = registry
        self._state_actor = state_actor
        self._bus = bus
        self._clock = clock
        self._log = logger
        self._breaker_threshold = breaker_threshold
        self._intake_limits = intake_limits or IntakeLimits()
        self._failures: dict[str, int] = {}
        self._broken: set[str] = set()

    def tick(self) -> TickReport:
        now = self._clock.now()
        state = self._state_actor.state
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
            ctx = TickContext(state=state, now=now, bus=self._bus, signals=tuple(available))
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
        new_state = self._state_actor.apply(intents)

        return TickReport(
            tick=new_state.tick_count,
            ran=tuple(ran),
            skipped_broken=tuple(sorted(self._broken)),
            failed=tuple(failed),
            committed=new_state is not state,
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

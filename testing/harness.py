"""Integration harness — drive the REAL CoreLoop + real components through fakes (spec §6).

This is the heart of the rebuild: the simulation no longer has a third tick model.
The harness runs the actual spine — ``PresenceNeuron → SolitudeDrive →
ContactAggregation → CognitionLauncher`` (the same code the live being runs) — over
the real SQLite store, through fake ports, so a green scenario HONESTLY predicts
live behaviour. The one thing that is not real here is the async act-gate (the
being's Hermes turn): the harness scripts its verdict (FULFILL on a message, REJECT
on ``[SILENT]``) by feeding a ``verdict`` signal into the SAME read-back path the
``post_llm`` hook uses live.

A scenario is a list of :class:`Step` (advance the fake clock to accumulate
silence, optionally publish an exchange, optionally script the act-gate verdict).
Each step runs ONE ``proactive_tick`` (real pipeline → real backstop → recording
egress) and records what happened: the live desire's state, whether a launch
reached the egress (and the impulse text), the delivery outcome, and the
suppression-span reasons emitted that tick — the span tree that makes a quiet tick
as debuggable as a loud one (spec §5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..composition import build_lifemodel
from ..core.desire_view import read_live_contact_desire
from ..core.proactive import proactive_tick
from ..core.taxonomy import exchange_signal, verdict_signal
from ..domain.egress import ReachOutcome, Verdict
from ..events import EventRing
from ..ports.memory import MemoryPort
from ..sim.quality import Actor, Label
from ..state.model import State
from .fakes import FakeClock


@dataclass(frozen=True)
class Step:
    """One harness tick: advance the clock, then optional events, then run the tick.

    ``advance`` accumulates silence (the drive rises by ``Δt``). ``exchange``
    publishes a real inbound exchange (``(actor, label)`` — never
    ``proactive_internal``). ``act_gate`` scripts the async Hermes turn's verdict
    for the turn currently in flight (``pending_proactive_id``) — the read-back path.
    """

    advance: timedelta = timedelta(0)
    exchange: tuple[Actor, Label] | None = None
    act_gate: Verdict | None = None


@dataclass(frozen=True)
class TickRecord:
    """What happened on one harness tick — the outcome AND the span tree."""

    tick: int
    outcome: ReachOutcome | None  # delivery outcome; None = the core stayed quiet
    desire_state: str | None  # live contact-desire state (active/deferred), else None
    launched: bool  # a LaunchProactive reached the egress this tick
    delivered_impulse: str | None  # the impulse text handed to the egress (if launched)
    suppressions: tuple[str, ...]  # reasons of the suppression spans emitted this tick
    u: float  # the drive vital after the tick


class RecordingEgress:
    """A :class:`~lifemodel.ports.proactive.ProactiveEgressPort` that records
    ``reach_out`` calls without sending anything, returning a fixed outcome."""

    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, str]] = []

    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        self.calls.append((target, impulse))
        return self.outcome


@dataclass
class IntegrationHarness:
    """Drive the real being spine through fake ports over a tmp dir.

    Builds the real graph once (real components + the SQLite store) and reuses it
    across ticks — the CoreLoop/StateActor are reusable, state persists in the
    store, so this runs the identical tick code the live being runs. The fake clock
    is advanced per step; the recording egress catches launches without sending; the
    recording logger captures every suppression span (the span tree)."""

    base_dir: Path
    clock: FakeClock = field(default_factory=lambda: FakeClock(datetime(2026, 1, 1, tzinfo=UTC)))
    event_ring: EventRing = field(default_factory=EventRing)
    egress: RecordingEgress = field(default_factory=RecordingEgress)
    target: dict[str, str | None] = field(
        default_factory=lambda: {"platform": "test", "chat_id": "1", "thread_id": None}
    )
    initial_state: State | None = None
    records: list[TickRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lm = build_lifemodel(
            base_dir=self.base_dir,
            clock=self.clock,
            event_ring=self.event_ring,
        )
        # Seed the start state (loaded lazily by the StateActor on the first tick).
        # The default is a rested start with last_tick_at set, so the FIRST clock
        # advance yields a real Δt (else minutes_between(None, now) = 0 and the drive
        # never rises on step 0) and cognition can afford a launch. A caller may pass
        # ``initial_state`` to land a tick inside a specific gate (a recent exchange,
        # an active decline backoff, a rate-limiting send log) without driving the
        # whole flow there — the components then run for real on that state.
        initial = self.initial_state
        if initial is None:
            initial = State(
                u=0.0, energy=1.0, fatigue=0.0, last_tick_at=self.clock.now().isoformat()
            )
        self._lm.state.commit(initial)

    def run(self, steps: Sequence[Step]) -> list[TickRecord]:
        """Run each step in order, appending a :class:`TickRecord` per tick."""
        for step in steps:
            self.records.append(self._step(step))
        return self.records

    def _memory(self) -> MemoryPort:
        # The live SQLite store is both StatePort and MemoryPort; narrow for the
        # typed readers (desire/intention views) that take a MemoryPort.
        memory = self._lm.state
        assert isinstance(memory, MemoryPort), "harness store must be a MemoryPort"
        return memory

    def _step(self, step: Step) -> TickRecord:
        self.clock.advance(step.advance)
        now = self.clock.now()
        # Feed this step's events into the same bus the live tick consumes.
        if step.exchange is not None:
            actor, label = step.exchange
            self._lm.bus.publish(
                exchange_signal(
                    origin_id=f"exchange-{len(self.records)}",
                    actor=actor,
                    label=label,
                    timestamp=now.isoformat(),
                )
            )
        if step.act_gate is not None:
            # Script the async act-gate: feed its verdict into the read-back path
            # (the verdict signal), correlated to the turn currently in flight.
            pending = self._lm.state.load().pending_proactive_id
            if pending:
                self._lm.bus.publish(
                    verdict_signal(
                        origin_id=f"verdict-{pending}",
                        verdict=step.act_gate,
                        timestamp=now.isoformat(),
                        correlation_id=pending,
                    )
                )
        ring_before = len(self.event_ring.read())
        egress_before = len(self.egress.calls)
        # The real delivery path: pipeline → backstop → recording egress.
        outcome = proactive_tick(self._lm, self.egress, self.target)
        # Suppression spans route through the SpanLogger onto the freshness ring
        # (spec §4.2/§5), not the ad-hoc logger — read this step's slice back.
        new_ring = self.event_ring.read()[ring_before:]
        new_egress = self.egress.calls[egress_before:]
        suppressions = tuple(
            rec["reason"]
            for rec in new_ring
            if rec.get("event") == "suppression" and "reason" in rec
        )
        desire = read_live_contact_desire(self._memory())
        final = self._lm.state.load()
        return TickRecord(
            tick=final.tick_count,
            outcome=outcome,
            desire_state=desire.state if desire is not None else None,
            launched=bool(new_egress),
            delivered_impulse=new_egress[-1][1] if new_egress else None,
            suppressions=suppressions,
            u=final.u,
        )

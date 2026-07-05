"""The simulation harness — the tick-stepper that certifies the model (spec §9).

Pure Python, no Hermes. The harness composes the three certified primitives —
the :class:`~lifemodel.sim.drive.Drive` (the continuous urge ``u``), the
wake-decision gates (:func:`~lifemodel.sim.wake.evaluate_wake`), and the
:class:`~lifemodel.sim.aggregation.Aggregator` (the desire lifecycle) — and
drives them over a trace of lane events plus a *scripted cognition*.

Scripted cognition (spec §9's ``cognition_verdict`` column): cognition is the
LLM and is out of scope for the model, so the harness scripts its verdicts as a
FIFO ``verdicts`` list consumed by successive wakes, falling back to
``default_verdict`` once exhausted. ``user_available`` is a latched environment
signal the release conditions read. Both are *inputs*, never inferred here.

Each tick the harness: (1) applies the tick's events (satiation, clock, desire
clearing, latched availability/in-flight); (2) rises the drive in genuine
silence; (3) updates the neuron-owned ``duration_over_θ``; (4) runs the
wake-decision (create / dedup / deferred-release); (5) applies the scripted
verdict; and records the end-of-tick state. Time unit and ``Δt`` are the
caller's; the shipped scenarios use minutes with ``Δt = 1``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .aggregation import Aggregator, DesireStatus, Verdict
from .drive import Drive
from .quality import Actor, Label, quality_of
from .wake import GateParams, LaneState, evaluate_wake


@dataclass(frozen=True)
class SimEvent:
    """One row of the trace (spec §9). An event may be an exchange, an
    availability signal, an in-flight marker, or any combination.

    - ``actor`` + ``label`` → an exchange (``proactive_internal`` never counts).
    - ``user_available`` → latched presence/availability the release reads.
    - ``in_flight`` → latched "a turn is running/queued" marker for the gate.
    """

    time: float
    actor: Actor | None = None
    label: Label | None = None
    user_available: bool | None = None
    in_flight: bool | None = None
    text: str = ""


@dataclass(frozen=True)
class SimConfig:
    """Drive + gate + release constants and initial conditions for one run."""

    alpha: float
    theta_u: float = 1.0
    beta: float = 1.0
    w: float = 15.0
    r0: float = 30.0
    k: float = 2.0
    r_max: float = 1440.0
    u_max: float = math.inf
    dt: float = 1.0
    horizon: float = 60.0
    # Deprivation-escalation bar: a deferred desire whose duration_over_θ reaches
    # this is re-presented even without an availability signal (anti-neglect).
    deprivation_release: float = 10_000.0
    default_verdict: Verdict = Verdict.REJECT
    # Initial conditions (also used to resume a persisted run — scenario 7).
    u0: float = 0.0
    duration_over_theta0: float = 0.0
    last_exchange_at0: float | None = None
    decline_count0: int = 0
    declined_at0: float | None = None
    initial_status: DesireStatus = DesireStatus.NONE
    initial_user_available: bool = False
    initial_in_flight: bool = False

    @property
    def gate_params(self) -> GateParams:
        return GateParams(theta_u=self.theta_u, w=self.w, r0=self.r0, k=self.k, r_max=self.r_max)


@dataclass
class TickRecord:
    """The observed end-of-tick state — the row the invariants are asserted on."""

    time: float
    u: float
    duration_over_theta: float
    last_exchange_at: float | None
    exchange_clock: float | None
    status: DesireStatus
    decline_count: int
    user_available: bool
    in_flight: bool
    outcome: str
    verdict: Verdict | None
    woke: bool


@dataclass
class SimResult:
    """The full per-tick record of a run, plus wake-focused accessors."""

    records: list[TickRecord] = field(default_factory=list)

    @property
    def wakes(self) -> list[TickRecord]:
        return [r for r in self.records if r.woke]

    def wake_times(self) -> list[float]:
        return [r.time for r in self.wakes]

    def at(self, time: float) -> TickRecord:
        for r in self.records:
            if r.time == time:
                return r
        raise KeyError(f"no record at t={time}")


# Outcome labels for a wake (the two ways a desire reaches cognition).
_WAKE_CREATE = "WAKE_CREATE"
_WAKE_RELEASE = "WAKE_RELEASE"
_HELD_DEFERRED = "held_deferred"


def run(
    config: SimConfig,
    events: Iterable[SimEvent] = (),
    verdicts: Sequence[Verdict] = (),
) -> SimResult:
    """Simulate the desire model over ``events`` and return the per-tick record.

    ``verdicts`` is the scripted cognition (FIFO, consumed per wake); once
    exhausted, ``config.default_verdict`` is used. See the module docstring.
    """
    drive = Drive(alpha=config.alpha, beta=config.beta, u_max=config.u_max, u=config.u0)
    lane = LaneState(
        last_exchange_at=config.last_exchange_at0,
        in_flight=config.initial_in_flight,
        declined_at=config.declined_at0,
        decline_count=config.decline_count0,
    )
    agg = Aggregator(status=config.initial_status)
    gates = config.gate_params

    duration_over_theta = config.duration_over_theta0
    user_available = config.initial_user_available
    pending = list(verdicts)

    by_tick = _bucket_events(events, config.dt)
    result = SimResult()

    n_steps = round(config.horizon / config.dt)
    for i in range(n_steps + 1):
        t = i * config.dt

        # (1) apply this tick's events
        had_exchange = False
        for ev in by_tick.get(i, ()):
            if ev.user_available is not None:
                user_available = ev.user_available
            if ev.in_flight is not None:
                lane.in_flight = ev.in_flight
            if ev.actor is None or ev.label is None:
                continue
            if ev.actor == "proactive_internal":
                continue  # the being's own nudge: never an exchange (spec §6)
            had_exchange = True
            lane.last_exchange_at = t
            q = quality_of(actor=ev.actor, label=ev.label)
            if ev.actor == "user":
                if q > 0.0:
                    drive.satiate(q=q)
                agg.on_exchange()  # inbound contact clears any live desire
                lane.declined_at = None  # a new exchange wipes the reject record
                lane.decline_count = 0

        # (2) rise in genuine silence (no exchange this tick; t=0 has no elapsed time)
        if i > 0 and not had_exchange:
            drive.rise(dt=config.dt)

        # (3) neuron-owned duration over threshold
        duration_over_theta = duration_over_theta + config.dt if drive.u >= config.theta_u else 0.0

        # (4) wake-decision
        outcome = ""
        verdict: Verdict | None = None
        woke = False
        if agg.status is DesireStatus.DEFERRED:
            release_cond = user_available or duration_over_theta >= config.deprivation_release
            gates_pass = not lane.in_flight and (
                lane.last_exchange_at is None or t - lane.last_exchange_at >= config.w
            )
            if release_cond and gates_pass and agg.on_release():
                woke, outcome = True, _WAKE_RELEASE
            else:
                outcome = _HELD_DEFERRED
        elif agg.status is DesireStatus.NONE:
            wake_out = evaluate_wake(u=drive.u, now=t, state=lane, params=gates)
            if wake_out.is_urge and agg.on_urge():
                woke, outcome = True, _WAKE_CREATE
            else:
                outcome = wake_out.value

        # the conversation clock the wake-decision actually saw, captured before a
        # FULFILL below advances it to `t` — this is what the window invariant checks.
        exchange_clock = lane.last_exchange_at

        # (5) apply the scripted verdict on a wake
        if woke:
            verdict = pending.pop(0) if pending else config.default_verdict
            agg.apply_verdict(verdict)
            if verdict is Verdict.FULFILL:
                drive.satiate(q=1.0)  # a delivered outreach is contact → satiates
                duration_over_theta = 0.0
                lane.last_exchange_at = t
            elif verdict is Verdict.REJECT:
                lane.declined_at = t
                lane.decline_count += 1
            # DEFER: held by apply_verdict; u, duration, and clock untouched.

        result.records.append(
            TickRecord(
                time=t,
                u=drive.u,
                duration_over_theta=duration_over_theta,
                last_exchange_at=lane.last_exchange_at,
                exchange_clock=exchange_clock,
                status=agg.status,
                decline_count=lane.decline_count,
                user_available=user_available,
                in_flight=lane.in_flight,
                outcome=outcome,
                verdict=verdict,
                woke=woke,
            )
        )

    return result


def _bucket_events(events: Iterable[SimEvent], dt: float) -> dict[int, list[SimEvent]]:
    """Group events by tick index ``round(time/dt)`` (scenarios align to the grid)."""
    buckets: dict[int, list[SimEvent]] = {}
    for ev in events:
        buckets.setdefault(round(ev.time / dt), []).append(ev)
    return buckets

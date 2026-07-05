"""The live decision adapter — State ↔ certified sim primitives (spec §5/§7/§8).

Hermes' host persists a plain :class:`~lifemodel.state.model.State` (JSON-native
fields, ISO-8601 timestamps) between ticks; the certified desire model in
``lifemodel.sim`` works in abstract "minutes" over in-memory dataclasses
(``Drive``, ``LaneState``, ``Aggregator``). This module is the *sole* place
that bridges the two: each call reconstructs the sim primitives from ``State``,
advances them by the elapsed wall-clock time, and writes the result back —
**never reimplementing** the drive/gate/lifecycle rules those primitives own.

Time-unit bridge (load-bearing): the sim primitives compare timestamps as
plain floats in a caller-chosen unit. Rather than picking a fixed epoch (which
would drift/overflow across long-running processes), every gate quantity is
expressed as *minutes relative to `now`* — `now` is always `0.0`, and any
earlier instant is negative (`-_minutes_between(instant, now)`). This keeps
the arithmetic exact regardless of how long the process has been alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lifemodel.sim.aggregation import Aggregator, DesireStatus, Verdict
from lifemodel.sim.drive import Drive
from lifemodel.sim.quality import Actor, Label, quality_of
from lifemodel.sim.wake import GateParams, LaneState, evaluate_wake
from lifemodel.state.model import State

#: The BASE prior (``tests/sim/test_sim_scenarios.py::BASE``, spec §9), hardcoded
#: for Phase 1 (disk hot-reload is Phase 2, bead follow-up).
ALPHA = 1 / 240
BETA = 1.0
U_MAX = 100.0
THETA = 1.0

BASE_PARAMS = GateParams(theta_u=THETA, w=15.0, r0=30.0, k=2.0, r_max=1440.0)


@dataclass(frozen=True)
class ReachoutDecision:
    """The result of one :func:`decide_reachout` evaluation."""

    wake: bool
    reason: str


def _minutes_between(a_iso: str | None, b: datetime) -> float:
    """Minutes from ``a_iso`` to ``b`` (``0.0`` if ``a_iso`` is ``None``)."""
    if a_iso is None:
        return 0.0
    a = datetime.fromisoformat(a_iso)
    return (b - a).total_seconds() / 60.0


def decide_reachout(state: State, *, now: datetime, busy: bool) -> ReachoutDecision:
    """Rise the drive by elapsed silence, evaluate the wake gates, and decide.

    Mutates ``state`` in place (``u``, ``duration_over_theta``, ``desire_status``,
    ``last_tick_at``) and creates ONE ``active`` desire on a clean URGE — a second
    urge while one is already live (active or deferred) is deduped, never a
    second wake (the anti-drum guarantee).
    """
    dt = _minutes_between(state.last_tick_at, now)
    drive = Drive(alpha=ALPHA, beta=BETA, u_max=U_MAX, u=state.u)
    if dt > 0:
        drive.rise(dt=dt)

    state.duration_over_theta = state.duration_over_theta + dt if drive.u >= THETA else 0.0

    # Every gate quantity is expressed as minutes relative to `now` (now=0.0);
    # `None` (no prior exchange/reject) must stay `None`, not `-0.0`, so the
    # corresponding gate is skipped entirely rather than spuriously tripped.
    exch_min = (
        -_minutes_between(state.last_exchange_at, now)
        if state.last_exchange_at is not None
        else None
    )
    decl_min = -_minutes_between(state.declined_at, now) if state.declined_at is not None else None

    lane = LaneState(
        last_exchange_at=exch_min,
        in_flight=busy,
        declined_at=decl_min,
        decline_count=state.decline_count,
    )
    outcome = evaluate_wake(u=drive.u, now=0.0, state=lane, params=BASE_PARAMS)

    agg = Aggregator(status=DesireStatus(state.desire_status))
    wake = agg.on_urge() if outcome.is_urge else False

    state.u = drive.u
    state.last_tick_at = now.isoformat()
    state.desire_status = agg.status.value

    return ReachoutDecision(wake=wake, reason=outcome.value)


def observe_exchange(state: State, *, actor: Actor, label: Label, now: datetime) -> None:
    """Record a lane exchange: satiate on a positive quality, reset silence/reject.

    An internal ``proactive_internal`` impulse is a no-op — it never satiates
    the drive and never touches the exchange clock (spec §6's load-bearing
    rule: the being must not satiate its own urge with its own nudge).
    """
    if actor == "proactive_internal":
        return

    q = quality_of(actor=actor, label=label)
    drive = Drive(alpha=ALPHA, beta=BETA, u_max=U_MAX, u=state.u)
    drive.satiate(q=q)
    state.u = drive.u

    state.last_exchange_at = now.isoformat()
    state.declined_at = None
    state.decline_count = 0

    agg = Aggregator(status=DesireStatus(state.desire_status))
    agg.on_exchange()
    state.desire_status = agg.status.value


def apply_verdict(state: State, verdict: Verdict, *, now: datetime) -> None:
    """Resolve a woken desire by cognition's verdict (FULFILL/DEFER/REJECT).

    FULFILL satiates fully, resets the over-threshold duration, and stamps both
    the exchange and contact clocks. REJECT records the growing-backoff
    bookkeeping with no satiation. DEFER is unreachable live in Phase 1 (no
    availability signal yet) but the status transition is kept for parity with
    the certified ``Aggregator``.
    """
    agg = Aggregator(status=DesireStatus(state.desire_status))
    agg.apply_verdict(verdict)
    state.desire_status = agg.status.value

    if verdict is Verdict.FULFILL:
        drive = Drive(alpha=ALPHA, beta=BETA, u_max=U_MAX, u=state.u)
        drive.satiate(q=1.0)
        state.u = drive.u
        state.duration_over_theta = 0.0
        state.last_exchange_at = now.isoformat()
        state.last_contact_at = now.isoformat()
        state.pending_proactive_id = None
        state.pending_proactive_since = None
    elif verdict is Verdict.REJECT:
        state.declined_at = now.isoformat()
        state.decline_count += 1
        state.pending_proactive_id = None
        state.pending_proactive_since = None

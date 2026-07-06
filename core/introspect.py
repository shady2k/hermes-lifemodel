"""Pure, Hermes-free readings for the /lifemodel debug view (spec §16).

``compute_readings`` turns a persisted :class:`State` into a frozen
:class:`Readings` snapshot for the renderer — computed directly from state and
the new pure helpers (circadian, inhibition, effective pressure, wake gates,
backstop), never from the (deleted) decision monolith. The calibration arrives
in a :class:`DebugConfig` so this module keeps its import boundary (no Hermes, no
composition, no debug).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..sim.wake import GateParams, LaneState, backoff_interval, evaluate_wake
from ..state.model import State
from .backstop import allow_send
from .circadian import circadian
from .pressure import effective_pressure, inhibition_at
from .timeutil import minutes_between

#: Max age (minutes) of ``last_tick_at`` before the brain reads as STALE. The loop
#: ticks ~every 60s (stamping ``last_tick_at`` via ``coreloop.tick()``), so a gap
#: past this means the loop is probably down and the gateway should have
#: reconnected the adapter. Two ticks of grace.
BRAIN_STALE_MIN = 2.0


@dataclass(frozen=True)
class DebugConfig:
    """Calibration the readings need (built by ``debug`` from the composition root)."""

    params: GateParams
    theta: float
    i0: float
    grace_min: float
    halflife_min: float
    peak_hour_utc: float
    max_per_day: int
    min_interval_min: float
    alpha: float
    u_max: float


@dataclass(frozen=True)
class Readings:
    schema_version: int
    tick_count: int
    # physiology
    energy: float
    fatigue: float
    circadian: float
    alertness: float
    # drive
    u: float
    inhibition: float
    action_pending_phase: str  # "none" | "grace" | "decaying"
    action_pending_remaining_min: float | None  # grace remaining, if in grace
    effective: float
    theta: float
    pct_to_wake: float
    duration_over_theta: float
    # desire lifecycle
    desire_status: str
    pending: bool
    pending_since: str | None
    last_contact_at: str | None
    last_exchange_at: str | None
    # gates
    would_wake: bool
    wake_reason: str
    silence_window_remaining_min: float | None
    decline_count: int
    backoff_remaining_min: float | None
    # backstop
    sends_today: int
    sends_cap: int
    send_allowed: bool
    # health / timing
    last_tick_at: str | None
    last_tick_ago_min: float | None
    brain_alive: bool  # last tick fresh (<= BRAIN_STALE_MIN) => the loop is ticking


def _ago(iso: str | None, now: datetime) -> float | None:
    return None if iso is None else minutes_between(iso, now)


def _action_pending(
    state: State, now: datetime, cfg: DebugConfig
) -> tuple[float, str, float | None]:
    if state.action_pending_since is None:
        return 0.0, "none", None
    inh = inhibition_at(
        state.action_pending_since,
        now,
        i0=cfg.i0,
        grace_min=cfg.grace_min,
        halflife_min=cfg.halflife_min,
    )
    elapsed = minutes_between(state.action_pending_since, now)
    if elapsed <= cfg.grace_min:
        return inh, "grace", cfg.grace_min - elapsed
    return inh, "decaying", None


def _silence_remaining(state: State, now: datetime, w: float) -> float | None:
    if state.last_exchange_at is None:
        return None
    left = w - minutes_between(state.last_exchange_at, now)
    return left if left > 0 else None


def _backoff_remaining(state: State, now: datetime, cfg: DebugConfig) -> float | None:
    if state.declined_at is None:
        return None
    r_n = backoff_interval(
        decline_count=state.decline_count, r0=cfg.params.r0, k=cfg.params.k, r_max=cfg.params.r_max
    )
    left = r_n - minutes_between(state.declined_at, now)
    return left if left > 0 else None


def _sends_today(send_log: list[str], now: datetime) -> int:
    from datetime import timedelta

    day_ago = now - timedelta(hours=24)
    count = 0
    for ts in send_log:
        try:
            t = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if t.tzinfo is not None and t >= day_ago:
            count += 1
    return count


def compute_readings(state: State, *, now: datetime, cfg: DebugConfig) -> Readings:
    last_tick_ago = _ago(state.last_tick_at, now)
    dt = max(0.0, minutes_between(state.last_tick_at, now))
    u = min(cfg.u_max, state.u + dt * cfg.alpha)
    inhibition, phase, grace_left = _action_pending(state, now, cfg)
    effective = effective_pressure(u, inhibition)

    exch_min = (
        -minutes_between(state.last_exchange_at, now)
        if state.last_exchange_at is not None
        else None
    )
    decl_min = -minutes_between(state.declined_at, now) if state.declined_at is not None else None
    lane = LaneState(
        last_exchange_at=exch_min,
        in_flight=False,
        declined_at=decl_min,
        decline_count=state.decline_count,
    )
    outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=cfg.params)

    return Readings(
        schema_version=state.schema_version,
        tick_count=state.tick_count,
        energy=state.energy,
        fatigue=state.fatigue,
        circadian=circadian(now, peak_hour_utc=cfg.peak_hour_utc),
        alertness=max(
            0.0, min(1.0, circadian(now, peak_hour_utc=cfg.peak_hour_utc) - state.fatigue)
        ),
        u=u,
        inhibition=inhibition,
        action_pending_phase=phase,
        action_pending_remaining_min=grace_left,
        effective=effective,
        theta=cfg.theta,
        pct_to_wake=(effective / cfg.theta) if cfg.theta else 0.0,
        duration_over_theta=state.duration_over_theta,
        desire_status=state.desire_status,
        pending=state.pending_proactive_id is not None,
        pending_since=state.pending_proactive_since,
        last_contact_at=state.last_contact_at,
        last_exchange_at=state.last_exchange_at,
        would_wake=outcome.is_urge,
        wake_reason=outcome.value,
        silence_window_remaining_min=_silence_remaining(state, now, cfg.params.w),
        decline_count=state.decline_count,
        backoff_remaining_min=_backoff_remaining(state, now, cfg),
        sends_today=_sends_today(state.proactive_send_log, now),
        sends_cap=cfg.max_per_day,
        send_allowed=allow_send(
            state.proactive_send_log,
            now,
            max_per_day=cfg.max_per_day,
            min_interval_min=cfg.min_interval_min,
        ),
        last_tick_at=state.last_tick_at,
        last_tick_ago_min=last_tick_ago,
        brain_alive=last_tick_ago is not None and last_tick_ago <= BRAIN_STALE_MIN,
    )

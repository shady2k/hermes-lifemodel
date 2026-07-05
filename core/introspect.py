"""Pure, Hermes-free personality readings for the debug view (lm-zmf, spec §3.1).

:func:`compute_readings` turns a persisted :class:`~lifemodel.state.model.State`
into a frozen :class:`PersonalityReadings` snapshot — everything the
:mod:`lifemodel.debug` renderer needs, already computed. It is the read-only
introspection counterpart of the live decision adapter
(:mod:`lifemodel.core.decision`): where the adapter *mutates* state on a tick,
this module runs the **same** logic on a **deep copy** and reports what it would
do, touching nothing on disk.

No drift by construction (spec §2.2): every quantity is read from ``State`` or
imported from its owning module — never restated. The honest wake verdict is the
real :func:`lifemodel.core.decision.decide_reachout` run on the copy, so it
captures, for free, the three things a naive ``evaluate_wake`` call would miss —
stale-pending recovery, the drive rise "as of now", and the ``Aggregator`` dedup
(anti-drum). Derived display quantities are simple arithmetic over the imported
constants and ``_minutes_between`` helper.

Import boundary (spec §3.1): this module imports only ``core.decision``,
``sim.*``, and ``state.model`` — no Hermes, no ``debug``, no ``composition`` — so
it cannot participate in an import cycle and stays unit-testable off-host.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime

from ..sim.wake import GateParams, WakeOutcome, backoff_interval
from ..state.model import State
from .decision import (
    ALPHA,
    BASE_PARAMS,
    BETA,
    PENDING_TIMEOUT_MIN,
    THETA,
    U_MAX,
    ReachoutDecision,
    _minutes_between,
    decide_reachout,
)

#: The ``busy`` / in-flight assumption every reading is labeled with. The debug
#: path cannot know true runtime ``in_flight`` (it is deliberately not persisted
#: — see spec §5), so it always evaluates as "no turn executing right now" and
#: labels the verdict accordingly. See :func:`compute_readings`.
_DEFAULT_BUSY = False


@dataclass(frozen=True)
class Temperament:
    """The being's fixed calibration (spec §4 TEMPERAMENT) — imported constants.

    These are *who this being is by nature*: the thresholds and rates the
    wake-decision is tuned to. Every value is imported from its owning module,
    never restated, so a calibration change propagates to the dump automatically.
    """

    theta: float
    alpha: float
    beta: float
    u_max: float
    base_params: GateParams
    pending_timeout_min: float
    #: The growing decline-backoff schedule ``R_n`` (``r0, r0·k, …`` capped at
    #: ``r_max``), computed once via :func:`sim.wake.backoff_interval` for display.
    backoff_schedule: tuple[float, ...]


@dataclass(frozen=True)
class GateRung:
    """One rung of the wake-gate precedence ladder, as the renderer prints it.

    ``status`` is the rung's disposition this evaluation (``BLOCKS HERE``,
    ``clear``, ``would-block``, ``UNKNOWN``, ``n/a``, ``—``, ``reached``);
    ``detail`` is the terse, visibly-derived note appended after it.
    """

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class PersonalityReadings:
    """Everything the debug renderer needs, already computed (spec §3.1/§4).

    Frozen and pure: building it never touches disk. The temperament block is
    constant; the drive / lifecycle / timing blocks echo persisted ``State``; the
    wake-readiness block reflects the deep-copy decision run. See module docstring
    for the no-drift and read-only invariants.
    """

    # --- META ---
    schema_version: int
    tick_count: int

    # --- TEMPERAMENT (fixed nature) ---
    temperament: Temperament

    # --- DRIVE (the contact urge now; persisted u + risen u) ---
    u_persisted: float
    u_risen: float
    duration_over_theta: float
    energy: float
    #: Minutes of continued silence until ``u`` reaches θ; ``None`` once at/over.
    time_to_theta_min: float | None

    # --- DESIRE LIFECYCLE (persisted) ---
    desire_status: str
    pending: bool
    decline_count: int
    declined_at: str | None
    #: Minutes left in the active decline backoff; ``None`` if none is active.
    backoff_remaining_min: float | None
    #: True when a stale ``active``+pending desire would be recovered as REJECT.
    stale_pending_recovery: bool
    stale_pending_age_min: float | None

    # --- TIMING (persisted timestamps + "ago" deltas) ---
    last_exchange_at: str | None
    last_exchange_ago_min: float | None
    last_contact_at: str | None
    last_contact_ago_min: float | None
    last_tick_at: str | None
    last_tick_ago_min: float | None

    # --- WAKE READINESS (the deep-copy decision run) ---
    #: The real ``ReachoutDecision.reason`` (a ``WakeOutcome`` value).
    gate_verdict: str
    #: The real ``ReachoutDecision.wake`` — whether outreach would actually launch.
    would_launch: bool
    risen_over_theta: bool
    #: Minutes left in the silence window; ``None`` if no prior exchange / past w.
    silence_window_remaining_min: float | None
    gate_ladder: tuple[GateRung, ...]

    # --- AUTONOMOUS LOOP (raw liveness stamp + ago; alive-bool is the renderer's,
    #     computed against tick.SERVICE_LIVENESS_MAX_AGE which this module cannot
    #     import without breaking its import boundary — spec §3.1/§3.3) ---
    egress_service_alive_at: str | None
    egress_service_ago_min: float | None


def temperament() -> Temperament:
    """The fixed temperament snapshot, built purely from imported constants.

    Needs no ``State`` — useful both inside :func:`compute_readings` and on the
    debug path's degraded branch (an unreadable store still has a temperament).
    """
    return Temperament(
        theta=THETA,
        alpha=ALPHA,
        beta=BETA,
        u_max=U_MAX,
        base_params=BASE_PARAMS,
        pending_timeout_min=PENDING_TIMEOUT_MIN,
        backoff_schedule=_backoff_schedule(),
    )


def compute_readings(
    state: State, *, now: datetime, busy: bool = _DEFAULT_BUSY
) -> PersonalityReadings:
    """Compute the full :class:`PersonalityReadings` for *state* as of *now*.

    Runs the real :func:`decide_reachout` on a **deep copy** of *state* (read-only
    wrt both disk and the caller's ``State`` — the copy is the only thing that
    moves), so the verdict, the risen urge, and the dedup/reflected lifecycle all
    match exactly what the next live tick would produce. ``busy`` defaults to
    ``False``: the debug path cannot know true runtime in-flight (spec §5), so the
    verdict is always labeled "assuming in_flight=false".
    """
    temp = temperament()

    # Stale-pending detection runs against the PERSISTED state (before the copy),
    # so the renderer can flag "the next tick would recover this as REJECT" even
    # though the copy's own verdict already reflects the post-rejection backoff.
    stale_recovery, stale_age = _stale_pending(state, now=now)

    # The honest verdict: the real decision, on a copy, as of now.
    snapshot = copy.deepcopy(state)
    decision: ReachoutDecision = decide_reachout(snapshot, now=now, busy=busy)
    u_risen = snapshot.u
    risen_over_theta = u_risen >= THETA

    return PersonalityReadings(
        schema_version=state.schema_version,
        tick_count=state.tick_count,
        temperament=temp,
        u_persisted=state.u,
        u_risen=u_risen,
        duration_over_theta=state.duration_over_theta,
        energy=state.energy,
        time_to_theta_min=None if risen_over_theta else (THETA - u_risen) / ALPHA,
        desire_status=state.desire_status,
        pending=state.pending_proactive_id is not None,
        decline_count=state.decline_count,
        declined_at=state.declined_at,
        backoff_remaining_min=_backoff_remaining(state, now=now),
        stale_pending_recovery=stale_recovery,
        stale_pending_age_min=stale_age,
        last_exchange_at=state.last_exchange_at,
        last_exchange_ago_min=_opt_minutes(state.last_exchange_at, now),
        last_contact_at=state.last_contact_at,
        last_contact_ago_min=_opt_minutes(state.last_contact_at, now),
        last_tick_at=state.last_tick_at,
        last_tick_ago_min=_opt_minutes(state.last_tick_at, now),
        gate_verdict=decision.reason,
        would_launch=decision.wake,
        risen_over_theta=risen_over_theta,
        silence_window_remaining_min=_silence_window_remaining(
            state, now=now, w=temp.base_params.w
        ),
        gate_ladder=_gate_ladder(
            decision=decision, snapshot=snapshot, snapshot_u=u_risen, now=now, temp=temp
        ),
        egress_service_alive_at=state.egress_service_alive_at,
        egress_service_ago_min=_opt_minutes(state.egress_service_alive_at, now),
    )


# --- helpers (all simple arithmetic over imported constants / _minutes_between) --------------


def _backoff_schedule() -> tuple[float, ...]:
    """``R_n`` for ``n = 1, 2, …`` until it saturates at ``r_max`` (then stop).

    Composed from :func:`sim.wake.backoff_interval` so the schedule can never
    drift from the real gate arithmetic.
    """
    terms: list[float] = []
    n = 1
    while True:
        rn = backoff_interval(
            decline_count=n,
            r0=BASE_PARAMS.r0,
            k=BASE_PARAMS.k,
            r_max=BASE_PARAMS.r_max,
        )
        if terms and rn <= terms[-1]:  # saturated at the cap → stop after including it once
            terms.append(rn)
            break
        terms.append(rn)
        if rn >= BASE_PARAMS.r_max:  # reached the cap; include it and stop
            break
        n += 1
        if n > 64:  # defensive backstop (k>1 saturates in a handful of steps)
            break
    return tuple(terms)


def _stale_pending(state: State, *, now: datetime) -> tuple[bool, float | None]:
    """Whether the next live tick would recover a stale pending desire as REJECT.

    Mirrors the predicate ``decide_reachout`` applies first (an ``active`` desire
    whose ``pending_proactive_since`` is at/over :data:`PENDING_TIMEOUT_MIN`),
    exposing it as a boolean + age so the renderer can print an explicit note.
    """
    if state.desire_status != "active" or state.pending_proactive_since is None:
        return False, None
    age = _minutes_between(state.pending_proactive_since, now)
    return (age >= PENDING_TIMEOUT_MIN, age)


def _opt_minutes(iso: str | None, now: datetime) -> float | None:
    """Minutes between *iso* and *now*, or ``None`` when *iso* is absent/unparseable."""
    if iso is None:
        return None
    minutes = _minutes_between(iso, now)
    # ``_minutes_between`` defends against malformed/naive values with 0.0; for
    # display we keep the raw result (the engine treats it the same way).
    return minutes


def _silence_window_remaining(state: State, *, now: datetime, w: float) -> float | None:
    """Minutes left in the active-silence window, or ``None`` if it does not apply.

    ``None`` when there was no prior exchange or the window has already elapsed.
    """
    if state.last_exchange_at is None:
        return None
    since = _minutes_between(state.last_exchange_at, now)
    remaining = w - since
    return remaining if remaining > 0 else None


def _backoff_remaining(state: State, *, now: datetime) -> float | None:
    """Minutes left in the decline backoff, or ``None`` if none is active.

    ``None`` when there is no decline on record or its backoff has already elapsed.
    """
    if state.declined_at is None:
        return None
    r_n = backoff_interval(
        decline_count=state.decline_count,
        r0=BASE_PARAMS.r0,
        k=BASE_PARAMS.k,
        r_max=BASE_PARAMS.r_max,
    )
    since = _minutes_between(state.declined_at, now)
    remaining = r_n - since
    return remaining if remaining > 0 else None


def _gate_ladder(
    *,
    decision: ReachoutDecision,
    snapshot: State,
    snapshot_u: float,
    now: datetime,
    temp: Temperament,
) -> tuple[GateRung, ...]:
    """Render the fixed-precedence gate ladder (spec §4), one :class:`GateRung` each.

    The ladder reads *snapshot* — the SAME deep-copy state the verdict ran on —
    so its facts (decline record, last exchange) match the verdict exactly. This
    matters for stale-pending recovery: the verdict (``no_wake_decline_backoff``)
    reflects the REJECT the copy just applied, so the ladder must see that freshly
    stamped decline too, not the persisted state which has none yet.

    Each rung is annotated with its own disposition *and* the rung that actually
    decided the verdict (from ``decision.reason``) is marked ``BLOCKS HERE`` (or
    ``would-block`` when the runtime-unknown ``in_flight`` rung sits above it).
    Rungs past the deciding one are ``—`` (not reached). ``in_flight`` is the one
    runtime-only rung: ``n/a`` while ``u < θ`` (precedence means it cannot matter)
    and ``UNKNOWN`` once ``u ≥ θ`` (if true at runtime, it would block here).
    """
    w = temp.base_params.w
    reason = decision.reason
    below_blocks = reason == WakeOutcome.BELOW_THRESHOLD.value
    since_exch = _opt_minutes(snapshot.last_exchange_at, now)
    since_decl = _opt_minutes(snapshot.declined_at, now)
    r_n = (
        backoff_interval(
            decline_count=snapshot.decline_count,
            r0=temp.base_params.r0,
            k=temp.base_params.k,
            r_max=temp.base_params.r_max,
        )
        if snapshot.declined_at is not None
        else None
    )

    rungs: list[GateRung] = []

    # 1. below_threshold — the only definitive blocker (nothing runtime above it).
    if below_blocks:
        rungs.append(
            GateRung("below_threshold", "BLOCKS HERE", f"u {_g(snapshot_u)} < θ {_g(temp.theta)}")
        )
    else:
        rungs.append(
            GateRung("below_threshold", "clear", f"u {_g(snapshot_u)} ≥ θ {_g(temp.theta)}")
        )

    # 2. in_flight — runtime-only; never knowable from persisted State.
    if below_blocks:
        rungs.append(GateRung("in_flight", "n/a", "(u < θ → cannot matter)"))
    else:
        rungs.append(GateRung("in_flight", "UNKNOWN", "runtime-only; if true, BLOCKS HERE"))

    # 3. silence_window.
    if reason == WakeOutcome.SILENCE_WINDOW.value:
        left = w - (since_exch or 0.0)
        rungs.append(GateRung("silence_window", "would-block", f"~{_fmt_min(left)} of w left"))
    elif below_blocks:
        rungs.append(
            GateRung(
                "silence_window", "—", f"(not reached; {_silence_hypothetical(since_exch, w)})"
            )
        )
    else:
        note = "last exchange > w ago" if since_exch is not None else "no prior exchange"
        rungs.append(GateRung("silence_window", "clear", f"({note})"))

    # 4. decline_backoff.
    if reason == WakeOutcome.DECLINE_BACKOFF.value:
        left = (r_n or 0.0) - (since_decl or 0.0)
        rungs.append(
            GateRung(
                "decline_backoff",
                "would-block",
                f"~{_fmt_min(left)} of R_{snapshot.decline_count} left",
            )
        )
    elif reason in (WakeOutcome.BELOW_THRESHOLD.value, WakeOutcome.SILENCE_WINDOW.value):
        rungs.append(
            GateRung(
                "decline_backoff",
                "—",
                f"(not reached; {_decline_hypothetical(snapshot.declined_at, r_n)})",
            )
        )
    else:
        note = "no active decline" if snapshot.declined_at is None else "backoff expired"
        rungs.append(GateRung("decline_backoff", "clear", f"({note})"))

    # 5. urge — terminal; reached only when every gate above cleared (busy=false).
    if reason == WakeOutcome.URGE.value:
        rungs.append(GateRung("urge", "reached", "conditional on in_flight=false"))
    else:
        rungs.append(GateRung("urge", "—", "conditional on in_flight=false"))

    return tuple(rungs)


def _silence_hypothetical(since_exch: float | None, w: float) -> str:
    """What the silence-window rung *would* say if the ladder reached it."""
    if since_exch is None:
        return "no prior exchange"
    remaining = w - since_exch
    if remaining > 0:
        return f"~{_fmt_min(remaining)} of w would remain"
    return "window already elapsed"


def _decline_hypothetical(declined_at: str | None, r_n: float | None) -> str:
    """What the decline-backoff rung *would* say if the ladder reached it."""
    if declined_at is None or r_n is None:
        return "no active decline"
    return f"~{_fmt_min(r_n)} of R_n would remain"


def _g(value: float) -> str:
    """Compact general-purpose number formatting (≤4 sig figs)."""
    return f"{value:.4g}"


def _fmt_min(minutes: float) -> str:
    """A short human duration for the ladder notes: ``5.0 min`` or ``2h 10m``."""
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    return f"{hours}h {mins:02d}m"

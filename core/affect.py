"""Core-affect: the derivation math BUILT IN beside its AUTONOMIC component (lm-ukc.2).

The being's core affect on Russell's circumplex — VALENCE in [-1, 1] (unpleasant …
pleasant) and AROUSAL in [0, 1] (calm … keyed-up) — derived each tick as a cheap
projection of the body, then eased toward that target with inertia (a leaky
integrator). The pure math (:func:`affect_target` + :func:`ease`) lives HERE, next
to the ``AffectSense`` component that owns it — no separate ``sim`` package — so a
simulation drives the real component through fake ports, and the pure functions
stay directly callable to reason about a trajectory without a live run.

Design (codex review, grounded on the real drive scale θ_u=1, α=1/240 → u=1 at 4h
silence): each valence contribution is CAPPED and returned separately (so no single
signal — loneliness above all — can own the axis), loneliness uses a ``sqrt`` curve
so ordinary silence tints mood before it saturates, rejection/exchange decay by
half-life, and arousal's urgency comes from ``duration_over_theta`` (how long the
pull has been sustained), never a bare ``u/θ`` that would saturate at threshold.
Everything here only COLORS the being's voice; nothing downstream reads it (the
one-way invariant is enforced structurally by ``AffectSense`` emitting no signal).
All weights/caps/taus are the params below — tunable on disk, calibrated live.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..state.model import State
from .circadian import circadian
from .component import TickContext
from .intents import Intent, UpdateState
from .metrics import MetricSpec
from .timeutil import minutes_between, to_iso


@dataclass(frozen=True)
class AffectParams:
    """Tunable weights, caps, references, and inertia constants for the derivation.

    Starting values are calibration seeds ("try it and tune on the live being"),
    not tuned truth. Grounded on the real contact scale (θ_u=1, α=1/240): ``u_ref``
    is the loneliness reference in drive-units (~1 day of silence at α=1/240)."""

    # --- valence (hedonic, [-1, 1]) ---
    u_ref: float = 6.0  # loneliness "peak colour" reference (~1 day silence); NOT θ
    u_valence_cap: float = 0.35
    reject_cap: float = 0.25
    reject_half_life_min: float = 180.0
    reject_count_ref: float = 3.0
    unanswered_cap: float = 0.08
    unanswered_ref: float = 2.0
    fatigue_valence_cap: float = 0.10
    exchange_cap: float = 0.35
    exchange_half_life_min: float = 240.0
    # --- arousal (activation, [0, 1]) ---
    arousal_base: float = 0.15
    arousal_alertness_w: float = 0.45
    arousal_energy_w: float = 0.20
    arousal_fatigue_w: float = 0.20
    arousal_urgency_w: float = 0.25
    urgency_duration_ref_min: float = 180.0
    # --- inertia (leaky integrator, minutes) ---
    tau_valence_min: float = 120.0  # valence lingers (mood)
    tau_arousal_min: float = 45.0  # arousal tracks the body faster (state)
    deadband: float = 0.005  # drop sub-deadband moves (no jitter)


@dataclass(frozen=True)
class AffectContributions:
    """The per-signal breakdown behind a target, for observability (lm-ukc.6).

    Each value is that signal's SIGNED push on the axis (already weighted/capped),
    so ``sum(valence.values())`` is the pre-clamp valence target and a debug view
    can show "what is pulling the being's mood right now"."""

    valence: dict[str, float] = field(default_factory=dict)
    arousal: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AffectBody:
    """The interoceptive INPUT the affect organ reads — its 'organ facade' (lm-ukc.2).

    Groups the slice of ``AgentState`` the derivation is a pure function of, plus the
    clock-derived readings (elapsed minutes, circadian phase). :meth:`from_state` is
    the ONE place that knows WHICH state fields feed affect and how the clock turns
    them into inputs, so the kernel below takes one cohesive value (not ten scalars)
    and the component step stays tiny. Storage stays flat + atomic — this is a read
    facade over the snapshot, it owns no mutation (the frame committer does)."""

    u: float
    decline_count: int
    minutes_since_declined: float | None
    unanswered_outbound_count: int
    fatigue: float
    minutes_since_exchange: float | None
    energy: float
    circadian: float
    duration_over_theta: float

    @classmethod
    def from_state(cls, state: State, *, now: datetime, peak_hour_utc: float) -> AffectBody:
        """Read the affect organ's inputs from the start-of-tick snapshot + clock."""
        return cls(
            u=state.u,
            decline_count=state.decline_count,
            minutes_since_declined=(
                minutes_between(state.declined_at, now) if state.declined_at is not None else None
            ),
            unanswered_outbound_count=state.unanswered_outbound_count,
            fatigue=state.fatigue,
            minutes_since_exchange=(
                minutes_between(state.last_exchange_at, now)
                if state.last_exchange_at is not None
                else None
            ),
            energy=state.energy,
            circadian=circadian(now, peak_hour_utc=peak_hour_utc),
            duration_over_theta=state.duration_over_theta,
        )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sat01(x: float) -> float:
    return _clamp(x, 0.0, 1.0)


def _decay(age_min: float, half_life_min: float) -> float:
    """Exponential recency weight: 1.0 at age 0, 0.5 at one half-life, → 0."""
    return math.exp(-math.log(2.0) * age_min / half_life_min)


def _smoothstep(x: float) -> float:
    """Smooth 0→1 ramp on [0,1] (x²(3−2x)); gentle onset, no hard corner."""
    x = _sat01(x)
    return x * x * (3.0 - 2.0 * x)


def affect_target(
    body: AffectBody, params: AffectParams
) -> tuple[float, float, AffectContributions]:
    """Project the organ's :class:`AffectBody` onto (valence, arousal), with the breakdown.

    Pure and clock-free: the body already carries the elapsed minutes (computed by
    :meth:`AffectBody.from_state` against the clock), so this kernel is fully
    unit-testable. A ``None`` recency means "no such event on record" → that term
    contributes nothing. Ranges are enforced here only by the final clamp;
    individual contributions are capped by their weights."""
    # --- valence: capped, separately-returned contributions (sum → clamp) ---
    u_v = -params.u_valence_cap * math.sqrt(_sat01(body.u / params.u_ref))

    reject_recent = (
        _decay(body.minutes_since_declined, params.reject_half_life_min)
        if body.minutes_since_declined is not None
        else 0.0
    )
    reject_v = (
        -params.reject_cap * _sat01(body.decline_count / params.reject_count_ref) * reject_recent
    )

    unanswered_v = -params.unanswered_cap * _sat01(
        body.unanswered_outbound_count / params.unanswered_ref
    )

    fatigue_v = -params.fatigue_valence_cap * _sat01(body.fatigue)

    exchange_recent = (
        _decay(body.minutes_since_exchange, params.exchange_half_life_min)
        if body.minutes_since_exchange is not None
        else 0.0
    )
    exchange_v = params.exchange_cap * exchange_recent

    valence = _clamp(u_v + reject_v + unanswered_v + fatigue_v + exchange_v, -1.0, 1.0)

    # --- arousal: baseline + alertness/energy − fatigue + sustained-pull urgency ---
    alertness = _clamp(body.circadian - _sat01(body.fatigue), 0.0, 1.0)
    alertness_a = params.arousal_alertness_w * alertness
    energy_a = params.arousal_energy_w * _sat01(body.energy)
    fatigue_a = -params.arousal_fatigue_w * _sat01(body.fatigue)
    pull_a = params.arousal_urgency_w * _smoothstep(
        body.duration_over_theta / params.urgency_duration_ref_min
    )
    arousal = _clamp(params.arousal_base + alertness_a + energy_a + fatigue_a + pull_a, 0.0, 1.0)

    contributions = AffectContributions(
        valence={
            "u": u_v,
            "reject": reject_v,
            "unanswered": unanswered_v,
            "fatigue": fatigue_v,
            "exchange": exchange_v,
        },
        arousal={
            "base": params.arousal_base,
            "alertness": alertness_a,
            "energy": energy_a,
            "fatigue": fatigue_a,
            "pull": pull_a,
        },
    )
    return valence, arousal, contributions


def ease(*, current: float, target: float, dt_min: float, tau_min: float, deadband: float) -> float:
    """One leaky-integrator step toward *target* over *dt_min* with time-constant *tau_min*.

    ``new = current + (target − current)·(1 − e^(−dt/τ))``. A move smaller than
    *deadband* is dropped (returns *current* unchanged) so micro-jitter never
    churns the stored value. ``dt_min ≤ 0`` yields no movement."""
    if dt_min <= 0.0:
        return current
    k = 1.0 - math.exp(-dt_min / tau_min)
    new = current + (target - current) * k
    if abs(new - current) < deadband:
        return current
    return new


AFFECT_VALENCE = "lifemodel_affect_valence"
AFFECT_AROUSAL = "lifemodel_affect_arousal"
AFFECT_VALENCE_SPEC = MetricSpec(
    name=AFFECT_VALENCE,
    kind="gauge",
    unit="",
    help="The being's current core-affect valence (-1..1), eased by AffectSense.",
)
AFFECT_AROUSAL_SPEC = MetricSpec(
    name=AFFECT_AROUSAL,
    kind="gauge",
    unit="",
    help="The being's current core-affect arousal (0..1), eased by AffectSense.",
)


@dataclass(frozen=True)
class AffectState:
    """The affect organ's OUTPUT slice — the eased affect this tick commits.

    :meth:`to_state_patch` is the ONE place that knows WHICH ``AgentState`` fields
    affect writes, mirroring :meth:`AffectBody.from_state` on the input side. It
    returns a patch for an ``UpdateState`` intent — the organ proposes, only the
    frame committer mutates."""

    valence: float
    arousal: float
    updated_at: str

    def to_state_patch(self) -> dict[str, object]:
        return {
            "affect_valence": self.valence,
            "affect_arousal": self.arousal,
            "affect_updated_at": self.updated_at,
        }


class AffectSense:
    """AUTONOMIC integrator that owns and eases the being's core affect.

    The affect sibling of ``SolitudeDrive``: it reads the start-of-tick snapshot,
    projects the body onto (valence, arousal) with the built-in kernel
    (:func:`affect_target`), eases the stored affect toward that target with inertia
    (:func:`ease`, split tau), and emits ONLY an ``UpdateState`` — never a signal.
    That absence is the one-way invariant made STRUCTURAL: nothing downstream can read
    affect, so affect can never feed the wake/contact decision (contrast
    ``SolitudeDrive``, which DOES emit ``contact_pressure`` because aggregation must
    read ``u``). Elapsed minutes come from ``last_tick_at`` — the same physiology clock
    as energy/fatigue — so the first tick (no prior stamp → dt 0) makes no move.
    """

    def __init__(
        self, *, params: AffectParams, peak_hour_utc: float, id: str = "affect-sense"
    ) -> None:
        self.id = id
        self._params = params
        self._peak_hour_utc = peak_hour_utc

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        p = self._params
        # Read the organ's inputs (state slice + clock readings) through the facade,
        # project to a target, then ease the stored affect toward it. dt comes from
        # last_tick_at (the physiology clock), so the first tick (dt 0) makes no move.
        body = AffectBody.from_state(state, now=ctx.now, peak_hour_utc=self._peak_hour_utc)
        target_valence, target_arousal, _contributions = affect_target(body, p)
        dt = minutes_between(state.last_tick_at, ctx.now)
        valence = ease(
            current=state.affect_valence,
            target=target_valence,
            dt_min=dt,
            tau_min=p.tau_valence_min,
            deadband=p.deadband,
        )
        arousal = ease(
            current=state.affect_arousal,
            target=target_arousal,
            dt_min=dt,
            tau_min=p.tau_arousal_min,
            deadband=p.deadband,
        )
        # Domain-metric channel (guarded, fail-open): publish the eased affect so the
        # debug view (lm-ukc.6) and live telemetry can watch it. No signal is emitted.
        if ctx.observe is not None:
            ctx.observe.set(AFFECT_VALENCE, valence)
            ctx.observe.set(AFFECT_AROUSAL, arousal)
        eased = AffectState(valence=valence, arousal=arousal, updated_at=to_iso(ctx.now))
        return [UpdateState(eased.to_state_patch())]

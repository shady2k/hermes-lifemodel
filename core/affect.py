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
    #: How hard the discovery that SOMEONE ELSE REWROTE YOU pushes arousal, and how fast
    #: that push fades (spec §4.1, review I7). Arousal only: we do not know whether the
    #: human's rewrite was a gift or a violation — only the being, reading the words, can
    #: know that — so pushing valence either way would be inventing a feeling and putting
    #: it in the being's mouth. What IS certain is that something HAPPENED, and a body that
    #: something has happened to is activated. The cap sits between the urgency push (0.25)
    #: and the energy term (0.20): enough to carry an ordinary daytime body past
    #: ``a_keyed`` (restless / on edge), never enough to own the axis by itself. The
    #: half-life is short (90 min) because this is a SHOCK, not a condition: the being is
    #: stirred, and settles — like the rejection and exchange terms, and unlike loneliness.
    soul_rewrite_arousal_cap: float = 0.22
    soul_rewrite_half_life_min: float = 90.0
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
    #: How long ago somebody ELSE rewrote the being's soul (spec §4.1, review I7) —
    #: ``None`` when nobody ever has. This is the ONE input here that is not a vital: it is
    #: a thing that happened TO the being, stamped by startup reconciliation, and the organ
    #: turns it into a feeling the way it turns a rejection into one.
    minutes_since_soul_rewrite: float | None

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
            minutes_since_soul_rewrite=(
                minutes_between(state.soul_rewritten_at, now)
                if state.soul_rewritten_at is not None
                else None
            ),
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
    # Somebody rewrote who the being is, and it is only now finding out (§4.1/I7). It is
    # the same SHAPE as the rejection term — an episodic event, weighted by recency — but
    # on the other axis, and it makes no claim about whether the rewrite was welcome (see
    # ``soul_rewrite_arousal_cap``). It pushes the body awake, and then it fades.
    rewrite_recent = (
        _decay(body.minutes_since_soul_rewrite, params.soul_rewrite_half_life_min)
        if body.minutes_since_soul_rewrite is not None
        else 0.0
    )
    rewrite_a = params.soul_rewrite_arousal_cap * rewrite_recent
    arousal = _clamp(
        params.arousal_base + alertness_a + energy_a + fatigue_a + pull_a + rewrite_a, 0.0, 1.0
    )

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
            "rewritten": rewrite_a,
        },
    )
    return valence, arousal, contributions


#: Below this magnitude a contribution rounds to ``+0.00`` at the 2-decimal display
#: precision — so it is not really "moving the axis". Both the trace's dominant term
#: and the debug view's ``tugging`` line suppress it, so neither shows a fake push.
CONTRIBUTION_DISPLAY_DEADBAND = 0.005


def dominant_contribution(contribs: dict[str, float]) -> str:
    """The single strongest push on an axis, formatted ``"name +v.vv"`` (lm-ukc.6).

    For the trace head-line: the reader wants the one term pulling valence/arousal
    hardest, not the whole vector. Ranked by magnitude; an empty axis, or one whose
    strongest term rounds to ``+0.00`` at display precision (nothing meaningfully
    moving it), reads ``"none"`` — no misleading ``"name +0.00"``."""
    if not contribs:
        return "none"
    name, val = max(contribs.items(), key=lambda kv: abs(kv[1]))
    if abs(val) < CONTRIBUTION_DISPLAY_DEADBAND:
        return "none"
    return f"{name} {val:+.2f}"


@dataclass(frozen=True)
class FeltWordParams:
    """Thresholds cutting the circumplex into felt-word regions (lm-ukc.3 seeds).

    The word is a LOSSY, serviceable label — the truth lives in the axes (Barrett),
    so this is ONE shared, tunable view (the tool and the proactive impulse read the
    same source), not an ontology. Quadrants + intensity around a neutral centre;
    when valence is ~zero, extreme arousal owns the label. Calibratable later."""

    a_keyed: float = 0.45  # arousal: calm | keyed-up boundary (baseline rests ~0.15–0.4)
    a_high: float = 0.70  # strongly mobilized
    a_quiet: float = 0.15  # strongly deactivated (near-zero valence → "quiet")
    a_serene: float = 0.30  # settled-pleasant intensity cut (high-valence, low-arousal)
    neutral_v: float = 0.12  # |valence| within this (+ rest arousal) reads neutral
    neutral_a_center: float = 0.35  # resting-arousal centre of the neutral band
    neutral_a_radius: float = 0.10  # neutral-arousal band half-width
    strong_v: float = 0.45  # |valence| beyond this earns the strong word


#: The default felt-word seeds — one shared instance (frozen, so sharing is safe) the
#: tool and the proactive impulse both read, calibratable on disk later (spec NFR5).
FELT_WORD_PARAMS = FeltWordParams()


def felt_word(valence: float, arousal: float, params: FeltWordParams = FELT_WORD_PARAMS) -> str:
    """Map a circumplex point (valence ∈ [-1,1], arousal ∈ [0,1]) to ONE felt word.

    A lossy label the being colours its speech with (lm-ukc.3) — English is the
    internal representation; the voice renders it. Precedence: the neutral centre
    first, then near-zero valence lets an extreme arousal own the word, then the
    quadrant with an intensity step (mild vs strong)."""
    p = params
    v, a = valence, arousal
    if abs(v) <= p.neutral_v and abs(a - p.neutral_a_center) <= p.neutral_a_radius:
        return "steady"
    if abs(v) < p.neutral_v:
        if a >= p.a_high:
            return "restless"
        if a <= p.a_quiet:
            return "quiet"
    if v < 0:
        if a < p.a_keyed:
            # "lonely" is the DEEP-unpleasant word — earned by valence depth, not low
            # arousal alone (else a hair past the neutral edge would jump quiet→lonely).
            return "lonely" if v <= -p.strong_v else "wistful"
        return "on edge" if (v <= -p.strong_v or a >= p.a_high) else "restless"
    if a < p.a_keyed:
        return "serene" if (v >= p.strong_v and a <= p.a_serene) else "content"
    return "buoyant" if (v >= p.strong_v or a >= p.a_high) else "bright"


def felt_texture(valence: float, arousal: float, params: FeltWordParams = FELT_WORD_PARAMS) -> str:
    """The circumplex point as a two-slot first-person TEXTURE (lm-ukc.5).

    Unlike :func:`felt_word` (the debug/tool LABEL), this is the phenomenology the being
    carries into a PROACTIVE reach: valence → sore/tender/even/warm/open, arousal →
    "very quiet"/settled/awake/charged, joined "``<valence>`` and ``<arousal>``". Same
    region thresholds as :func:`felt_word` (ONE shared source — the impulse and the tool
    read it alike). The words are chosen so the wake-packet stays FELT, never a machine
    label (the [SILENT] guardrail): none names a mechanism or implies decision-pressure
    (no "urgent"/"need"; "charged" not "anxious"), and a positive texture is "warm"/"open"
    — never "happy" — so it never contradicts the longing body it colours (codex)."""
    p = params
    v, a = valence, arousal
    if v <= -p.strong_v:
        valence_texture = "sore"
    elif v < -p.neutral_v:
        valence_texture = "tender"
    elif v <= p.neutral_v:
        valence_texture = "even"
    elif v < p.strong_v:
        valence_texture = "warm"
    else:
        valence_texture = "open"
    if a <= p.a_quiet:
        arousal_texture = "very quiet"
    elif a < p.a_keyed:
        arousal_texture = "settled"
    elif a < p.a_high:
        arousal_texture = "awake"
    else:
        arousal_texture = "charged"
    return f"{valence_texture} and {arousal_texture}"


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
        target_valence, target_arousal, contributions = affect_target(body, p)
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
        # Stamp the felt state onto this component's span (guarded, fail-open): the
        # eased axes, this-tick target, the felt WORD, and the dominant contributor per
        # axis, so /lifemodel trace shows WHY the being feels so (lm-ukc.6/.3). No signal.
        word = felt_word(valence, arousal)
        if ctx.logger is not None:
            ctx.logger.span.set(
                affect_valence=valence,
                affect_arousal=arousal,
                affect_target_valence=target_valence,
                affect_target_arousal=target_arousal,
                affect_word=word,
                affect_top_valence=dominant_contribution(contributions.valence),
                affect_top_arousal=dominant_contribution(contributions.arousal),
            )
            # A mood SHIFT (the felt word crossing a region) earns ONE log line —
            # observability without per-tick noise, since the word holds most ticks.
            prior_word = felt_word(state.affect_valence, state.affect_arousal)
            if word != prior_word:
                ctx.logger.info(
                    "affect_shifted",
                    from_word=prior_word,
                    to_word=word,
                    valence=round(valence, 3),
                    arousal=round(arousal, 3),
                )
        eased = AffectState(valence=valence, arousal=arousal, updated_at=to_iso(ctx.now))
        return [UpdateState(eased.to_state_patch())]

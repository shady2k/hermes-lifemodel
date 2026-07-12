"""Pure, Hermes-free readings for the /lifemodel debug view (spec §16).

``compute_readings`` turns a persisted :class:`State` into a frozen
:class:`Readings` snapshot for the renderer — computed directly from state and
the new pure helpers (circadian, inhibition, effective pressure, wake gates,
backstop), never from the (deleted) decision monolith. The calibration arrives
in a :class:`DebugConfig` so this module keeps its import boundary (no Hermes, no
composition, no debug).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..domain.objects import Thought, ThoughtState, UserModel
from ..state.model import State
from .affect import AffectBody, AffectParams, affect_target, felt_word
from .backstop import allow_send
from .circadian import circadian
from .pressure import effective_pressure, inhibition_at
from .receptivity import appraise_receptivity
from .timeutil import from_iso, minutes_between
from .user_model_view import DEFAULT_USER_MODEL
from .wake import GateParams, LaneState, backoff_interval, evaluate_wake
from .why_graph import WhyNode, display_id

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
    #: The core-affect derivation seeds (lm-ukc.6): the debug view recomputes the
    #: affect target/contributions from the snapshot with these, mirroring the live
    #: ``AffectSense``. Defaulted so a config built without them still reads affect.
    affect_params: AffectParams = AffectParams()


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
    #: Where the live contact desire sprang from (lm-27n.9): ``drive`` (bottom-up
    #: pressure), ``thought`` (a crystallized deliberation), or ``mixed`` — the
    #: "why" behind a proactive reach, the [SILENT] audit.
    desire_spring: str
    #: The source thought id(s) a top-down/mixed desire crystallized from (empty for
    #: a pure drive spring) — the concrete reason carried for lm-8o3's wake framing.
    desire_source_thought_ids: tuple[str, ...]
    intention_status: str  # the live decision record: active/deferred/none
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
    # receptivity (owner appropriateness — lm-27n.5)
    receptivity_allowed: bool
    receptivity_multiplier: float
    receptivity_confidence: float
    receptivity_hard_reasons: tuple[str, ...]
    receptivity_soft_reasons: tuple[str, ...]
    receptivity_constraints: tuple[str, ...]
    # thoughts (what the being is turning over — lm-27n.6/.7)
    #: live thoughts rendered "content [id] score=… np=… [parked]", most-salient
    #: first — the attention engine's score/loop/park state (lm-27n.7) surfaced.
    thoughts: tuple[str, ...]
    # health / timing
    last_tick_at: str | None
    last_tick_ago_min: float | None
    brain_alive: bool  # last tick fresh (<= BRAIN_STALE_MIN) => the loop is ticking
    #: One COMPACT "why did I write" line — the current contact intention's causal
    #: chain (lm-27n.10), e.g. ``intention:contact:owner <- desire:contact:owner
    #: (source)``. ``"no current outreach"`` when there is no live/recent intention.
    #: The FULL graph lives behind ``/lifemodel why`` (not re-walked into every dump).
    contact_chain: str = "no current outreach"
    # affect (core-affect self-model — lm-ukc.6). CURRENT axes are the stored eased
    # values; TARGET + contributions are recomputed from the snapshot (like ``u``).
    #: The felt WORD (lm-ukc.3) — a lossy label over the CURRENT eased axes.
    affect_word: str = "steady"
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    affect_updated_at: str | None = None
    affect_target_valence: float = 0.0
    affect_target_arousal: float = 0.0
    #: Per-signal signed pushes behind each target, ranked by magnitude (strongest
    #: first) so the dump shows "what tugs valence/arousal most" (lm-ukc.6).
    affect_valence_contributions: tuple[tuple[str, float], ...] = ()
    affect_arousal_contributions: tuple[tuple[str, float], ...] = ()


#: How many links the COMPACT contact-chain line follows down its primary lineage —
#: keeps the debug dump's one-liner one line, however deep the full graph runs.
_CHAIN_SUMMARY_HOPS = 4


def contact_chain_summary(node: WhyNode | None) -> str:
    """A one-line "why did I write" summary of the contact-intention chain (lm-27n.10).

    Follows the primary (first-edge) lineage down a few hops, e.g. ``intention:
    contact:owner <- desire:contact:owner (source)``. ``None`` (no live/recent
    intention) → ``"no current outreach"``. Cycle/missing edges terminate the line
    with an explicit marker; it is always a single, bounded line."""
    if node is None:
        return "no current outreach"
    parts = [display_id(node.kind, node.id)]
    current: WhyNode | None = node
    for _ in range(_CHAIN_SUMMARY_HOPS):
        if current is None or not current.edges:
            break
        edge = current.edges[0]
        if edge.node is not None:
            parts.append(f"<- {display_id(edge.node.kind, edge.node.id)} ({edge.label})")
            current = edge.node
        elif edge.cycle:
            parts.append(f"<- [cycle] ({edge.label})")
            break
        elif edge.missing_ref is not None:
            parts.append(f"<- {edge.missing_ref} ({edge.label}) [missing]")
            break
        else:  # pragma: no cover - an edge always has node/cycle/missing_ref
            break
    return " ".join(parts)


def _ago(iso: str | None, now: datetime) -> float | None:
    return None if iso is None else minutes_between(iso, now)


def _render_thought(t: Thought) -> str:
    """Render one live thought for the audit: content, id, the attention engine's
    last score, its no-progress (loop) counter, and a ``parked`` marker (lm-27n.7).

    Keeps the ``"{content} [{id}]"`` prefix the .6 dump established, then appends
    the .7 attention state — so a reader sees not just *what* is turned over but
    how hard it is competing and whether the brake has parked it."""
    parked = " [parked]" if t.state == ThoughtState.PARKED.value else ""
    return f"{t.content} [{t.id}] score={t.attention_score:.2f} np={t.no_progress_count}{parked}"


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


def _silence_anchor(state: State) -> str | None:
    """The timestamp the silence-window gate measures from (lm-md6.1): an admin
    override (:attr:`~lifemodel.state.model.State.silence_anchor_at`) if set, else
    the real :attr:`~lifemodel.state.model.State.last_exchange_at` — mirroring the
    live gate in ``core/aggregation.py`` so the debug readings never disagree with it."""
    if state.silence_anchor_at is not None:
        return state.silence_anchor_at
    return state.last_exchange_at


def _silence_remaining(state: State, now: datetime, w: float) -> float | None:
    anchor = _silence_anchor(state)
    if anchor is None:
        return None
    left = w - minutes_between(anchor, now)
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
            t = from_iso(ts)  # strict: malformed/naive both raise -> skipped
        except (ValueError, TypeError):
            continue
        if t >= day_ago:
            count += 1
    return count


def compute_readings(
    state: State,
    *,
    now: datetime,
    cfg: DebugConfig,
    desire_state: str = "none",
    desire_spring: str = "drive",
    desire_source_thought_ids: tuple[str, ...] = (),
    intention_state: str = "none",
    user_model: UserModel | None = None,
    thoughts: Sequence[Thought] = (),
    contact_chain: str = "no current outreach",
) -> Readings:
    """Compute the debug readings. ``desire_state`` is the live contact-desire's
    lifecycle state (``active``/``deferred``/``none``), read by the caller from
    the typed ``kind='desire'`` row (lm-27n.3 — no longer a ``State`` flag);
    ``intention_state`` is the live contact-intention's (the Bratman decision
    record, lm-27n.4) — the "why did I send" audit line. ``user_model`` is the
    live owner user_model (lm-27n.5); ``None`` falls back to the permissive
    :data:`~lifemodel.core.user_model_view.DEFAULT_USER_MODEL`, so the
    receptivity readings surface ``allowed=True / multiplier=1.0`` — the "why
    silent" audit is behaviour-neutral until the owner sets prefs. ``thoughts``
    are the being's live thoughts (lm-27n.6), most-salient first — the "what am I
    turning over" audit; empty until one is seeded/generated."""
    appraisal = appraise_receptivity(user_model or DEFAULT_USER_MODEL, state, now)
    last_tick_ago = _ago(state.last_tick_at, now)
    dt = max(0.0, minutes_between(state.last_tick_at, now))
    u = min(cfg.u_max, state.u + dt * cfg.alpha)
    inhibition, phase, grace_left = _action_pending(state, now, cfg)
    effective = effective_pressure(u, inhibition)

    # The would-wake reading measures the silence window from the gate's anchor
    # (admin override or the real last exchange), matching core/aggregation.py so the
    # dump's would_wake never contradicts the live gate (lm-md6.1).
    silence_anchor = _silence_anchor(state)
    exch_min = -minutes_between(silence_anchor, now) if silence_anchor is not None else None
    decl_min = -minutes_between(state.declined_at, now) if state.declined_at is not None else None
    lane = LaneState(
        last_exchange_at=exch_min,
        in_flight=False,
        declined_at=decl_min,
        decline_count=state.decline_count,
    )
    outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=cfg.params)

    # Core affect (lm-ukc.6): recompute this-tick target + contributions from the
    # snapshot with the live seeds, mirroring AffectSense (which reads state.u at the
    # start of the tick, before the drive integrates) — so the "target" the dump shows
    # is exactly what the being would ease toward. Current axes stay the stored values.
    affect_body = AffectBody.from_state(state, now=now, peak_hour_utc=cfg.peak_hour_utc)
    affect_tv, affect_ta, affect_contribs = affect_target(affect_body, cfg.affect_params)
    affect_v_ranked = tuple(
        sorted(affect_contribs.valence.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    affect_a_ranked = tuple(
        sorted(affect_contribs.arousal.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

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
        desire_status=desire_state,
        desire_spring=desire_spring,
        desire_source_thought_ids=desire_source_thought_ids,
        intention_status=intention_state,
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
        receptivity_allowed=appraisal.allowed,
        receptivity_multiplier=appraisal.pressure_multiplier,
        receptivity_confidence=appraisal.confidence,
        receptivity_hard_reasons=appraisal.hard_reasons,
        receptivity_soft_reasons=appraisal.soft_reasons,
        receptivity_constraints=appraisal.constraints,
        thoughts=tuple(_render_thought(t) for t in thoughts),
        last_tick_at=state.last_tick_at,
        last_tick_ago_min=last_tick_ago,
        brain_alive=last_tick_ago is not None and last_tick_ago <= BRAIN_STALE_MIN,
        contact_chain=contact_chain,
        affect_word=felt_word(state.affect_valence, state.affect_arousal),
        affect_valence=state.affect_valence,
        affect_arousal=state.affect_arousal,
        affect_updated_at=state.affect_updated_at,
        affect_target_valence=affect_tv,
        affect_target_arousal=affect_ta,
        affect_valence_contributions=affect_v_ranked,
        affect_arousal_contributions=affect_a_ranked,
    )

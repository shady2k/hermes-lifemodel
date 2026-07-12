"""Unit tests for the pure core-affect derivation (lm-ukc.2).

``affect_target`` projects the being's body scalars onto Russell's circumplex
(valence in [-1,1], arousal in [0,1]) with capped, separately-returned
contributions; ``ease`` is the leaky-integrator step (inertia). Both are pure,
Hermes-free, clock-free (elapsed minutes are passed in), so the formula is
fully unit-testable and calibratable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from lifemodel.core.affect import (
    AffectBody,
    AffectParams,
    AffectSense,
    affect_target,
    dominant_contribution,
    ease,
)
from lifemodel.core.component import TickContext
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.timeutil import to_iso
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeActiveSpan, FakeSpanLogger, FakeTracer

P = AffectParams()


def _target(**over: float) -> tuple[float, float]:
    """affect_target for a resting body, overriding only the named scalars."""
    fields: dict[str, object] = dict(
        u=0.0,
        decline_count=0,
        minutes_since_declined=None,
        unanswered_outbound_count=0,
        fatigue=0.0,
        minutes_since_exchange=None,
        energy=1.0,
        circadian=0.5,
        duration_over_theta=0.0,
    )
    fields.update(over)
    v, a, _contrib = affect_target(AffectBody(**fields), P)  # type: ignore[arg-type]
    return v, a


def test_resting_body_has_neutral_valence() -> None:
    # No loneliness, no rejection, no fresh exchange → valence sits at ~0 (neutral).
    v, _a = _target()
    assert abs(v) < 1e-9


def test_loneliness_lowers_valence_with_sqrt_gradation() -> None:
    # u normalized by u_ref (not θ): sqrt gives visible movement early, capped at u_ref.
    # 4h silence (u=1), ~11h (u=2.71), and ≥1 day (u=u_ref) grade down toward the cap.
    v_4h, _ = _target(u=1.0)
    v_11h, _ = _target(u=2.71)
    v_day, _ = _target(u=P.u_ref)
    v_more, _ = _target(u=P.u_ref * 5)
    assert v_4h < 0 and v_11h < v_4h < 0  # deeper silence → lower valence
    assert v_day == pytest.approx(-P.u_valence_cap, abs=1e-9)  # reaches the cap at u_ref
    assert v_more == pytest.approx(-P.u_valence_cap, abs=1e-9)  # never exceeds the cap
    # sqrt shape: 4h loneliness is a fraction of the cap, not near it.
    assert -P.u_valence_cap < v_4h < -0.05


def test_u_does_not_monopolize_valence() -> None:
    # Even at extreme u, the u-term is bounded by its cap, so other signals stay audible
    # (codex: no single lonely attractor). A fresh exchange still lifts a lonely valence.
    lonely, _ = _target(u=P.u_ref * 10)
    lonely_but_talked, _ = _target(u=P.u_ref * 10, minutes_since_exchange=0.0)
    assert lonely_but_talked > lonely  # the +exchange term is not drowned by u


def test_fresh_exchange_lifts_valence_and_decays() -> None:
    just_talked, _ = _target(minutes_since_exchange=0.0)
    a_while_ago, _ = _target(minutes_since_exchange=P.exchange_half_life_min)
    assert just_talked == pytest.approx(P.exchange_cap, abs=1e-9)  # full warmth at t=0
    assert a_while_ago == pytest.approx(P.exchange_cap / 2, abs=1e-3)  # half-life decay


def test_valence_is_clamped_to_unit_range() -> None:
    # Pile every downward contribution on at once; the sum still clamps to [-1, 1].
    v, _ = _target(
        u=P.u_ref * 5,
        decline_count=99,
        minutes_since_declined=0.0,
        unanswered_outbound_count=99,
        fatigue=1.0,
    )
    assert -1.0 <= v <= 1.0


def test_arousal_baseline_and_energy_calm_vs_keyed_up() -> None:
    # Rested daytime = moderate arousal; night + fatigue = calmer; sustained pull = up.
    _v, day_rested = _target(circadian=1.0, energy=1.0, fatigue=0.0)
    _v, night_tired = _target(circadian=0.0, energy=0.2, fatigue=1.0)
    _v, restless = _target(
        circadian=1.0, energy=1.0, fatigue=0.0, duration_over_theta=P.urgency_duration_ref_min
    )
    assert 0.0 <= night_tired < day_rested <= 1.0
    assert restless > day_rested  # sustained over-threshold pressure raises arousal
    assert 0.0 <= restless <= 1.0


def test_arousal_urgency_driven_by_duration_not_bare_u() -> None:
    # With θ=1, bare u/θ would saturate instantly; urgency must come from how LONG the
    # being has been over threshold (duration_over_theta), not from u crossing 1.
    _v, just_over = _target(u=1.01, duration_over_theta=0.0)
    _v, long_over = _target(u=1.01, duration_over_theta=P.urgency_duration_ref_min)
    assert long_over > just_over


def test_ease_moves_toward_target_by_leaky_step() -> None:
    # new = cur + (target-cur)*(1-exp(-dt/tau)); one tau-worth of dt closes ~63%.
    new = ease(
        current=0.0,
        target=1.0,
        dt_min=P.tau_valence_min,
        tau_min=P.tau_valence_min,
        deadband=P.deadband,
    )
    assert new == pytest.approx(1 - math.exp(-1.0), abs=1e-9)  # ≈0.632


def test_ease_deadband_suppresses_micro_moves() -> None:
    # A move smaller than the deadband is dropped (no jitter): tiny dt, far-ish target.
    new = ease(
        current=0.5, target=0.5001, dt_min=0.001, tau_min=P.tau_valence_min, deadband=P.deadband
    )
    assert new == 0.5


def test_ease_arousal_faster_than_valence() -> None:
    # Split tau: arousal tracks the body faster than valence lingers.
    dt = 30.0
    val_step = ease(
        current=0.0, target=1.0, dt_min=dt, tau_min=P.tau_valence_min, deadband=P.deadband
    )
    aro_step = ease(
        current=0.0, target=1.0, dt_min=dt, tau_min=P.tau_arousal_min, deadband=P.deadband
    )
    assert aro_step > val_step


# --- AffectSense: the AUTONOMIC integrator that wires the kernel into the tick ---

_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)
_PEAK = 13.0


def _sense() -> AffectSense:
    return AffectSense(params=P, peak_hour_utc=_PEAK)


def _ctx(state: State, now: datetime) -> TickContext:
    return TickContext(state=state, now=now, signals=(), trace=_TRACE)


def _update(intents: object) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes  # type: ignore[attr-defined]


def test_affect_sense_eases_valence_down_when_lonely() -> None:
    # A day of silence (u=u_ref) → the valence target is negative; one tick eases the
    # stored valence toward it (below its starting 0), and stamps affect_updated_at.
    state = State(u=P.u_ref, affect_valence=0.0, last_tick_at="2026-07-12T12:00:00+00:00")
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)  # +60 min
    changes = _update(_sense().step(_ctx(state, now)))
    assert changes["affect_valence"] < 0.0
    assert changes["affect_updated_at"] == to_iso(now)


def test_affect_sense_warms_valence_after_fresh_exchange() -> None:
    # Just talked → the warm exchange term lifts the valence target above 0.
    state = State(
        last_exchange_at="2026-07-12T12:55:00+00:00", last_tick_at="2026-07-12T12:00:00+00:00"
    )
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)
    changes = _update(_sense().step(_ctx(state, now)))
    assert changes["affect_valence"] > 0.0


def test_affect_sense_first_tick_holds_neutral() -> None:
    # No last_tick_at → no elapsed dt → inertia makes no move: affect holds its value.
    state = State(u=P.u_ref, affect_valence=0.0, affect_arousal=0.0, last_tick_at=None)
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)
    changes = _update(_sense().step(_ctx(state, now)))
    assert changes["affect_valence"] == 0.0
    assert changes["affect_arousal"] == 0.0


def test_affect_sense_emits_no_signal() -> None:
    # The one-way invariant, made STRUCTURAL: affect writes state but emits NO signal,
    # so nothing downstream (aggregation/wake) can read it. Contrast SolitudeDrive,
    # which DOES emit contact_pressure precisely because aggregation must read u.
    state = State(u=P.u_ref, last_tick_at="2026-07-12T12:00:00+00:00")
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)
    intents = _sense().step(_ctx(state, now))
    assert any(isinstance(i, UpdateState) for i in intents)
    assert all(not isinstance(i, EmitSignal) for i in intents)


def _logged_ctx(state: State, now: datetime) -> tuple[TickContext, FakeSpanLogger]:
    # The live tick hands AffectSense a span-bound logger over its child span; the
    # FakeActiveSpan records .set(...) attrs so the trace surfacing reads back.
    trace = FakeTracer().start_root()
    logger = FakeSpanLogger(FakeActiveSpan(trace, component="affect", tick=state.tick_count + 1))
    ctx = TickContext(state=state, now=now, signals=(), trace=trace, logger=logger)
    return ctx, logger


def test_affect_sense_stamps_felt_state_on_its_span() -> None:
    # lm-ukc.6: the being's felt state must be legible in the trace. AffectSense stamps
    # its component span with the eased axes, this-tick target, and the dominant
    # contributor per axis — so /lifemodel trace shows "why it feels so".
    state = State(u=P.u_ref, affect_valence=0.0, last_tick_at="2026-07-12T12:00:00+00:00")
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)  # +60 min
    ctx, logger = _logged_ctx(state, now)
    _sense().step(ctx)
    attrs = logger.span.attrs
    assert {
        "affect_valence",
        "affect_arousal",
        "affect_target_valence",
        "affect_target_arousal",
        "affect_top_valence",
        "affect_top_arousal",
    } <= set(attrs)
    # a day of silence with no other pull → loneliness ('u') dominates valence
    assert attrs["affect_target_valence"] < 0.0
    assert attrs["affect_top_valence"].startswith("u ")


def test_affect_sense_span_surfacing_is_optional() -> None:
    # ctx.logger is None in a bare context (no graph): step must not crash, mirroring
    # the ctx.observe guard — surfacing is fail-open, never a tick crash.
    state = State(u=P.u_ref, last_tick_at="2026-07-12T12:00:00+00:00")
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)
    intents = _sense().step(_ctx(state, now))  # _ctx has no logger
    assert any(isinstance(i, UpdateState) for i in intents)


def test_dominant_contribution_suppresses_sub_deadband_as_none() -> None:
    # A push that rounds to +0.00 at display precision is not "moving the axis": it
    # reads "none", matching the debug view's deadband filter — no misleading "x +0.00".
    assert dominant_contribution({"u": -0.002, "exchange": 0.001}) == "none"
    assert dominant_contribution({}) == "none"
    assert dominant_contribution({"u": -0.30, "exchange": 0.01}).startswith("u ")

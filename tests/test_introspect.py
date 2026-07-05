"""Tests for ``core/introspect`` — pure, Hermes-free personality readings (lm-zmf).

Contract under test (spec §7):
* ``compute_readings`` runs the REAL ``decide_reachout`` on a deep copy and
  surfaces its honest verdict — one crafted ``State`` per gate outcome plus the
  dedup and stale-pending-recovery cases;
* derived quantities (time-to-θ, silence-window / decline-backoff remaining)
  match hand-computed values built from the IMPORTED constants;
* the input ``State`` is never mutated (the copy is the only thing that moves);
* the temperament snapshot echoes the imported constants (no restated formulas).

Offline: no Hermes, no network, no Anthropic env required. Stdlib only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lifemodel.core.decision import ALPHA, BASE_PARAMS, PENDING_TIMEOUT_MIN, THETA
from lifemodel.core.introspect import PersonalityReadings, compute_readings, temperament
from lifemodel.sim.wake import WakeOutcome, backoff_interval
from lifemodel.state.model import State


def at(mins: float) -> datetime:
    return datetime(2026, 7, 5, 0, 0, tzinfo=UTC) + timedelta(minutes=mins)


# --- temperament: imported constants, not restated -------------------------------------------


def test_temperament_echoes_imported_constants():
    temp = temperament()

    assert temp.theta == THETA
    assert temp.alpha == ALPHA
    assert temp.base_params == BASE_PARAMS
    assert temp.pending_timeout_min == PENDING_TIMEOUT_MIN
    # The backoff schedule is the real R_n series from sim.wake, grown to the cap.
    expected_first = backoff_interval(
        decline_count=1, r0=BASE_PARAMS.r0, k=BASE_PARAMS.k, r_max=BASE_PARAMS.r_max
    )
    assert temp.backoff_schedule[0] == expected_first
    # Monotonically growing until it hits the cap, then constant at the cap.
    assert temp.backoff_schedule[-1] == BASE_PARAMS.r_max
    assert all(b <= BASE_PARAMS.r_max + 1e-9 for b in temp.backoff_schedule)


# --- one crafted State per gate outcome -------------------------------------------------------


def test_below_threshold_verdict_and_derived_time_to_theta():
    s = State(last_tick_at=at(0).isoformat())  # u starts at 0
    r = compute_readings(s, now=at(10), busy=False)

    assert r.gate_verdict == WakeOutcome.BELOW_THRESHOLD.value
    assert r.would_launch is False
    assert not r.risen_over_theta
    # u rose by 10 min of silence: 10 · α.
    assert r.u_risen == pytest.approx(10 * ALPHA, abs=1e-9)
    # (θ − u) / α minutes remain.
    assert r.time_to_theta_min == pytest.approx((THETA - r.u_risen) / ALPHA, abs=1e-6)
    # input untouched
    assert s.u == 0.0
    assert s.desire_status == "none"
    assert s.last_tick_at == at(0).isoformat()


def test_over_theta_inside_silence_window():
    s = State(u=50.0, last_tick_at=at(0).isoformat(), last_exchange_at=at(0).isoformat())
    r = compute_readings(s, now=at(10), busy=False)  # 10 < w=15

    assert r.gate_verdict == WakeOutcome.SILENCE_WINDOW.value
    assert r.would_launch is False
    assert r.risen_over_theta
    assert r.silence_window_remaining_min == pytest.approx(BASE_PARAMS.w - 10.0, abs=1e-9)


def test_over_theta_inside_decline_backoff():
    s = State(
        u=50.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1
    )
    r = compute_readings(s, now=at(20), busy=False)  # 20 < r0=30

    assert r.gate_verdict == WakeOutcome.DECLINE_BACKOFF.value
    assert r.would_launch is False
    expected_R1 = backoff_interval(
        decline_count=1, r0=BASE_PARAMS.r0, k=BASE_PARAMS.k, r_max=BASE_PARAMS.r_max
    )
    assert r.backoff_remaining_min == pytest.approx(expected_R1 - 20.0, abs=1e-9)


def test_clean_urge_launches_outreach():
    s = State(last_tick_at=at(0).isoformat())
    r = compute_readings(s, now=at(240), busy=False)  # 240 min → u crosses θ

    assert r.gate_verdict == WakeOutcome.URGE.value
    assert r.would_launch is True
    assert r.risen_over_theta
    # persisted lifecycle is still "none" — the wake is what the NEXT tick would do.
    assert s.desire_status == "none"


def test_dedup_gate_urge_but_would_not_launch():
    # Gate clears (URGE) but a desire is already live → Aggregator dedup, no launch.
    s = State(u=50.0, desire_status="active", last_tick_at=at(0).isoformat())
    r = compute_readings(s, now=at(10), busy=False)

    assert r.gate_verdict == WakeOutcome.URGE.value  # gate said URGE
    assert r.would_launch is False  # …but outreach does not launch (dedup)
    # input untouched
    assert s.desire_status == "active"


def test_stale_pending_recovery_is_flagged_and_verdict_reflects_reject():
    s = State(
        u=99.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    r = compute_readings(s, now=at(PENDING_TIMEOUT_MIN), busy=False)

    assert r.stale_pending_recovery is True
    assert r.stale_pending_age_min == pytest.approx(PENDING_TIMEOUT_MIN, abs=1e-9)
    # The recovered REJECT stamps a fresh decline → the verdict reflects the backoff.
    assert r.gate_verdict == WakeOutcome.DECLINE_BACKOFF.value
    assert r.would_launch is False

    # SNAPSHOT CONSISTENCY (must-fix 1/3): the lifecycle readings are derived from
    # the post-decision snapshot, so they reflect the recovery the verdict saw —
    # not the persisted state, which still has the stale active+pending desire.
    assert r.desire_status == "none"  # REJECT resolved the desire
    assert r.pending is False  # …and cleared the pending bookkeeping
    assert r.decline_count == 1  # the recovery's first decline
    assert r.declined_at is not None
    expected_r1 = backoff_interval(
        decline_count=1, r0=BASE_PARAMS.r0, k=BASE_PARAMS.k, r_max=BASE_PARAMS.r_max
    )
    assert r.backoff_remaining_min == pytest.approx(expected_r1, abs=1e-9)

    # The original (persisted) state is untouched — recovery only ran on the copy.
    assert s.desire_status == "active"
    assert s.decline_count == 0
    assert s.pending_proactive_id == "p1"


def test_stale_last_tick_rises_u_so_drive_and_wake_agree():
    # A stale `last_tick_at` means the elapsed-since-last-tick rise is large: the
    # persisted u is 0, but "right now" u has risen past θ. DRIVE (u now) and WAKE
    # READINESS (urge now risen) must read the SAME post-decision urge — the bug
    # they regress is DRIVE showing persisted 0% while WAKE says ≥θ (must-fix 2).
    s = State(u=0.0, last_tick_at=at(0).isoformat())  # persisted u = 0
    r = compute_readings(s, now=at(300), busy=False)  # 300 min of silence → u ≥ θ

    assert r.u_risen >= THETA  # risen past θ, even though persisted u was 0
    assert r.u_risen > 0.0
    assert r.gate_verdict == WakeOutcome.URGE.value
    # The persisted state is untouched (no rise leaked back).
    assert s.u == 0.0


def test_fresh_pending_is_not_flagged_as_stale():
    s = State(
        u=99.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    r = compute_readings(s, now=at(PENDING_TIMEOUT_MIN - 1), busy=False)

    assert r.stale_pending_recovery is False
    assert r.stale_pending_age_min == pytest.approx(PENDING_TIMEOUT_MIN - 1, abs=1e-9)


# --- the input State is never mutated ---------------------------------------------------------


def test_compute_readings_never_mutates_input_state():
    s = State(
        u=42.0,
        desire_status="active",
        declined_at=at(0).isoformat(),
        decline_count=2,
        pending_proactive_id="p9",
        pending_proactive_since=at(1).isoformat(),
        last_tick_at=at(0).isoformat(),
        last_exchange_at=at(2).isoformat(),
        tick_count=17,
    )
    snapshot_fields = (
        "u",
        "duration_over_theta",
        "desire_status",
        "declined_at",
        "decline_count",
        "pending_proactive_id",
        "pending_proactive_since",
        "last_tick_at",
        "last_exchange_at",
        "tick_count",
    )
    before = {f: getattr(s, f) for f in snapshot_fields}

    compute_readings(s, now=at(20), busy=False)

    after = {f: getattr(s, f) for f in snapshot_fields}
    assert after == before


# --- gate ladder: statuses per branch ---------------------------------------------------------


def _rung(r: PersonalityReadings, name: str) -> str:
    match = [x for x in r.gate_ladder if x.name == name]
    assert len(match) == 1, f"no single rung named {name!r}"
    return match[0].status


def test_gate_ladder_below_threshold_branch():
    r = compute_readings(State(last_tick_at=at(0).isoformat()), now=at(10), busy=False)
    assert _rung(r, "below_threshold") == "BLOCKS HERE"
    assert _rung(r, "in_flight") == "n/a"  # u < θ → cannot matter
    assert _rung(r, "urge") == "—"


def test_gate_ladder_clean_urge_branch():
    r = compute_readings(State(last_tick_at=at(0).isoformat()), now=at(240), busy=False)
    assert _rung(r, "below_threshold") == "clear"
    assert _rung(r, "in_flight") == "UNKNOWN"  # runtime-only
    assert _rung(r, "silence_window") == "clear"
    assert _rung(r, "decline_backoff") == "clear"
    assert _rung(r, "urge") == "reached"


def test_gate_ladder_decline_backoff_branch():
    s = State(
        u=50.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1
    )
    r = compute_readings(s, now=at(20), busy=False)
    assert _rung(r, "decline_backoff") == "would-block"
    assert _rung(r, "urge") == "—"  # blocked above


def test_gate_ladder_stale_recovery_uses_snapshot_decline():
    # The verdict (run on the copy) reflects the REJECT stale-recovery just applied,
    # so the ladder must see that freshly-stamped decline — not the persisted state
    # (which still has no decline). Regression: previously this rendered "R_0 ... ~0.0".
    s = State(
        u=99.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    r = compute_readings(s, now=at(PENDING_TIMEOUT_MIN), busy=False)

    decline_rung = [x for x in r.gate_ladder if x.name == "decline_backoff"][0]
    assert decline_rung.status == "would-block"
    assert "R_1" in decline_rung.detail  # the recovery's first decline, not R_0
    assert "R_0" not in decline_rung.detail
    # A real backoff remains (the just-stamped decline's full R_1), not ~0 left.
    assert not decline_rung.detail.startswith("~0.0 min")

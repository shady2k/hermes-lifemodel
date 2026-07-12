"""The wake-decision: hard policy gates over the drive (spec §7).

An urge (``u ≥ θ_u``) is *necessary but not sufficient* to wake cognition. Four
gates, applied in a fixed precedence, decide whether an urge is allowed to wake:
the active-silence window ``W``, no-wake-while-in-flight, the *growing* decline
backoff ``R``, and the internal-impulse exclusion (handled at state-update, not
here). Crossing the threshold never sends a message; it can only produce an URGE
that wakes cognition.
"""

from __future__ import annotations

from lifemodel.core.wake import (
    GateParams,
    LaneState,
    WakeOutcome,
    backoff_interval,
    evaluate_wake,
)

# A calm default gate config for the tests; behaviour-under-test overrides fields.
PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)


def test_first_decline_backoff_is_the_base_interval() -> None:
    # decline_count = 1 (one decline so far) → the base R0, no growth yet.
    assert backoff_interval(decline_count=1, r0=30.0, k=2.0, r_max=1000.0) == 30.0


def test_backoff_grows_geometrically_with_consecutive_declines() -> None:
    assert backoff_interval(decline_count=2, r0=30.0, k=2.0, r_max=1000.0) == 60.0
    assert backoff_interval(decline_count=3, r0=30.0, k=2.0, r_max=1000.0) == 120.0


def test_backoff_is_capped_at_r_max() -> None:
    assert backoff_interval(decline_count=10, r0=30.0, k=2.0, r_max=100.0) == 100.0


def test_backoff_is_non_decreasing_in_decline_count() -> None:
    intervals = [
        backoff_interval(decline_count=n, r0=30.0, k=1.7, r_max=5000.0) for n in range(1, 8)
    ]
    assert intervals == sorted(intervals)


def test_below_threshold_never_wakes() -> None:
    state = LaneState(last_exchange_at=None)
    out = evaluate_wake(u=0.99, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.BELOW_THRESHOLD


def test_at_threshold_with_clean_state_wakes() -> None:
    # No prior exchange, no turn in flight, no decline → an urge is allowed to wake.
    state = LaneState(last_exchange_at=None)
    out = evaluate_wake(u=1.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.URGE


def test_in_flight_vetoes_even_above_threshold() -> None:
    state = LaneState(last_exchange_at=None, in_flight=True)
    out = evaluate_wake(u=5.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.IN_FLIGHT


def test_within_active_silence_window_vetoes() -> None:
    # Exchange 5 min ago, W = 15 min → still inside the window, no interruption.
    state = LaneState(last_exchange_at=995.0)
    out = evaluate_wake(u=2.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.SILENCE_WINDOW


def test_just_outside_active_silence_window_wakes() -> None:
    state = LaneState(last_exchange_at=984.0)  # 16 min ago, W = 15
    out = evaluate_wake(u=2.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.URGE


def test_decline_backoff_vetoes_within_R() -> None:
    # One decline 10 min ago; base backoff R0 = 30 min → still suppressed.
    state = LaneState(last_exchange_at=None, declined_at=990.0, decline_count=1)
    out = evaluate_wake(u=3.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.DECLINE_BACKOFF


def test_decline_backoff_expires_after_R_then_wakes() -> None:
    # One decline 31 min ago; base backoff R0 = 30 min → expired, urge may wake.
    state = LaneState(last_exchange_at=None, declined_at=969.0, decline_count=1)
    out = evaluate_wake(u=3.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.URGE


def test_growing_backoff_suppresses_longer_after_more_declines() -> None:
    # Two declines: backoff = R0·k = 60 min; 40 min elapsed → still suppressed,
    # whereas a fixed R0 = 30 would have released it (this is the anti-drum lever).
    state = LaneState(last_exchange_at=None, declined_at=960.0, decline_count=2)
    out = evaluate_wake(u=3.0, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.DECLINE_BACKOFF


def test_precedence_below_threshold_beats_every_veto() -> None:
    # u below θ_u → BELOW_THRESHOLD regardless of in-flight / window / decline.
    state = LaneState(last_exchange_at=999.0, in_flight=True, declined_at=999.0, decline_count=3)
    out = evaluate_wake(u=0.5, now=1000.0, state=state, params=PARAMS)
    assert out is WakeOutcome.BELOW_THRESHOLD

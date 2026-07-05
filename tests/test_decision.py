from datetime import UTC, datetime, timedelta

from lifemodel.core.decision import (
    PENDING_TIMEOUT_MIN,
    THETA,
    _minutes_between,
    apply_verdict,
    decide_reachout,
    observe_exchange,
)
from lifemodel.sim.aggregation import Verdict
from lifemodel.state.model import State


def at(mins):
    return datetime(2026, 7, 5, 0, 0, tzinfo=UTC) + timedelta(minutes=mins)


def test_rises_in_silence_and_wakes_after_urge_matures():
    # BASE α=1/240 → ~240 min of silence to cross θ=1. No prior exchange, no reject.
    s = State(last_tick_at=at(0).isoformat())
    d = decide_reachout(s, now=at(239), busy=False)
    assert d.wake is False and s.u < THETA
    d = decide_reachout(s, now=at(240), busy=False)
    assert d.wake is True and s.desire_status == "active"


def test_dedup_no_second_wake_while_desire_active():
    s = State(last_tick_at=at(0).isoformat())
    decide_reachout(s, now=at(240), busy=False)  # active
    d = decide_reachout(s, now=at(300), busy=False)  # still active
    assert d.wake is False and s.desire_status == "active"


def test_no_wake_within_active_silence_window():
    s = State(u=50.0, last_tick_at=at(0).isoformat(), last_exchange_at=at(0).isoformat())
    d = decide_reachout(s, now=at(10), busy=False)  # 10 < W=15
    assert d.wake is False and d.reason == "no_wake_silence_window"


def test_no_wake_while_busy():
    s = State(u=50.0, last_tick_at=at(0).isoformat())
    d = decide_reachout(s, now=at(30), busy=True)
    assert d.wake is False and d.reason == "no_wake_in_flight"


def test_user_exchange_satiates_and_clears_desire_and_reject():
    s = State(
        u=99.0,
        desire_status="active",
        declined_at=at(0).isoformat(),
        decline_count=3,
        last_tick_at=at(0).isoformat(),
    )
    observe_exchange(s, actor="user", label="two_way", now=at(100))
    assert s.u < 99.0 and s.desire_status == "none" and s.decline_count == 0
    assert s.declined_at is None and s.last_exchange_at == at(100).isoformat()


def test_internal_impulse_never_satiates_or_touches_clock():
    s = State(u=50.0, last_tick_at=at(0).isoformat())
    observe_exchange(s, actor="proactive_internal", label="monologue", now=at(5))
    assert s.u == 50.0 and s.last_exchange_at is None


def test_fulfill_satiates_resets_and_clears_pending():
    s = State(
        u=99.0,
        duration_over_theta=40.0,
        desire_status="active",
        pending_proactive_id="p1",
        last_tick_at=at(0).isoformat(),
    )
    apply_verdict(s, Verdict.FULFILL, now=at(50))
    assert s.desire_status == "none" and s.duration_over_theta == 0.0
    assert s.u < 99.0 and s.pending_proactive_id is None and s.last_contact_at == at(50).isoformat()


def test_reject_records_growing_backoff_no_satiation():
    s = State(
        u=99.0,
        desire_status="active",
        decline_count=1,
        pending_proactive_id="p1",
        last_tick_at=at(0).isoformat(),
    )
    apply_verdict(s, Verdict.REJECT, now=at(50))
    assert s.desire_status == "none" and s.decline_count == 2
    assert s.u == 99.0 and s.declined_at == at(50).isoformat() and s.pending_proactive_id is None


def test_reject_backoff_suppresses_then_releases():
    # After a reject at t, evaluate_wake's growing backoff must veto within R then wake after.
    s = State(
        u=99.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1
    )
    assert decide_reachout(s, now=at(20), busy=False).wake is False  # 20 < r0=30
    s2 = State(
        u=99.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1
    )
    assert decide_reachout(s2, now=at(31), busy=False).wake is True  # 31 > 30


# --- BLOCKER 1: stale-pending recovery (a lost post_llm_call must not silence forever) ----


def test_stale_pending_resolves_silently_and_can_wake_again():
    # An `active` desire whose pending_proactive_since is old (>= the timeout)
    # means the post_llm_call verdict never arrived (lost or mis-correlated).
    # decide_reachout must release it silently (no reject) rather than dedup
    # every future urge against a desire nothing will ever resolve.
    s = State(
        u=0.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    d = decide_reachout(s, now=at(PENDING_TIMEOUT_MIN), busy=False)
    assert d.wake is False  # urge hasn't matured yet at this point
    assert s.desire_status == "none"
    assert s.pending_proactive_id is None
    assert s.pending_proactive_since is None
    # Silent release, not a reject:
    assert s.decline_count == 0
    assert s.declined_at is None

    # And it can wake again once a fresh urge matures.
    d2 = decide_reachout(s, now=at(PENDING_TIMEOUT_MIN + 240), busy=False)
    assert d2.wake is True
    assert s.desire_status == "active"


def test_pending_below_timeout_is_not_recovered():
    # Below PENDING_TIMEOUT_MIN, the pending desire is left alone — dedup still
    # applies (no second wake while a verdict may still legitimately arrive).
    s = State(
        u=99.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    d = decide_reachout(s, now=at(PENDING_TIMEOUT_MIN - 1), busy=False)
    assert d.wake is False
    assert s.desire_status == "active"
    assert s.pending_proactive_id == "p1"


# --- BLOCKER 2a: observe_exchange must also clear pending proactive bookkeeping ----


def test_user_exchange_clears_pending_proactive_bookkeeping():
    s = State(
        u=50.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    observe_exchange(s, actor="user", label="two_way", now=at(10))
    assert s.desire_status == "none"
    assert s.pending_proactive_id is None
    assert s.pending_proactive_since is None


# --- BLOCKER 3: _minutes_between must be defensive, never crash the tick ----


def test_minutes_between_defensive_on_unparseable_string():
    assert _minutes_between("not-a-date", at(0)) == 0.0


def test_minutes_between_defensive_on_naive_timestamp():
    assert _minutes_between("2026-07-05T00:00:00", at(0)) == 0.0


def test_decide_reachout_survives_malformed_last_tick_at():
    # State.from_dict never validates last_tick_at as a parsed instant (it is
    # documented as an opaque display string) — so a malformed or tz-naive
    # value can legitimately reach decide_reachout. It must not crash the tick.
    s = State(last_tick_at="not-a-date")
    d = decide_reachout(s, now=at(0), busy=False)
    assert d.reason == "no_wake_below_threshold"


def test_decide_reachout_survives_naive_last_tick_at():
    s = State(last_tick_at="2026-07-05T00:00:00")  # tz-naive
    d = decide_reachout(s, now=at(0), busy=False)
    assert d.reason == "no_wake_below_threshold"

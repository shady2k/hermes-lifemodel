# tests/test_introspect.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.introspect import DebugConfig, Readings, compute_readings
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State

CFG = DebugConfig(
    params=GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0),
    theta=1.0,
    i0=1.0,
    grace_min=45.0,
    halflife_min=60.0,
    peak_hour_utc=13.0,
    max_per_day=3,
    min_interval_min=60.0,
)
NOW = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)


def test_reads_physiology_and_drive() -> None:
    state = State(u=2.0, energy=0.7, fatigue=0.3, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert isinstance(r, Readings)
    assert r.energy == 0.7
    assert r.fatigue == 0.3
    assert 0.0 <= r.circadian <= 1.0
    assert r.u == 2.0
    assert r.inhibition == 0.0  # no ActionPending
    assert abs(r.effective - 2.0) < 1e-9  # u*(1-0)
    assert r.would_wake is True  # effective >= theta, no gates


def test_action_pending_suppresses_effective_and_wake() -> None:
    state = State(
        u=3.0,
        action_pending_since="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)  # 10 min ago -> in grace -> inhibition 1
    assert r.inhibition == 1.0
    assert r.action_pending_phase == "grace"
    assert r.effective == 0.0
    assert r.would_wake is False
    assert r.wake_reason == "no_wake_below_threshold"  # effective 0 < theta


def test_backstop_readings() -> None:
    log = ["2026-07-06T03:30:00+00:00", "2026-07-06T02:00:00+00:00"]  # 2 today, last 30m ago
    state = State(u=2.0, proactive_send_log=log, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.sends_today == 2
    assert r.sends_cap == 3
    assert r.send_allowed is False  # last send 30 min ago < 60 min interval


def test_silence_window_and_backoff() -> None:
    state = State(
        u=2.0,
        last_exchange_at="2026-07-06T03:55:00+00:00",  # 5 min ago, w=15 -> 10 left
        declined_at="2026-07-06T03:40:00+00:00",
        decline_count=1,  # 20 min ago, r0=30 -> 10 left
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert abs((r.silence_window_remaining_min or 0) - 10.0) < 1e-6
    assert abs((r.backoff_remaining_min or 0) - 10.0) < 1e-6
    assert r.would_wake is False  # silence window blocks

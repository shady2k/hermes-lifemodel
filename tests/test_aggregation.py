# tests/test_aggregation.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.component import TickContext
from lifemodel.core.intents import UpdateState
from lifemodel.core.taxonomy import (
    contact_signal,
    exchange_signal,
    in_flight_signal,
    verdict_signal,
)
from lifemodel.sim.aggregation import Verdict
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State

PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)


def _agg() -> ContactAggregation:
    return ContactAggregation(params=PARAMS, theta=1.0, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals))


def _changes(intents) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def test_urge_over_threshold_creates_active_desire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"


def test_below_threshold_stays_none(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=0.5, delta=0.0, timestamp=None)  # < theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_second_urge_is_deduped_no_refire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"  # still one desire — dedup


def test_silence_window_suppresses_wake(tmp_path) -> None:
    # exchange 5 min ago (< w=15) → SILENCE_WINDOW, no wake even with high u
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        desire_status="none",
        last_exchange_at="2026-07-06T03:55:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_in_flight_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    busy = in_flight_signal(origin_id="f1", value=True, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, busy], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_decline_backoff_suppresses_then_allows(tmp_path) -> None:
    # declined 10 min ago, decline_count=1 → backoff r0=30 min active → no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        desire_status="none",
        decline_count=1,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # inside backoff


def test_duration_over_theta_accumulates(tmp_path) -> None:
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5 min
    state = State(
        u=2.0,
        desire_status="active",
        duration_over_theta=10.0,
        last_tick_at="2026-07-06T00:00:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)  # >= theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert abs(changes["duration_over_theta"] - 15.0) < 1e-9


def test_aggregation_does_not_write_u_on_normal_tick(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert "u" not in changes  # neuron owns u; aggregation only writes it on FULFILL (Task 4)


def test_exchange_clears_desire_and_resets_clocks(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        desire_status="active",
        decline_count=2,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # desire cleared
    assert changes["decline_count"] == 0
    assert changes["declined_at"] is None
    assert changes["last_exchange_at"] == now.isoformat()


def test_exchange_this_tick_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # fresh exchange → SILENCE_WINDOW


def test_internal_impulse_is_not_an_exchange(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    own = exchange_signal(
        origin_id="e1", actor="proactive_internal", label="two_way", timestamp=None
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, own], tmp_path=tmp_path)))
    assert changes["last_exchange_at"] is None  # own nudge did not reset the clock
    assert changes["desire_status"] == "active"  # desire not cleared by own nudge


def test_fulfill_satiates_u_and_stamps_contact(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.5,
        desire_status="active",
        duration_over_theta=99.0,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["u"] == 0.5  # 1.5 - beta*1.0
    assert changes["duration_over_theta"] == 0.0
    assert changes["last_contact_at"] == now.isoformat()
    assert changes["last_exchange_at"] == now.isoformat()


def test_reject_records_growing_backoff(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.5,
        desire_status="active",
        decline_count=1,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["decline_count"] == 2
    assert changes["declined_at"] == now.isoformat()
    assert "u" not in changes  # reject does not satiate


def test_defer_holds_desire_and_keeps_pressure(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.DEFER, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "deferred"
    assert "u" not in changes  # pressure not dropped


def test_fulfill_resets_duration_even_when_u_stays_high(tmp_path) -> None:
    # FULFILL resets duration_over_theta unconditionally (matching decision.py),
    # NOT merely because the satiated u fell below theta. Here u=5.0 -> satiate
    # to 4.0 (still >= theta) but duration must still reset to 0.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=5.0,
        desire_status="active",
        duration_over_theta=500.0,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=5.0, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["u"] == 4.0
    assert changes["duration_over_theta"] == 0.0  # reset regardless of u


def test_reject_then_backoff_blocks_immediate_rewake(tmp_path) -> None:
    # after a REJECT this tick, the fresh declined_at must veto a wake in the same tick
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=5.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=5.0, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # rejected + backoff vetoes re-wake


# --- Phase C1: effective pressure gates ---


def test_action_pending_grace_suppresses_wake_despite_high_latent(tmp_path) -> None:
    # latent u=3 (>= theta) but a send 10 min ago (within 45-min grace) -> effective ~0 -> no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        desire_status="none",
        action_pending_since="2026-07-06T03:50:00+00:00",  # 10 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # inhibited during grace


def test_pressure_recovers_after_grace_and_decay(tmp_path) -> None:
    # send ~3h ago: grace(45m)+ ~2 half-lives -> inhibition ~0.06 -> effective ~ u -> wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        desire_status="none",
        action_pending_since="2026-07-06T01:00:00+00:00",  # 180 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"  # ignored long enough -> loneliness returns


def test_duration_over_theta_uses_latent_not_effective(tmp_path) -> None:
    # even fully inhibited (effective 0), latent u>=theta so duration keeps accruing
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5
    state = State(
        u=2.0,
        desire_status="none",
        duration_over_theta=10.0,
        action_pending_since="2026-07-06T00:04:00+00:00",  # in grace -> inhibition 1
        last_tick_at="2026-07-06T00:00:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert (
        abs(changes["duration_over_theta"] - 15.0) < 1e-9
    )  # latent-based, accrues under inhibition
    assert changes["desire_status"] == "none"  # but no wake (effective suppressed)

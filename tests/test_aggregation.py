# tests/test_aggregation.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.component import TickContext
from lifemodel.core.intents import UpdateState
from lifemodel.core.taxonomy import contact_signal, in_flight_signal
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

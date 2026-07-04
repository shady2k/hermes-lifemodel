from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import ThresholdAggregator
from lifemodel.logging import get_logger
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore
from lifemodel.tick import SERVICE_LIVENESS_MAX_AGE, run_tick, service_is_alive

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)


def test_service_alive_when_stamp_fresh() -> None:
    st = State(egress_service_alive_at=(_T0 - timedelta(seconds=30)).isoformat())
    assert service_is_alive(st, now=_T0) is True


def test_service_dead_when_stamp_stale_or_absent() -> None:
    assert service_is_alive(State(), now=_T0) is False
    stale = State(egress_service_alive_at=(_T0 - SERVICE_LIVENESS_MAX_AGE - timedelta(seconds=1)).isoformat())
    assert service_is_alive(stale, now=_T0) is False


def test_run_tick_defers_and_does_not_touch_pressure_when_service_alive() -> None:
    fresh = State(pressure=28.0, egress_service_alive_at=(_T0 - timedelta(seconds=10)).isoformat())
    lm = build_lifemodel(
        base_dir=__import__("pathlib").Path("/unused"),
        state=FakeStateStore(fresh),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=ThresholdAggregator(threshold=10.0),
        neurons=(),
    )
    decision = run_tick(lm, logger=get_logger("t"))
    assert decision.wake is False
    assert lm.state.load().pressure == 28.0  # NOT accumulated/drained while deferring

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.state.model import State
from lifemodel.tick import SERVICE_LIVENESS_MAX_AGE, service_is_alive

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=UTC)


def test_service_alive_when_stamp_fresh() -> None:
    st = State(egress_service_alive_at=(_T0 - timedelta(seconds=30)).isoformat())
    assert service_is_alive(st, now=_T0) is True


def test_service_dead_when_stamp_stale_or_absent() -> None:
    assert service_is_alive(State(), now=_T0) is False
    stale = State(
        egress_service_alive_at=(_T0 - SERVICE_LIVENESS_MAX_AGE - timedelta(seconds=1)).isoformat()
    )
    assert service_is_alive(stale, now=_T0) is False


# NOTE: the old test_run_tick_defers_and_does_not_touch_pressure_when_service_alive
# asserted run_tick's now-removed State.pressure accumulation/deferral behaviour —
# obsolete per the wire-desire-model plan (Task 3). Task 4 demotes run_tick to a
# silent watchdog that never wakes on pressure at all, so a pressure-based defer
# test has no replacement here; run_tick's own behaviour is Task 4's test file.

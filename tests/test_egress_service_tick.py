from __future__ import annotations

from datetime import datetime, timezone

from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import ThresholdAggregator
from lifemodel.domain.egress import ReachOutcome
from lifemodel.egress_service import run_proactive_tick
from lifemodel.logging import get_logger
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)
_TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class _RecordingEgress:
    def __init__(self, outcome: ReachOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[dict[str, str | None], str]] = []

    def reach_out(self, target: dict[str, str | None], impulse: str) -> ReachOutcome:
        self.calls.append((dict(target), impulse))
        return self.outcome


def _lm(pressure: float) -> object:
    return build_lifemodel(
        base_dir=__import__("pathlib").Path("/unused"),
        state=FakeStateStore(State(pressure=pressure)),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=ThresholdAggregator(threshold=10.0),
        neurons=(),
    )


def test_below_threshold_does_not_reach_out() -> None:
    lm = _lm(pressure=1.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert egress.calls == []
    assert lm.state.load().pressure > 0.0  # pressure NOT drained


def test_delivered_drains_pressure_and_stamps_contact() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert outcome is ReachOutcome.DELIVERED
    assert len(egress.calls) == 1
    st = lm.state.load()
    assert st.pressure == 0.0
    assert st.last_contact_at is not None
    assert st.cooldown_until is not None


def test_failed_delivery_keeps_pressure() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.FAILED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert outcome is ReachOutcome.FAILED
    st = lm.state.load()
    assert st.pressure == 28.0        # NOT drained — retry next tick
    assert st.last_contact_at is None


def test_busy_skips_delivery_and_keeps_pressure() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"), busy=True)
    assert outcome is ReachOutcome.SKIPPED_BUSY
    assert egress.calls == []
    assert lm.state.load().pressure == 28.0

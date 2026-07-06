"""Tests for :func:`lifemodel.egress_service.run_proactive_tick` (spec §13/§14, model A).

Phase E3: ``run_proactive_tick`` now drives ``coreloop.tick()`` (the layered pipeline),
consumes surfaced ``LaunchProactive`` intents, applies the global backstop
(``allow_send``), and reaches out with ``IMPULSE_LABEL_PREFIX + launch.prompt``.
Blocked/failed launches roll pending back. Liveness always stamped.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import build_lifemodel
from lifemodel.domain.egress import ReachOutcome
from lifemodel.egress_service import run_proactive_tick
from lifemodel.impulse import IMPULSE_LABEL_PREFIX
from lifemodel.log import get_logger
from lifemodel.state.model import State

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class FakeEgress:
    def __init__(self, outcome=ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


class FixedClock:
    def __init__(self, m):
        self._m = m

    def now(self):
        return self._m


def _lm(tmp_path, state: State, now: datetime):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(now))
    lm.state.commit(state)
    return lm


def test_active_desire_launches_native_turn(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(
        desire_status="active",
        u=2.0,
        energy=1.0,
        pending_proactive_id=None,
        last_tick_at="2026-07-06T11:59:00+00:00",
    )
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert len(egress.calls) == 1
    _, impulse = egress.calls[0]
    assert impulse.startswith(IMPULSE_LABEL_PREFIX)  # correlation marker prepended
    assert lm.state.load().pending_proactive_id is not None  # a turn is in flight


def test_no_active_desire_does_not_reach_out(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="none", u=0.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert egress.calls == []


def test_backstop_blocks_when_cap_reached(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    log = [
        "2026-07-06T11:00:00+00:00",
        "2026-07-06T10:00:00+00:00",
        "2026-07-06T09:00:00+00:00",
    ]  # 3 today
    state = State(
        desire_status="active",
        u=2.0,
        energy=1.0,
        proactive_send_log=log,
        last_tick_at="2026-07-06T11:59:00+00:00",
    )
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert egress.calls == []  # backstop blocked the send
    assert lm.state.load().desire_status == "deferred"  # held, not sent


def test_failed_launch_rolls_back_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(
        desire_status="active",
        u=2.0,
        energy=1.0,
        last_tick_at="2026-07-06T11:59:00+00:00",
    )
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress(outcome=ReachOutcome.UNAVAILABLE)
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert final.pending_proactive_id is None  # rolled back
    assert final.desire_status == "active"  # kept to retry (not rejected)


def test_liveness_is_stamped(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="none", last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    run_proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    assert lm.state.load().egress_service_alive_at == now.isoformat()

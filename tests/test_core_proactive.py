"""Tests for :func:`lifemodel.core.proactive.proactive_tick` (spec §13/§14).

The Hermes-free decision+delivery tick: drives ``coreloop.tick()``, consumes a
surfaced ``LaunchProactive``, applies the global backstop, and reaches out via the
injected ``ProactiveEgressPort``. Unlike the deleted ``egress_service`` version it
stamps NO liveness (``last_tick_at`` — the dt clock — is stamped by
``coreloop.tick()``; liveness is derived from its freshness elsewhere) and imports
no Hermes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import build_lifemodel
from lifemodel.core.proactive import proactive_tick
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.egress import ReachOutcome
from lifemodel.log import get_logger
from lifemodel.state.model import State

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class FakeEgress:
    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


class FixedClock:
    def __init__(self, m: datetime) -> None:
        self._m = m

    def now(self) -> datetime:
        return self._m


def _lm(tmp_path, state: State, now: datetime):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(now))
    lm.state.commit(state)
    return lm


def _active(**over) -> State:
    base = dict(desire_status="active", u=2.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00")
    base.update(over)
    return State(**base)


NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
CAP_LOG = ["2026-07-06T11:00:00+00:00", "2026-07-06T10:00:00+00:00", "2026-07-06T09:00:00+00:00"]


def test_active_desire_launches_native_turn(tmp_path) -> None:
    lm = _lm(tmp_path, _active(), NOW)
    egress = FakeEgress()
    out = proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert out is ReachOutcome.DELIVERED
    assert len(egress.calls) == 1
    _, impulse = egress.calls[0]
    assert impulse.startswith(IMPULSE_LABEL_PREFIX)
    assert lm.state.load().pending_proactive_id is not None  # a turn is in flight


def test_no_active_desire_does_not_reach_out(tmp_path) -> None:
    idle = State(desire_status="none", u=0.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, idle, NOW)
    egress = FakeEgress()
    proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert egress.calls == []


def test_backstop_block_defers_and_refunds(tmp_path) -> None:
    lm = _lm(tmp_path, _active(proactive_send_log=CAP_LOG), NOW)
    egress = FakeEgress()
    proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert egress.calls == []  # backstop blocked the send
    assert final.desire_status == "deferred"  # held, not sent
    assert final.energy >= 0.99  # reservation refunded


def test_failed_delivery_rolls_pending_active_and_refunds(tmp_path) -> None:
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(outcome=ReachOutcome.UNAVAILABLE), TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert final.pending_proactive_id is None  # rolled back
    assert final.desire_status == "active"  # kept to retry
    assert final.energy >= 0.99  # refunded — no turn ran


def test_delivered_launch_keeps_the_cost(tmp_path) -> None:
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    assert lm.state.load().energy < 1.0  # the turn ran -> energy spent, not refunded


def test_does_not_stamp_egress_service_alive_at(tmp_path) -> None:
    lm = _lm(tmp_path, State(desire_status="none", last_tick_at="2026-07-06T11:59:00+00:00"), NOW)
    proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    # liveness is NOT a separate stamp anymore; last_tick_at (dt clock) carries it
    assert getattr(lm.state.load(), "egress_service_alive_at", None) is None
    assert lm.state.load().last_tick_at == NOW.isoformat()  # coreloop stamped the tick

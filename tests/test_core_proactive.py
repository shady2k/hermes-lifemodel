"""Tests for :func:`lifemodel.core.proactive.proactive_tick` (spec §13/§14).

The Hermes-free decision+delivery tick: drives ``coreloop.tick()``, consumes a
surfaced ``LaunchProactive``, applies the global backstop, and reaches out via the
injected ``ProactiveEgressPort``. Unlike the deleted ``egress_service`` version it
stamps NO liveness (``last_tick_at`` — the dt clock — is stamped by
``coreloop.tick()``; liveness is derived from its freshness elsewhere) and imports
no Hermes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import lifemodel.log as lm_logging
from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.intention_view import read_live_contact_intention
from lifemodel.core.proactive import proactive_tick
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import DesireState
from lifemodel.events import EventRing, EventSink
from lifemodel.log import EventTee, get_logger
from lifemodel.state.model import State


class _RecordingLogger:
    """A minimal :class:`~lifemodel.log.EventLogger` that records calls per level."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict[str, Any]]] = []
        self.info_calls: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **fields: Any) -> None:
        self.debug_calls.append((event, dict(fields)))

    def info(self, event: str, **fields: Any) -> None:
        self.info_calls.append((event, dict(fields)))

    def warning(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass

    def error(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass

    def critical(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass


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


def _lm(
    tmp_path,
    state: State,
    now: datetime,
    *,
    desire: DesireState | None = DesireState.ACTIVE,
    event_ring: EventRing | None = None,
):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(now), event_ring=event_ring)
    lm.state.commit(state)
    if desire is not None:
        # the contact desire is a typed row now (lm-27n.3), not a State flag
        lm.state.put(encode_contact_desire(build_contact_desire(state=desire, salience=state.u)))
    return lm


def _supp_reasons(ring: EventRing) -> list[str]:
    """The suppression reason codes recorded on the graph's freshness ring (spec §5).

    Suppression spans route through the SpanLogger onto the durable writer + ring,
    NOT the caller's ad-hoc logger — so a test reads them back here."""
    return [r["reason"] for r in ring.read() if r.get("event") == "suppression"]


def _active(**over) -> State:
    base = dict(u=2.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00")
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
    idle = State(u=0.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, idle, NOW, desire=None)  # no live desire row
    egress = FakeEgress()
    out = proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert out is None  # T5: a quiet tick returns None, not a false 'busy' outcome
    assert egress.calls == []


def test_backstop_block_defers_and_refunds(tmp_path) -> None:
    ring = EventRing()
    lm = _lm(tmp_path, _active(proactive_send_log=CAP_LOG), NOW, event_ring=ring)
    egress = FakeEgress()
    out = proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    final = lm.state.load()
    desire = read_live_contact_desire(lm.state)
    assert out is None  # T5: a backstop-held launch returns None, not a 'busy' outcome
    assert egress.calls == []  # backstop blocked the send
    assert desire is not None and desire.state == "deferred"  # held, not sent
    assert final.energy >= 0.99  # reservation refunded
    # the hold is a logged suppression span (backstop_rate_limited), not a silent busy
    assert _supp_reasons(ring)[-1] == "backstop_rate_limited"


def test_no_launch_returns_none_and_logs_a_suppression_reason(tmp_path) -> None:
    # T5 acceptance: a quiet tick (no launch) returns None — NOT a false 'busy'
    # egress outcome — and its reason is a logged suppression span. Here the drive
    # sits below threshold, so aggregation (running in-tick under its span-bound
    # logger) emits a below_threshold suppression; proactive_tick then returns None.
    ring = EventRing()
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(NOW), event_ring=ring)
    lm.state.commit(State(u=0.0, last_tick_at="2026-07-06T11:59:00+00:00"))  # below theta
    egress = FakeEgress()
    out = proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert out is None  # quiet — no egress outcome
    assert egress.calls == []
    assert _supp_reasons(ring)[-1] == "below_threshold"


def test_failed_delivery_rolls_pending_active_and_refunds(tmp_path) -> None:
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(outcome=ReachOutcome.UNAVAILABLE), TARGET, logger=get_logger("t"))
    final = lm.state.load()
    desire = read_live_contact_desire(lm.state)
    assert final.pending_proactive_id is None  # rolled back
    assert desire is not None and desire.state == "active"  # kept to retry
    assert final.energy >= 0.99  # refunded — no turn ran


def test_egress_unavailable_emits_egress_unavailable_suppression(tmp_path) -> None:
    # A non-DELIVERED egress outcome is a first-class suppression span naming the gate.
    ring = EventRing()
    lm = _lm(tmp_path, _active(), NOW, event_ring=ring)
    proactive_tick(lm, FakeEgress(outcome=ReachOutcome.UNAVAILABLE), TARGET, logger=get_logger("t"))
    assert _supp_reasons(ring)[-1] == "egress_unavailable"


def test_egress_failed_emits_egress_failed_suppression(tmp_path) -> None:
    ring = EventRing()
    lm = _lm(tmp_path, _active(), NOW, event_ring=ring)
    proactive_tick(lm, FakeEgress(outcome=ReachOutcome.FAILED), TARGET, logger=get_logger("t"))
    assert _supp_reasons(ring)[-1] == "egress_failed"


def test_delivered_launch_keeps_the_cost(tmp_path) -> None:
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    assert lm.state.load().energy < 1.0  # the turn ran -> energy spent, not refunded


# --- lm-27n.4: the intention (decision record) rides the same rollback ---


def test_delivered_launch_leaves_intention_active_in_flight(tmp_path) -> None:
    # A delivered launch crystallized the decision record; it stays ``active`` (the
    # verdict resolves it next tick), mirroring the desire.
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    intention = read_live_contact_intention(lm.state)
    assert intention is not None and intention.state == "active"


def test_backstop_block_defers_intention_with_desire(tmp_path) -> None:
    # A backstop block holds BOTH the desire AND the intention active -> deferred,
    # atomically with the pending-clear + energy refund.
    lm = _lm(tmp_path, _active(proactive_send_log=CAP_LOG), NOW)
    egress = FakeEgress()
    proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert egress.calls == []  # backstop blocked the send
    assert read_live_contact_desire(lm.state).state == "deferred"
    assert read_live_contact_intention(lm.state).state == "deferred"  # held in lockstep
    assert final.pending_proactive_id is None  # pending cleared
    assert final.energy >= 0.99  # reservation refunded


def test_failed_delivery_keeps_intention_active_to_retry(tmp_path) -> None:
    # A delivery failure keeps BOTH rows active to retry; only pending clears + the
    # reservation refunds — no transition, no split-brain.
    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(outcome=ReachOutcome.UNAVAILABLE), TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert read_live_contact_desire(lm.state).state == "active"
    assert read_live_contact_intention(lm.state).state == "active"  # kept to retry
    assert final.pending_proactive_id is None
    assert final.energy >= 0.99


def test_does_not_stamp_egress_service_alive_at(tmp_path) -> None:
    lm = _lm(tmp_path, State(last_tick_at="2026-07-06T11:59:00+00:00"), NOW, desire=None)
    proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    # liveness is NOT a separate stamp anymore; last_tick_at (dt clock) carries it
    assert getattr(lm.state.load(), "egress_service_alive_at", None) is None
    assert lm.state.load().last_tick_at == NOW.isoformat()  # coreloop stamped the tick


# --- B3 (lm-j2w): the FULL assembled prompt, logged at DEBUG only ------------


def test_proactive_prompt_logged_at_debug_with_full_untruncated_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))

    lm = _lm(tmp_path, _active(), NOW)
    egress = FakeEgress()
    proactive_tick(lm, egress, TARGET, logger=logger)

    assert len(egress.calls) == 1
    _, delivered_impulse = egress.calls[0]  # the exact text handed to egress

    prompt_events = [c for c in base.debug_calls if c[0] == "proactive_prompt"]
    assert len(prompt_events) == 1
    _, fields = prompt_events[0]
    # Complete, untruncated — byte-identical to what was actually delivered.
    assert fields["prompt"] == delivered_impulse
    assert fields["prompt"].startswith(IMPULSE_LABEL_PREFIX)
    assert fields["correlation_id"]


def test_proactive_prompt_not_emitted_when_effective_level_is_info(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "info-events.jsonl"))

    lm = _lm(tmp_path, _active(), NOW)
    proactive_tick(lm, FakeEgress(), TARGET, logger=logger)

    assert base.debug_calls == []  # gated — DEBUG never reaches the base logger/sink

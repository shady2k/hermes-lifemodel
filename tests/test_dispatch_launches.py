"""Tests for :func:`lifemodel.core.proactive.dispatch_launches` (lm-705.6, codex #2).

Extracted from ``proactive_tick`` (Task 4 of the internal-cognition-seam plan) so
that ANY frame's ``report.launches`` gets dispatched — not just a proactive-tick's
own. The regression this exists to prove: a completion frame that is NOT itself a
``proactive_tick`` call (e.g. the internal-cognition completion frame,
``core/internal_cognition.py``) can still incidentally surface a ``LaunchProactive``
(``CoreLoop`` runs *every* enabled component regardless of trigger — CognitionLauncher
included), and that launch must still reach the egress, never get silently dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import build_lifemodel
from lifemodel.core.coreloop import TickReport
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.frame import FrameTrigger
from lifemodel.core.intents import LaunchProactive
from lifemodel.core.proactive import dispatch_launches
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import DesireState
from lifemodel.state.model import State

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}
BORN_AT = "2026-07-01T10:00:00+00:00"
_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
CAP_LOG = ["2026-07-16T11:00:00+00:00", "2026-07-16T10:00:00+00:00", "2026-07-16T09:00:00+00:00"]


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class FakeEgress:
    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


def _lm(tmp_path, state: State, now: datetime):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(now))
    lm.state.commit(state)
    lm.state.put(
        encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=state.u))
    )
    return lm


def _launch(correlation_id: str = "c-1") -> LaunchProactive:
    return LaunchProactive(
        prompt="hi",
        correlation_id=correlation_id,
        origin_traceparent=_ORIGIN_TP,
        reserved_energy=0.05,
    )


def _report(*, trigger: FrameTrigger, launches: tuple[LaunchProactive, ...]) -> TickReport:
    return TickReport(
        tick=1,
        ran=(),
        skipped_broken=(),
        failed=(),
        committed=True,
        launches=launches,
        trigger=trigger,
    )


def _pending_state(**over: object) -> State:
    base = dict(
        genesis_completed_at=BORN_AT,
        u=2.0,
        energy=0.95,
        pending_proactive_id="c-1",
        pending_proactive_since="2026-07-16T11:59:00+00:00",
        pending_proactive_origin_traceparent=_ORIGIN_TP,
        last_tick_at="2026-07-16T11:59:00+00:00",
    )
    base.update(over)
    return State(**base)  # type: ignore[arg-type]


def test_no_launches_returns_none(tmp_path) -> None:
    lm = _lm(tmp_path, _pending_state(pending_proactive_id=None), NOW)
    egress = FakeEgress()
    outcome = dispatch_launches(
        lm, _report(trigger=FrameTrigger.HEARTBEAT, launches=()), egress, TARGET
    )
    assert outcome is None
    assert egress.calls == []


def test_dispatch_launches_delivers_the_launch(tmp_path) -> None:
    lm = _lm(tmp_path, _pending_state(), NOW)
    egress = FakeEgress()
    report = _report(trigger=FrameTrigger.HEARTBEAT, launches=(_launch(),))

    outcome = dispatch_launches(lm, report, egress, TARGET)

    assert outcome is ReachOutcome.DELIVERED
    assert len(egress.calls) == 1
    # Launch != fulfilment (core/proactive.py's own contract): a delivered launch
    # LEAVES pending set until the async outcome read-back resolves it.
    assert lm.state.load().pending_proactive_id == "c-1"


def test_dispatch_launches_backstop_holds_and_rolls_back(tmp_path) -> None:
    lm = _lm(tmp_path, _pending_state(proactive_send_log=CAP_LOG), NOW)
    egress = FakeEgress()
    report = _report(trigger=FrameTrigger.HEARTBEAT, launches=(_launch(),))

    outcome = dispatch_launches(lm, report, egress, TARGET)

    assert outcome is None  # backstop held — no delivery outcome
    assert egress.calls == []
    final = lm.state.load()
    assert final.pending_proactive_id is None  # rolled back
    assert final.energy >= 0.99  # reservation refunded
    desire = read_live_contact_desire(lm.state)
    assert desire is not None and desire.state == "deferred"  # held, not sent


def test_dispatch_launches_from_a_non_proactive_completion_frame_still_dispatches(
    tmp_path,
) -> None:
    # THE regression (codex #2): the report did not come from proactive_tick at all —
    # it is what a completion frame's run_frame(..., trigger=ASYNC_COMPLETION) would
    # return when CognitionLauncher incidentally also woke this tick. dispatch_launches
    # must still deliver it — a naive completion executor that only applied the
    # completion's OWN intents and ignored report.launches would strand this launch
    # (pending_proactive_id set with nothing injected, blocking real outreach forever).
    lm = _lm(tmp_path, _pending_state(), NOW)
    egress = FakeEgress()
    report = _report(trigger=FrameTrigger.ASYNC_COMPLETION, launches=(_launch(),))

    outcome = dispatch_launches(lm, report, egress, TARGET)

    assert outcome is ReachOutcome.DELIVERED
    assert len(egress.calls) == 1
    assert egress.calls[0] == (TARGET, "hi")


def test_dispatch_launches_egress_failure_rolls_back_keeping_desire_active(tmp_path) -> None:
    lm = _lm(tmp_path, _pending_state(), NOW)
    egress = FakeEgress(outcome=ReachOutcome.UNAVAILABLE)
    report = _report(trigger=FrameTrigger.HEARTBEAT, launches=(_launch(),))

    outcome = dispatch_launches(lm, report, egress, TARGET)

    assert outcome is ReachOutcome.UNAVAILABLE
    final = lm.state.load()
    assert final.pending_proactive_id is None
    assert final.energy >= 0.99
    desire = read_live_contact_desire(lm.state)
    assert desire is not None and desire.state == "active"  # kept, to retry

"""Tests for :func:`lifemodel.egress_service.run_proactive_tick` (spec §3.2/§5/§6).

The old pressure/aggregator/cooldown decision path is gone — ``run_proactive_tick``
now decides purely via :func:`lifemodel.core.decision.decide_reachout` (already
pinned by ``tests/test_decision.py``) and only adds the delivery-launch behaviour
this module owns: on a wake, record a pending proactive id and call
``egress.reach_out``; roll the desire back (no reject) on anything short of
``DELIVERED``. The verdict itself (fulfill/reject) is applied later by the
``post_llm_call`` observer — never here.

Local fakes/helpers (no shared ``conftest.py`` exists in this repo; every test
module builds its own, matching the existing style — see e.g.
``tests/test_tick.py``'s ``_build``):

* ``make_lm(**state_kwargs)`` — a ``LifeModel`` wired from a fresh ``State``
  seeded with *state_kwargs*, a ``FakeStateStore``/``FakeSignalBus`` and a
  ``FakeClock`` pinned at ``_T0``. Neurons/aggregator are irrelevant to the new
  decision path (it never reads ``lm.bus``/``lm.aggregator``/``lm.neurons``), so
  they are wired with inert defaults.
* ``make_lm_high_u()`` — a ``LifeModel`` whose urge is already mature (``u`` far
  past ``THETA``) and past the active-silence window, with no reject on record —
  i.e. one tick away from a wake.
* ``fake_egress()`` / ``fake_egress_failing()`` — a ``ProactiveEgressPort`` fake
  recording every ``reach_out`` call, returning ``DELIVERED`` / ``FAILED``
  respectively.
* ``NULL_LOGGER`` — a no-op ``EventLogger`` so tests don't depend on structlog
  output.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.aggregator import SilentAggregator
from lifemodel.domain.egress import ReachOutcome
from lifemodel.egress_service import run_proactive_tick
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=UTC)
_TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class _NullLogger:
    def info(self, event: str, **fields: Any) -> None:
        pass


NULL_LOGGER = _NullLogger()


def make_lm(**state_kwargs: Any) -> LifeModel:
    """A ``LifeModel`` over a fresh state seeded with *state_kwargs* (no kwargs =
    the documented ``State()`` defaults — silence not yet matured)."""
    return build_lifemodel(
        base_dir=Path("/unused"),
        state=FakeStateStore(State(**state_kwargs)),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=SilentAggregator(),
        neurons=(),
    )


def make_lm_high_u() -> LifeModel:
    """``u`` well past ``THETA``, past the active-silence window, no reject —
    ``decide_reachout`` wakes on the very next call with ``busy=False``."""
    return make_lm(u=50.0, last_exchange_at=(_T0 - timedelta(minutes=20)).isoformat())


class _FakeEgress:
    def __init__(self, outcome: ReachOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[dict[str, str | None], str]] = []

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        self.calls.append((dict(target), impulse))
        return self.outcome


def fake_egress() -> _FakeEgress:
    return _FakeEgress(ReachOutcome.DELIVERED)


def fake_egress_failing() -> _FakeEgress:
    return _FakeEgress(ReachOutcome.FAILED)


def test_no_reach_out_below_threshold() -> None:
    lm = make_lm()  # fresh state — urge has not matured
    egress = fake_egress()
    run_proactive_tick(lm, egress, _TARGET, logger=NULL_LOGGER, busy=False)
    assert egress.calls == []  # urge not matured -> no reach-out
    assert lm.state.load().egress_service_alive_at is not None  # liveness always stamped


def test_wake_launches_turn_records_pending_and_does_not_apply_verdict() -> None:
    lm = make_lm_high_u()  # u high, past W, no reject
    egress = fake_egress()
    run_proactive_tick(lm, egress, _TARGET, logger=NULL_LOGGER, busy=False)
    assert len(egress.calls) == 1
    s = lm.state.load()
    assert s.desire_status == "active"
    assert s.pending_proactive_id is not None
    assert s.decline_count == 0  # verdict NOT applied here


def test_busy_gate_blocks_reach_out() -> None:
    lm = make_lm_high_u()
    egress = fake_egress()
    run_proactive_tick(lm, egress, _TARGET, logger=NULL_LOGGER, busy=True)
    assert egress.calls == []


def test_failed_launch_rolls_back_desire() -> None:
    lm = make_lm_high_u()
    egress = fake_egress_failing()
    run_proactive_tick(lm, egress, _TARGET, logger=NULL_LOGGER, busy=False)
    s = lm.state.load()
    assert s.desire_status == "none" and s.pending_proactive_id is None  # rolled back, no reject


def test_liveness_stamped_even_on_failed_launch() -> None:
    lm = make_lm_high_u()
    egress = fake_egress_failing()
    run_proactive_tick(lm, egress, _TARGET, logger=NULL_LOGGER, busy=False)
    assert lm.state.load().egress_service_alive_at is not None

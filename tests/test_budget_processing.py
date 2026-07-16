from datetime import datetime, timedelta

from lifemodel.core.budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from lifemodel.state.model import State


def _now(day="2026-07-16", hh=12, mm=0):
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00+00:00")


def test_default_ceiling_is_positive():
    assert DEFAULT_DAILY_INTERNAL_CALL_CEILING == 50
    assert timedelta(0) < DEFAULT_MIN_INTERPROCESSING_INTERVAL


def test_budget_available_below_ceiling_and_rolls_day():
    s = State(internal_calls_today=2, internal_calls_day="2026-07-16")
    assert internal_budget_available(s, now=_now(), daily_ceiling=3) is True
    at_cap = State(internal_calls_today=3, internal_calls_day="2026-07-16")
    assert internal_budget_available(at_cap, now=_now(), daily_ceiling=3) is False
    # a new day resets the count → available again
    assert internal_budget_available(at_cap, now=_now(day="2026-07-17"), daily_ceiling=3) is True


def test_interval_elapsed_when_never_run_or_past_window():
    fresh = State(last_internal_call_at=None)
    assert internal_interval_elapsed(fresh, now=_now(), min_interval=timedelta(minutes=30)) is True
    recent = State(last_internal_call_at="2026-07-16T11:45:00+00:00")
    assert (
        internal_interval_elapsed(recent, now=_now(hh=12, mm=0), min_interval=timedelta(minutes=30))
        is False
    )
    assert (
        internal_interval_elapsed(
            recent, now=_now(hh=12, mm=20), min_interval=timedelta(minutes=30)
        )
        is True
    )


def test_new_state_defaults_are_neutral():
    s = State()
    assert s.pending_internal_subject_id is None
    assert s.last_internal_call_at is None
    # additive round-trip through from_dict
    assert State.from_dict(s.to_dict()).pending_internal_subject_id is None
    assert State.from_dict(s.to_dict()).last_internal_call_at is None

"""Tests for :mod:`lifemodel.core.budget` — the FR20 durable daily call ceiling.

A pure function over :class:`~lifemodel.state.model.State`: the caller commits the
returned ``State`` (with the reservation folded in) via an ``UpdateState`` inside a
frame (see :mod:`lifemodel.adapters.internal_runner`). No I/O, no clock port — the
day is derived straight from the injected ``now``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.budget import reserve_internal_call
from lifemodel.state.model import State


def _now(day: str = "2026-07-16") -> datetime:
    return datetime.fromisoformat(f"{day}T12:00:00+00:00")


def test_reserve_increments_and_rolls_day() -> None:
    s0 = State(internal_calls_today=0, internal_calls_day="")
    s1 = reserve_internal_call(s0, now=_now(), daily_ceiling=3)
    assert s1 is not None
    assert s1.internal_calls_today == 1
    assert s1.internal_calls_day == "2026-07-16"


def test_reserve_denies_at_ceiling() -> None:
    s = State(internal_calls_today=3, internal_calls_day="2026-07-16")
    assert reserve_internal_call(s, now=_now(), daily_ceiling=3) is None


def test_new_day_resets_counter() -> None:
    s = State(internal_calls_today=3, internal_calls_day="2026-07-15")
    s2 = reserve_internal_call(s, now=_now("2026-07-16"), daily_ceiling=3)
    assert s2 is not None
    assert s2.internal_calls_today == 1
    assert s2.internal_calls_day == "2026-07-16"


def test_reserve_does_not_mutate_pending_internal_id() -> None:
    # reserve_internal_call is the FR20 quota half only; setting pending_internal_id
    # is the runner's job (a separate UpdateState in the same commit).
    s0 = State(pending_internal_id="already-something")
    s1 = reserve_internal_call(s0, now=_now(), daily_ceiling=3)
    assert s1 is not None
    assert s1.pending_internal_id == "already-something"


def test_reserve_day_key_is_the_plain_iso_date() -> None:
    # reserve_internal_call reads only .date() off `now` (a tz-aware ISO datetime,
    # per ClockPort) — this pins the day key format: plain ISO date, no time/offset.
    s0 = State()
    s1 = reserve_internal_call(s0, now=datetime(2026, 1, 5, 0, 0, 1, tzinfo=UTC), daily_ceiling=1)
    assert s1 is not None
    assert s1.internal_calls_day == "2026-01-05"

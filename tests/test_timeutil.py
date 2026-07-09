from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.timeutil import minutes_between


def test_minutes_between_counts_forward() -> None:
    a = "2026-07-06T00:00:00+00:00"
    b = datetime(2026, 7, 6, 0, 30, tzinfo=UTC)
    assert minutes_between(a, b) == 30.0


def test_minutes_between_none_is_zero() -> None:
    assert minutes_between(None, datetime(2026, 7, 6, tzinfo=UTC)) == 0.0


def test_minutes_between_unparseable_is_zero() -> None:
    assert minutes_between("not-a-date", datetime(2026, 7, 6, tzinfo=UTC)) == 0.0


def test_minutes_between_naive_is_zero() -> None:
    assert minutes_between("2026-07-06T00:00:00", datetime(2026, 7, 6, 1, 0, tzinfo=UTC)) == 0.0

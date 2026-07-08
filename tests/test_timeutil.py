from __future__ import annotations

import re
from datetime import UTC, datetime

from lifemodel.core.timeutil import humanize_elapsed, minutes_between


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


def test_humanize_elapsed_never_talked() -> None:
    assert humanize_elapsed(None) == "вы ещё толком не общались"


def test_humanize_elapsed_bands() -> None:
    assert humanize_elapsed(0.0) == "совсем недавно"
    assert humanize_elapsed(59.0) == "совсем недавно"
    assert humanize_elapsed(60.0) == "пару часов назад"
    assert humanize_elapsed(179.0) == "пару часов назад"
    assert humanize_elapsed(180.0) == "несколько часов назад"
    assert humanize_elapsed(479.0) == "несколько часов назад"
    assert humanize_elapsed(480.0) == "сегодня, но уже порядочно прошло"
    assert humanize_elapsed(1439.0) == "сегодня, но уже порядочно прошло"
    assert humanize_elapsed(1440.0) == "со вчерашнего дня"
    assert humanize_elapsed(2880.0) == "уже несколько дней"
    assert humanize_elapsed(5760.0) == "около недели"
    assert humanize_elapsed(11520.0) == "не одну неделю"
    assert humanize_elapsed(43200.0) == "очень давно"


def test_humanize_elapsed_has_no_digits() -> None:
    for m in (None, 0.0, 60.0, 500.0, 1440.0, 3000.0, 50000.0):
        assert re.search(r"\d", humanize_elapsed(m)) is None


def test_humanize_elapsed_negative_is_recent() -> None:
    # a clock-skew negative elapsed is treated as "just now", never crashes
    assert humanize_elapsed(-5.0) == "совсем недавно"

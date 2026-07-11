from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from lifemodel.core.timeutil import minutes_between, to_iso

#: The load-bearing normalization invariant (spec §3/§6.2): fixed-width, always
#: 6-digit microseconds, always ``+00:00`` — the shape that makes TEXT order ==
#: chronological order.
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")


@pytest.mark.parametrize("microsecond", [0, 1, 500000, 999999])
def test_to_iso_is_fixed_width_normalized(microsecond: int) -> None:
    dt = datetime(2026, 7, 11, 8, 5, 3, microsecond, tzinfo=UTC)
    assert _ISO_UTC_RE.match(to_iso(dt))


@pytest.mark.parametrize("dt", [datetime.min, datetime.max])
def test_to_iso_min_max_are_well_formed_fixed_width(dt: datetime) -> None:
    assert _ISO_UTC_RE.match(to_iso(dt.replace(tzinfo=UTC)))


def test_to_iso_text_sort_matches_datetime_sort() -> None:
    # A deliberately tricky, out-of-order mix: whole-second, .5 -> 500000,
    # .000001, adjacent second/minute/hour rollovers, and two instants that
    # differ ONLY in microseconds. This is the load-bearing invariant: lexical
    # order of the serialized text must equal chronological order.
    datetimes = [
        datetime(2026, 7, 11, 8, 5, 3, 500000, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 5, 3, 0, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 5, 3, 1, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 5, 4, 0, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 6, 0, 0, tzinfo=UTC),
        datetime(2026, 7, 11, 9, 0, 0, 0, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 5, 3, 999999, tzinfo=UTC),
        datetime(2026, 7, 11, 8, 5, 3, 999998, tzinfo=UTC),
    ]
    iso_list = [to_iso(dt) for dt in datetimes]
    assert sorted(iso_list) == [to_iso(dt) for dt in sorted(datetimes)]


def test_to_iso_rejects_tz_naive() -> None:
    with pytest.raises(ValueError):
        to_iso(datetime(2026, 7, 11, 8, 5, 3))


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

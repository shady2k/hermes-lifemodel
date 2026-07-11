from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from lifemodel.core.timeutil import (
    from_iso,
    minutes_between,
    to_epoch_seconds,
    to_iso,
)

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


def test_from_iso_parses_to_aware_utc() -> None:
    parsed = from_iso("2026-07-11T08:05:03.500000+00:00")
    assert parsed == datetime(2026, 7, 11, 8, 5, 3, 500000, tzinfo=UTC)
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def test_from_iso_converts_offset_to_utc() -> None:
    # An +03:00 wall time is the same instant as 05:05:03 UTC.
    assert from_iso("2026-07-11T08:05:03.500000+03:00") == datetime(
        2026, 7, 11, 5, 5, 3, 500000, tzinfo=UTC
    )


def test_from_iso_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        from_iso("not-a-timestamp")


def test_from_iso_raises_on_naive_string() -> None:
    with pytest.raises(ValueError):
        from_iso("2026-07-11T08:05:03.500000")


def test_round_trip_from_iso_to_iso_preserves_utc_instant() -> None:
    dt = datetime(2026, 7, 11, 8, 5, 3, 500000, tzinfo=UTC)
    assert from_iso(to_iso(dt)) == dt


def test_round_trip_from_iso_to_iso_preserves_offset_instant() -> None:
    dt = datetime(2026, 7, 11, 8, 5, 3, 500000, tzinfo=timezone(timedelta(hours=3)))
    assert from_iso(to_iso(dt)) == dt  # equal as instants (both aware)


def test_round_trip_to_iso_from_iso_is_identity_for_normalized() -> None:
    s = "2026-07-11T08:05:03.500000+00:00"
    assert to_iso(from_iso(s)) == s


def test_to_epoch_seconds_rejects_tz_naive() -> None:
    with pytest.raises(ValueError):
        to_epoch_seconds(datetime(2026, 7, 11, 8, 5, 3))


def test_to_epoch_seconds_round_trips_within_a_microsecond() -> None:
    s = "2026-07-11T08:05:03.500000+00:00"
    epoch = to_epoch_seconds(from_iso(s))
    reconstructed = datetime.fromtimestamp(epoch, UTC)
    assert abs((reconstructed - from_iso(s)).total_seconds()) < 1e-6


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

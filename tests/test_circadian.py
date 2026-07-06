from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.circadian import circadian

PEAK = 13.0  # 16:00 MSK peak, 04:00 MSK trough


def test_peak_at_peak_hour() -> None:
    c = circadian(datetime(2026, 7, 6, 13, 0, tzinfo=UTC), peak_hour_utc=PEAK)
    assert abs(c - 1.0) < 1e-9


def test_trough_twelve_hours_later() -> None:
    c = circadian(
        datetime(2026, 7, 6, 1, 0, tzinfo=UTC), peak_hour_utc=PEAK
    )  # 01:00 UTC = 04:00 MSK
    assert abs(c - 0.0) < 1e-9


def test_midpoint_at_quarter_phase() -> None:
    c = circadian(datetime(2026, 7, 6, 19, 0, tzinfo=UTC), peak_hour_utc=PEAK)  # +6h from peak
    assert abs(c - 0.5) < 1e-9


def test_always_in_unit_interval() -> None:
    for hour in range(24):
        c = circadian(datetime(2026, 7, 6, hour, 0, tzinfo=UTC), peak_hour_utc=PEAK)
        assert 0.0 <= c <= 1.0

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.backstop import allow_send, record_send
from lifemodel.core.timeutil import to_iso

NOW = datetime(2026, 7, 6, 20, 0, tzinfo=UTC)


def _ago(minutes: float) -> str:
    return to_iso(NOW - timedelta(minutes=minutes))


def test_allows_when_log_empty() -> None:
    assert allow_send([], NOW) is True


def test_denies_when_daily_cap_reached() -> None:
    log = [_ago(600), _ago(400), _ago(200)]  # 3 sends within 24h -> cap (default 3)
    assert allow_send(log, NOW) is False


def test_denies_within_min_interval() -> None:
    log = [_ago(30)]  # last send 30 min ago < 60 min
    assert allow_send(log, NOW) is False


def test_allows_after_min_interval_and_under_cap() -> None:
    log = [_ago(90)]  # 90 min ago, only 1 today
    assert allow_send(log, NOW) is True


def test_old_sends_outside_24h_do_not_count() -> None:
    log = [_ago(60 * 25), _ago(60 * 26), _ago(60 * 27)]  # all >24h ago
    assert allow_send(log, NOW) is True


def test_malformed_entries_are_ignored_not_crashing() -> None:
    assert allow_send(["not-a-date", _ago(90)], NOW) is True  # bad entry skipped, 1 valid @90m
    assert allow_send(["garbage"], NOW) is True  # no valid recent send


def test_record_send_appends_and_bounds() -> None:
    log = [_ago(1000 + i) for i in range(25)]
    new = record_send(log, NOW, keep=20)
    assert len(new) == 20
    assert new[-1] == to_iso(NOW)  # newest last

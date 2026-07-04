from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lifemodel.domain.wake import WakePacket
from lifemodel.impulse import IMPULSE_LABEL_PREFIX, compose_impulse

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)


def _packet() -> WakePacket:
    return WakePacket(reason="silence", pressure_kind="idle", pressure=28.0, threshold=10.0)


def test_impulse_is_labeled_internal_and_not_user_authored() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=_T0 - timedelta(hours=5))
    assert text.startswith(IMPULSE_LABEL_PREFIX)
    lowered = text.lower()
    assert "not from the user" in lowered or "не от пользователя" in lowered
    # never starts with a slash (would enter Hermes command routing — spec §5 guard f)
    assert not text.lstrip().startswith("/")


def test_impulse_reports_whole_hours_of_silence() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=_T0 - timedelta(hours=5, minutes=40))
    assert "5" in text  # floor(5h40m) == 5 hours


def test_impulse_handles_unknown_last_contact() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=None)
    assert text.startswith(IMPULSE_LABEL_PREFIX)
    assert "/" != text.strip()[0]

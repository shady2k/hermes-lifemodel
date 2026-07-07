from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.invalidation import is_verdict_stale

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
PENDING = "proactive-2026-07-06T11:55:00+00:00"


def _call(**over: object) -> tuple[bool, str]:
    kw = dict(
        desire_state="active",
        pending_id=PENDING,
        verdict_correlation_id=PENDING,
        last_exchange_at=None,
        pending_since="2026-07-06T11:55:00+00:00",
        effective=2.0,
        threshold=1.0,
        now=NOW,
        deadline_min=30.0,
    )
    kw.update(over)
    return is_verdict_stale(**kw)  # type: ignore[arg-type]


def test_fresh_verdict_is_applied() -> None:
    stale, reason = _call()
    assert stale is False and reason == "fresh"


def test_resolved_desire_is_stale() -> None:
    assert _call(desire_state="none")[0] is True


def test_correlation_mismatch_is_stale() -> None:
    stale, reason = _call(verdict_correlation_id="proactive-OTHER")
    assert stale is True and reason == "stale_desire_id"


def test_user_reply_after_launch_is_stale() -> None:
    # exchange at 11:58 is after pending_since 11:55 -> reactive path already answered
    stale, reason = _call(last_exchange_at="2026-07-06T11:58:00+00:00")
    assert stale is True and reason == "user_replied"


def test_exchange_before_launch_is_not_stale() -> None:
    # exchange at 11:50 predates the launch -> not a during-think reply
    assert _call(last_exchange_at="2026-07-06T11:50:00+00:00")[0] is False


def test_pressure_satisfied_is_stale() -> None:
    stale, reason = _call(effective=0.5, threshold=1.0)
    assert stale is True and reason == "pressure_satisfied"


def test_deadline_elapsed_is_stale() -> None:
    # pending since 11:00, now 12:00 -> 60 min > 30 min deadline
    stale, reason = _call(pending_since="2026-07-06T11:00:00+00:00", deadline_min=30.0)
    assert stale is True and reason == "deadline"

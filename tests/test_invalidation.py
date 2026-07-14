from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.invalidation import is_proactive_outcome_stale

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
PENDING = "proactive-2026-07-06T11:55:00+00:00"


def _call(**over: object) -> tuple[bool, str]:
    kw = dict(
        desire_state="active",
        pending_id=PENDING,
        outcome_correlation_id=PENDING,
        last_exchange_at=None,
        pending_since="2026-07-06T11:55:00+00:00",
        effective=2.0,
        threshold=1.0,
        now=NOW,
        deadline_min=30.0,
    )
    kw.update(over)
    return is_proactive_outcome_stale(**kw)  # type: ignore[arg-type]


def test_fresh_verdict_is_applied() -> None:
    stale, reason = _call()
    assert stale is False and reason == "fresh"


def test_resolved_desire_is_stale() -> None:
    assert _call(desire_state="none")[0] is True


def test_correlation_mismatch_is_stale() -> None:
    stale, reason = _call(outcome_correlation_id="proactive-OTHER")
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


def test_a_wake_that_never_came_from_pressure_is_not_stale_for_lacking_it() -> None:
    # A genesis first-waking (spec §6.2) is not sprung by the drive: a newborn's u is 0
    # < θ BY CONSTRUCTION. Judging its outcome by "the pressure was satisfied while I was
    # composing" would discard EVERY genesis outcome — not a lost message but a DEADLOCK:
    # the desire stays active, pending_proactive_id never clears, and the launcher holds
    # every future launch for the rest of the being's life.
    stale, reason = _call(effective=0.0, threshold=1.0, pressure_sprung=False)
    assert stale is False and reason == "fresh"


def test_a_pressure_free_wake_is_still_stale_for_every_other_reason() -> None:
    # The waiver is narrow: a resolved desire, a mismatched correlation, a user who
    # replied while it composed, and the deadline are just as true of a birth.
    assert _call(pressure_sprung=False, desire_state="none")[0] is True
    assert _call(pressure_sprung=False, outcome_correlation_id="other")[0] is True
    assert _call(pressure_sprung=False, last_exchange_at="2026-07-06T11:58:00+00:00")[0] is True
    assert _call(pressure_sprung=False, pending_since="2026-07-06T11:00:00+00:00")[0] is True

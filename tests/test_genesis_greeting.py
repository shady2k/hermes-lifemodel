from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.genesis import should_greet, stamp_greeted
from lifemodel.core.timeutil import to_iso
from lifemodel.domain.egress import ReachOutcome
from lifemodel.state.model import State

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
UNBORN = State()
GREETED = State(genesis_greeted_at="2026-07-13T09:00:00+00:00")
BORN = State(genesis_completed_at="2026-07-13T10:00:00+00:00")


def test_an_unborn_being_greets_without_waiting_for_the_drive() -> None:
    # u = 0 at birth, and it must NOT wait to cross theta. The drive models a contact
    # deficit inside an EXISTING relationship and a newborn has none: there is nobody to
    # miss. Birth is not longing.
    assert UNBORN.u == 0.0
    assert should_greet(UNBORN) is True


def test_a_being_greets_once_across_a_restart_storm() -> None:
    # connect() runs on EVERY gateway restart and the SupervisedLoop reconnects after a
    # loop death. Without the stamp, every `make deploy` re-introduces the being.
    assert should_greet(GREETED) is False


def test_a_born_being_never_greets_again() -> None:
    assert should_greet(BORN) is False


def test_an_undelivered_greeting_is_NOT_stamped_and_is_retried() -> None:
    # Stamping on the ATTEMPT would silence forever the being of a human who installed
    # the plugin before configuring a channel — they would never be greeted at all.
    # (Same lesson as lm-2gi: count on confirmed delivery, never on a verdict.)
    assert stamp_greeted(UNBORN, outcome=ReachOutcome.UNAVAILABLE, now=NOW) is None
    assert stamp_greeted(UNBORN, outcome=ReachOutcome.FAILED, now=NOW) is None
    assert should_greet(UNBORN) is True  # so the next connect tries again


def test_a_delivered_greeting_is_stamped_once() -> None:
    after = stamp_greeted(UNBORN, outcome=ReachOutcome.DELIVERED, now=NOW)
    assert after is not None
    # Canonical serializer (core.timeutil.to_iso), matching genesis_completed_at's own
    # stamping convention (hooks.py) and every other *_at field in the suite — fixed-width
    # 6-digit microseconds, not a hand-rolled literal, so TEXT order == chronological order.
    assert after.genesis_greeted_at == to_iso(NOW)
    assert should_greet(after) is False


def test_greeting_does_not_touch_the_contact_model() -> None:
    # Genesis must NOT fake a desire or a pending-proactive id to borrow the proactive
    # machinery — that would pollute the contact model with a longing that does not exist.
    after = stamp_greeted(UNBORN, outcome=ReachOutcome.DELIVERED, now=NOW)
    assert after is not None
    assert after.u == 0.0
    assert after.pending_proactive_id is None
    assert after.unanswered_outbound_count == 0
    assert after.genesis_completed_at is None  # greeting is not birth

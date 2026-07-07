"""The contact-intention view — reader predicate + constructor (lm-27n.4).

Mirrors the desire view: the singleton ``kind='intention'`` record ``contact:owner``
read back as a typed :class:`Intention`, with the live/terminal predicate every
site shares. A live intention is ``pending``/``active``/``deferred``; a terminal one
(``completed``/``dropped``/``expired``) reads as absence.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.intention_view import (
    build_contact_intention,
    encode_contact_intention,
    live_contact_intention,
    read_live_contact_intention,
)
from lifemodel.domain.objects import CONTACT_INTENTION_ID, IntentionState
from lifemodel.testing import contact_intention_objects, contact_intention_record
from lifemodel.testing.fakes import FakeClock, FakeMemoryStore

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def test_active_intention_is_live_in_snapshot() -> None:
    intention = live_contact_intention(contact_intention_objects("active"))
    assert intention is not None
    assert intention.id == CONTACT_INTENTION_ID
    assert intention.state == "active"


def test_deferred_intention_is_live_in_snapshot() -> None:
    # active AND deferred are both live (a backstop-held decision is not gone).
    intention = live_contact_intention(contact_intention_objects("deferred"))
    assert intention is not None and intention.state == "deferred"


def test_completed_intention_is_absence() -> None:
    assert live_contact_intention(contact_intention_objects("completed")) is None


def test_dropped_intention_is_absence() -> None:
    assert live_contact_intention(contact_intention_objects("dropped")) is None


def test_empty_snapshot_has_no_intention() -> None:
    assert live_contact_intention(()) is None


def test_build_carries_rubicon_fields_and_round_trips() -> None:
    intention = build_contact_intention(
        state=IntentionState.ACTIVE, commitment_strength=1.75, salience=1.75, source_drive=1.5
    )
    assert intention.commitment_strength == 1.75
    assert intention.goal
    assert intention.reconsideration_triggers  # recorded for auditability
    # encode goes through the registry (the single write door) and round-trips.
    draft = encode_contact_intention(intention)
    assert draft.kind == "intention"
    assert draft.state == "active"
    assert draft.payload["commitment_strength"] == 1.75


def test_read_live_from_memory_port_point_in_time() -> None:
    store = FakeMemoryStore(clock=FakeClock(NOW))
    assert read_live_contact_intention(store) is None  # nothing yet
    store.put(encode_contact_intention(build_contact_intention(state=IntentionState.ACTIVE)))
    live = read_live_contact_intention(store)
    assert live is not None and live.state == "active"
    # a terminal row reads as absence
    store.transition("intention", CONTACT_INTENTION_ID, "active", "completed")
    assert read_live_contact_intention(store) is None


def test_other_kind_record_is_ignored() -> None:
    # a desire record in the snapshot must not decode as an intention
    from lifemodel.testing import contact_desire_record

    assert live_contact_intention((contact_desire_record("active"),)) is None
    # sanity: a real intention record still reads
    assert live_contact_intention((contact_intention_record("active"),)) is not None

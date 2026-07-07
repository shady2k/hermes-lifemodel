# tests/test_relationship_view.py
#
# The owner-relationship view (lm-27n.5): the single registry door onto the
# singleton kind='relationship' row "owner", mirroring the desire/intention views.
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.relationship_view import (
    DEFAULT_RELATIONSHIP,
    EXPLICIT_CONFIDENCE,
    RELATIONSHIP_KIND,
    build_owner_relationship,
    encode_owner_relationship,
    live_owner_relationship,
    read_owner_relationship,
)
from lifemodel.domain.objects import OWNER_RELATIONSHIP_ID, RelationshipState
from lifemodel.testing import (
    FakeClock,
    FakeMemoryStore,
    contact_desire_record,
    owner_relationship_objects,
    owner_relationship_record,
)

_CLOCK = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))


def test_owner_relationship_id_and_kind() -> None:
    assert OWNER_RELATIONSHIP_ID == "owner"
    rel = build_owner_relationship()
    assert rel.id == OWNER_RELATIONSHIP_ID
    assert rel.KIND == RELATIONSHIP_KIND
    assert rel.state == RelationshipState.ACTIVE.value


def test_default_relationship_is_permissive_and_low_confidence() -> None:
    d = DEFAULT_RELATIONSHIP
    assert d.bad_hours == ()
    assert d.privacy_boundaries == ()
    assert d.topic_sensitivity == ()
    assert d.acceptable_styles == ()
    assert d.cadence == ""
    assert (d.confidence or 0.0) < EXPLICIT_CONFIDENCE


def test_live_owner_relationship_reads_the_snapshot() -> None:
    objects = owner_relationship_objects(bad_hours=(2, 3), confidence=EXPLICIT_CONFIDENCE)
    rel = live_owner_relationship(objects)
    assert rel is not None
    assert rel.bad_hours == (2, 3)
    assert rel.confidence == EXPLICIT_CONFIDENCE


def test_live_owner_relationship_ignores_non_owner_records() -> None:
    # A desire row in the snapshot is not the owner relationship.
    assert live_owner_relationship((contact_desire_record("active"),)) is None
    assert live_owner_relationship(()) is None


def test_archived_relationship_reads_as_absence() -> None:
    assert live_owner_relationship(owner_relationship_objects("archived")) is None


def test_encode_round_trips_through_the_store() -> None:
    store = FakeMemoryStore(clock=_CLOCK)
    draft = encode_owner_relationship(
        build_owner_relationship(bad_hours=(1,), cadence="2h", confidence=EXPLICIT_CONFIDENCE)
    )
    store.put(draft)
    rel = read_owner_relationship(store)
    assert rel is not None
    assert rel.bad_hours == (1,)
    assert rel.cadence == "2h"


def test_read_owner_relationship_absent_is_none() -> None:
    assert read_owner_relationship(FakeMemoryStore(clock=_CLOCK)) is None


def test_builder_record_has_owner_id() -> None:
    rec = owner_relationship_record(confidence=EXPLICIT_CONFIDENCE)
    assert rec.kind == RELATIONSHIP_KIND
    assert rec.id == OWNER_RELATIONSHIP_ID

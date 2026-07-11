# tests/test_user_model_view.py
#
# The owner user-model view (spec §8): the single registry door onto the
# singleton kind='user_model' row "owner", mirroring the desire/intention views.
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.user_model_view import (
    DEFAULT_USER_MODEL,
    EXPLICIT_CONFIDENCE,
    USER_MODEL_KIND,
    build_owner_user_model,
    encode_owner_user_model,
    live_owner_user_model,
    read_owner_user_model,
)
from lifemodel.domain.objects import OWNER_USER_MODEL_ID, UserModelState
from lifemodel.testing import (
    FakeClock,
    FakeMemoryStore,
    contact_desire_record,
    owner_user_model_objects,
    owner_user_model_record,
)

_CLOCK = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))


def test_owner_user_model_id_and_kind() -> None:
    assert OWNER_USER_MODEL_ID == "owner"
    um = build_owner_user_model()
    assert um.id == OWNER_USER_MODEL_ID
    assert um.KIND == USER_MODEL_KIND
    assert um.state == UserModelState.ACTIVE.value


def test_default_user_model_is_permissive_and_low_confidence() -> None:
    d = DEFAULT_USER_MODEL
    assert d.bad_hours.value == ()
    assert d.privacy_boundaries.value == ()
    assert d.topic_sensitivity.value == ()
    assert d.acceptable_styles.value == ()
    assert d.cadence.value == ""
    assert (d.confidence or 0.0) < EXPLICIT_CONFIDENCE


def test_live_owner_user_model_reads_the_snapshot() -> None:
    objects = owner_user_model_objects(bad_hours=(2, 3), confidence=EXPLICIT_CONFIDENCE)
    um = live_owner_user_model(objects)
    assert um is not None
    assert um.bad_hours.value == (2, 3)
    assert um.confidence == EXPLICIT_CONFIDENCE


def test_live_owner_user_model_ignores_non_owner_records() -> None:
    # A desire row in the snapshot is not the owner user-model.
    assert live_owner_user_model((contact_desire_record("active"),)) is None
    assert live_owner_user_model(()) is None


def test_archived_user_model_reads_as_absence() -> None:
    assert live_owner_user_model(owner_user_model_objects("archived")) is None


def test_encode_round_trips_through_the_store() -> None:
    store = FakeMemoryStore(clock=_CLOCK)
    draft = encode_owner_user_model(
        build_owner_user_model(bad_hours=(1,), cadence="2h", confidence=EXPLICIT_CONFIDENCE)
    )
    store.put(draft)
    um = read_owner_user_model(store)
    assert um is not None
    assert um.bad_hours.value == (1,)
    assert um.cadence.value == "2h"


def test_read_owner_user_model_absent_is_none() -> None:
    assert read_owner_user_model(FakeMemoryStore(clock=_CLOCK)) is None


def test_builder_record_has_owner_id() -> None:
    rec = owner_user_model_record(confidence=EXPLICIT_CONFIDENCE)
    assert rec.kind == USER_MODEL_KIND
    assert rec.id == OWNER_USER_MODEL_ID

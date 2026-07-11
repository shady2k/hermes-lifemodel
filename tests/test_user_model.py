# tests/test_user_model.py
#
# The UserModel is our DERIVED cache of the Other (spec §8): each field carries
# per-field inference metadata {value, inferred_at, ttl}. A field whose
# inferred_at + ttl has passed is STALE and reads as UNKNOWN ("стухло →
# неизвестно"), never as the old value. Owner-SET prefs are authoritative (no
# ttl) and never go stale.
from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from lifemodel.core.user_model_view import (
    build_owner_user_model,
    encode_owner_user_model,
    read_owner_user_model,
)
from lifemodel.domain.objects import UserModel
from lifemodel.domain.objects.inference import UNKNOWN, InferredField
from lifemodel.testing import FakeClock, FakeMemoryStore

_INFERRED_AT = "2026-07-11T12:00:00+00:00"
_BASE = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
_CLOCK = FakeClock(_BASE)


def test_field_is_a_value_inferred_at_ttl_bundle() -> None:
    # Setting a field records inferred_at (and ttl); reading it back yields the
    # {value, inferred_at, ttl} bundle.
    um = build_owner_user_model(cadence="daily", inferred_at=_INFERRED_AT, ttl_seconds=3600.0)
    assert isinstance(um.cadence, InferredField)
    assert um.cadence.value == "daily"
    assert um.cadence.inferred_at == _INFERRED_AT
    assert um.cadence.ttl_seconds == 3600.0


def test_fresh_field_returns_value_stale_field_returns_unknown() -> None:
    um = build_owner_user_model(cadence="daily", inferred_at=_INFERRED_AT, ttl_seconds=3600.0)
    fresh = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)  # within the hour ttl
    stale = datetime(2026, 7, 11, 14, 0, tzinfo=UTC)  # past inferred_at + ttl
    assert um.cadence.resolve(fresh) == "daily"
    assert um.cadence.resolve(stale) is UNKNOWN
    assert um.cadence.resolve(stale) != "daily"


def test_per_field_freshness_is_independent() -> None:
    # At one `now`, a long-ttl field is still fresh while a short-ttl field is
    # already stale — each field carries its OWN inference metadata.
    base = build_owner_user_model(cadence="daily", inferred_at=_INFERRED_AT, ttl_seconds=86400.0)
    um = dataclasses.replace(base, bad_hours=InferredField((2, 3), _INFERRED_AT, 60.0))
    now = datetime(2026, 7, 11, 12, 30, tzinfo=UTC)  # 30 min later
    assert um.cadence.resolve(now) == "daily"  # 1-day ttl -> fresh
    assert um.bad_hours.resolve(now) is UNKNOWN  # 60-second ttl -> stale


def test_round_trips_through_memory_port_preserving_metadata() -> None:
    store = FakeMemoryStore(clock=_CLOCK)
    store.put(
        encode_owner_user_model(
            build_owner_user_model(
                cadence="2h", bad_hours=(1,), inferred_at=_INFERRED_AT, ttl_seconds=7200.0
            )
        )
    )
    um = read_owner_user_model(store)
    assert um is not None
    assert isinstance(um, UserModel)
    assert um.cadence.value == "2h"
    assert um.cadence.inferred_at == _INFERRED_AT
    assert um.cadence.ttl_seconds == 7200.0
    assert um.bad_hours.value == (1,)


def test_owner_set_prefs_are_authoritative_and_never_stale() -> None:
    # No inference stamp -> ttl None -> the owner's boundary never silently expires.
    um = build_owner_user_model(bad_hours=(2, 3))
    far_future = datetime(3000, 1, 1, tzinfo=UTC)
    assert um.bad_hours.is_stale(far_future) is False
    assert um.bad_hours.resolve(far_future) == (2, 3)

# tests/test_inference.py
#
# InferredField (spec §8): per-field inference metadata for the derived UserModel.
# A field carries {value, inferred_at, ttl}; once ``inferred_at + ttl`` is in the
# past it is STALE and must read as UNKNOWN ("стухло → неизвестно"), never as the
# stale value. ``ttl_seconds=None`` means "never expires" (an authoritative /
# owner-set value, not a time-boxed inference).

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.domain.objects.inference import UNKNOWN, InferredField

_INFERRED_AT = "2026-07-11T12:00:00+00:00"
_BASE = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)


def test_fresh_field_resolves_to_its_value() -> None:
    field = InferredField(value="daily", inferred_at=_INFERRED_AT, ttl_seconds=3600.0)
    now = _BASE + timedelta(minutes=30)  # within the hour ttl
    assert field.is_stale(now) is False
    assert field.resolve(now) == "daily"


def test_stale_field_resolves_to_unknown_not_the_old_value() -> None:
    field = InferredField(value="daily", inferred_at=_INFERRED_AT, ttl_seconds=3600.0)
    now = _BASE + timedelta(hours=2)  # past inferred_at + ttl
    assert field.is_stale(now) is True
    assert field.resolve(now) is UNKNOWN
    assert field.resolve(now) != "daily"


def test_resolve_or_returns_default_when_stale() -> None:
    field = InferredField(value=(2, 3), inferred_at=_INFERRED_AT, ttl_seconds=60.0)
    fresh = _BASE + timedelta(seconds=30)
    stale = _BASE + timedelta(seconds=120)
    assert field.resolve_or(fresh, default=()) == (2, 3)
    assert field.resolve_or(stale, default=()) == ()


def test_ttl_none_never_goes_stale() -> None:
    # An authoritative / owner-set value (no ttl) must never silently vanish.
    field: InferredField[str] = InferredField(value="explicit", inferred_at=_INFERRED_AT)
    far_future = _BASE + timedelta(days=3650)
    assert field.is_stale(far_future) is False
    assert field.resolve(far_future) == "explicit"


def test_no_inferred_at_never_goes_stale() -> None:
    # A plain value with no inference stamp is timeless (the permissive default).
    field: InferredField[tuple[int, ...]] = InferredField(value=())
    assert field.is_stale(_BASE) is False
    assert field.resolve(_BASE) == ()


def test_the_bundle_is_readable_back() -> None:
    field = InferredField(value="hourly", inferred_at=_INFERRED_AT, ttl_seconds=120.0)
    assert field.value == "hourly"
    assert field.inferred_at == _INFERRED_AT
    assert field.ttl_seconds == 120.0

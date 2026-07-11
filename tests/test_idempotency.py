"""Unit tests for the external-event idempotency ring (spec §8 / lm-fib.8.5).

``dedupe_external_events`` is a pure function: given the durable ring, a frame's
seed signals, and ``now``, it returns the signals to actually process (duplicate
external events dropped) plus the updated ring (fresh ids recorded, oldest /
TTL-expired evicted to stay bounded). No I/O, no state — the coreloop persists
the returned ring via the state-actor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.idempotency import (
    DEFAULT_RING_CAP,
    DEFAULT_RING_TTL,
    dedupe_external_events,
)
from lifemodel.core.taxonomy import contact_observed_signal, proactive_outcome_signal
from lifemodel.domain.egress import ProactiveOutcome

_NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _contact(origin_id: str):
    return contact_observed_signal(
        origin_id=origin_id, actor="user", label="two_way", timestamp=None
    )


def test_new_external_id_passes_through_and_is_recorded() -> None:
    kept, ring = dedupe_external_events({}, [_contact("m-1")], _NOW)
    assert [s.origin_id for s in kept] == ["m-1"]  # the fresh event is processed
    assert set(ring) == {"m-1"}  # ...and remembered for next time
    assert ring["m-1"] == _NOW.isoformat()  # stamped with when we first saw it


def test_duplicate_external_id_is_dropped_and_ring_unchanged() -> None:
    seen = {"m-1": _NOW.isoformat()}
    kept, ring = dedupe_external_events(seen, [_contact("m-1")], _NOW)
    assert kept == ()  # the retry never reaches the drive/aggregation
    assert ring == seen  # nothing new recorded — same stamp preserved


def test_different_external_id_after_first_still_passes() -> None:
    seen = {"m-1": _NOW.isoformat()}
    kept, ring = dedupe_external_events(seen, [_contact("m-2")], _NOW)
    assert [s.origin_id for s in kept] == ["m-2"]  # dedup is by id, not blanket-suppress
    assert set(ring) == {"m-1", "m-2"}


def test_ttl_expired_id_can_fire_again() -> None:
    ttl = timedelta(hours=1)
    old = (_NOW - timedelta(hours=2)).isoformat()  # older than the TTL
    kept, ring = dedupe_external_events({"m-1": old}, [_contact("m-1")], _NOW, ttl=ttl)
    assert [s.origin_id for s in kept] == ["m-1"]  # expired → treated as new, processed
    assert ring["m-1"] == _NOW.isoformat()  # re-recorded with a fresh stamp


def test_ttl_expired_entries_are_evicted_even_with_no_new_events() -> None:
    ttl = timedelta(hours=1)
    old = (_NOW - timedelta(hours=2)).isoformat()
    fresh = (_NOW - timedelta(minutes=5)).isoformat()
    kept, ring = dedupe_external_events({"stale": old, "recent": fresh}, [], _NOW, ttl=ttl)
    assert kept == ()
    assert set(ring) == {"recent"}  # the stale entry is swept, the recent one stays


def test_ring_stays_bounded_evicting_oldest_first() -> None:
    # A cap of 2: after recording m-3 the oldest (m-1) is evicted, so the ring is
    # bounded and m-1 could fire again later.
    seen = {"m-1": _NOW.isoformat(), "m-2": _NOW.isoformat()}
    kept, ring = dedupe_external_events(seen, [_contact("m-3")], _NOW, cap=2)
    assert [s.origin_id for s in kept] == ["m-3"]
    assert list(ring) == ["m-2", "m-3"]  # oldest (m-1) evicted, insertion order kept


def test_non_contact_signals_pass_through_untouched() -> None:
    # A proactive_outcome is not an external inbound — it is never deduped or recorded.
    outcome = proactive_outcome_signal(
        origin_id="o-1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id="p-1"
    )
    kept, ring = dedupe_external_events({}, [outcome], _NOW)
    assert kept == (outcome,)
    assert ring == {}  # nothing recorded for a non-external signal


def test_mixed_batch_drops_only_the_duplicate_contact() -> None:
    seen = {"m-1": _NOW.isoformat()}
    outcome = proactive_outcome_signal(
        origin_id="o-1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id="p-1"
    )
    kept, ring = dedupe_external_events(seen, [_contact("m-1"), outcome], _NOW)
    assert kept == (outcome,)  # the duplicate contact dropped; the outcome survives
    assert ring == seen


def test_defaults_are_sane_bounds() -> None:
    assert DEFAULT_RING_CAP > 0
    assert DEFAULT_RING_TTL.total_seconds() > 0

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
    filter_external_events,
    record_external_events,
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


# --- the split (lm-fib): filter is pure (never records); record is separate ---


def test_filter_reports_fresh_ids_without_touching_the_ring() -> None:
    # The pure filter drops duplicates and REPORTS the fresh ids to record later —
    # but it never mutates the ring itself (so a downstream fault can leave it clean).
    ring = {"m-1": _NOW.isoformat()}
    kept, fresh = filter_external_events(ring, [_contact("m-1"), _contact("m-2")], _NOW)
    assert [s.origin_id for s in kept] == ["m-2"]  # the duplicate m-1 dropped
    assert fresh == ("m-2",)  # only the genuinely new id is reported fresh
    assert ring == {"m-1": _NOW.isoformat()}  # the input ring is untouched


def test_filter_dedups_repeats_within_one_batch() -> None:
    # The same fresh id twice in ONE batch is reported once and kept once.
    kept, fresh = filter_external_events({}, [_contact("m-9"), _contact("m-9")], _NOW)
    assert [s.origin_id for s in kept] == ["m-9"]
    assert fresh == ("m-9",)


def test_record_adds_fresh_ids_and_stays_bounded() -> None:
    seen = {"m-1": _NOW.isoformat(), "m-2": _NOW.isoformat()}
    ring = record_external_events(seen, ["m-3"], _NOW, cap=2)
    assert list(ring) == ["m-2", "m-3"]  # oldest (m-1) evicted, fresh recorded
    assert ring["m-3"] == _NOW.isoformat()


def test_record_with_no_fresh_ids_still_sweeps_ttl() -> None:
    ttl = timedelta(hours=1)
    old = (_NOW - timedelta(hours=2)).isoformat()
    fresh = (_NOW - timedelta(minutes=5)).isoformat()
    ring = record_external_events({"stale": old, "recent": fresh}, [], _NOW, ttl=ttl)
    assert set(ring) == {"recent"}  # bounded/TTL upkeep runs even with nothing new


def test_record_is_a_noop_ring_for_a_pure_duplicate_frame() -> None:
    # filter drops the dup → no fresh ids → record returns a ring EQUAL to the input,
    # so the coreloop writes no churn on a retry.
    seen = {"m-1": _NOW.isoformat()}
    _kept, fresh = filter_external_events(seen, [_contact("m-1")], _NOW)
    assert fresh == ()
    assert record_external_events(seen, fresh, _NOW) == seen

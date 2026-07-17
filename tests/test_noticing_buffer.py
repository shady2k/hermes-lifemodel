"""Unit tests for :class:`NoticingBuffer` (lm-705.5 Task 2, design §4.1).

A process-owned, lock-protected per-session pending→complete ring: at most one
open ``pending`` turn per session lane; completed turns land in a bounded ring;
``closed_segment`` honours the closed-prefix rule (never surveys mid-turn); a
stale ``pending`` ages out via TTL so a dropped turn can't wedge the lane shut
forever.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from lifemodel.core.noticing_buffer import BufferEntry, NoticingBuffer

T0 = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


def test_open_stamp_complete_yields_closed_segment_with_source_id() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi there", now=T0)
    buf.stamp_source("s1", "m1")
    buf.complete("s1", "t1", assistant_text="hello!", now=T0 + timedelta(seconds=1))

    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=2))

    assert segment == [
        BufferEntry(
            session_id="s1",
            turn_id="t1",
            source_ids=("m1",),
            user_text="hi there",
            assistant_text="hello!",
            ts=segment[0].ts,
        )
    ]
    assert segment[0].source_ids == ("m1",)


def test_stamp_source_can_record_multiple_ids_before_completion() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.stamp_source("s1", "m1")
    buf.stamp_source("s1", "m2")
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=2))

    assert segment[0].source_ids == ("m1", "m2")


def test_open_pending_with_no_completion_yields_empty_closed_segment() -> None:
    # Closed-prefix rule: an unmatched pending turn must never be surveyed.
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)

    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=1)) == []


def test_closed_segment_is_empty_while_a_later_pending_is_open() -> None:
    # Even with prior complete entries in the ring, a NEW open pending closes
    # the lane again — never survey mid-turn.
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=2)) != []

    buf.open_pending("s1", user_text="second turn", now=T0 + timedelta(seconds=3))
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=4)) == []


def test_stale_pending_ages_to_abandoned_and_lane_reopens() -> None:
    buf = NoticingBuffer(pending_ttl=timedelta(minutes=30))
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    # A second turn opens and is dropped (never completes) — e.g. a crashed call.
    buf.open_pending("s1", user_text="dropped", now=T0 + timedelta(seconds=2))

    # Still within TTL: the dropped turn keeps the lane closed.
    just_under_ttl = T0 + timedelta(seconds=2) + timedelta(minutes=30) - timedelta(seconds=1)
    assert buf.closed_segment("s1", now=just_under_ttl) == []

    # Past TTL: the stale pending ages to abandoned and is dropped, so the
    # earlier complete entry (t1) is no longer wedged behind it.
    past_ttl = T0 + timedelta(seconds=2) + timedelta(minutes=30) + timedelta(seconds=1)
    segment = buf.closed_segment("s1", now=past_ttl)
    assert [e.turn_id for e in segment] == ["t1"]

    # The lane has genuinely reopened: a fresh turn can complete normally.
    buf.open_pending("s1", user_text="third turn", now=past_ttl + timedelta(seconds=1))
    buf.complete("s1", "t3", assistant_text="ok", now=past_ttl + timedelta(seconds=2))
    segment2 = buf.closed_segment("s1", now=past_ttl + timedelta(seconds=3))
    assert [e.turn_id for e in segment2] == ["t1", "t3"]


def test_complete_with_a_naive_now_raises_without_dropping_the_pending() -> None:
    # M3: ts must be validated (to_iso rejects a naive datetime) BEFORE the
    # pending slot is popped, so a bad clock call fails loud with the pending
    # still intact — never silently destroyed on the way to raising.
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    naive_now = datetime(2026, 7, 17, 12, 0, 1)  # no tzinfo

    with pytest.raises(ValueError):
        buf.complete("s1", "t1", assistant_text="hello", now=naive_now)

    # still pending (closed-prefix rule keeps the lane shut) -- never dropped.
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=2)) == []

    # a later, valid complete lands normally on the SAME (never-destroyed) pending.
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=3))
    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=4))
    assert [e.turn_id for e in segment] == ["t1"]
    assert segment[0].user_text == "hi"


def test_complete_with_no_open_pending_is_a_defensive_noop() -> None:
    buf = NoticingBuffer()
    buf.complete("s1", "t1", assistant_text="hello", now=T0)
    assert buf.closed_segment("s1", now=T0) == []


def test_stamp_source_with_no_open_pending_is_a_noop() -> None:
    buf = NoticingBuffer()
    # Must not raise, must not create a phantom pending/entry.
    buf.stamp_source("s1", "m1")
    assert buf.closed_segment("s1", now=T0) == []


def test_ring_bounds_length_and_evicts_oldest() -> None:
    buf = NoticingBuffer(max_entries=3)
    for i in range(5):
        turn_id = f"t{i}"
        buf.open_pending("s1", user_text=f"u{i}", now=T0 + timedelta(seconds=i))
        buf.complete(
            "s1", turn_id, assistant_text=f"a{i}", now=T0 + timedelta(seconds=i, milliseconds=1)
        )

    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=10))
    assert [e.turn_id for e in segment] == ["t2", "t3", "t4"]


def test_abandon_pending_drops_the_open_slot() -> None:
    # review-2 G2: a post_llm decline must be able to release a pending slot
    # without ever completing it, so a later closed_segment isn't blocked by
    # a turn that never really landed.
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)

    buf.abandon_pending("s1")

    # The lane reopens immediately -- no stale pending left to gate it shut.
    assert buf.closed_segment("s1", now=T0) == []  # empty ring, but NOT gated
    buf.open_pending("s1", user_text="second try", now=T0 + timedelta(seconds=1))
    buf.complete("s1", "t1", assistant_text="ok", now=T0 + timedelta(seconds=2))
    assert [e.turn_id for e in buf.closed_segment("s1", now=T0 + timedelta(seconds=3))] == ["t1"]


def test_abandon_pending_with_no_open_pending_is_a_noop() -> None:
    buf = NoticingBuffer()
    buf.complete("s1", "t1", assistant_text="a", now=T0)  # defensive no-op, no pending exists

    buf.abandon_pending("s1")  # must not raise, must not disturb anything

    assert buf.closed_segment("s1", now=T0) == []


def test_abandon_pending_does_not_touch_the_completed_ring() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="first", now=T0)
    buf.complete("s1", "t1", assistant_text="reply1", now=T0 + timedelta(seconds=1))
    # A second turn opens on the same lane, then is abandoned (declined).
    buf.open_pending("s1", user_text="second", now=T0 + timedelta(seconds=2))

    buf.abandon_pending("s1")

    # t1 -- already safely closed before the abandoned pending ever opened --
    # is immediately visible again, not gated behind pending_ttl.
    assert [e.turn_id for e in buf.closed_segment("s1", now=T0 + timedelta(seconds=3))] == ["t1"]


def test_two_sessions_are_fully_independent() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi from s1", now=T0)
    buf.stamp_source("s1", "m-s1")
    buf.complete("s1", "t1", assistant_text="reply s1", now=T0 + timedelta(seconds=1))

    # s2 has an open pending — must not affect s1's closed segment, and s1 must
    # not affect s2's (still-empty) one.
    buf.open_pending("s2", user_text="hi from s2", now=T0)

    seg1 = buf.closed_segment("s1", now=T0 + timedelta(seconds=2))
    seg2 = buf.closed_segment("s2", now=T0 + timedelta(seconds=2))

    assert [e.turn_id for e in seg1] == ["t1"]
    assert seg1[0].source_ids == ("m-s1",)
    assert seg2 == []

    buf.complete("s2", "u1", assistant_text="reply s2", now=T0 + timedelta(seconds=3))
    seg2_after = buf.closed_segment("s2", now=T0 + timedelta(seconds=4))
    assert [e.turn_id for e in seg2_after] == ["u1"]
    # s1's ring is untouched by s2's activity.
    assert [e.turn_id for e in buf.closed_segment("s1", now=T0 + timedelta(seconds=4))] == ["t1"]


def test_session_ids_reports_every_known_lane() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s-pending-only", user_text="mid-turn", now=T0)
    buf.open_pending("s-complete", user_text="hi", now=T0)
    buf.complete("s-complete", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    assert buf.session_ids() == ["s-complete", "s-pending-only"]


def test_session_ids_is_empty_for_a_fresh_buffer() -> None:
    assert NoticingBuffer().session_ids() == []


def test_concurrent_open_stamp_complete_across_sessions_is_thread_safe() -> None:
    # Each thread owns its own session lane (the real usage shape — distinct
    # conversations proceed concurrently). This exercises the shared lock
    # protecting the two internal dicts against concurrent mutation.
    buf = NoticingBuffer()
    num_threads = 16
    errors: list[BaseException] = []
    barrier = threading.Barrier(num_threads)

    def worker(i: int) -> None:
        session_id = f"session-{i}"
        try:
            barrier.wait(timeout=5)
            buf.open_pending(session_id, user_text=f"hi {i}", now=T0)
            buf.stamp_source(session_id, f"m{i}")
            buf.complete(session_id, f"t{i}", assistant_text=f"reply {i}", now=T0)
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    for i in range(num_threads):
        session_id = f"session-{i}"
        segment = buf.closed_segment(session_id, now=T0)
        assert [e.turn_id for e in segment] == [f"t{i}"]
        assert segment[0].source_ids == (f"m{i}",)

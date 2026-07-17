"""Tests for ``NoticingBuffer`` delegating to a ``BufferStore`` (lm-705.14 Task 2).

Two backings exercised through the SAME ``NoticingBuffer`` API:

* the default in-memory store (mirroring a couple of ``test_noticing_buffer.py``'s
  own scenarios plus the new claim/finalize/release lifecycle) — proving the
  extraction into ``InMemoryBufferStore`` changed nothing observable;
* ``SqliteBufferStore`` over a temp dir — proving the durable half: the SAME
  buffer API, and a captured turn that survives a freshly-built
  ``NoticingBuffer`` over the SAME directory (a simulated restart).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.buffer_store import BufferEntry, InMemoryBufferStore
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.state.sqlite_store import SqliteBufferStore
from lifemodel.testing import FakeClock

T0 = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


def _sqlite_buffer(tmp_path: Path) -> NoticingBuffer:
    return NoticingBuffer(store=SqliteBufferStore(tmp_path, clock=FakeClock(T0)))


# ---- default (in-memory) store: mirrors test_noticing_buffer.py's own scenarios --


def test_default_store_is_in_memory_and_behaves_as_before() -> None:
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


def test_explicit_in_memory_buffer_store_behaves_identically_to_the_default() -> None:
    buf = NoticingBuffer(store=InMemoryBufferStore(max_entries=3))
    buf.open_pending("s1", user_text="hi", now=T0)

    # closed-prefix rule: a live pending gates the whole lane.
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=1)) == []

    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=2))
    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=3))
    assert [e.turn_id for e in segment] == ["t1"]


def test_claim_claimed_finalize_round_trip_over_in_memory_store() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    buf.claim("s1", ("t1",), "survey-1")

    # claimed rows leave closed_segment's result (immune to further ring pressure)...
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=2)) == []
    # ...and become visible via claimed().
    assert [e.turn_id for e in buf.claimed("survey-1")] == ["t1"]

    buf.finalize("survey-1")
    assert buf.claimed("survey-1") == []


def test_release_returns_a_claim_to_complete_over_in_memory_store() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))
    buf.claim("s1", ("t1",), "survey-1")

    buf.release("survey-1")

    assert buf.claimed("survey-1") == []
    assert [e.turn_id for e in buf.closed_segment("s1", now=T0 + timedelta(seconds=2))] == ["t1"]


def test_recover_stale_claims_releases_everything_over_in_memory_store() -> None:
    buf = NoticingBuffer()
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))
    buf.claim("s1", ("t1",), "survey-1")

    buf.recover_stale_claims()

    assert buf.claimed("survey-1") == []
    assert [e.turn_id for e in buf.closed_segment("s1", now=T0 + timedelta(seconds=2))] == ["t1"]


# ---- durable store (SqliteBufferStore): same buffer API, survives a restart -----


def test_sqlite_backed_buffer_behaves_like_the_in_memory_one(tmp_path: Path) -> None:
    buf = _sqlite_buffer(tmp_path)
    buf.open_pending("s1", user_text="hi there", now=T0)
    buf.stamp_source("s1", "m1")

    # closed-prefix rule holds over the durable store too.
    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=1)) == []

    buf.complete("s1", "t1", assistant_text="hello!", now=T0 + timedelta(seconds=2))
    segment = buf.closed_segment("s1", now=T0 + timedelta(seconds=3))

    assert [e.turn_id for e in segment] == ["t1"]
    assert segment[0].source_ids == ("m1",)
    assert segment[0].user_text == "hi there"
    assert segment[0].assistant_text == "hello!"


def test_sqlite_backed_buffer_survives_a_freshly_built_noticing_buffer(tmp_path: Path) -> None:
    # Capture a completed turn through one NoticingBuffer instance...
    first = _sqlite_buffer(tmp_path)
    first.open_pending("s1", user_text="remember me", now=T0)
    first.complete("s1", "t1", assistant_text="I will", now=T0 + timedelta(seconds=1))

    # ...then rebuild an entirely new NoticingBuffer (a fresh process/restart,
    # simulated) over the SAME temp dir: the captured turn must still be there.
    second = NoticingBuffer(store=SqliteBufferStore(tmp_path, clock=FakeClock(T0)))
    segment = second.closed_segment("s1", now=T0 + timedelta(seconds=2))

    assert [e.turn_id for e in segment] == ["t1"]
    assert segment[0].user_text == "remember me"
    assert segment[0].assistant_text == "I will"


def test_sqlite_backed_buffer_abandon_pending_and_session_ids(tmp_path: Path) -> None:
    buf = _sqlite_buffer(tmp_path)
    buf.open_pending("s1", user_text="hi", now=T0)

    buf.abandon_pending("s1")

    assert buf.closed_segment("s1", now=T0) == []  # empty, not gated
    assert buf.session_ids() == []  # abandoned before ever completing -- nothing durable yet

    buf.open_pending("s1", user_text="second try", now=T0 + timedelta(seconds=1))
    buf.complete("s1", "t1", assistant_text="ok", now=T0 + timedelta(seconds=2))
    assert buf.session_ids() == ["s1"]


def test_claim_claimed_finalize_round_trip_over_sqlite_store(tmp_path: Path) -> None:
    buf = _sqlite_buffer(tmp_path)
    buf.open_pending("s1", user_text="hi", now=T0)
    buf.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    buf.claim("s1", ("t1",), "survey-1")

    assert buf.closed_segment("s1", now=T0 + timedelta(seconds=2)) == []
    assert [e.turn_id for e in buf.claimed("survey-1")] == ["t1"]

    buf.finalize("survey-1")
    assert buf.claimed("survey-1") == []

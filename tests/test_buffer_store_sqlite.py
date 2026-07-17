"""Tests for :class:`SqliteBufferStore` + the ``conversation_buffer`` migration
(lm-705.14 Task 1, durable NoticingBuffer + claim/finalize).

Covers what the store layer must guarantee: the pending -> complete -> claimed
lifecycle over the ``conversation_buffer`` table (migration v4), the
closed-prefix + TTL-abandon semantics matching today's in-memory
``NoticingBuffer`` (``core/noticing_buffer.py``), and durability across a
rebuilt store — a restart must never wipe a captured-but-not-yet-noticed
conversation.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.buffer_store import BufferEntry, BufferStore
from lifemodel.state.sqlite_store import SqliteBufferStore, SQLiteRuntimeStore
from lifemodel.testing import FakeClock

T0 = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
TTL = timedelta(minutes=30)
DB_FILENAME = "lifemodel.sqlite"


def _store(tmp_path: Path) -> SqliteBufferStore:
    return SqliteBufferStore(tmp_path, clock=FakeClock(T0))


def _two_completed_turns(store: SqliteBufferStore, session_id: str = "s1") -> None:
    store.open_pending(session_id, user_text="u0", now=T0)
    store.complete(session_id, "t1", assistant_text="a0", now=T0 + timedelta(seconds=1))
    store.open_pending(session_id, user_text="u1", now=T0 + timedelta(seconds=2))
    store.complete(session_id, "t2", assistant_text="a1", now=T0 + timedelta(seconds=3))


# ---- protocol conformance ----------------------------------------------------


def test_sqlite_buffer_store_satisfies_the_buffer_store_protocol(tmp_path: Path) -> None:
    assert isinstance(_store(tmp_path), BufferStore)


# ---- migration v4 -------------------------------------------------------------


def test_migration_v4_creates_the_conversation_buffer_table(tmp_path: Path) -> None:
    SQLiteRuntimeStore(tmp_path, clock=FakeClock(T0))

    with closing(sqlite3.connect(str(tmp_path / DB_FILENAME))) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert "conversation_buffer" in tables
    assert versions == [1, 2, 3, 4]


def test_migration_v4_is_idempotent_on_a_second_construction(tmp_path: Path) -> None:
    clock = FakeClock(T0)
    SQLiteRuntimeStore(tmp_path, clock=clock)
    SQLiteRuntimeStore(tmp_path, clock=clock)  # no re-apply, no error

    with closing(sqlite3.connect(str(tmp_path / DB_FILENAME))) as conn:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert versions == [1, 2, 3, 4]


def test_sqlite_buffer_store_works_standalone_without_a_prior_runtime_store(
    tmp_path: Path,
) -> None:
    # No SQLiteRuntimeStore constructed first: SqliteBufferStore must ensure its
    # own table exists (the documented alternative to relying on migration order).
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)
    store.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    entries = store.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1"]


# ---- open -> stamp -> complete -> completed() ---------------------------------


def test_open_stamp_complete_yields_completed_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi there", now=T0)
    store.stamp_source("s1", "m1")
    store.complete("s1", "t1", assistant_text="hello!", now=T0 + timedelta(seconds=1))

    entries = store.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL)

    assert entries == [
        BufferEntry(
            session_id="s1",
            turn_id="t1",
            source_ids=("m1",),
            user_text="hi there",
            assistant_text="hello!",
            ts=entries[0].ts,
        )
    ]


def test_stamp_source_can_record_multiple_ids_before_completion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)
    store.stamp_source("s1", "m1")
    store.stamp_source("s1", "m2")
    store.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    entries = store.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL)
    assert entries[0].source_ids == ("m1", "m2")


def test_live_pending_yields_empty_completed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)

    assert store.completed("s1", now=T0 + timedelta(seconds=1), ttl=TTL) == []


def test_a_new_pending_recloses_the_lane_even_with_prior_complete_entries(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)
    store.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))
    assert store.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL) != []

    store.open_pending("s1", user_text="second turn", now=T0 + timedelta(seconds=3))

    assert store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL) == []


def test_ttl_stale_pending_is_abandoned_and_lane_reopens(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)
    store.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))
    # A second turn opens and is dropped (never completes) -- e.g. a crashed call.
    store.open_pending("s1", user_text="dropped", now=T0 + timedelta(seconds=2))

    just_under_ttl = T0 + timedelta(seconds=2) + TTL - timedelta(seconds=1)
    assert store.completed("s1", now=just_under_ttl, ttl=TTL) == []

    past_ttl = T0 + timedelta(seconds=2) + TTL + timedelta(seconds=1)
    entries = store.completed("s1", now=past_ttl, ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1"]

    # The lane has genuinely reopened: a fresh turn completes normally.
    store.open_pending("s1", user_text="third turn", now=past_ttl + timedelta(seconds=1))
    store.complete("s1", "t3", assistant_text="ok", now=past_ttl + timedelta(seconds=2))
    entries2 = store.completed("s1", now=past_ttl + timedelta(seconds=3), ttl=TTL)
    assert [e.turn_id for e in entries2] == ["t1", "t3"]


def test_complete_with_no_open_pending_is_a_defensive_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.complete("s1", "t1", assistant_text="hello", now=T0)
    assert store.completed("s1", now=T0, ttl=TTL) == []


def test_stamp_source_with_no_open_pending_is_a_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.stamp_source("s1", "m1")  # must not raise, must not create a phantom entry
    assert store.completed("s1", now=T0, ttl=TTL) == []


def test_abandon_pending_drops_the_open_slot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi", now=T0)

    store.abandon_pending("s1")

    assert store.completed("s1", now=T0, ttl=TTL) == []  # empty, but NOT gated
    store.open_pending("s1", user_text="second try", now=T0 + timedelta(seconds=1))
    store.complete("s1", "t1", assistant_text="ok", now=T0 + timedelta(seconds=2))
    entries = store.completed("s1", now=T0 + timedelta(seconds=3), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1"]


def test_abandon_pending_with_no_open_pending_is_a_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.complete("s1", "t1", assistant_text="a", now=T0)  # defensive no-op, no pending exists

    store.abandon_pending("s1")  # must not raise, must not disturb anything

    assert store.completed("s1", now=T0, ttl=TTL) == []


def test_two_sessions_are_fully_independent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s1", user_text="hi s1", now=T0)
    store.stamp_source("s1", "m-s1")
    store.complete("s1", "t1", assistant_text="reply s1", now=T0 + timedelta(seconds=1))

    store.open_pending("s2", user_text="hi s2", now=T0)

    seg1 = store.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL)
    seg2 = store.completed("s2", now=T0 + timedelta(seconds=2), ttl=TTL)

    assert [e.turn_id for e in seg1] == ["t1"]
    assert seg1[0].source_ids == ("m-s1",)
    assert seg2 == []

    store.complete("s2", "u1", assistant_text="reply s2", now=T0 + timedelta(seconds=3))
    seg2_after = store.completed("s2", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in seg2_after] == ["u1"]
    # s1's set is untouched by s2's activity.
    assert [e.turn_id for e in store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)] == [
        "t1"
    ]


def test_session_ids_reports_every_known_lane(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.open_pending("s-pending-only", user_text="mid-turn", now=T0)
    store.open_pending("s-complete", user_text="hi", now=T0)
    store.complete("s-complete", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    assert store.session_ids() == ["s-complete", "s-pending-only"]


def test_session_ids_is_empty_for_a_fresh_store(tmp_path: Path) -> None:
    assert _store(tmp_path).session_ids() == []


# ---- claim / claimed / finalize / release / recover_stale_claims -------------


def test_claim_marks_rows_visible_via_claimed_and_removes_them_from_completed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)

    store.claim("s1", ("t1", "t2"), "survey-1")

    assert [e.turn_id for e in store.claimed("survey-1")] == ["t1", "t2"]
    assert store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL) == []


def test_claim_only_the_named_turns_leaves_the_rest_completed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)

    store.claim("s1", ("t1",), "survey-1")

    assert [e.turn_id for e in store.claimed("survey-1")] == ["t1"]
    remaining = store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in remaining] == ["t2"]


def test_claim_of_an_unknown_turn_id_is_silently_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)

    store.claim("s1", ("t1", "does-not-exist"), "survey-1")

    assert [e.turn_id for e in store.claimed("survey-1")] == ["t1"]


def test_claim_with_no_turn_ids_is_a_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)

    store.claim("s1", (), "survey-1")

    assert store.claimed("survey-1") == []
    entries = store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1", "t2"]


def test_finalize_drops_the_claimed_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)
    store.claim("s1", ("t1", "t2"), "survey-1")

    store.finalize("survey-1")

    assert store.claimed("survey-1") == []
    assert store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL) == []


def test_release_returns_claimed_rows_to_complete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)
    store.claim("s1", ("t1", "t2"), "survey-1")

    store.release("survey-1")

    assert store.claimed("survey-1") == []
    entries = store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1", "t2"]


def test_recover_stale_claims_releases_every_leftover_claim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)
    store.claim("s1", ("t1", "t2"), "survey-1")

    store.recover_stale_claims()

    assert store.claimed("survey-1") == []
    entries = store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1", "t2"]


def test_recover_stale_claims_across_multiple_survey_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _two_completed_turns(store)
    store.claim("s1", ("t1",), "survey-1")
    store.claim("s1", ("t2",), "survey-2")

    store.recover_stale_claims()

    assert store.claimed("survey-1") == []
    assert store.claimed("survey-2") == []
    entries = store.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1", "t2"]


# ---- durability: a second store on the same base_dir sees the rows ----------


def test_a_second_store_over_the_same_base_dir_sees_the_rows(tmp_path: Path) -> None:
    first = _store(tmp_path)
    first.open_pending("s1", user_text="hi", now=T0)
    first.stamp_source("s1", "m1")
    first.complete("s1", "t1", assistant_text="hello", now=T0 + timedelta(seconds=1))

    reopened = _store(tmp_path)  # simulates a restart -- a fresh instance, same dir

    entries = reopened.completed("s1", now=T0 + timedelta(seconds=2), ttl=TTL)
    assert [e.turn_id for e in entries] == ["t1"]
    assert entries[0].source_ids == ("m1",)
    assert entries[0].user_text == "hi"
    assert entries[0].assistant_text == "hello"


def test_durability_survives_a_claim(tmp_path: Path) -> None:
    first = _store(tmp_path)
    _two_completed_turns(first)
    first.claim("s1", ("t1",), "survey-1")

    reopened = _store(tmp_path)

    assert [e.turn_id for e in reopened.claimed("survey-1")] == ["t1"]
    assert [
        e.turn_id for e in reopened.completed("s1", now=T0 + timedelta(seconds=4), ttl=TTL)
    ] == ["t2"]


def test_durability_of_a_still_open_pending(tmp_path: Path) -> None:
    first = _store(tmp_path)
    first.open_pending("s1", user_text="mid-flight", now=T0)

    reopened = _store(tmp_path)

    # the closed-prefix rule survives the "restart" too -- still gated.
    assert reopened.completed("s1", now=T0 + timedelta(seconds=1), ttl=TTL) == []
    assert reopened.session_ids() == ["s1"]

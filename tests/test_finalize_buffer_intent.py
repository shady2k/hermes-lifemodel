"""Atomic ``FinalizeBuffer`` through ``commit_tick`` (lm-705.13, codex I3).

The cursor-advance half of a noticing pass — dropping the ``conversation_buffer``
rows claimed under a ``survey_id`` — MUST land in the SAME ``BEGIN IMMEDIATE``
transaction as the pass's ``PutRecord(thought)`` + consumed-ring ``UpdateState``.
These tests drive the REAL :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`
(as both ``StatePort`` and ``TickCommitPort``) and a real
:class:`~lifemodel.state.sqlite_store.SqliteBufferStore` over ONE physical
``lifemodel.sqlite`` (D7 — one store), so the DELETE the committer issues actually
reaches the rows the buffer store reads back. Mirrors
``tests/test_sqlite_store.py``'s own ``commit_tick`` rollback pattern
(``test_commit_tick_stale_transition_rolls_back_everything``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.core.intents import FinalizeBuffer, PutRecord, TransitionRecord, UpdateState
from lifemodel.core.state_actor import StateActor
from lifemodel.core.thought_view import build_thought, encode_thought, seed_thought_id
from lifemodel.domain.memory import PutOp, StaleTransition, TransitionOp
from lifemodel.state.sqlite_store import SqliteBufferStore, SQLiteRuntimeStore
from lifemodel.testing import FakeClock

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
SURVEY_ID = "t1@2026-07-17T12:00:00.000000+00:00"


def _claimed_buffer(base_dir: Path, clock: FakeClock) -> SqliteBufferStore:
    """A buffer store with one turn ``t1`` on lane ``s1`` claimed under
    :data:`SURVEY_ID` — the exact snapshot a launched noticing pass surveyed."""
    buf = SqliteBufferStore(base_dir, clock=clock)
    buf.open_pending("s1", user_text="I have an interview Friday", now=NOW)
    buf.complete("s1", "t1", assistant_text="good luck", now=NOW)
    buf.claim("s1", ("t1",), SURVEY_ID)
    assert [e.turn_id for e in buf.claimed(SURVEY_ID)] == ["t1"]  # precondition
    return buf


def _thought_put(content: str) -> PutRecord:
    thought = build_thought(id=seed_thought_id(content), content=content, salience=0.6)
    return PutRecord(op=PutOp(draft=encode_thought(thought)))


def test_finalize_lands_atomically_with_the_thought_and_ring(tmp_path: Path) -> None:
    clock = FakeClock(NOW)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    buf = _claimed_buffer(tmp_path, clock)
    actor = StateActor(store)

    content = "they have an interview Friday"
    actor.apply(
        [
            _thought_put(content),
            UpdateState({"noticed_source_ids": ("t1",)}),
            FinalizeBuffer(SURVEY_ID),
        ]
    )

    # A fresh store instance proves ALL THREE committed together, durably.
    fresh = SQLiteRuntimeStore(tmp_path, clock=clock)
    thought = fresh.get("thought", seed_thought_id(content))
    assert thought is not None and thought.payload["content"] == content
    assert fresh.load().noticed_source_ids == ("t1",)
    # the claimed rows are gone — the cursor advanced in the SAME transaction.
    assert buf.claimed(SURVEY_ID) == []


def test_finalize_only_batch_still_advances_the_cursor(tmp_path: Path) -> None:
    # A genuinely-surveyed-but-fruitless pass emits ONLY a FinalizeBuffer — no
    # thought, no consumed id — yet the claimed rows must still be dropped, or the
    # segment would be re-surveyed forever.
    clock = FakeClock(NOW)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    buf = _claimed_buffer(tmp_path, clock)
    actor = StateActor(store)

    actor.apply([FinalizeBuffer(SURVEY_ID)])

    assert buf.claimed(SURVEY_ID) == []


def test_stale_transition_rolls_back_the_finalize_and_the_thought(tmp_path: Path) -> None:
    # ATOMICITY (codex I3): a stale transition anywhere in the batch rolls back
    # EVERYTHING — the thought never lands AND the claimed rows survive, so the
    # segment is left intact for a retry / boot recovery rather than a half-applied
    # pass (thoughts without their cursor-advance, or vice versa).
    clock = FakeClock(NOW)
    store = SQLiteRuntimeStore(tmp_path, clock=clock)
    buf = _claimed_buffer(tmp_path, clock)
    actor = StateActor(store)

    content = "should vanish on rollback"
    with pytest.raises(StaleTransition):
        actor.apply(
            [
                _thought_put(content),
                # no such row → a stale transition mid-batch aborts + rolls back.
                TransitionRecord(
                    TransitionOp(
                        kind="desire", id="ghost", from_state="active", to_state="archived"
                    )
                ),
                FinalizeBuffer(SURVEY_ID),
            ]
        )

    fresh = SQLiteRuntimeStore(tmp_path, clock=clock)
    assert fresh.get("thought", seed_thought_id(content)) is None  # thought rolled back
    assert fresh.load().noticed_source_ids == ()  # state untouched
    # the claim SURVIVES — nothing was finalized.
    assert [e.turn_id for e in buf.claimed(SURVEY_ID)] == ["t1"]

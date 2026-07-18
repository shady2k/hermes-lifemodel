"""PutOp.create_only — atomic create-if-absent (lm-705.11 Task 1)."""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.domain.memory import MemoryDraft, PutOp
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeMemoryStore, FakeStateStore


def _draft(id: str, state: str, *, source: str) -> MemoryDraft:
    return MemoryDraft(kind="thought", id=id, state=state, payload={"content": id}, source=source)


def _store() -> FakeStateStore:
    clock = FakeClock(datetime(2026, 7, 18, tzinfo=UTC))
    return FakeStateStore(initial=State(), memory=FakeMemoryStore(clock=clock))


def test_create_only_inserts_when_absent() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="a"), create_only=True)])
    assert store._memory.get("thought", "t1").source == "a"


def test_create_only_is_noop_on_existing_terminal_row() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "resolved", source="orig"))])  # normal put
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="new"), create_only=True)])
    row = store._memory.get("thought", "t1")
    assert row.state == "resolved" and row.source == "orig"  # NOT resurrected, NOT overwritten


def test_normal_put_still_upserts() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="a"))])
    store.commit_tick(None, [PutOp(draft=_draft("t1", "parked", source="b"))])  # create_only=False
    assert store._memory.get("thought", "t1").state == "parked"

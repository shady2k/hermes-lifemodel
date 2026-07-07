"""Tests for the shipped test fakes (HLA §13, "imitations before code").

Later tasks (1.1–1.4) build on these, so their behaviour is pinned here: the
clock is controllable, delivery records, the state store isolates and persists,
and the in-memory signal bus honours the same dedup contract as the file bus.
Imports no Hermes.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lifemodel.domain.memory import (
    MemoryDraft,
    PutOp,
    StaleTransition,
    TransitionOp,
)
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import (
    FakeClock,
    FakeDelivery,
    FakeMemoryStore,
    FakeSignalBus,
    FakeStateStore,
)

BASE_TIME = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _draft(id: str, state: str = "active", **overrides: object) -> MemoryDraft:
    base: dict[str, object] = dict(
        kind="desire", id=id, state=state, payload={"note": id}, source="t"
    )
    base.update(overrides)
    return MemoryDraft(**base)  # type: ignore[arg-type]


def test_fake_clock_is_controllable() -> None:
    start = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
    clock = FakeClock(start)
    assert clock.now() == start
    clock.advance(timedelta(minutes=5))
    assert clock.now() == start + timedelta(minutes=5)
    clock.set(start)
    assert clock.now() == start


def test_fake_delivery_records_sends() -> None:
    delivery = FakeDelivery()
    delivery.send("author", "hello")
    delivery.send("author", "again")
    assert delivery.sent == [("author", "hello"), ("author", "again")]


def test_fake_state_store_defaults_and_round_trips() -> None:
    store = FakeStateStore()
    assert store.load() == State()
    store.commit(State(u=2.0))
    assert store.load().u == 2.0


def test_fake_state_store_isolates_from_caller_mutation() -> None:
    store = FakeStateStore()
    state = State(u=1.0)
    store.commit(state)
    # Mutating the caller's object must not reach into the store (copy-in).
    state.u = 99.0
    assert store.load().u == 1.0
    # ...and a loaded copy is likewise detached (copy-out).
    loaded = store.load()
    loaded.u = 42.0
    assert store.load().u == 1.0


def test_fake_signal_bus_matches_dedup_contract() -> None:
    bus = FakeSignalBus()
    bus.publish(Signal(origin_id="m1", kind="x"))
    bus.publish(Signal(origin_id="m1", kind="y"))  # dup id
    bus.publish(Signal(origin_id="m2", kind="x"))
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m1", "m2"]
    # Idempotent across calls; re-published id does not re-fire.
    bus.publish(Signal(origin_id="m1", kind="z"))
    assert bus.consume_unprocessed() == []


# --- lm-27n.2: FakeStateStore.commit_tick (atomic State+memory) ---


def _fake_committer() -> tuple[FakeStateStore, FakeMemoryStore]:
    mem = FakeMemoryStore(clock=FakeClock(BASE_TIME))
    store = FakeStateStore(memory=mem)
    return store, mem


def test_fake_commit_tick_applies_state_and_mutations() -> None:
    store, mem = _fake_committer()
    mem.put(_draft("d1"))
    store.commit_tick(
        State(u=1.5),
        [
            PutOp(_draft("d2")),
            TransitionOp(kind="desire", id="d1", from_state="active", to_state="archived"),
        ],
    )
    assert store.load().u == 1.5
    d1 = mem.get("desire", "d1")
    assert d1 is not None and d1.state == "archived"
    assert mem.get("desire", "d2") is not None


def test_fake_commit_tick_stale_transition_rolls_back_everything() -> None:
    store, mem = _fake_committer()
    store.commit(State(u=1.0))
    with pytest.raises(StaleTransition):
        store.commit_tick(
            State(u=9.0),
            [
                PutOp(_draft("ghost")),
                TransitionOp(kind="desire", id="ghost", from_state="archived", to_state="active"),
            ],
        )
    assert store.load().u == 1.0  # state rolled back
    assert mem.get("desire", "ghost") is None  # earlier put rolled back


def test_fake_commit_tick_applies_in_list_order() -> None:
    store, mem = _fake_committer()
    store.commit_tick(
        None,
        [
            PutOp(_draft("loop")),
            TransitionOp(kind="desire", id="loop", from_state="active", to_state="archived"),
        ],
    )
    record = mem.get("desire", "loop")
    assert record is not None and record.state == "archived"


def test_fake_commit_tick_without_memory_rejects_mutations() -> None:
    store = FakeStateStore()  # no memory backing
    with pytest.raises(TypeError):
        store.commit_tick(None, [PutOp(_draft("x"))])


def test_fake_commit_tick_state_only_needs_no_memory() -> None:
    store = FakeStateStore()
    store.commit_tick(State(u=2.0), [])
    assert store.load().u == 2.0


def test_fake_and_real_commit_tick_agree_on_rollback(tmp_path: Path) -> None:
    # The parity guarantee: the same batch produces the same end-state on the
    # fake and the real SQLite store, and both roll back all-or-nothing.
    def scenario(committer, memory, load_state) -> tuple[float, bool]:  # type: ignore[no-untyped-def]
        memory.put(_draft("d1"))
        committer.commit(State(u=1.0))
        with contextlib.suppress(StaleTransition):
            committer.commit_tick(
                State(u=5.0),
                [
                    PutOp(_draft("d2")),
                    TransitionOp(kind="desire", id="d1", from_state="archived", to_state="active"),
                ],
            )
        return load_state().u, memory.get("desire", "d2") is None

    mem = FakeMemoryStore(clock=FakeClock(BASE_TIME))
    fake = FakeStateStore(memory=mem)
    fake_result = scenario(fake, mem, fake.load)

    real = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    real_result = scenario(real, real, real.load)

    assert fake_result == real_result == (1.0, True)


def test_fake_memory_store_honors_draft_schema_version() -> None:
    mem = FakeMemoryStore(clock=FakeClock(BASE_TIME))
    mem.put(_draft("v2", schema_version=2))
    record = mem.get("desire", "v2")
    assert record is not None and record.schema_version == 2

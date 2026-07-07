from __future__ import annotations

import pytest

from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.state_actor import StateActor, UnknownStateField
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


class RecordingStore:
    """Minimal StatePort double that counts commits."""

    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)

    def reset(self) -> State:
        self._state = State()
        self.commits.append(self._state)
        return self._state


def test_apply_merges_updates_and_commits_once() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    result = actor.apply([UpdateState({"u": 0.5}), UpdateState({"tick_count": 2})])
    assert result.u == 0.5
    assert result.tick_count == 2
    assert actor.state is result
    assert len(store.commits) == 1


def test_apply_without_state_changes_does_not_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    before = actor.state
    result = actor.apply([EmitSignal(Signal(origin_id="n1", kind="contact"))])
    assert result is before
    assert store.commits == []


def test_apply_empty_batch_does_not_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    result = actor.apply([])
    assert result is actor.state
    assert store.commits == []


def test_unknown_field_raises_before_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    with pytest.raises(UnknownStateField):
        actor.apply([UpdateState({"u": 0.5}), UpdateState({"not_a_field": 1})])
    assert store.commits == []  # all-or-nothing: nothing committed


def test_actor_loads_initial_state_from_store() -> None:
    store = RecordingStore(State(u=0.9, tick_count=7))
    actor = StateActor(store)
    assert actor.state.u == 0.9
    assert actor.state.tick_count == 7


def test_injected_state_overrides_store_load() -> None:
    store = RecordingStore(State(u=0.1))
    actor = StateActor(store, state=State(u=0.4))
    assert actor.state.u == 0.4

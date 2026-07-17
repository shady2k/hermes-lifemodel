from __future__ import annotations

from collections.abc import Sequence

import pytest

from lifemodel.core.intents import (
    EmitSignal,
    FinalizeBuffer,
    PutRecord,
    TransitionRecord,
    UpdateState,
)
from lifemodel.core.state_actor import StateActor, UnknownStateField
from lifemodel.domain.memory import MemoryDraft, MemoryMutation, PutOp, TransitionOp
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


class RecordingStore:
    """Minimal StatePort + TickCommitPort double that records commits/mutations."""

    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []
        self.tick_calls: list[tuple[State | None, list[MemoryMutation]]] = []
        self.finalize_calls: list[str | None] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)

    def reset(self) -> State:
        self._state = State()
        self.commits.append(self._state)
        return self._state

    def commit_tick(
        self,
        state: State | None,
        mutations: Sequence[MemoryMutation],
        *,
        finalize_survey_id: str | None = None,
    ) -> None:
        self.tick_calls.append((state, list(mutations)))
        self.finalize_calls.append(finalize_survey_id)
        if state is not None:
            self.commit(state)


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


# --- lm-27n.2: the atomic State+memory committer ---


def _put(id: str) -> PutRecord:
    return PutRecord(
        PutOp(MemoryDraft(kind="desire", id=id, state="active", payload={}, source="t"))
    )


def _transition(id: str) -> TransitionRecord:
    return TransitionRecord(
        TransitionOp(kind="desire", id=id, from_state="active", to_state="archived")
    )


def test_apply_collects_state_and_mutations_into_one_commit_tick() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([_put("a"), UpdateState({"u": 0.5}), _transition("a")])

    assert len(store.tick_calls) == 1  # exactly one atomic commit
    committed_state, mutations = store.tick_calls[0]
    assert committed_state is not None and committed_state.u == 0.5
    # Mutations preserved in emission order (put before transition).
    assert isinstance(mutations[0], PutOp)
    assert isinstance(mutations[1], TransitionOp)


def test_apply_mutation_only_commits_with_none_state() -> None:
    # A tick with mutations but no state patch still commits (something changed),
    # passing state=None so the state row/revision is untouched.
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([_put("solo")])

    assert len(store.tick_calls) == 1
    committed_state, mutations = store.tick_calls[0]
    assert committed_state is None
    assert len(mutations) == 1
    assert store.commits == []  # state row never rewritten


def test_apply_no_state_and_no_mutations_does_not_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([EmitSignal(Signal(origin_id="n", kind="contact"))])
    assert store.tick_calls == []


def test_unknown_field_raises_before_commit_even_with_mutations() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    with pytest.raises(UnknownStateField):
        actor.apply([_put("a"), UpdateState({"not_a_field": 1})])
    assert store.tick_calls == []  # all-or-nothing: nothing committed


def test_apply_state_only_passes_empty_mutations() -> None:
    # Behavior-neutral: a state-only tick commits new_state with an empty batch.
    store = RecordingStore()
    actor = StateActor(store)
    result = actor.apply([UpdateState({"u": 0.5})])

    assert result.u == 0.5
    committed_state, mutations = store.tick_calls[0]
    assert committed_state is not None and committed_state.u == 0.5
    assert mutations == []


# --- lm-705.13: FinalizeBuffer threads the survey_id into the atomic commit ---


def test_apply_threads_finalize_survey_id_alongside_thoughts() -> None:
    # A genuine noticing pass: a thought put + a consumed-ring patch + the
    # FinalizeBuffer, all collected into the ONE commit_tick so the claimed-row
    # DELETE lands atomically with the thought (codex I3).
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([_put("thought:x"), UpdateState({"u": 0.5}), FinalizeBuffer("t7@iso")])

    assert len(store.tick_calls) == 1  # exactly one atomic commit
    committed_state, mutations = store.tick_calls[0]
    assert committed_state is not None and committed_state.u == 0.5
    assert isinstance(mutations[0], PutOp)
    assert store.finalize_calls == ["t7@iso"]


def test_apply_finalize_only_still_commits() -> None:
    # A genuinely-surveyed-but-fruitless pass emits ONLY a FinalizeBuffer (no
    # state patch, no memory mutation) — it must STILL commit so the cursor
    # advances and the segment is never re-shown forever. state=None: the row is
    # untouched, but the finalize DELETE still runs.
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([FinalizeBuffer("t1@iso")])

    assert len(store.tick_calls) == 1
    committed_state, mutations = store.tick_calls[0]
    assert committed_state is None
    assert mutations == []
    assert store.finalize_calls == ["t1@iso"]
    assert store.commits == []  # state row never rewritten


def test_apply_without_finalize_passes_none() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    actor.apply([UpdateState({"u": 0.5})])
    assert store.finalize_calls == [None]


def test_state_actor_requires_a_committer_or_committing_store() -> None:
    class StateOnly:
        def load(self) -> State:
            return State()

        def commit(self, state: State) -> None: ...

        def reset(self) -> State:
            return State()

    with pytest.raises(TypeError):
        StateActor(StateOnly())  # type: ignore[arg-type]


def test_state_actor_accepts_a_separate_committer() -> None:
    class StateOnly:
        def load(self) -> State:
            return State(u=0.2)

        def commit(self, state: State) -> None: ...

        def reset(self) -> State:
            return State()

    committer = RecordingStore()
    actor = StateActor(StateOnly(), committer=committer)  # type: ignore[arg-type]
    actor.apply([UpdateState({"u": 0.9})])
    assert committer.tick_calls[0][0] is not None

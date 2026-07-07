"""StateActor — the single owner of model-state mutation (spec §6, §7.1, §15).

Every other producer (neurons, layers, Hermes hooks, cron) only *returns* or
*enqueues* intents; this is the one place that mutates :class:`State` and calls
:meth:`StatePort.commit`. It merges a batch of :class:`UpdateState` intents into
one patch, applies it atomically with :func:`dataclasses.replace`, and commits
(checkpoints) exactly once — and only if something actually changed (spec §6:
"Checkpoint — это интент, который state-actor генерирует сам в конце тика, если
были мутации"). Intents it does not own (e.g. ``EmitSignal``) are ignored here.

State is loaded lazily on first access (``.state`` or ``.apply``) so that
constructing an actor never raises — the composition root can wire one
without forcing a store read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from ..domain.memory import MemoryMutation
from ..log import EventLogger
from ..ports.tick_commit import TickCommitPort
from ..state.model import State
from ..state.port import StatePort
from .intents import Intent, PutRecord, TransitionRecord, UpdateState

_STATE_FIELDS = frozenset(f.name for f in fields(State))


class UnknownStateField(KeyError):
    """An ``UpdateState`` intent named a field that ``State`` does not declare."""


class StateActor:
    """The single owner of the tick's atomic State + memory commit.

    Collects the batch's :class:`UpdateState` patch *and* its ``PutRecord``/
    ``TransitionRecord`` mutations (in emission order) and hands both to the one
    :class:`~lifemodel.ports.tick_commit.TickCommitPort` in a single
    ``commit_tick`` — so vitals and entities move atomically (no split-brain,
    HLA §4.1). ``StatePort`` is kept only for the lazy initial load. The
    committer defaults to *store* (the live store implements both ports); tests
    inject a fake that satisfies both.
    """

    def __init__(
        self,
        store: StatePort,
        *,
        committer: TickCommitPort | None = None,
        state: State | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self._store = store
        if committer is not None:
            self._committer: TickCommitPort = committer
        elif isinstance(store, TickCommitPort):
            self._committer = store
        else:
            raise TypeError(
                "StateActor needs a committer, or a store that implements TickCommitPort"
            )
        self._provided_state = state
        self._state: State | None = None
        self._log = logger
        self._checkpoint_id = 0

    @property
    def state(self) -> State:
        """The current in-memory state (last committed, or the initial load)."""
        if self._state is None:
            self._state = (
                self._provided_state if self._provided_state is not None else self._store.load()
            )
        return self._state

    def apply(self, intents: Sequence[Intent]) -> State:
        """Apply a batch atomically. Commits once (one ``commit_tick``) iff the
        merged patch is non-empty OR there is >=1 memory mutation; validates all
        field names *before* committing (all-or-nothing)."""
        patch: dict[str, Any] = {}
        mutations: list[MemoryMutation] = []
        for intent in intents:
            if isinstance(intent, UpdateState):
                for name, value in intent.changes.items():
                    if name not in _STATE_FIELDS:
                        raise UnknownStateField(name)
                    patch[name] = value
            elif isinstance(intent, PutRecord | TransitionRecord):
                mutations.append(intent.op)
        if not patch and not mutations:
            return self.state

        # Rewrite the state row only when the patch changed something; a
        # mutation-only tick passes ``None`` so the row (and its revision) is
        # untouched. A state-only tick (``mutations == []``) is byte-identical to
        # the old single ``commit`` path.
        new_state = replace(self.state, **patch) if patch else self.state
        self._committer.commit_tick(new_state if patch else None, mutations)
        self._state = new_state
        self._checkpoint_id += 1
        if self._log is not None:
            self._log.info(
                "state_checkpoint",
                checkpoint_id=self._checkpoint_id,
                fields=sorted(patch),
            )
        return new_state

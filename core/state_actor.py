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

from ..log import EventLogger
from ..state.model import State
from ..state.port import StatePort
from .intents import Intent, UpdateState

_STATE_FIELDS = frozenset(f.name for f in fields(State))


class UnknownStateField(KeyError):
    """An ``UpdateState`` intent named a field that ``State`` does not declare."""


class StateActor:
    def __init__(
        self,
        store: StatePort,
        *,
        state: State | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self._store = store
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
        """Apply a batch atomically. Commits once iff the merged patch is
        non-empty; validates all field names *before* committing."""
        patch: dict[str, Any] = {}
        for intent in intents:
            if isinstance(intent, UpdateState):
                for name, value in intent.changes.items():
                    if name not in _STATE_FIELDS:
                        raise UnknownStateField(name)
                    patch[name] = value
        if not patch:
            return self.state

        new_state = replace(self.state, **patch)
        self._store.commit(new_state)
        self._state = new_state
        self._checkpoint_id += 1
        if self._log is not None:
            self._log.info(
                "state_checkpoint",
                checkpoint_id=self._checkpoint_id,
                fields=sorted(patch),
            )
        return new_state

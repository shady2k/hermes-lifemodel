"""``StatePort`` — the hexagon boundary for persisting the being's state (§13).

The core (neurons, aggregator, cognition) depends only on this Protocol, never
on a concrete store, so it stays host- and storage-agnostic and tests inject a
fake. The JSON file adapter (:mod:`lifemodel.state.json_store`) is one
implementation; a later phase could add another (e.g. SQLite for high-frequency
signal traces, HLA §4/D3) without touching the core.

Kept deliberately small: only the two operations Phase 1 needs. Richer ops
(short-lock snapshot read, conflict-checked commit — HLA §9) arrive with the
concurrency work in Phase 7.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .model import State


@runtime_checkable
class StatePort(Protocol):
    """The single writer/reader contract for the being's state (§9)."""

    def load(self) -> State:
        """Return the current persisted state, or a default on first run."""
        ...

    def commit(self, state: State) -> None:
        """Atomically persist *state* as the new source of truth."""
        ...

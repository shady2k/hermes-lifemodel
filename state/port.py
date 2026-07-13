"""``StatePort`` — the hexagon boundary for persisting the being's state (§13).

The core (neurons, aggregator, cognition) depends only on this Protocol, never
on a concrete store, so it stays host- and storage-agnostic and tests inject a
fake (:class:`~lifemodel.testing.fakes.FakeStateStore`). The live adapter is
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (lm-fib.6.2), which
persists ``State`` as one JSON blob in a SQLite singleton row — a settled
design (HLA §4.1/D7 v0.7): ``State`` still owns its own ``to_dict``/``from_dict``
validation and is actively reshaping, so typed-per-field columns would be a
migration treadmill. The retired ``JsonStateStore`` (a single ``state.json``
file) preceded it.

Kept deliberately small: only the operations the being's live wiring needs.
Richer ops (short-lock snapshot read, conflict-checked commit — HLA §9) arrive
with the concurrency work in Phase 7.
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

    def reset(self) -> State:
        """Factory-wipe the persisted state to a fresh default, returning it.

        Must succeed even when the previously-persisted state is unreadable
        (corrupt, or from an unsupported schema version) — it does not require
        a prior successful :meth:`load`.

        NOT what the owner-facing ``/lifemodel reset`` subcommand uses (Phase 4,
        spec §6.4): a factory wipe there commits a freshly-BORN body
        (:func:`~lifemodel.core.genesis.newborn`) via plain :meth:`commit` instead —
        this fresh-default's zero arousal is exactly the lifeless body that command
        exists to stop persisting. :meth:`commit` is itself an unconditional UPSERT
        (never a read-modify-write), which is what actually gives ``reset_for_dir``
        its "works on an unreadable row" guarantee now — not this method. Kept as a
        port capability in its own right (a bare factory-default reset, unit-tested
        directly in :mod:`tests.test_sqlite_store`), just no longer the one the owner
        command routes through.
        """
        ...

"""``TickCommitPort`` — the atomic State+memory end-of-tick committer (HLA §4.1).

A narrow, single-method port (HLA §13, "pragmatism, not ceremony") so the
:class:`~lifemodel.core.state_actor.StateActor` can persist the being's ``State``
*and* a batch of memory mutations in **one** transaction, without depending on
:class:`~lifemodel.ports.memory.MemoryPort`'s full CRUD surface. The HLA point
(§4.1): one adapter (:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`)
implements every port, so a single transaction spans the being's vitals
(``runtime_state``) and its entities (``memory_records``) — split-brain (state
advanced but memory dropped, or vice versa) is impossible by construction.

**All-or-nothing.** A stale transition mid-batch, or a serialization error, rolls
back *everything* — the state row included — and propagates. Returns ``None``: the
tick model has no in-tick read-your-writes (that is the future MemoryWorkspace;
this port deliberately does not preclude it). Tests inject a fake that applies to
its in-memory maps with the same all-or-nothing semantics, so fake and real agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..domain.memory import MemoryMutation
from ..state.model import State


@runtime_checkable
class TickCommitPort(Protocol):
    """Persist a tick's ``State`` change + memory mutations in one transaction."""

    def commit_tick(
        self,
        state: State | None,
        mutations: Sequence[MemoryMutation],
        *,
        finalize_survey_id: str | None = None,
    ) -> None:
        """Apply *state* (if not ``None``) then each mutation in list order,
        atomically. When *finalize_survey_id* is not ``None``, the noticing
        conversation-buffer rows claimed under it are dropped in the SAME
        transaction (lm-705.13, codex I3) — so a noticing pass's thoughts and its
        cursor-advance commit or roll back together. Raises on a stale transition
        or serialization error, rolling back the whole batch (state, mutations, and
        finalize alike). Returns ``None``."""
        ...

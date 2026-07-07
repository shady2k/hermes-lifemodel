"""``MemoryPort`` — the boundary for durable memory-record CRUD (HLA §4.1/D7).

The being's growing memory (desires, facts, open loops — HLA §4) lives behind
this Protocol, never behind a concrete store, so the core stays storage-
agnostic and unit-testable off-host (HLA §13). The real implementation is
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`; tests inject
:class:`~lifemodel.testing.fakes.FakeMemoryStore`. Purely additive as of
lm-fib.6.1 — nothing in the live tick depends on this port yet.

Records are entities identified by ``(kind, id)`` (e.g. ``kind="desire"``) that
move through a caller-defined ``state`` machine (e.g. ``"active"`` ->
``"archived"``); :meth:`MemoryPort.transition` is the sole guarded state
change, giving callers optimistic-concurrency-style safety without a generic
compare-and-swap API.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from ..domain.memory import MemoryDraft, MemoryPatch, MemoryRecord

#: The ``find`` sort orders every ``MemoryPort`` implementation must support,
#: each deterministic via an ``id ASC`` tiebreak (HLA §4.1).
OrderBy = Literal["updated_desc", "created_desc", "salience_desc"]


@runtime_checkable
class MemoryPort(Protocol):
    """CRUD + guarded state transitions over memory records (§4.1)."""

    def put(self, draft: MemoryDraft) -> str:
        """Upsert *draft* keyed by ``(kind, id)``. Returns the record's ``id``.

        On insert: ``created_at == updated_at == now`` (from the store's
        ``ClockPort``) and ``revision == 0``. On conflict (a row already
        exists for ``(kind, id)``): ``created_at`` is kept from the original
        insert, ``updated_at`` is bumped to ``now``, and ``revision`` is
        incremented — every other field is replaced wholesale with *draft*'s.
        """
        ...

    def get(self, kind: str, id: str) -> MemoryRecord | None:
        """Return the record for ``(kind, id)``, or ``None`` if it does not exist."""
        ...

    def find(
        self,
        kind: str | None = None,
        state: str | None = None,
        limit: int | None = None,
        order_by: OrderBy = "updated_desc",
    ) -> list[MemoryRecord]:
        """Return records matching the given filters, in deterministic order.

        ``kind``/``state`` are optional equality filters (``None`` = no
        filter on that field); both may be combined. ``limit`` caps the result
        length when given. Ordering is always deterministic, tiebroken by
        ``id`` ascending: ``updated_desc`` = updated-time descending,
        ``created_desc`` = created-time descending, ``salience_desc`` =
        salience descending.
        """
        ...

    def transition(
        self,
        kind: str,
        id: str,
        from_state: str,
        to_state: str,
        patch: MemoryPatch | None = None,
    ) -> MemoryRecord:
        """Guarded state change: only applies if the record is in ``from_state``.

        Equivalent to ``UPDATE ... WHERE kind=? AND id=? AND state=from_state``:
        if no record matches — because it does not exist, or it exists but is
        in a different state — nothing is written and
        :class:`~lifemodel.domain.memory.StaleTransition` is raised. On
        success: ``state`` becomes *to_state*, *patch* is applied
        (``payload_merge`` shallow-merges into the existing payload; every
        other non-``None`` patch field replaces the existing value), and
        ``revision``/``updated_at`` are bumped. Soft-delete is the convention
        ``to_state="archived"`` — there is no hard delete on this port.
        """
        ...

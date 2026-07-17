"""``BufferStore`` — the durable-backing port for the noticing conversation buffer
(design §4.1/§4.2, lm-705.14/lm-705.13).

Hermes-free, stdlib-only (mirrors the ``ports/`` boundary discipline, HLA §13).
:class:`~lifemodel.core.noticing_buffer.NoticingBuffer` is the process-owned
buffer used by the hooks/adapter; it currently owns its ``pending``/``complete``
state directly (``dict``/``deque``), which a plugin/gateway restart wipes clean.
A later task in this plan (lm-705.14 Task 2) makes it delegate persistence to a
:class:`BufferStore` instead, so the captured-but-not-yet-noticed conversation
survives a restart. This module only defines the port and its value type —
:class:`~lifemodel.state.sqlite_store.SqliteBufferStore` (the durable,
SQLite-backed implementation) lives at the store layer (``state/sqlite_store.py``,
the one place allowed to import ``sqlite3``), mirroring the existing split
between :class:`~lifemodel.ports.memory.MemoryPort` and
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`.

**Lifecycle (design §4.1/§4.2).** Per session ("lane") there is at most one open
``pending`` turn: :meth:`BufferStore.open_pending`/:meth:`BufferStore.stamp_source`
build it up, :meth:`BufferStore.complete` moves it into the ordered ``complete``
set returned by :meth:`BufferStore.completed` — **iff** the lane is closed (the
closed-prefix rule: a lane with a live pending yields no segment, so a noticing
pass never surveys mid-turn). A pending that outlives its TTL (a dropped/crashed
turn) is abandoned the next time :meth:`BufferStore.completed` observes it, so
one lost turn can never wedge a lane shut forever. :meth:`BufferStore.abandon_pending`
lets a caller decline an open pending explicitly (e.g. an empty assistant reply)
rather than waiting out the TTL.

**Claim/finalize (codex I2/I3, lm-705.13).** A noticing pass claims the surveyed
prefix under a ``survey_id`` via :meth:`BufferStore.claim` — an immutable
snapshot immune to further ring/store pressure — and :meth:`BufferStore.claimed`
returns exactly that snapshot regardless of what has completed since.
:meth:`BufferStore.finalize` drops the claimed rows (the durable half of a
successful pass, applied atomically with the thought commit by a later task);
:meth:`BufferStore.release` returns them to ``complete`` (a transient failure, so
the segment is re-surveyed later); :meth:`BufferStore.recover_stale_claims`
releases every outstanding claim at boot (a pass that died mid-flight with the
process).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BufferEntry:
    """One completed conversation turn, eligible for a closed-prefix segment."""

    session_id: str
    turn_id: str
    source_ids: tuple[str, ...]
    user_text: str
    assistant_text: str
    ts: str


@runtime_checkable
class BufferStore(Protocol):
    """The persistence boundary the noticing conversation buffer delegates to.

    Two implementations: durable
    (:class:`~lifemodel.state.sqlite_store.SqliteBufferStore`, over the
    ``conversation_buffer`` table) and, from a later task, an in-memory fake
    carrying today's ``NoticingBuffer`` logic verbatim.
    """

    def open_pending(self, session_id: str, *, user_text: str, now: datetime) -> None:
        """Open (or refresh) *session_id*'s single pending slot.

        Refreshing replaces any prior pending outright — a fresh inbound turn
        supersedes whatever was there; the platform never opens two turns on
        one lane at once.
        """
        ...

    def stamp_source(self, session_id: str, message_id: str) -> None:
        """Append a platform message id to the open pending slot; no-op if none."""
        ...

    def complete(
        self, session_id: str, turn_id: str, *, assistant_text: str, now: datetime
    ) -> None:
        """Move the pending slot into a durable ``complete`` entry; no-op if none open.

        Defensive: a ``complete`` with no matching ``open_pending`` (e.g. a
        duplicate/late callback) does nothing rather than fabricate an entry
        with no real user turn behind it.
        """
        ...

    def abandon_pending(self, session_id: str) -> None:
        """Drop *session_id*'s pending slot, if any; a no-op otherwise."""
        ...

    def completed(self, session_id: str, *, now: datetime, ttl: timedelta) -> list[BufferEntry]:
        """The ordered ``complete`` entries for *session_id*, iff its lane is closed.

        FIRST TTL-abandons a stale pending (``now - opened_at > ttl``), dropping
        it, so a turn that never completes can't wedge the lane shut forever.
        THEN applies the closed-prefix rule: any pending still open (fresh,
        within ``ttl``) yields ``[]`` — a noticing pass must never survey
        mid-turn.
        """
        ...

    def claim(self, session_id: str, turn_ids: tuple[str, ...], survey_id: str) -> None:
        """Mark the given ``complete`` *turn_ids* ``claimed`` under *survey_id*.

        Claimed rows leave :meth:`completed`'s result (immune to further ring
        pressure) and become visible via :meth:`claimed`. A *turn_ids* entry
        that is not currently ``complete`` (already claimed, or unknown) is
        silently skipped.
        """
        ...

    def claimed(self, survey_id: str) -> list[BufferEntry]:
        """The ordered entries claimed under *survey_id* — the immutable
        snapshot a noticing pass actually surveyed, regardless of any ring/store
        pressure since the claim."""
        ...

    def finalize(self, survey_id: str) -> None:
        """Drop the rows claimed under *survey_id* — the durable half of a
        successful noticing pass's atomic commit."""
        ...

    def release(self, survey_id: str) -> None:
        """Return the rows claimed under *survey_id* to ``complete`` (un-claim)
        — a transient noticing failure, so the segment is re-surveyed later."""
        ...

    def recover_stale_claims(self) -> None:
        """Release every ``claimed`` row back to ``complete`` — boot recovery for
        a noticing pass that died mid-flight with the process."""
        ...

    def session_ids(self) -> list[str]:
        """Every session lane this store currently knows of, sorted for a
        deterministic iteration order."""
        ...

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

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from .timeutil import to_iso

#: Default per-session cap on the ``complete`` ring (design §4.1). Shared by
#: :class:`InMemoryBufferStore`, :class:`~lifemodel.core.noticing_buffer.NoticingBuffer`,
#: and the durable :class:`~lifemodel.state.sqlite_store.SqliteBufferStore` so the
#: in-memory and durable backings can never drift on how much captured-but-not-yet-
#: noticed conversation a lane retains before the oldest turns are dropped.
DEFAULT_BUFFER_MAX_ENTRIES = 256


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


@dataclass
class _PendingTurn:
    """A session's single in-flight turn — at most one per lane."""

    user_text: str
    opened_at: datetime
    source_ids: list[str] = field(default_factory=list)


class InMemoryBufferStore:
    """In-process :class:`BufferStore` fake — the exact pending/complete/
    claimed logic that used to live directly on
    :class:`~lifemodel.core.noticing_buffer.NoticingBuffer` (lm-705.5), moved
    here VERBATIM (lm-705.14 Task 2) so the SAME buffer API can be backed by
    either this (every existing test, and today's production default) or
    :class:`~lifemodel.state.sqlite_store.SqliteBufferStore` (durable — a
    plugin/gateway restart wipes THIS store clean; that gap is exactly what
    the durable store closes).

    Two sessions are fully independent: each has its own optional pending
    slot and its own bounded ``complete`` ring (``max_entries`` applies per
    session). One :class:`threading.Lock` guards every mutation, mirroring
    ``NoticingBuffer``'s own former lock one-for-one.
    """

    def __init__(self, *, max_entries: int = DEFAULT_BUFFER_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingTurn] = {}
        self._complete: dict[str, deque[BufferEntry]] = {}
        self._claimed: dict[str, list[BufferEntry]] = {}

    # ---- pending lifecycle -------------------------------------------------

    def open_pending(self, session_id: str, *, user_text: str, now: datetime) -> None:
        with self._lock:
            self._pending[session_id] = _PendingTurn(user_text=user_text, opened_at=now)

    def stamp_source(self, session_id: str, message_id: str) -> None:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is not None:
                pending.source_ids.append(message_id)

    def complete(
        self, session_id: str, turn_id: str, *, assistant_text: str, now: datetime
    ) -> None:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is None:
                return
            ts = to_iso(now)  # validate BEFORE mutating anything (M3, ported verbatim)
            del self._pending[session_id]
            entry = BufferEntry(
                session_id=session_id,
                turn_id=turn_id,
                source_ids=tuple(pending.source_ids),
                user_text=pending.user_text,
                assistant_text=assistant_text,
                ts=ts,
            )
            ring = self._complete.get(session_id)
            if ring is None:
                ring = deque(maxlen=self._max_entries)
                self._complete[session_id] = ring
            ring.append(entry)

    def abandon_pending(self, session_id: str) -> None:
        with self._lock:
            self._pending.pop(session_id, None)

    def completed(self, session_id: str, *, now: datetime, ttl: timedelta) -> list[BufferEntry]:
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is not None and now - pending.opened_at > ttl:
                del self._pending[session_id]
                pending = None
            if pending is not None:
                return []
            ring = self._complete.get(session_id)
            if ring is None:
                return []
            return list(ring)

    # ---- claim / finalize / release lifecycle (lm-705.13) ------------------

    def claim(self, session_id: str, turn_ids: tuple[str, ...], survey_id: str) -> None:
        if not turn_ids:
            return
        wanted = set(turn_ids)
        with self._lock:
            ring = self._complete.get(session_id)
            if ring is None:
                return
            remaining: list[BufferEntry] = []
            claimed_entries: list[BufferEntry] = []
            for entry in ring:
                (claimed_entries if entry.turn_id in wanted else remaining).append(entry)
            if not claimed_entries:
                return  # none of turn_ids were actually `complete` -- silently skipped
            ring.clear()
            ring.extend(remaining)
            self._claimed.setdefault(survey_id, []).extend(claimed_entries)

    def claimed(self, survey_id: str) -> list[BufferEntry]:
        with self._lock:
            return list(self._claimed.get(survey_id, []))

    def finalize(self, survey_id: str) -> None:
        with self._lock:
            self._claimed.pop(survey_id, None)

    def release(self, survey_id: str) -> None:
        with self._lock:
            entries = self._claimed.pop(survey_id, None)
            if entries:
                self._restore_locked(entries)

    def recover_stale_claims(self) -> None:
        with self._lock:
            all_entries = [entry for entries in self._claimed.values() for entry in entries]
            self._claimed.clear()
            if all_entries:
                self._restore_locked(all_entries)

    def _restore_locked(self, entries: list[BufferEntry]) -> None:
        """Return *entries* to their session's ``complete`` ring, ahead of
        anything that completed while they were claimed away — a claim only
        ever takes a closed PREFIX, so a released/recovered entry is always
        chronologically earlier than whatever the ring holds now. Callers
        MUST already hold ``self._lock``."""
        by_session: dict[str, list[BufferEntry]] = {}
        for entry in entries:
            by_session.setdefault(entry.session_id, []).append(entry)
        for session_id, session_entries in by_session.items():
            ring = self._complete.get(session_id)
            if ring is None:
                ring = deque(maxlen=self._max_entries)
                self._complete[session_id] = ring
            ring.extendleft(reversed(session_entries))

    def session_ids(self) -> list[str]:
        with self._lock:
            return sorted(set(self._pending) | set(self._complete))

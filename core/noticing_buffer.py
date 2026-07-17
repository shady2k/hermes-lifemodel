"""``NoticingBuffer`` — the process-owned conversation buffer (design §4.1, lm-705.5).

A **single process-owned, lock-protected** service — injected into the hooks
*and* the adapter (Task 3), NOT a field on a freshly-built ``LifeModel`` (graphs
are rebuilt per call, so a per-graph buffer would lose every in-flight turn).

Per **session** ("lane") there is at most ONE open ``pending`` slot — a turn in
flight. ``open_pending``/``stamp_source`` build it up; ``complete`` moves it
into a bounded per-session ring of ``complete`` :class:`BufferEntry` values.
The **closed-prefix rule** (design §4.2): a lane with a live ``pending`` yields
NO segment from :meth:`NoticingBuffer.closed_segment` — a noticing pass must
never survey mid-turn (a long tool-heavy reply must not be read before its
``post_llm``). A ``pending`` that outlives ``pending_ttl`` (a dropped/crashed
turn) ages to ``abandoned`` and is dropped the moment it is next observed, so
one lost turn can never wedge a lane shut forever.

Pure, stdlib-only (``threading``, ``collections.deque``, ``datetime``) — no
Hermes, no ``LifeModel``. The only intra-repo import is :mod:`.timeutil`, for
the same fixed-width UTC ISO stamp every other durable timestamp in this
codebase uses (`ts` on :class:`BufferEntry`), which is why *now* must always be
an aware ``datetime`` (:func:`~lifemodel.core.timeutil.to_iso` rejects a naive
one) — the same convention the ``ClockPort`` boundary already enforces
elsewhere, so no caller should ever be passing a naive one anyway.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .timeutil import to_iso


@dataclass(frozen=True)
class BufferEntry:
    """One completed conversation turn, eligible for a closed-prefix segment."""

    session_id: str
    turn_id: str
    source_ids: tuple[str, ...]
    user_text: str
    assistant_text: str
    ts: str


@dataclass
class _PendingTurn:
    """A session's single in-flight turn — at most one per lane."""

    user_text: str
    opened_at: datetime
    source_ids: list[str] = field(default_factory=list)


class NoticingBuffer:
    """Per-session pending→complete ring, one :class:`threading.Lock` for all mutation.

    Two sessions are fully independent: each has its own optional pending slot
    and its own bounded ``complete`` ring (``max_entries`` applies per session).
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
        pending_ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        self._max_entries = max_entries
        self._pending_ttl = pending_ttl
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingTurn] = {}
        self._complete: dict[str, deque[BufferEntry]] = {}

    def open_pending(self, session_id: str, *, user_text: str, now: datetime) -> None:
        """Open (or refresh) *session_id*'s single pending slot.

        Refreshing replaces any prior pending outright (a fresh inbound turn
        supersedes whatever was there — the platform never opens two turns on
        one lane at once).
        """
        with self._lock:
            self._pending[session_id] = _PendingTurn(user_text=user_text, opened_at=now)

    def stamp_source(self, session_id: str, message_id: str) -> None:
        """Append a platform message id to the open pending slot; no-op if none."""
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is not None:
                pending.source_ids.append(message_id)

    def complete(
        self, session_id: str, turn_id: str, *, assistant_text: str, now: datetime
    ) -> None:
        """Move the pending slot into a ``complete`` ring entry; no-op if none open.

        Defensive: a ``complete`` with no matching ``open_pending`` (e.g. a
        duplicate/late callback) does nothing rather than fabricate an entry
        with no real user turn behind it.
        """
        with self._lock:
            pending = self._pending.pop(session_id, None)
            if pending is None:
                return
            entry = BufferEntry(
                session_id=session_id,
                turn_id=turn_id,
                source_ids=tuple(pending.source_ids),
                user_text=pending.user_text,
                assistant_text=assistant_text,
                ts=to_iso(now),
            )
            ring = self._complete.get(session_id)
            if ring is None:
                ring = deque(maxlen=self._max_entries)
                self._complete[session_id] = ring
            ring.append(entry)

    def closed_segment(self, session_id: str, *, now: datetime) -> list[BufferEntry]:
        """The ordered ``complete`` entries for *session_id*, iff its lane is closed.

        FIRST ages a stale pending (``now - opened_at > pending_ttl``) to
        abandoned, dropping it, so a turn that never completes can't wedge the
        lane shut forever. THEN applies the closed-prefix rule: any pending
        still open (fresh, within TTL) yields ``[]`` — never survey mid-turn.
        """
        with self._lock:
            pending = self._pending.get(session_id)
            if pending is not None and now - pending.opened_at > self._pending_ttl:
                del self._pending[session_id]
                pending = None
            if pending is not None:
                return []
            ring = self._complete.get(session_id)
            if ring is None:
                return []
            return list(ring)

    def session_ids(self) -> list[str]:
        """Every session lane the buffer currently knows of (an open pending, a
        non-empty ``complete`` ring, or both), lock-guarded and sorted for a
        deterministic iteration order. There is no separate "live sessions"
        registry elsewhere — a caller that needs to sweep every lane (e.g.
        :class:`~lifemodel.core.noticing.NoticingTrigger`) reads this rather
        than track session ids itself.
        """
        with self._lock:
            return sorted(set(self._pending) | set(self._complete))

    def segment_through(self, session_id: str, turn_id: str) -> list[BufferEntry]:
        """The ``complete`` ring PREFIX for *session_id* up to and including
        *turn_id* — unlike :meth:`closed_segment`, this does NOT apply the
        closed-prefix (pending) gate.

        For a caller re-deriving what an EARLIER :meth:`closed_segment` read
        already gated once (a noticing pass's async completion, recovering the
        exact segment its own trigger surveyed): a NEW pending opening on this
        lane in the meantime must not retroactively hide entries that were
        already safely closed at trigger time. Mirrors :meth:`clear_through`'s
        own scan so "what was surveyed" and "what gets cleared" never drift
        apart. Empty if *turn_id* is not found (already cleared, or never
        present) — the caller treats that as nothing to do.
        """
        with self._lock:
            ring = self._complete.get(session_id)
            if ring is None:
                return []
            entries = list(ring)
            for i, entry in enumerate(entries):
                if entry.turn_id == turn_id:
                    return entries[: i + 1]
            return []

    def clear_through(self, session_id: str, turn_id: str) -> None:
        """Cursor: drop *session_id*'s ``complete`` entries up to and including *turn_id*.

        Entries after it (a newer turn the surveyed pass did not consume)
        survive. A *turn_id* not found in the ring (already cleared, or never
        present) is a no-op — there is nothing to advance the cursor past.
        """
        with self._lock:
            ring = self._complete.get(session_id)
            if ring is None:
                return
            entries = list(ring)
            for i, entry in enumerate(entries):
                if entry.turn_id == turn_id:
                    ring.clear()
                    ring.extend(entries[i + 1 :])
                    return

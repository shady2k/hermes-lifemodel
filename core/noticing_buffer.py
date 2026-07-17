"""``NoticingBuffer`` — the process-owned conversation buffer (design §4.1, lm-705.5).

A **single process-owned service** — injected into the hooks *and* the adapter
(Task 3), NOT a field on a freshly-built ``LifeModel`` (graphs are rebuilt per
call, so a per-graph buffer would lose every in-flight turn).

Per **session** ("lane") there is at most ONE open ``pending`` slot — a turn in
flight. ``open_pending``/``stamp_source`` build it up; ``complete`` moves it
into a bounded per-session ring of ``complete`` :class:`BufferEntry` values.
The **closed-prefix rule** (design §4.2): a lane with a live ``pending`` yields
NO segment from :meth:`NoticingBuffer.closed_segment` — a noticing pass must
never survey mid-turn (a long tool-heavy reply must not be read before its
``post_llm``). A ``pending`` that outlives ``pending_ttl`` (a dropped/crashed
turn) ages to ``abandoned`` and is dropped the moment it is next observed, so
one lost turn can never wedge a lane shut forever.

**Delegation (lm-705.14 Task 2).** ``NoticingBuffer`` itself holds NO buffer
state and no lock of its own anymore — every method is a thin pass-through to
an injected :class:`~lifemodel.core.buffer_store.BufferStore`, which owns the
actual pending/complete/claimed data and its own lock-guarded mutation. The
default store (``store=None``) is
:class:`~lifemodel.core.buffer_store.InMemoryBufferStore` — the exact
dict/deque logic this class used to own directly, moved verbatim — so every
existing caller (today's production wiring, and the whole existing test suite)
sees byte-identical behaviour. Injecting
:class:`~lifemodel.state.sqlite_store.SqliteBufferStore` instead makes the SAME
API durable: the captured-but-not-yet-noticed conversation survives a
plugin/gateway restart.

Pure, stdlib-only (``datetime``) — no Hermes, no ``LifeModel``. The only
intra-repo import is :mod:`.buffer_store`, for :class:`BufferEntry` and the
:class:`BufferStore` port/default fake.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .buffer_store import (
    DEFAULT_BUFFER_MAX_ENTRIES,
    BufferEntry,
    BufferStore,
    InMemoryBufferStore,
)

__all__ = ["BufferEntry", "NoticingBuffer"]


class NoticingBuffer:
    """Per-session pending→complete API, delegating all persistence to an
    injected :class:`BufferStore` (lm-705.14 Task 2).

    Two sessions are fully independent: each has its own optional pending slot
    and its own bounded ``complete`` ring (``max_entries`` applies per session,
    for the default in-memory store).
    """

    def __init__(
        self,
        *,
        store: BufferStore | None = None,
        max_entries: int = DEFAULT_BUFFER_MAX_ENTRIES,
        pending_ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        # ``max_entries`` bounds ONLY the default in-memory store built here; an
        # INJECTED store owns its own cap (the durable ``SqliteBufferStore`` prunes
        # its ``complete`` rows to its own ``max_entries``), so this arg is ignored
        # when ``store`` is not None.
        self._store: BufferStore = (
            store if store is not None else InMemoryBufferStore(max_entries=max_entries)
        )
        self._pending_ttl = pending_ttl

    def open_pending(self, session_id: str, *, user_text: str, now: datetime) -> None:
        """Open (or refresh) *session_id*'s single pending slot.

        Refreshing replaces any prior pending outright (a fresh inbound turn
        supersedes whatever was there — the platform never opens two turns on
        one lane at once).
        """
        self._store.open_pending(session_id, user_text=user_text, now=now)

    def stamp_source(self, session_id: str, message_id: str) -> None:
        """Append a platform message id to the open pending slot; no-op if none."""
        self._store.stamp_source(session_id, message_id)

    def complete(
        self, session_id: str, turn_id: str, *, assistant_text: str, now: datetime
    ) -> None:
        """Move the pending slot into a ``complete`` entry; no-op if none open.

        Defensive: a ``complete`` with no matching ``open_pending`` (e.g. a
        duplicate/late callback) does nothing rather than fabricate an entry
        with no real user turn behind it. *now* is validated (rejecting a
        tz-naive value) BEFORE the pending slot is popped (M3) — a bad clock
        call fails loud with the pending still intact for a later, valid
        retry. Enforced by the store implementation.
        """
        self._store.complete(session_id, turn_id, assistant_text=assistant_text, now=now)

    def closed_segment(self, session_id: str, *, now: datetime) -> list[BufferEntry]:
        """The ordered ``complete`` entries for *session_id*, iff its lane is closed.

        FIRST ages a stale pending (older than ``pending_ttl``) to abandoned,
        dropping it, so a turn that never completes can't wedge the lane shut
        forever. THEN applies the closed-prefix rule: any pending still open
        (fresh, within TTL) yields ``[]`` — never survey mid-turn.
        """
        return self._store.completed(session_id, now=now, ttl=self._pending_ttl)

    def abandon_pending(self, session_id: str) -> None:
        """Drop *session_id*'s pending slot, if any; a no-op otherwise (review-2 G2).

        For a caller that opened a pending turn (``open_pending``) and then, at
        ``post_llm``, DECLINES it — an empty assistant response, or the turn
        turning out not to be a genuine reactive exchange after all — rather
        than completing it. Without this, the declined pending would otherwise
        only clear via :meth:`closed_segment`'s stale-pending aging, silently
        blocking the WHOLE lane (the closed-prefix rule) for up to
        ``pending_ttl`` even though nothing is actually in flight on it
        anymore.
        """
        self._store.abandon_pending(session_id)

    def session_ids(self) -> list[str]:
        """Every session lane the buffer currently knows of (an open pending, a
        non-empty ``complete`` ring, or both), sorted for a deterministic
        iteration order. There is no separate "live sessions" registry
        elsewhere — a caller that needs to sweep every lane (e.g.
        :class:`~lifemodel.core.noticing.NoticingTrigger`) reads this rather
        than track session ids itself.
        """
        return self._store.session_ids()

    # ---- claim / finalize / release lifecycle (lm-705.13, wired from Task 3) -

    def claim(self, session_id: str, turn_ids: tuple[str, ...], survey_id: str) -> None:
        """Mark the given ``complete`` *turn_ids* ``claimed`` under *survey_id*
        — see :meth:`~lifemodel.core.buffer_store.BufferStore.claim`."""
        self._store.claim(session_id, turn_ids, survey_id)

    def claimed(self, survey_id: str) -> list[BufferEntry]:
        """The ordered entries claimed under *survey_id* — the immutable
        snapshot a noticing pass actually surveyed, regardless of any
        ring/store pressure since the claim. See
        :meth:`~lifemodel.core.buffer_store.BufferStore.claimed`."""
        return self._store.claimed(survey_id)

    def finalize(self, survey_id: str) -> None:
        """Drop the rows claimed under *survey_id* — the durable half of a
        successful noticing pass's atomic commit. See
        :meth:`~lifemodel.core.buffer_store.BufferStore.finalize`."""
        self._store.finalize(survey_id)

    def release(self, survey_id: str) -> None:
        """Return the rows claimed under *survey_id* to ``complete`` (un-claim)
        — a transient noticing failure, so the segment is re-surveyed later.
        See :meth:`~lifemodel.core.buffer_store.BufferStore.release`."""
        self._store.release(survey_id)

    def recover_stale_claims(self) -> None:
        """Release every outstanding claim — boot recovery for a noticing pass
        that died mid-flight with the process. See
        :meth:`~lifemodel.core.buffer_store.BufferStore.recover_stale_claims`."""
        self._store.recover_stale_claims()

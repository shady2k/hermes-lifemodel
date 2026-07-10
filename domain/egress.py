"""``ReachOutcome`` — the result of one proactive reach-out attempt (HLA §7/§9).

A small enumerated result the egress port returns so callers (the proactive
service loop, the cron fallback) can react without catching exceptions. It is
PURELY the delivery boundary: a *delivered* native turn, an *unavailable*
primitive, or a fail-closed *failed* attempt. It is NOT a "did the being stay
silent" verdict — a quiet tick (no launch, or a backstop-held launch) produces no
``ReachOutcome`` at all (the reason is a logged suppression span, spec §5); and
even ``DELIVERED`` only means the turn was queued — whether the being actually
spoke is the async ``proactive_outcome`` read-back from ``post_llm_call``
(spec §9). Pure stdlib value type; imports nothing from Hermes.
"""

from __future__ import annotations

from enum import Enum


class ReachOutcome(Enum):
    """Result of one proactive reach-out attempt (HLA §7/§9, egress boundary)."""

    DELIVERED = "delivered"  # native reach-in turn injected into the live session
    UNAVAILABLE = "unavailable"  # reach-in primitive not available (no runner / version drift)
    FAILED = "failed"  # attempted but errored (already logged, fail-closed)

    @property
    def ok(self) -> bool:
        """True only when a native proactive turn was actually delivered (queued).

        Note (spec §9): ``ok`` means the turn reached the live session's queue —
        NOT that the being spoke. The real "spoke vs stayed silent" verdict is the
        async ``proactive_outcome`` read-back from ``post_llm_call``."""
        return self is ReachOutcome.DELIVERED


class ProactiveOutcome(Enum):
    """The outcome of a woken, launched proactive turn — the efference copy (spec §5/§6).

    Not a "verdict" but the *fact of what I did* (spec §5): the being's own Hermes
    turn either spoke (``SENT`` — a message went out), chose silence (``SILENT`` — a
    ``[SILENT]`` marker), failed to deliver (``FAILED``), or went stale/superseded
    while it was composing (``STALE``). The outcome arrives as a ``proactive_outcome``
    signal in its OWN ExecutionFrame the moment the async turn finishes (the
    ``post_llm_call`` read-back in :mod:`lifemodel.hooks`), and aggregation resolves
    the pending desire from it — immediately, not at the next heartbeat (spec §3).
    ``SENT`` starts an ActionPending window but does NOT satiate ``u`` (send ≠
    contact); ``SILENT`` records a decline + growing backoff."""

    SENT = "sent"  # the being sent a message → contact happened (u unsatiated)
    SILENT = "silent"  # chose silence → clear it (+ growing decline backoff)
    FAILED = "failed"  # the turn could not be delivered → clear it, no backoff
    STALE = "stale"  # superseded while composing (user replied) → clear it, no backoff

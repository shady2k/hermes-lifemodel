"""``ReachOutcome`` — the result of one proactive reach-out attempt (HLA §7).

A small enumerated result the egress port returns so callers (the proactive
service loop, the cron fallback) can react without catching exceptions: a
*delivered* native turn, a deliberate skip, an unavailable primitive, or a
fail-closed error. Pure stdlib value type; imports nothing from Hermes.
"""

from __future__ import annotations

from enum import Enum


class ReachOutcome(Enum):
    """Result of one proactive reach-out attempt (HLA §7, egress)."""

    DELIVERED = "delivered"  # native reach-in turn injected into the live session
    SKIPPED_BUSY = "skipped_busy"  # a turn is active in the session; retry next tick
    UNAVAILABLE = "unavailable"  # reach-in primitive not available (no runner / version drift)
    FAILED = "failed"  # attempted but errored (already logged, fail-closed)

    @property
    def ok(self) -> bool:
        """True only when a native proactive turn was actually delivered."""
        return self is ReachOutcome.DELIVERED

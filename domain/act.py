"""``Decision`` — the act-gate's verdict on whether to act (HLA §1/§7).

Outgoing speech is gated by *our* restraint (HLA §7): the act-gate weighs
receptivity, quiet-hours, budget and whether there is a real reason, then decides
to speak, act, or stay silent/defer. This is the immutable value it returns; the
gating *logic* is Phase 2 (2.3). Text-only in Phase 1 — the only "act" is a
delivered message. Imports nothing from Hermes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """Allow or suppress an outgoing act, with the reason for observability.

    ``reason`` is always populated (both when allowing and suppressing) so the
    debug trace (HLA §12, NFR9) can explain *why* the being spoke or held back.
    """

    allow: bool
    reason: str = ""

    @classmethod
    def allowed(cls, reason: str = "") -> Decision:
        """The act may proceed (deliver the message / take the action)."""
        return cls(allow=True, reason=reason)

    @classmethod
    def suppressed(cls, reason: str) -> Decision:
        """Hold back — silence or defer (the ``[SILENT]`` path, HLA §5)."""
        return cls(allow=False, reason=reason)

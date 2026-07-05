"""``DeliveryPort`` — the boundary for outbound speech (HLA §7/§13).

The act-gate *decides* whether to speak; the gateway *delivers* (HLA §7). This
port is that delivery boundary: the core hands a channel and text to a
``DeliveryPort`` and never touches Hermes' ``DeliveryRouter`` directly. Behind it
sits a no-op stub (:class:`~lifemodel.adapters.delivery.NoopDelivery`) or a
recording :class:`~lifemodel.testing.fakes.FakeDelivery` in tests. Note the
Phase-1.4 *proactive* outbound is delivered by Hermes' cron on a wake (``deliver``,
HLA §7/D4), so this port is not on that path yet — it is the seam for a future
*direct*-from-cognition send.

**Phase-1 scope (roadmap 1.4):** text-only, author/home channel only. ``send``
takes a channel id and a plain-text body — no attachments, no tools, no rich
payloads. Quiet-hours / cooldown / one-message-per-cycle live *above* this port:
in Phase 1.4 they are enforced by the tick's drain + cooldown (which gate the
wake itself); a fuller act-gate is Phase 2.3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DeliveryPort(Protocol):
    """Deliver an outbound text message to a channel (HLA §7)."""

    def send(self, channel: str, text: str) -> None:
        """Send *text* to *channel* (Phase 1: author/home channel, text-only)."""
        ...

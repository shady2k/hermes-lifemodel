"""``DeliveryPort`` — the boundary for outbound speech (HLA §7/§13).

The act-gate *decides* whether to speak; the gateway *delivers* (HLA §7). This
port is that delivery boundary: the core hands a channel and text to a
``DeliveryPort`` and never touches Hermes' ``DeliveryRouter`` directly. Behind it
sits a real Hermes-gateway adapter (Phase 1.4), a no-op stub
(:class:`~lifemodel.adapters.delivery.NoopDelivery`), or a recording
:class:`~lifemodel.testing.fakes.FakeDelivery` in tests.

**Phase-1 scope (roadmap 1.4):** text-only, author/home channel only. ``send``
takes a channel id and a plain-text body — no attachments, no tools, no rich
payloads. Quiet-hours / cooldown / one-message-per-cycle live *above* this port
in the act-gate, not here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DeliveryPort(Protocol):
    """Deliver an outbound text message to a channel (HLA §7)."""

    def send(self, channel: str, text: str) -> None:
        """Send *text* to *channel* (Phase 1: author/home channel, text-only)."""
        ...

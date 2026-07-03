"""``NoopDelivery`` — a do-nothing :class:`DeliveryPort` stub (HLA §7/§13).

The walking skeleton needs *a* delivery to wire, but the real Hermes-gateway
adapter is Phase 1.4. This stub lets the object graph construct and run without
speaking to the world: it accepts a ``send`` and drops it, logging a structured
event so a debug trace still shows the intent (HLA §12, NFR9). Swap in the real
gateway adapter at the composition root — nothing else changes. Stdlib + our own
logging only; no Hermes.
"""

from __future__ import annotations

from ..logging import EventLogger, get_logger


class NoopDelivery:
    """A :class:`~lifemodel.ports.delivery.DeliveryPort` that delivers nothing.

    Used as the default delivery in the composition root until the real gateway
    adapter lands (Phase 1.4). The dropped send is logged, not silent, so the
    skeleton stays observable.
    """

    def __init__(self, *, logger: EventLogger | None = None) -> None:
        self._log = logger or get_logger("lifemodel.delivery")

    def send(self, channel: str, text: str) -> None:
        """Drop the message; record that it would have been sent (text-only)."""
        self._log.info("delivery_noop", channel=channel, text_len=len(text))

"""Domain models — the plain, JSON-native values the whole system speaks.

Pure stdlib dataclasses with no behaviour beyond (de)serialization and their own
invariants. Both the core (neurons, layers, aggregator, act-gate) and the
adapters (signal bus, clock, delivery) depend on these; the models depend on
nothing (not Hermes, not the ports). This is the shared vocabulary of the
hexagon (HLA §13).
"""

from __future__ import annotations

from .act import Decision
from .layer import LayerResult
from .signal import Signal, SignalDecodeError
from .wake import (
    WAKE_PACKET_VERSION,
    WakeDecision,
    WakePacket,
    WakePacketError,
)

__all__ = [
    "WAKE_PACKET_VERSION",
    "Decision",
    "LayerResult",
    "Signal",
    "SignalDecodeError",
    "WakeDecision",
    "WakePacket",
    "WakePacketError",
]

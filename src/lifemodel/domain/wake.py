"""``WakePacket`` and ``WakeDecision`` — the proactive wake path (HLA §11).

The **wake-packet** is the compact JSON the neuron ``--script`` emits on stdout
when a pressure crosses its threshold (HLA §11 / D4): it is *the* schema the
awakened cognition turn reads as context — a small, versioned bag (wake reason +
which pressure crossed + energy/budget/last-contact), deliberately **not**
arbitrary stdout, to guard against token bloat and debug leaking into cognition.

A **wake-decision** is what the aggregator returns: whether to wake cognition
and, if so, the packet to hand it. Both are immutable JSON-native values so the
script's stdout and the state files round-trip with :mod:`json` alone. Imports
nothing from Hermes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Version of the wake-packet wire shape. Bump when the stdout schema changes so
#: an awakened turn can reject a packet it does not understand (HLA §11).
WAKE_PACKET_VERSION = 1


@dataclass(frozen=True)
class WakePacket:
    """Why the being is waking now — the neuron script's stdout schema (§11).

    Minimal Phase-1 slice of the soul-packet's wake half. The full soul-packet
    (identity/temperament/desires/loops) is layered in later phases; here we
    carry only the wake *reason* and the physiological context an act-gate needs.
    """

    reason: str
    #: Which pressure/neuron crossed its threshold (e.g. ``"connection"``).
    pressure_kind: str
    #: The pressure value that crossed the threshold.
    pressure: float
    #: Current energy (HLA §4/§11); a slot the wake path reads.
    energy: float = 1.0
    #: Remaining cost budget, or ``None`` when budgeting is not yet wired.
    budget: float | None = None
    #: ISO-8601 UTC timestamp of the last outbound contact, for cooldown.
    last_contact_at: str | None = None
    #: Wire-shape version header (serializes first).
    version: int = WAKE_PACKET_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a compact JSON-native dict, ``version`` header first."""
        return {
            "version": self.version,
            "reason": self.reason,
            "pressure_kind": self.pressure_kind,
            "pressure": self.pressure,
            "energy": self.energy,
            "budget": self.budget,
            "last_contact_at": self.last_contact_at,
        }

    def to_json(self) -> str:
        """Render the compact JSON the neuron script writes to stdout (§11)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WakePacket:
        """Rebuild a ``WakePacket`` from a parsed mapping, validating types."""
        return cls(
            reason=_as_str(data.get("reason"), "reason"),
            pressure_kind=_as_str(data.get("pressure_kind"), "pressure_kind"),
            pressure=_as_float(data.get("pressure"), "pressure"),
            energy=_as_float(data.get("energy", 1.0), "energy"),
            budget=_as_opt_float(data.get("budget"), "budget"),
            last_contact_at=_as_opt_str(data.get("last_contact_at"), "last_contact_at"),
            version=_as_int(data.get("version", WAKE_PACKET_VERSION), "version"),
        )

    @classmethod
    def from_json(cls, text: str) -> WakePacket:
        """Parse a wake-packet from the neuron script's stdout line (§11)."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise WakePacketError(f"wake-packet must be a JSON object, got {type(data).__name__}")
        return cls.from_dict(data)


@dataclass(frozen=True)
class WakeDecision:
    """The aggregator's verdict: wake cognition or stay quiet (HLA §5/§10).

    Invariant: if ``wake`` is true a ``packet`` must be present — an awakened
    turn is meaningless without the wake-packet that explains why. Use the
    :meth:`stay_asleep` / :meth:`wake_with` constructors rather than building
    the pair by hand.
    """

    wake: bool
    packet: WakePacket | None = None

    def __post_init__(self) -> None:
        if self.wake and self.packet is None:
            raise ValueError("a waking WakeDecision must carry a WakePacket")

    @classmethod
    def stay_asleep(cls) -> WakeDecision:
        """The common case: below threshold, no wake, zero LLM (HLA §1)."""
        return cls(wake=False, packet=None)

    @classmethod
    def wake_with(cls, packet: WakePacket) -> WakeDecision:
        """Wake cognition and hand it *packet* as context (HLA §11)."""
        return cls(wake=True, packet=packet)


class WakePacketError(ValueError):
    """Raised when a wake-packet cannot be parsed from its JSON form."""


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise WakePacketError(f"'{name}' must be a str, got {type(value).__name__}")
    return value


def _as_opt_str(value: object, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise WakePacketError(f"'{name}' must be a str or null, got {type(value).__name__}")


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WakePacketError(f"'{name}' must be a number, got {type(value).__name__}")
    return float(value)


def _as_opt_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, name)


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WakePacketError(f"'{name}' must be an int, got {type(value).__name__}")
    return value

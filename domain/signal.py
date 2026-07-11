"""``Signal`` — the one nervous-impulse model for the whole flow (HLA §1/§2).

Every input (incoming message, elapsed time, an unmet need, a motor result)
becomes a :class:`Signal`. Producers are the neuron tick and the gateway turn;
the consumer is the aggregator (HLA §2/§10). A signal is an immutable record and
carries a **stable origin id** so the aggregator can dedup it — a message must
not be counted twice, once as the gateway-turn signal and again as the next
tick's signal (HLA §10).

Kept minimal-but-extensible (YAGNI): only the fields Phase 1 needs. New fields
slot in later because :meth:`from_dict` is tolerant of missing keys. Every field
is JSON-native so a ``Signal`` round-trips through :mod:`json` alone. There is no
durable signal log: the nervous flow is ephemeral (spec §2/§3) — a signal lives
inside one ExecutionFrame's in-memory :class:`~lifemodel.core.frame.SignalFrame`
and dies with it. Imports nothing from Hermes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class SignalDecodeError(ValueError):
    """Raised when a persisted signal record cannot be read back as a ``Signal``."""


@dataclass(frozen=True)
class Signal:
    """An immutable nervous impulse flowing into the aggregator (HLA §2).

    * ``origin_id`` — stable id for dedup (``message_id``/``turn_id``/hash, §10).
    * ``kind`` — coarse category (e.g. ``"incoming"``, ``"connection"``,
      ``"thought"``, ``"overdue"``) so the aggregator can weigh signals by type.
    * ``payload`` — small JSON-native detail bag; kept optional and flat.
    * ``timestamp`` — ISO-8601 UTC string from the clock (see ``ClockPort``);
      a string (not ``datetime``) to keep the wire format trivial.
    * ``salience`` — weight/importance the aggregator accumulates (HLA §2).
    """

    origin_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    salience: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-native dict (one line in the durable log)."""
        return {
            "origin_id": self.origin_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
            "salience": self.salience,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Signal:
        """Rebuild a ``Signal`` from a parsed mapping, validating field types.

        Tolerant of missing optional keys; strict on present keys of the wrong
        type (a corrupt log line), raising :class:`SignalDecodeError`.
        """
        origin_id = data.get("origin_id")
        if not isinstance(origin_id, str):
            raise SignalDecodeError(f"'origin_id' must be a str, got {_type(origin_id)}")
        kind = data.get("kind")
        if not isinstance(kind, str):
            raise SignalDecodeError(f"'kind' must be a str, got {_type(kind)}")

        payload = data.get("payload", {})
        if not isinstance(payload, Mapping):
            raise SignalDecodeError(f"'payload' must be an object, got {_type(payload)}")

        timestamp = data.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise SignalDecodeError(f"'timestamp' must be a str or null, got {_type(timestamp)}")

        salience = data.get("salience", 1.0)
        if isinstance(salience, bool) or not isinstance(salience, int | float):
            raise SignalDecodeError(f"'salience' must be a number, got {_type(salience)}")

        return cls(
            origin_id=origin_id,
            kind=kind,
            payload=dict(payload),
            timestamp=timestamp,
            salience=float(salience),
        )


def _type(value: object) -> str:
    return type(value).__name__

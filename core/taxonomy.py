"""Signal taxonomy — the typed vocabulary of the pipeline (spec §4).

Phase B1 defines two kinds:
- ``contact`` — the neuron's *transient* intra-tick output: the unipolar drive
  value ``[0..u_max]`` plus its per-tick ``delta``. Never persisted.
- ``exchange`` — a *durable* external input: a real lane event (actor + label,
  per :mod:`lifemodel.sim.quality`) that the neuron reads to satiate the drive.

Builders keep payloads JSON-native and uniform; readers validate on the way out.
"""

from __future__ import annotations

from typing import cast

from ..domain.signal import Signal
from ..sim.quality import Actor, Label

KIND_CONTACT = "contact"
KIND_EXCHANGE = "exchange"

_ACTORS: frozenset[str] = frozenset({"user", "assistant", "proactive_internal"})
_LABELS: frozenset[str] = frozenset({"two_way", "ack", "monologue", "rejection"})


def is_kind(signal: Signal, kind: str) -> bool:
    return signal.kind == kind


def contact_signal(*, origin_id: str, value: float, delta: float, timestamp: str | None) -> Signal:
    """Build a transient contact signal carrying the drive value and its delta."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_CONTACT,
        payload={"value": float(value), "delta": float(delta)},
        timestamp=timestamp,
    )


def exchange_signal(*, origin_id: str, actor: Actor, label: Label, timestamp: str | None) -> Signal:
    """Build a durable exchange-input signal from a lane event."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_EXCHANGE,
        payload={"actor": actor, "label": label},
        timestamp=timestamp,
    )


def read_exchange(signal: Signal) -> tuple[Actor, Label]:
    """Validate and extract ``(actor, label)`` from an exchange signal."""
    if signal.kind != KIND_EXCHANGE:
        raise ValueError(f"not an exchange signal: kind={signal.kind!r}")
    actor = signal.payload.get("actor")
    label = signal.payload.get("label")
    if actor not in _ACTORS or label not in _LABELS:
        raise ValueError(f"invalid exchange payload: {signal.payload!r}")
    return cast(Actor, actor), cast(Label, label)

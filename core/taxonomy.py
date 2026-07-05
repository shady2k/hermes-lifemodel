"""Signal taxonomy — the typed vocabulary of the pipeline (spec §4).

Phase B1 defines two kinds:
- ``contact`` — the neuron's *transient* intra-tick output: the unipolar drive
  value ``[0..u_max]`` plus its per-tick ``delta``. Never persisted.
- ``exchange`` — a *durable* external input: a real lane event (actor + label,
  per :mod:`lifemodel.sim.quality`) that the neuron reads to satiate the drive.

Builders keep payloads JSON-native and uniform; readers validate on the way out.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from ..domain.signal import Signal
from ..sim.aggregation import Verdict
from ..sim.quality import Actor, Label

KIND_CONTACT = "contact"
KIND_EXCHANGE = "exchange"
KIND_VERDICT = "verdict"
KIND_IN_FLIGHT = "in_flight"

_VERDICTS: dict[str, Verdict] = {v.value: v for v in Verdict}

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


def verdict_signal(*, origin_id: str, verdict: Verdict, timestamp: str | None) -> Signal:
    """Build a durable verdict-input signal (cognition's decision on a desire)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_VERDICT,
        payload={"verdict": verdict.value},
        timestamp=timestamp,
    )


def read_verdict(signal: Signal) -> Verdict:
    """Validate and extract the ``Verdict`` from a verdict signal."""
    if signal.kind != KIND_VERDICT:
        raise ValueError(f"not a verdict signal: kind={signal.kind!r}")
    raw = signal.payload.get("verdict")
    if raw not in _VERDICTS:
        raise ValueError(f"invalid verdict payload: {signal.payload!r}")
    return _VERDICTS[raw]


def in_flight_signal(*, origin_id: str, value: bool, timestamp: str | None) -> Signal:
    """Build a durable in-flight input (a turn is running/queued)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_IN_FLIGHT,
        payload={"value": bool(value)},
        timestamp=timestamp,
    )


def is_in_flight(signals: Iterable[Signal]) -> bool:
    """True if any in-flight signal in the batch reports a running turn."""
    return any(s.kind == KIND_IN_FLIGHT and bool(s.payload.get("value")) for s in signals)


def contact_value(signals: Iterable[Signal], *, default: float) -> float:
    """The most recent transient contact value in the batch, or ``default``."""
    value = default
    for s in signals:
        if s.kind == KIND_CONTACT:
            raw = s.payload.get("value", default)
            value = float(raw) if isinstance(raw, int | float) else default
    return value

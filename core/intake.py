"""Backpressure intake — the aggregation flood guard (spec §5.1).

Bounds the per-tick signal batch so the aggregation is O(bounded) regardless of
input volume ("the layer never fails"). Signals are split by lane:

* **control** (``exchange``/``verdict``/``in_flight``/``delivery_result``) —
  lossless: kept in arrival order up to ``max_control``; any overflow is *counted*
  (the CoreLoop leaves it on the bus for next tick, not dropped).
* **sensor** (``contact`` and future ``ping``/``presence``) — coalesced per-kind
  to latest-wins, then bounded to ``max_sensor``.

Pure and total: never raises, whatever the input. Full salience-based shedding
(spec §5) slots in for the sensor lane once there are multiple noisy sensors; v1
keeps it to latest-wins coalescing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..domain.signal import Signal
from .taxonomy import Lane


@dataclass(frozen=True)
class IntakeLimits:
    """Per-lane intake caps (bootstrap values; tuned later, spec §22)."""

    max_control: int = 256
    max_sensor: int = 64


@dataclass(frozen=True)
class IntakeResult:
    kept: tuple[Signal, ...]
    shed_control: int
    shed_sensor: int
    coalesced_sensor: int


def apply_intake(
    signals: Iterable[Signal],
    *,
    limits: IntakeLimits,
    lane_of: Callable[[str], Lane],
) -> IntakeResult:
    """Apply lane-aware backpressure to a signal batch. Never raises."""
    control: list[Signal] = []
    sensor_latest: dict[str, Signal] = {}
    coalesced = 0
    for sig in signals:
        if lane_of(sig.kind) == "control":
            control.append(sig)
        else:
            if sig.kind in sensor_latest:
                coalesced += 1
            sensor_latest[sig.kind] = sig  # latest-wins

    control_kept = control[: limits.max_control]
    shed_control = len(control) - len(control_kept)

    sensor_all = list(sensor_latest.values())  # dict preserves first-seen kind order (py3.7+)
    sensor_kept = sensor_all[: limits.max_sensor]
    shed_sensor = len(sensor_all) - len(sensor_kept)

    return IntakeResult(
        kept=tuple(control_kept) + tuple(sensor_kept),
        shed_control=shed_control,
        shed_sensor=shed_sensor,
        coalesced_sensor=coalesced,
    )

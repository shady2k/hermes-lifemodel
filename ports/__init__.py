"""Ports — the hexagon boundaries the core depends on (HLA §13).

A *port* is an interface for a dependency that genuinely varies or must be faked:
the Hermes host, the state store, delivery, the clock (HLA §13, "pragmatism, not
ceremony" — we do not wrap everything). The core depends only on these
Protocols; concrete :mod:`lifemodel.adapters` (and test
:mod:`lifemodel.testing` fakes) implement them, wired by the one composition
root (:mod:`lifemodel.composition`).

Phase-1 ports: ``StatePort``, ``DeliveryPort``, ``ClockPort``. ``MemoryPort``
and ``PressureSensorPort`` arrived with lm-fib.6.1 (HLA §4.1/D7) — purely
additive, not yet wired into the live tick. ``LlmPort`` arrives with the phase
that needs it.

``StatePort`` is defined with its model and JSON adapter in the cohesive
:mod:`lifemodel.state` package (task 0.2); it is re-exported here so the ports
layer is the single catalogue of Phase-1 boundaries.
"""

from __future__ import annotations

from ..state.port import StatePort
from .clock import ClockPort
from .delivery import DeliveryPort
from .memory import MemoryPort
from .pressure import PressureSensorPort
from .proactive import ProactiveEgressPort
from .tick_commit import TickCommitPort

__all__ = [
    "ClockPort",
    "DeliveryPort",
    "MemoryPort",
    "PressureSensorPort",
    "ProactiveEgressPort",
    "StatePort",
    "TickCommitPort",
]

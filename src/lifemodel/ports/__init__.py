"""Ports — the hexagon boundaries the core depends on (HLA §13).

A *port* is an interface for a dependency that genuinely varies or must be faked:
the Hermes host, the state store, delivery, the clock (HLA §13, "pragmatism, not
ceremony" — we do not wrap everything). The core depends only on these
Protocols; concrete :mod:`lifemodel.adapters` (and test
:mod:`lifemodel.testing` fakes) implement them, wired by the one composition
root (:mod:`lifemodel.composition`).

Phase-1 ports only (YAGNI): ``StatePort``, ``DeliveryPort``, ``ClockPort``.
``LlmPort``/``MemoryPort`` arrive with the phases that need them (roadmap 0.4).

``StatePort`` is defined with its model and JSON adapter in the cohesive
:mod:`lifemodel.state` package (task 0.2); it is re-exported here so the ports
layer is the single catalogue of Phase-1 boundaries.
"""

from __future__ import annotations

from ..state.port import StatePort
from .clock import ClockPort
from .delivery import DeliveryPort

__all__ = [
    "ClockPort",
    "DeliveryPort",
    "StatePort",
]

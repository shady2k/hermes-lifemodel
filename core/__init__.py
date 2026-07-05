"""Core — the Hermes-free brain: extension points and pressure logic (HLA §13).

The core depends only on :mod:`lifemodel.domain` values and
:mod:`lifemodel.ports` interfaces — never on Hermes, never on a concrete
adapter. It holds the ABC extension points the system grows by subclassing:
:class:`~lifemodel.core.neuron.Neuron`, :class:`~lifemodel.core.layer.Layer`,
:class:`~lifemodel.core.aggregator.Aggregator`,
:class:`~lifemodel.core.act_gate.ActGate`, and
:class:`~lifemodel.core.signal_bus.SignalBus` (HLA §13, lego-swappability).
"""

from __future__ import annotations

from .act_gate import ActGate
from .aggregator import Aggregator, SilentAggregator
from .intents import CheckpointState, EmitSignal, Intent, UpdateState
from .layer import Layer, ProcessingLayer
from .neuron import Neuron
from .signal_bus import SignalBus

__all__ = [
    "ActGate",
    "Aggregator",
    "CheckpointState",
    "EmitSignal",
    "Intent",
    "Layer",
    "Neuron",
    "ProcessingLayer",
    "SignalBus",
    "SilentAggregator",
    "UpdateState",
]

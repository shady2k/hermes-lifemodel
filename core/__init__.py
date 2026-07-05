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
from .component import Component, TickContext
from .contact_neuron import ContactNeuron
from .coreloop import CoreLoop, TickReport
from .intents import CheckpointState, EmitSignal, Intent, UpdateState
from .layer import Layer, ProcessingLayer
from .neuron import Neuron
from .registry import (
    ComponentManifest,
    ComponentRegistry,
    DuplicateComponent,
    UnknownComponent,
)
from .signal_bus import SignalBus
from .state_actor import StateActor, UnknownStateField
from .taxonomy import (
    KIND_CONTACT,
    KIND_EXCHANGE,
    contact_signal,
    exchange_signal,
    is_kind,
    read_exchange,
)
from .timeutil import minutes_between

__all__ = [
    "ActGate",
    "Aggregator",
    "CheckpointState",
    "Component",
    "ComponentManifest",
    "ContactNeuron",
    "ComponentRegistry",
    "CoreLoop",
    "DuplicateComponent",
    "EmitSignal",
    "Intent",
    "KIND_CONTACT",
    "KIND_EXCHANGE",
    "Layer",
    "Neuron",
    "ProcessingLayer",
    "SignalBus",
    "SilentAggregator",
    "StateActor",
    "TickContext",
    "TickReport",
    "UnknownComponent",
    "UnknownStateField",
    "UpdateState",
    "contact_signal",
    "exchange_signal",
    "is_kind",
    "minutes_between",
    "read_exchange",
]

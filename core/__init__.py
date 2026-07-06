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
from .aggregation import ContactAggregation
from .aggregator import Aggregator, SilentAggregator
from .backstop import allow_send, record_send
from .circadian import circadian
from .cognition import Cognition
from .component import Component, TickContext
from .contact_neuron import ContactNeuron
from .coreloop import CoreLoop, TickReport
from .energy import Reservation, can_afford, cost_real, reserve, settle
from .intake import IntakeLimits, IntakeResult, apply_intake
from .intents import CheckpointState, EmitSignal, Intent, LaunchProactive, UpdateState
from .invalidation import is_verdict_stale
from .layer import Layer, ProcessingLayer
from .neuron import Neuron
from .output_lint import DEFAULT_MECHANICAL_PATTERNS, LintResult, lint_proactive
from .personality import Personality
from .pressure import effective_pressure, inhibition_at
from .projection import project_contact
from .registry import (
    ComponentManifest,
    ComponentRegistry,
    DuplicateComponent,
    UnknownComponent,
)
from .signal_bus import SignalBus
from .state_actor import StateActor, UnknownStateField
from .taxonomy import (
    CONTROL_KINDS,
    KIND_CONTACT,
    KIND_EXCHANGE,
    KIND_IN_FLIGHT,
    KIND_VERDICT,
    Lane,
    contact_signal,
    contact_value,
    exchange_signal,
    in_flight_signal,
    is_in_flight,
    is_kind,
    lane_of,
    read_exchange,
    read_verdict,
    verdict_signal,
)
from .timeutil import minutes_between
from .wake_packet import GUIDANCE, ProactivePrompt, build_wake_packet

__all__ = [
    "ActGate",
    "Aggregator",
    "allow_send",
    "CheckpointState",
    "can_afford",
    "circadian",
    "Component",
    "Cognition",
    "DEFAULT_MECHANICAL_PATTERNS",
    "cost_real",
    "ComponentManifest",
    "ContactAggregation",
    "ContactNeuron",
    "ComponentRegistry",
    "CONTROL_KINDS",
    "CoreLoop",
    "DuplicateComponent",
    "EmitSignal",
    "IntakeLimits",
    "IntakeResult",
    "is_verdict_stale",
    "Intent",
    "LaunchProactive",
    "KIND_CONTACT",
    "lint_proactive",
    "LintResult",
    "KIND_EXCHANGE",
    "KIND_IN_FLIGHT",
    "KIND_VERDICT",
    "Lane",
    "Layer",
    "Neuron",
    "Personality",
    "Reservation",
    "ProcessingLayer",
    "SignalBus",
    "SilentAggregator",
    "StateActor",
    "TickContext",
    "TickReport",
    "UnknownComponent",
    "UnknownStateField",
    "UpdateState",
    "apply_intake",
    "project_contact",
    "build_wake_packet",
    "GUIDANCE",
    "ProactivePrompt",
    "effective_pressure",
    "inhibition_at",
    "contact_signal",
    "contact_value",
    "exchange_signal",
    "in_flight_signal",
    "is_in_flight",
    "is_kind",
    "lane_of",
    "minutes_between",
    "read_exchange",
    "read_verdict",
    "record_send",
    "reserve",
    "settle",
    "verdict_signal",
]

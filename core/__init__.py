"""Core — the Hermes-free brain: the ExecutionFrame pipeline and pressure logic.

The core depends only on :mod:`lifemodel.domain` values and
:mod:`lifemodel.ports` interfaces — never on Hermes, never on a concrete
adapter. The nervous flow is **ephemeral** (spec §2/§3): a signal lives ``<=`` one
:class:`~lifemodel.core.frame.SignalFrame`, and only the body (``AgentState``),
memory and the trace record are durable. Components (sensor → drive → aggregation
→ cognition launcher) fold the frame's signals and return intents; the single
:class:`~lifemodel.core.state_actor.StateActor` commits them atomically at end of
frame.
"""

from __future__ import annotations

from .aggregation import ContactAggregation
from .backstop import allow_send, record_send
from .circadian import circadian
from .cognition import CognitionLauncher
from .component import (
    LAYER_BY_TYPE,
    Component,
    ComponentLayer,
    TickContext,
    layer_for_type,
)
from .contact_sensor import ContactSensor
from .coreloop import CoreLoop, TickReport
from .energy import Reservation, can_afford, cost_real, reserve, settle
from .frame import FrameTrigger, SignalFrame, run_frame
from .intents import CheckpointState, EmitSignal, Intent, LaunchProactive, UpdateState
from .invalidation import is_proactive_outcome_stale
from .personality import Personality
from .pressure import effective_pressure, inhibition_at
from .projection import project_contact
from .registry import (
    ComponentManifest,
    ComponentRegistry,
    DuplicateComponent,
    IncompleteManifest,
    UnknownComponent,
)
from .solitude_drive import SolitudeDrive
from .state_actor import StateActor, UnknownStateField
from .suppression import (
    EVENT_SUPPRESSION,
    SUPPRESSION_MIN_FIELDS,
    SuppressionReason,
    emit_suppression_span,
)
from .taxonomy import (
    CONTROL_KINDS,
    KIND_CONTACT,
    KIND_CONTACT_OBSERVED,
    KIND_CONTACT_PRESENCE,
    KIND_CONTACT_PRESSURE,
    KIND_IN_FLIGHT,
    KIND_PROACTIVE_OUTCOME,
    ContactPresenceReading,
    Lane,
    contact_observed_signal,
    contact_presence_signal,
    contact_pressure_signal,
    contact_pressure_value,
    contact_signal,
    contact_value,
    in_flight_signal,
    is_in_flight,
    is_kind,
    lane_of,
    proactive_outcome_signal,
    read_contact_observed,
    read_contact_presence,
    read_proactive_outcome,
    read_proactive_outcome_correlation,
)
from .timeutil import minutes_between
from .wake_packet import IMPULSE_LABEL_PREFIX, ProactivePrompt, build_wake_packet

__all__ = [
    "allow_send",
    "CheckpointState",
    "can_afford",
    "circadian",
    "Component",
    "ComponentLayer",
    "CognitionLauncher",
    "cost_real",
    "ComponentManifest",
    "IncompleteManifest",
    "LAYER_BY_TYPE",
    "layer_for_type",
    "ContactAggregation",
    "ContactPresenceReading",
    "ContactSensor",
    "SolitudeDrive",
    "ComponentRegistry",
    "CONTROL_KINDS",
    "CoreLoop",
    "DuplicateComponent",
    "EmitSignal",
    "EVENT_SUPPRESSION",
    "emit_suppression_span",
    "FrameTrigger",
    "SignalFrame",
    "run_frame",
    "SUPPRESSION_MIN_FIELDS",
    "SuppressionReason",
    "is_proactive_outcome_stale",
    "Intent",
    "LaunchProactive",
    "KIND_CONTACT",
    "KIND_CONTACT_OBSERVED",
    "KIND_CONTACT_PRESENCE",
    "KIND_CONTACT_PRESSURE",
    "KIND_IN_FLIGHT",
    "KIND_PROACTIVE_OUTCOME",
    "Lane",
    "Personality",
    "Reservation",
    "StateActor",
    "TickContext",
    "TickReport",
    "UnknownComponent",
    "UnknownStateField",
    "UpdateState",
    "project_contact",
    "build_wake_packet",
    "IMPULSE_LABEL_PREFIX",
    "ProactivePrompt",
    "effective_pressure",
    "inhibition_at",
    "contact_signal",
    "contact_value",
    "contact_presence_signal",
    "contact_pressure_signal",
    "contact_pressure_value",
    "contact_observed_signal",
    "in_flight_signal",
    "is_in_flight",
    "is_kind",
    "lane_of",
    "minutes_between",
    "read_contact_presence",
    "read_contact_observed",
    "read_proactive_outcome",
    "read_proactive_outcome_correlation",
    "record_send",
    "reserve",
    "settle",
    "proactive_outcome_signal",
]

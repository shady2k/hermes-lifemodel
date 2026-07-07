"""Composition root — the one place the object graph is wired (HLA §13, DI).

Clean architecture with dependency injection means exactly one module knows how
the pieces fit together. :func:`build_lifemodel` takes the injectable
dependencies (a state ``base_dir``, plus optional clock / delivery / neurons /
aggregator / bus / state overrides) and returns an assembled :class:`LifeModel`.
Everything the core needs arrives through its constructor; nothing reaches out
for a global.

Two call sites this must serve (roadmap 0.4):

* :func:`lifemodel.register` — wires the graph with **real Hermes adapters**
  (the gateway ``DeliveryPort``, constructed at the plugin boundary and injected
  in). Those Hermes-touching adapters are built *there*, not here.
* the cron ``--script`` entrypoint (task 1.1) — wires it from a ``base_dir`` and
  takes the Hermes-free defaults below.

The defaults are the concrete :class:`SystemClock`,
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (the ``StatePort``
adapter since lm-fib.6.2 — it replaced the retired ``JsonStateStore``/
``state.json``), and durable :class:`FileSignalBus`, with a
:class:`NoopDelivery` stub for the ``DeliveryPort``. Note the *proactive* outbound
does **not** go through this port: the supervised platform adapter's tick
(:mod:`lifemodel.adapters.being_platform` → :func:`lifemodel.core.proactive.proactive_tick`)
launches proactive turns directly via its own ``ProactiveEgressPort``. The
``DeliveryPort`` stays the seam for a future *direct*-from-cognition delivery path,
so ``NoopDelivery`` remains the default.

**Wire-desire-model plan (Task 4): no decision aggregator/neuron in the live
path.** The cron tick no longer runs a neuron loop or asks an aggregator to
decide — the in-process service uses :mod:`lifemodel.core.decision` (which
reconstructs the certified ``sim`` primitives from ``State`` directly, bypassing
this ``Aggregator``/``Neuron`` seam entirely). So the live default aggregator is
now :class:`SilentAggregator` (never wakes, whatever it is asked to decide —
matching "cron never decides") and the live default neuron list is empty. The
old ``ThresholdAggregator``/``StubTimerNeuron`` defaults were removed outright
by Task 8's cleanup pass once confirmed fully orphaned — the ``Aggregator`` and
``Neuron`` ABCs (plus ``SilentAggregator``) remain as the extension seam for any
future subclass.

Passing ``neurons`` or ``aggregator`` explicitly — including an empty ``()`` —
opts out of the default, so real Hermes wiring and the seam tests stay in full
control.

**This module imports no Hermes** — only Hermes-free adapters and the core — so
the whole graph is constructible (and testable) with injected fakes off-host.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .adapters.clock import SystemClock
from .adapters.delivery import NoopDelivery
from .adapters.signal_bus import FileSignalBus
from .core.aggregation import ContactAggregation
from .core.aggregator import Aggregator, SilentAggregator
from .core.cognition import Cognition
from .core.contact_neuron import ContactNeuron
from .core.coreloop import CoreLoop
from .core.neuron import Neuron
from .core.personality import Personality
from .core.registry import ComponentManifest, ComponentRegistry, UnknownComponent
from .core.signal_bus import SignalBus
from .core.state_actor import StateActor
from .log import EventLogger
from .ports.clock import ClockPort
from .ports.delivery import DeliveryPort
from .ports.memory import MemoryPort
from .ports.pressure import PressureSensorPort
from .ports.tick_commit import TickCommitPort
from .sim.wake import GateParams
from .state.port import StatePort
from .state.sqlite_store import SQLiteRuntimeStore

CONTACT_ALPHA = 1.0 / 240.0
CONTACT_BETA = 1.0
CONTACT_U_MAX = 100.0
CONTACT_PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)
CONTACT_I0 = 1.0
CONTACT_GRACE_MIN = 45.0
CONTACT_INHIBITION_HALFLIFE_MIN = 60.0

E_MAX = 1.0
ENERGY_RECOVERY_PER_MIN = 0.01
NIGHT_RECOVERY_BOOST = 0.5
FATIGUE_DECAY_PER_MIN = 0.002
CIRCADIAN_PEAK_UTC_HOUR = 13.0  # peak alertness 16:00 MSK, trough 04:00 MSK

COGNITION_FAST_COST = 0.02
COGNITION_SEND_COST = 0.03
COST_ALPHA = 2.0


@dataclass(frozen=True)
class LifeModel:
    """The assembled object graph — every dependency wired, none reached for.

    The tick orchestrator (1.1) and the awakened cognition turn (1.4) receive one
    of these and read the collaborators they need. Frozen: the graph is fixed
    once composed; mutable state lives behind ``state`` (the store), not here.
    """

    state: StatePort
    bus: SignalBus
    clock: ClockPort
    delivery: DeliveryPort
    aggregator: Aggregator
    neurons: tuple[Neuron, ...] = field(default_factory=tuple)
    registry: ComponentRegistry = field(default_factory=ComponentRegistry)
    state_actor: StateActor | None = None
    coreloop: CoreLoop | None = None


def build_lifemodel(
    *,
    base_dir: Path,
    state: StatePort | None = None,
    bus: SignalBus | None = None,
    clock: ClockPort | None = None,
    delivery: DeliveryPort | None = None,
    aggregator: Aggregator | None = None,
    neurons: Sequence[Neuron] | None = None,
    logger: EventLogger | None = None,
    registry: ComponentRegistry | None = None,
) -> LifeModel:
    """Assemble the :class:`LifeModel` graph from injected parts (HLA §13).

    *base_dir* is the profile-scoped state directory (from
    :func:`lifemodel.paths.state_dir`); the default state store and signal bus
    live under it. Every collaborator is overridable so ``register(ctx)`` can
    inject real Hermes adapters and tests can inject fakes — the wiring is the
    same, only the parts differ.

    ``neurons`` defaults to an empty tuple and ``aggregator`` defaults to
    :class:`SilentAggregator` (roadmap Task 4: the cron path decides nothing —
    the in-process service is the sole brain via ``core/decision``, bypassing
    this seam). ``None`` means "take the default"; passing an explicit value —
    including an empty ``()`` — overrides it, so callers keep full control.
    """
    resolved_clock: ClockPort = clock or SystemClock()
    resolved_state: StatePort = state or SQLiteRuntimeStore(
        base_dir, clock=resolved_clock, logger=logger
    )
    # The one live adapter (SQLiteRuntimeStore) implements StatePort + MemoryPort
    # + PressureSensorPort + TickCommitPort, so a tick's commit spans vitals and
    # entities in one transaction (HLA §4.1). Default the memory/pressure/commit
    # slots to that same instance when it satisfies the port; an injected fake
    # StatePort that does not (e.g. FakeStateStore) leaves them unwired — the
    # CoreLoop then reads empty snapshots and the StateActor uses the store's own
    # commit_tick (the fakes implement it).
    resolved_memory: MemoryPort | None = (
        resolved_state if isinstance(resolved_state, MemoryPort) else None
    )
    resolved_pressure: PressureSensorPort | None = (
        resolved_state if isinstance(resolved_state, PressureSensorPort) else None
    )
    resolved_committer: TickCommitPort | None = (
        resolved_state if isinstance(resolved_state, TickCommitPort) else None
    )
    resolved_bus: SignalBus = bus or FileSignalBus(base_dir, logger=logger)
    resolved_delivery: DeliveryPort = delivery or NoopDelivery(logger=logger)
    resolved_aggregator: Aggregator = aggregator or SilentAggregator()
    resolved_neurons: tuple[Neuron, ...] = () if neurons is None else tuple(neurons)

    resolved_registry: ComponentRegistry = registry if registry is not None else ComponentRegistry()
    try:
        resolved_registry.manifest("personality")
    except UnknownComponent:
        personality = Personality(
            e_max=E_MAX,
            recovery_per_min=ENERGY_RECOVERY_PER_MIN,
            night_boost=NIGHT_RECOVERY_BOOST,
            fatigue_decay_per_min=FATIGUE_DECAY_PER_MIN,
            peak_hour_utc=CIRCADIAN_PEAK_UTC_HOUR,
        )
        resolved_registry.register(
            personality, ComponentManifest(id=personality.id, type="personality")
        )
    try:
        resolved_registry.manifest("contact")
    except UnknownComponent:
        contact = ContactNeuron(alpha=CONTACT_ALPHA, beta=CONTACT_BETA, u_max=CONTACT_U_MAX)
        resolved_registry.register(contact, ComponentManifest(id=contact.id, type="neuron"))
    try:
        resolved_registry.manifest("contact-aggregation")
    except UnknownComponent:
        aggregation = ContactAggregation(
            params=CONTACT_PARAMS,
            theta=CONTACT_PARAMS.theta_u,
            beta=CONTACT_BETA,
            u_max=CONTACT_U_MAX,
            i0=CONTACT_I0,
            grace_min=CONTACT_GRACE_MIN,
            halflife_min=CONTACT_INHIBITION_HALFLIFE_MIN,
        )
        resolved_registry.register(
            aggregation, ComponentManifest(id=aggregation.id, type="aggregation")
        )
    try:
        resolved_registry.manifest("cognition")
    except UnknownComponent:
        cognition = Cognition(
            fast_cost=COGNITION_FAST_COST,
            send_cost=COGNITION_SEND_COST,
            alpha=COST_ALPHA,
        )
        resolved_registry.register(cognition, ComponentManifest(id=cognition.id, type="cognition"))
    resolved_state_actor = StateActor(resolved_state, committer=resolved_committer, logger=logger)
    resolved_coreloop = CoreLoop(
        registry=resolved_registry,
        state_actor=resolved_state_actor,
        bus=resolved_bus,
        clock=resolved_clock,
        logger=logger,
        pressure_sensor=resolved_pressure,
        memory=resolved_memory,
    )

    return LifeModel(
        state=resolved_state,
        bus=resolved_bus,
        clock=resolved_clock,
        delivery=resolved_delivery,
        aggregator=resolved_aggregator,
        neurons=resolved_neurons,
        registry=resolved_registry,
        state_actor=resolved_state_actor,
        coreloop=resolved_coreloop,
    )

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
from datetime import tzinfo
from pathlib import Path

from .adapters.clock import SystemClock
from .adapters.delivery import NoopDelivery
from .adapters.signal_bus import FileSignalBus
from .adapters.trace_export import make_trace_exporter
from .adapters.tracer import StdlibTracer
from .core.aggregation import ContactAggregation
from .core.aggregator import Aggregator, SilentAggregator
from .core.cognition import CognitionLauncher
from .core.contact_neuron import PresenceNeuron
from .core.coreloop import CoreLoop
from .core.neuron import Neuron
from .core.personality import Personality
from .core.registry import ComponentManifest, ComponentRegistry, UnknownComponent
from .core.signal_bus import SignalBus
from .core.solitude_drive import SolitudeDrive
from .core.state_actor import StateActor
from .domain.objects import default_registry
from .events import EventRing
from .ports.clock import ClockPort
from .ports.delivery import DeliveryPort
from .ports.memory import MemoryPort
from .ports.pressure import PressureSensorPort
from .ports.tick_commit import TickCommitPort
from .ports.trace_export import TraceExportPort
from .ports.tracer import TracerPort
from .sim.wake import GateParams
from .state.port import StatePort
from .state.sqlite_store import SQLiteRuntimeStore
from .state.trace_store import NULL_TRACE_SINK, TraceSink

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
    #: The execution tracer (lm-27n.11) the graph was built with — the SAME one
    #: wired into the CoreLoop. Exposed so out-of-tick paths (e.g. the egress
    #: backstop in :func:`proactive_tick`) can mint a span for a suppression.
    #: Always set by :func:`build_lifemodel`; ``None`` only for a hand-built graph.
    tracer: TracerPort | None = None
    #: The durable trace sink + in-memory freshness ring (spec §4.2) the CoreLoop
    #: fans SpanLoggers onto — exposed so an OUT-OF-TICK span (the egress backstop
    #: suppression in :func:`proactive_tick`) records through the SAME sinks as the
    #: in-tick pipeline. ``NULL_TRACE_SINK`` / a throwaway ring off the live being.
    trace_writer: TraceSink = NULL_TRACE_SINK
    event_ring: EventRing = field(default_factory=EventRing)


def build_lifemodel(
    *,
    base_dir: Path,
    state: StatePort | None = None,
    bus: SignalBus | None = None,
    clock: ClockPort | None = None,
    delivery: DeliveryPort | None = None,
    aggregator: Aggregator | None = None,
    neurons: Sequence[Neuron] | None = None,
    registry: ComponentRegistry | None = None,
    tracer: TracerPort | None = None,
    trace_exporter: TraceExportPort | None = None,
    trace_writer: TraceSink | None = None,
    event_ring: EventRing | None = None,
    display_tz: tzinfo | None = None,
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

    ``display_tz`` is the owner's local timezone for the wake-packet's temporal
    facts, resolved from Hermes at the adapter boundary (this module imports no
    Hermes) and forwarded to the :class:`CognitionLauncher` as a plain stdlib
    ``tzinfo``; ``None`` (the default, and every test/CLI caller) falls back to
    server-local then UTC.
    """
    resolved_clock: ClockPort = clock or SystemClock()
    resolved_state: StatePort = state or SQLiteRuntimeStore(base_dir, clock=resolved_clock)
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
    resolved_bus: SignalBus = bus or FileSignalBus(base_dir)
    resolved_delivery: DeliveryPort = delivery or NoopDelivery()
    # The execution tracer (lm-27n.11): the DEFAULT mints real W3C ids stdlib-only;
    # a test injects a deterministic FakeTracer. The CoreLoop mints ONE root trace per
    # tick from it and threads it through TickContext so creation sites stamp the born
    # object's provenance with the tick's trace.
    resolved_tracer: TracerPort = tracer or StdlibTracer()
    # The OPTIONAL tick-end trace exporter (lm-27n.10): the factory returns a real
    # OpenTelemetry exporter only if ``opentelemetry`` is importable, else a no-op —
    # so in the Hermes venv (no OTel) this is behaviour-neutral. The CoreLoop ships
    # each finished tick's root span to it best-effort, after the commit.
    resolved_trace_exporter: TraceExportPort = trace_exporter or make_trace_exporter()
    # The durable trace sink + freshness ring (spec §4.2). The live BeingAdapter
    # injects an acquired ``TraceWriter`` (writing ``observability.sqlite``); off
    # the being both default to no-op/throwaway. The SAME instances are wired into
    # the CoreLoop AND exposed on the returned LifeModel, so an out-of-tick span
    # (the egress backstop) records through the same sinks as the in-tick pipeline.
    resolved_writer: TraceSink = trace_writer if trace_writer is not None else NULL_TRACE_SINK
    resolved_ring: EventRing = event_ring if event_ring is not None else EventRing()
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
        # T2 split (spec §3): the instantaneous, stateless contact-channel sensor.
        # PresenceNeuron keeps the historical ``contact`` slot id; it measures the
        # channel now and emits a raw ``contact_presence`` reading — it writes NO u.
        presence = PresenceNeuron()
        resolved_registry.register(presence, ComponentManifest(id=presence.id, type="neuron"))
    try:
        resolved_registry.manifest("solitude-drive")
    except UnknownComponent:
        # T2 split: the AUTONOMIC integrator that OWNS and writes u. Registered
        # AFTER the sensor (it consumes the sensor's contact_presence reading) and
        # BEFORE aggregation (which reads the fresh u from the drive's contact
        # signal, since the drive's UpdateState is only visible after commit).
        drive = SolitudeDrive(alpha=CONTACT_ALPHA, beta=CONTACT_BETA, u_max=CONTACT_U_MAX)
        resolved_registry.register(drive, ComponentManifest(id=drive.id, type="drive"))
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
        resolved_registry.manifest("cognition-launcher")
    except UnknownComponent:
        # T4: the 0-LLM launcher (renamed from Cognition) — reserves energy, builds
        # the wake-packet, emits LaunchProactive + Intention. No synchronous LLM in
        # core; the real act-gate is the async Hermes turn (verdict read back next tick).
        launcher = CognitionLauncher(
            fast_cost=COGNITION_FAST_COST,
            send_cost=COGNITION_SEND_COST,
            alpha=COST_ALPHA,
            # The owner's local zone for the wake-packet's temporal facts. Resolved
            # from Hermes at the adapter boundary (this module imports no Hermes) and
            # threaded through as a plain stdlib ``tzinfo``; ``None`` → server-local
            # then UTC (see wake_packet._fmt_ts).
            display_tz=display_tz,
        )
        resolved_registry.register(launcher, ComponentManifest(id=launcher.id, type="launcher"))
    resolved_state_actor = StateActor(resolved_state, committer=resolved_committer)
    resolved_coreloop = CoreLoop(
        registry=resolved_registry,
        state_actor=resolved_state_actor,
        bus=resolved_bus,
        clock=resolved_clock,
        trace_writer=resolved_writer,
        event_ring=resolved_ring,
        pressure_sensor=resolved_pressure,
        memory=resolved_memory,
        # The registry-derived live (non-terminal) state-set drives the snapshot,
        # so parked thoughts + pending intentions are visible, not just active +
        # deferred (the object-core is the single source of what "live" means).
        live_states=default_registry().live_states(),
        tracer=resolved_tracer,
        trace_exporter=resolved_trace_exporter,
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
        tracer=resolved_tracer,
        trace_writer=resolved_writer,
        event_ring=resolved_ring,
    )

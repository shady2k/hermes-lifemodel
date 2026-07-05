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

For Phase 0.4 the defaults are the concrete :class:`SystemClock`,
:class:`JsonStateStore`, and durable :class:`FileSignalBus`, with a
:class:`NoopDelivery` stub for the ``DeliveryPort``. Note the Phase-1.4 *proactive*
outbound does **not** go through this port: the in-process egress service is the
sole decision brain and launches proactive turns directly via its own
``ProactiveEgressPort`` (see :mod:`lifemodel.egress_service`); the cron tick
never wakes at all (roadmap Task 4, the drum-killer). The ``DeliveryPort`` stays
the seam for a future *direct*-from-cognition delivery path, so ``NoopDelivery``
remains the default.

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
from .core.aggregator import Aggregator, SilentAggregator
from .core.neuron import Neuron
from .core.signal_bus import SignalBus
from .log import EventLogger
from .ports.clock import ClockPort
from .ports.delivery import DeliveryPort
from .state.json_store import JsonStateStore
from .state.port import StatePort


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
    resolved_state: StatePort = state or JsonStateStore(base_dir, logger=logger)
    resolved_bus: SignalBus = bus or FileSignalBus(base_dir, logger=logger)
    resolved_clock: ClockPort = clock or SystemClock()
    resolved_delivery: DeliveryPort = delivery or NoopDelivery(logger=logger)
    resolved_aggregator: Aggregator = aggregator or SilentAggregator()
    resolved_neurons: tuple[Neuron, ...] = () if neurons is None else tuple(neurons)

    return LifeModel(
        state=resolved_state,
        bus=resolved_bus,
        clock=resolved_clock,
        delivery=resolved_delivery,
        aggregator=resolved_aggregator,
        neurons=resolved_neurons,
    )

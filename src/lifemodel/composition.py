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
:class:`JsonStateStore`, and durable :class:`FileSignalBus`, with lightweight
stubs where real behaviour is a later task: a :class:`NoopDelivery` (real
gateway delivery is 1.4) and a pass-through :class:`SilentAggregator` (real
thresholding is 1.3). Phase 1.2 landed the first neuron: the default neuron list
is now a single :class:`StubTimerNeuron`, so every default graph accumulates
pressure each tick (real behavioural neurons arrive in Phase 2+). The graph
therefore constructs and is exercisable end-to-end now, and later tasks fill in
implementations rather than reshape the wiring.

Passing ``neurons`` explicitly — including an empty ``()`` — opts out of that
default, so real Hermes wiring and the seam tests stay in full control.

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
from .core.neuron import Neuron, StubTimerNeuron
from .core.signal_bus import SignalBus
from .logging import EventLogger
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

    ``neurons`` defaults to a single :class:`StubTimerNeuron` (the Phase-1.2
    autonomic layer). ``None`` means "take the default"; passing any sequence —
    including an empty ``()`` — overrides it, so callers keep full control.
    """
    resolved_state: StatePort = state or JsonStateStore(base_dir, logger=logger)
    resolved_bus: SignalBus = bus or FileSignalBus(base_dir, logger=logger)
    resolved_clock: ClockPort = clock or SystemClock()
    resolved_delivery: DeliveryPort = delivery or NoopDelivery(logger=logger)
    resolved_aggregator: Aggregator = aggregator or SilentAggregator()
    resolved_neurons: tuple[Neuron, ...] = (
        (StubTimerNeuron(logger=logger),) if neurons is None else tuple(neurons)
    )

    return LifeModel(
        state=resolved_state,
        bus=resolved_bus,
        clock=resolved_clock,
        delivery=resolved_delivery,
        aggregator=resolved_aggregator,
        neurons=resolved_neurons,
    )

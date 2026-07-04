"""``Neuron`` — the autonomic-layer extension point (HLA §1/§2/§13).

Neurons are the brain stem: zero-LLM monitors that read the being's state each
tick, accumulate pressure, and emit :class:`~lifemodel.domain.signal.Signal`
objects when a threshold is crossed (HLA §2). Each new sensor — connection,
thoughts, commitments — is a new :class:`Neuron` subclass, so the autonomic
layer grows by adding implementations, never by editing a dispatcher (HLA §13,
lego-swappability).

This module ships the contract plus the first concrete neuron,
:class:`StubTimerNeuron` (Phase 1.2) — mirroring how
:class:`~lifemodel.core.aggregator.SilentAggregator` ships beside its ABC. Real
behavioural neurons arrive in Phase 2+. A neuron reads state but does **not**
persist it — the tick orchestrator owns the single state commit (HLA §9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..domain.signal import Signal
from ..state.model import State

if TYPE_CHECKING:  # only a type reference — keep core's runtime imports minimal
    from ..logging import EventLogger

#: Signal ``kind`` the stub timer neuron stamps on every impulse (HLA §2).
TIMER_PRESSURE_KIND = "timer_pressure"

#: Structured event the stub timer neuron emits each time it fires — operational
#: observability (like ``plugin_registered``), so a debug trace shows neuron
#: activity. Not part of the queryable §12 debug vocabulary; the neuron's
#: *effect* (growing ``State.pressure``) is what the debug state section shows.
EVENT_NEURON_FIRED = "neuron_fired"

#: Default pressure delta each tick contributes. The neuron owns this weight;
#: the tick orchestrator only sums the emitted deltas into ``State.pressure``.
_DEFAULT_DELTA = 1.0


class Neuron(ABC):
    """A zero-LLM monitor that turns state into signals each tick (HLA §2)."""

    @abstractmethod
    def tick(self, state: State) -> list[Signal]:
        """Inspect *state* and return any signals fired this tick (may be empty).

        Pure with respect to persistence: read ``state``, return signals; do not
        write the store. Returning ``[]`` is the common, quiet case.
        """
        raise NotImplementedError


class StubTimerNeuron(Neuron):
    """A deterministic timer neuron: one fixed-delta pressure signal per tick.

    The first (engine-validation) neuron of Phase 1.2. It carries no behaviour —
    the meaningful time-since-contact neuron is Phase 2.1 — but it exercises the
    live autonomic seam end to end: every tick it emits **exactly one**
    :class:`Signal` whose ``salience`` is a fixed pressure *delta*. The tick
    orchestrator sums those deltas into :attr:`~lifemodel.state.model.State.pressure`,
    so a single number accumulates and persists across ticks at zero LLM cost
    (HLA §1/§2).

    The neuron **owns the delta** and nothing else. It does not mutate state (the
    tick is the single writer, HLA §9) and knows nothing of thresholds or waking
    (that stays the aggregator's job, Phase 1.3). Its signal carries an
    ``origin_id`` derived from ``state.tick_count`` — the pre-increment count the
    neuron reads — so the id is **stable within a tick** (dedup counts the
    impulse exactly once) yet **distinct across ticks** (two ticks are never
    collapsed together), HLA §10.
    """

    def __init__(self, *, delta: float = _DEFAULT_DELTA, logger: EventLogger | None = None) -> None:
        self._delta = delta
        self._logger = logger

    def tick(self, state: State) -> list[Signal]:
        """Emit one fixed-delta pressure signal, tagged with the tick's identity."""
        origin_id = f"timer-{state.tick_count}"
        if self._logger is not None:
            self._logger.info(
                EVENT_NEURON_FIRED,
                neuron="stub_timer",
                origin_id=origin_id,
                delta=self._delta,
            )
        return [Signal(origin_id=origin_id, kind=TIMER_PRESSURE_KIND, salience=self._delta)]

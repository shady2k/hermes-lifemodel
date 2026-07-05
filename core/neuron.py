"""``Neuron`` — the autonomic-layer extension point (HLA §1/§2/§13).

Neurons are the brain stem: zero-LLM monitors that read the being's state each
tick and emit :class:`~lifemodel.domain.signal.Signal` objects when a threshold
is crossed (HLA §2). Each new sensor — connection, thoughts, commitments — is a
new :class:`Neuron` subclass, so the autonomic layer grows by adding
implementations, never by editing a dispatcher (HLA §13, lego-swappability).

This module ships only the contract. The Phase-1.2 stub concrete neuron
(``StubTimerNeuron``) and the global-pressure model it fed were removed by the
wire-desire-model plan (Task 8): the certified desire model in
:mod:`lifemodel.sim` — reconstructed each tick by :mod:`lifemodel.core.decision`
— is now the sole source of the drive, bypassing this neuron/aggregator seam
entirely in the live path. Real behavioural neurons, if this seam is revived,
are a later-phase concern. A neuron reads state but does **not** persist it —
the tick orchestrator owns the single state commit (HLA §9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.signal import Signal
from ..state.model import State


class Neuron(ABC):
    """A zero-LLM monitor that turns state into signals each tick (HLA §2)."""

    @abstractmethod
    def tick(self, state: State) -> list[Signal]:
        """Inspect *state* and return any signals fired this tick (may be empty).

        Pure with respect to persistence: read ``state``, return signals; do not
        write the store. Returning ``[]`` is the common, quiet case.
        """
        raise NotImplementedError

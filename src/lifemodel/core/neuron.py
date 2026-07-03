"""``Neuron`` — the autonomic-layer extension point (HLA §1/§2/§13).

Neurons are the brain stem: zero-LLM monitors that read the being's state each
tick, accumulate pressure, and emit :class:`~lifemodel.domain.signal.Signal`
objects when a threshold is crossed (HLA §2). Each new sensor — connection,
thoughts, commitments — is a new :class:`Neuron` subclass, so the autonomic
layer grows by adding implementations, never by editing a dispatcher (HLA §13,
lego-swappability).

This is the contract only. Phase 1.2 fills in the first (stub-timer) neuron;
real behavioural neurons arrive in Phase 2+. A neuron reads state but does **not**
persist it — the tick orchestrator owns the single state commit (HLA §9).
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

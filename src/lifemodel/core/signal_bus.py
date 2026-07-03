"""``SignalBus`` — the durable signal log extension point (HLA §2/§10/§13).

The signal bus is the nervous system's wire: a **durable, append-only log** of
signals. Producers are the neuron tick and the gateway turn; the single consumer
is the aggregator, which reads *unprocessed* signals with a filter+dedup by
stable origin id (HLA §2/§10). Routing both inputs through one log is what keeps
the being's state consistent — the incoming turn and the proactive tick write to
one place (HLA §10).

This is the contract. The durable file adapter
(:class:`~lifemodel.adapters.signal_bus.FileSignalBus`) and the in-memory
:class:`~lifemodel.testing.fakes.FakeSignalBus` implement it. Dedup semantics
(HLA §10) that every implementation must honour:

* ``publish`` appends honestly — the same origin id may be logged twice (once by
  the gateway turn, once by the next tick); the *consumer* dedups.
* ``consume_unprocessed`` returns each origin id **at most once ever**: it skips
  ids already consumed on a prior call and collapses duplicates within a batch,
  then marks the returned ids consumed durably so a later call — or a fresh bus
  over the same storage — never returns them again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.signal import Signal


class SignalBus(ABC):
    """A durable append-only log of signals with dedup on consume (HLA §2/§10)."""

    @abstractmethod
    def publish(self, signal: Signal) -> None:
        """Append *signal* to the durable log (a producer wrote it)."""
        raise NotImplementedError

    @abstractmethod
    def consume_unprocessed(self) -> list[Signal]:
        """Return not-yet-consumed signals (deduped by origin id) and mark them.

        Idempotent across calls and restarts: once an origin id is returned it is
        never returned again (HLA §10). The returned list preserves publish order
        and holds each origin id at most once.
        """
        raise NotImplementedError

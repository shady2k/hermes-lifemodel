"""In-memory fakes for every Phase-1 port — "imitations before code" (HLA §13).

The DI contract says tests inject fakes, not real adapters. These are the
canonical ones: a fake clock, delivery, state store, and signal bus, each
satisfying its port/ABC. They are shipped in the package (not hidden in a test
folder) so later tasks (1.1–1.4) reuse the *same* fakes their upstream defined,
rather than re-rolling subtly different ones. Stdlib only; no Hermes.

Each fake is deliberately transparent — it exposes the recorded inputs
(``FakeDelivery.sent``, ``FakeClock`` mutators) so a test can assert on them.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta

from ..core.signal_bus import SignalBus
from ..domain.signal import Signal
from ..state.model import State


class FakeClock:
    """A :class:`~lifemodel.ports.clock.ClockPort` pinned to a controllable time.

    Construct with a timezone-aware UTC ``datetime``; move it with
    :meth:`advance` / :meth:`set` so tests exercise elapsed-time logic
    (cooldowns, the connection neuron) deterministically.
    """

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by *delta*."""
        self._now += delta

    def set(self, now: datetime) -> None:
        """Pin the clock to an absolute instant."""
        self._now = now


class FakeDelivery:
    """A :class:`~lifemodel.ports.delivery.DeliveryPort` that records sends.

    Nothing leaves the process; every ``send`` is appended to :attr:`sent` as a
    ``(channel, text)`` pair for the test to assert on.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, channel: str, text: str) -> None:
        self.sent.append((channel, text))


class FakeStateStore:
    """An in-memory :class:`~lifemodel.state.port.StatePort`.

    Holds one ``State`` in memory (the documented default until first commit).
    Deep-copies on the way in and out so a caller mutating its own ``State`` can
    never reach through and change what the store holds — matching the isolation
    a real serializing store gives.
    """

    def __init__(self, initial: State | None = None) -> None:
        self._state = copy.deepcopy(initial) if initial is not None else State()

    def load(self) -> State:
        return copy.deepcopy(self._state)

    def commit(self, state: State) -> None:
        self._state = copy.deepcopy(state)


class FakeSignalBus(SignalBus):
    """An in-memory :class:`SignalBus` with the same dedup contract (HLA §10).

    Mirrors :class:`~lifemodel.adapters.signal_bus.FileSignalBus` without the
    filesystem: ``publish`` appends to an in-memory log, ``consume_unprocessed``
    returns each origin id at most once ever (deduped within a batch and across
    calls). It has no on-disk state, so restart-persistence is out of scope —
    that behaviour is covered by the file bus.
    """

    def __init__(self) -> None:
        self._log: list[Signal] = []
        self._consumed: set[str] = set()

    def publish(self, signal: Signal) -> None:
        self._log.append(signal)

    def consume_unprocessed(self) -> list[Signal]:
        fresh = self._unprocessed()
        self._consumed.update(s.origin_id for s in fresh)
        return fresh

    def peek_unprocessed(self) -> list[Signal]:
        """Read-only view of the unprocessed signals — never marks them consumed.

        Parity with :meth:`FileSignalBus.peek_unprocessed` so the debug path's
        read-only bus inspection can be exercised with this fake.
        """
        return self._unprocessed()

    def _unprocessed(self) -> list[Signal]:
        fresh: list[Signal] = []
        seen = set(self._consumed)
        for signal in self._log:
            if signal.origin_id in seen:
                continue
            seen.add(signal.origin_id)
            fresh.append(signal)
        return fresh

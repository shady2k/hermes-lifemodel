"""``ClockPort`` — the boundary for reading wall-clock time (HLA §13).

Time is a real, impure dependency: the core must not call ``datetime.now()``
directly or its logic becomes untestable and non-deterministic. Neurons read the
clock to measure elapsed time (the "connection" neuron, 2.1), the tick stamps
``last_tick_at``, and the act-gate computes cooldowns (1.4). Injecting this port
lets tests pin time with a :class:`~lifemodel.testing.fakes.FakeClock`.

``now()`` returns a **timezone-aware UTC ``datetime``** — the richest primitive:
callers that persist format it to an ISO-8601 string (state timestamps are
strings, see :class:`~lifemodel.state.model.State`), and callers that measure
durations (cooldowns) subtract two ``datetime`` values. One method, both needs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class ClockPort(Protocol):
    """A source of the current instant, injected so time is fake-able (§13)."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC ``datetime``."""
        ...

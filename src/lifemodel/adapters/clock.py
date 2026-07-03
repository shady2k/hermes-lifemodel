"""``SystemClock`` — the real wall clock behind :class:`ClockPort` (HLA §13).

Reads the host's clock as timezone-aware UTC. This is the only place production
code touches ``datetime.now``; everything else takes a
:class:`~lifemodel.ports.clock.ClockPort`, so tests inject a
:class:`~lifemodel.testing.fakes.FakeClock` instead. Stdlib only, no Hermes.
"""

from __future__ import annotations

from datetime import UTC, datetime


class SystemClock:
    """A :class:`~lifemodel.ports.clock.ClockPort` backed by the host clock."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC ``datetime``."""
        return datetime.now(UTC)

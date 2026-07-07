"""``PressureSensorPort`` — a live read of the being's contact pressure (HLA §4.1).

A narrow, single-method port (HLA §13, "pragmatism, not ceremony") so a future
aggregation/cognition consumer can ask "is there a contact frame available,
and how strong is it?" without depending on :class:`~lifemodel.ports.memory.MemoryPort`'s
full CRUD surface, and without depending on Hermes. The real implementation is
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (a parametrized
``SELECT`` over ``memory_records`` — deliberately not a SQL ``VIEW``, so the
"now" it compares against is always the caller's injected instant, never
``CURRENT_TIMESTAMP``); tests inject
:class:`~lifemodel.testing.fakes.FakePressureSensor`. Purely additive as of
lm-fib.6.1 — nothing in the live tick reads this port yet (that lands with
lm-fib.6.3's DesireFrame minting).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..domain.memory import PressureIndex


@runtime_checkable
class PressureSensorPort(Protocol):
    """Read the live contact-pressure summary as of a given instant (§4.1)."""

    def read_pressure_index(self, now: datetime) -> PressureIndex:
        """Summarize active, unexpired ``kind='desire'`` records as of *now*.

        ``active_desire_count`` counts records with ``kind='desire'``,
        ``state='active'``, and ``expires_at`` either unset or strictly after
        *now*; ``max_desire_salience`` is the highest ``salience`` among them
        (``0.0`` when there are none); ``contact_frame_available`` is
        ``active_desire_count > 0``.

        **Fail-soft**: a transient/operational sensor error (e.g. the database
        is momentarily locked) returns the default ``PressureIndex()`` rather
        than raising — a stalled read must never crash the caller. A schema
        error (e.g. the table does not exist) is a genuine init bug, not a
        transient condition, and is *not* swallowed — it propagates.
        """
        ...

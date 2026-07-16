"""Test helper: a bare :class:`~lifemodel.core.component.TickContext` builder (§7.2).

Promotes ``tests/test_aggregation.py``'s local ``_ctx`` helper (a literal
:class:`~lifemodel.ports.tracer.TraceContext` + a ``State`` + a signals tuple) into
``testing/`` so every single-component unit test shares ONE builder instead of
re-rolling a local one per test module — lm-705.1 Task 3 needed this for
``ThoughtCapture``'s own test, and it is generic enough for any future component
test that only needs a bare, un-wired :class:`TickContext` (no live tracer, no
logger, no metrics — every field a component might read defaults to its neutral
value).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from ..core.component import TickContext
from ..domain.memory import MemoryRecord, PressureIndex
from ..domain.signal import Signal
from ..ports.tracer import TraceContext
from ..state.model import State

#: A fixed, valid W3C trace context for a bare unit-test TickContext. No live
#: tracer is needed: ``TickContext.trace`` is a plain value (spec §4.1), not a
#: ``TracerPort`` — mirrors ``tests/test_aggregation.py``'s module-level ``_TRACE``.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)

#: A fixed instant so a caller that does not care about "now" still gets a
#: deterministic, timezone-aware one.
_NOW = datetime(2026, 1, 1, tzinfo=UTC)

#: The neutral "nothing pressing" pressure index (frozen/immutable, so one shared
#: instance is safe to default to) — a module-level singleton rather than a call in
#: the signature default (ruff B008), mirroring :data:`_TRACE`/:data:`_NOW` above.
_PRESSURE = PressureIndex()


def make_tick_context(
    *,
    state: State | None = None,
    now: datetime = _NOW,
    signals: Iterable[Signal] = (),
    objects: Iterable[MemoryRecord] = (),
    pressure: PressureIndex = _PRESSURE,
    trace: TraceContext = _TRACE,
) -> TickContext:
    """A bare :class:`TickContext` for unit-testing ONE component in isolation.

    Every field defaults to its neutral/empty value, so a minimal
    ``make_tick_context(signals=[...])`` call is a plain frame carrying just the
    signals under test. ``logger``/``tracer``/``metrics``/``observe`` are left
    ``None`` (no harness) — a component under test must guard those the same way
    it does in a live tick with no graph.
    """
    return TickContext(
        state=state if state is not None else State(),
        now=now,
        signals=tuple(signals),
        objects=tuple(objects),
        pressure=pressure,
        trace=trace,
    )

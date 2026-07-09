"""The per-tick component seam (spec §3, §7.2).

A component is anything the CoreLoop schedules on a tick — a neuron, an
aggregation stage, the personality, cognition. It reads an immutable
:class:`TickContext` (state snapshot + clock + bus) and returns intents; it
never mutates state. Kept deliberately minimal so every layer can implement it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..domain.memory import MemoryRecord, PressureIndex
from ..domain.signal import Signal
from ..log import SpanBoundLogger
from ..ports.tracer import TraceContext
from ..state.model import State
from .intents import Intent
from .signal_bus import SignalBus


@dataclass(frozen=True)
class TickContext:
    """Read-only inputs handed to every component on a tick.

    ``pressure`` and ``objects`` are the being's *start-of-tick* snapshot — read
    once, before any component runs, so every component this tick sees the same
    consistent view (HLA §4.1). Both default to empty so the many existing
    construction sites keep compiling; no component reads them yet (lm-27n.2
    installs the snapshot; aggregation consumes it in .3).
    """

    state: State
    now: datetime
    bus: SignalBus
    #: THE component's active execution span's W3C ids (spec §5) — set by the
    #: CoreLoop to the CHILD span it minted for this component (parented on the
    #: tick's root), so a creation site stamps the born object's provenance with
    #: it. NON-OPTIONAL: tracing is mandatory (§4.1) — a tick without a span is
    #: structurally impossible, so every construction site MUST supply one (a bare
    #: unit test mints a ``FakeTracer().start_root()``). The mutable
    #: :class:`~lifemodel.ports.tracer.ActiveSpan` a component drops decision
    #: values onto is reached via :attr:`logger` (``ctx.logger.span``).
    trace: TraceContext
    signals: tuple[Signal, ...] = ()
    #: The being's live contact-pressure summary as of ``now`` (start of tick).
    pressure: PressureIndex = PressureIndex()
    #: A bounded snapshot of the being's ``state="active"`` memory records.
    objects: tuple[MemoryRecord, ...] = ()
    #: THE component's span-bound logger (spec §4.1), set by the CoreLoop to a
    #: :class:`~lifemodel.log.SpanLogger` over this component's child span. A
    #: component logs through it (self-stamping trace/span/tick) and drops decision
    #: values onto ``logger.span`` so the span is self-explaining. ``None`` in a
    #: bare unit-test ``TickContext`` (no graph) — observability emission is skipped.
    logger: SpanBoundLogger | None = None


@runtime_checkable
class Component(Protocol):
    """A schedulable unit. ``id`` is stable and unique within a registry."""

    id: str

    def step(self, ctx: TickContext) -> Sequence[Intent]: ...

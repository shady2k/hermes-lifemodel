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
from ..log import EventLogger
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
    installs the snapshot; aggregation consumes it in .3). Deliberately
    extensible: the ``trace`` field (lm-27n.11) slots in additively here.
    """

    state: State
    now: datetime
    bus: SignalBus
    signals: tuple[Signal, ...] = ()
    #: The being's live contact-pressure summary as of ``now`` (start of tick).
    pressure: PressureIndex = PressureIndex()
    #: A bounded snapshot of the being's ``state="active"`` memory records.
    objects: tuple[MemoryRecord, ...] = ()
    #: THE component's active execution span (spec §5) — set by the CoreLoop to the
    #: CHILD span it minted for this component (parented on the tick's root), so a
    #: creation site stamps the born object's provenance with it and the component's
    #: own logs bind to it. Tracing is mandatory in the live tick (the tracer is a
    #: required CoreLoop dependency), so the CoreLoop always supplies a span; the
    #: ``None`` default is only for direct unit-test construction of a ``TickContext``.
    trace: TraceContext | None = None
    #: The graph's shared :class:`EventLogger`, set by the CoreLoop so a component
    #: can emit observability (e.g. a suppression span, spec §5) through the SAME
    #: collaborator the rest of the graph uses. ``None`` in a bare unit-test
    #: ``TickContext`` (no graph) — observability emission is then skipped.
    logger: EventLogger | None = None


@runtime_checkable
class Component(Protocol):
    """A schedulable unit. ``id`` is stable and unique within a registry."""

    id: str

    def step(self, ctx: TickContext) -> Sequence[Intent]: ...

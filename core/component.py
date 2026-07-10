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
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from ..domain.memory import MemoryRecord, PressureIndex
from ..domain.signal import Signal
from ..events import EventRing
from ..log import SpanBoundLogger
from ..ports.tracer import TraceContext, TracerPort
from ..state.model import State
from ..state.trace_store import NULL_TRACE_SINK, TraceSink
from .intents import Intent
from .metrics import MetricRegistry
from .signal_bus import SignalBus


class ComponentLayer(StrEnum):
    """The CLOSED architectural layer a component belongs to (telemetry-core §4.2).

    The being's brain runs ``AUTONOMIC → AGGREGATION → COGNITION`` (HLA §1); an
    :data:`INFRA` layer names the out-of-band operational components (proactive
    egress, the trace writer) that are not a brain stage. This is the low-cardinality
    ``layer`` label carried by every component/tick metric — a :class:`~enum.StrEnum`
    so a member IS its own label value (``ComponentLayer.AUTONOMIC == "autonomic"``).

    Named ``ComponentLayer`` (not ``Layer``) deliberately: :class:`lifemodel.core.layer.Layer`
    is the unrelated brain-stage ABC extension point. This enum tags a *registered
    component*; that ABC is a *processing stage* base class.
    """

    AUTONOMIC = "autonomic"
    AGGREGATION = "aggregation"
    COGNITION = "cognition"
    INFRA = "infra"


#: Static ``ComponentManifest.type`` → :class:`ComponentLayer` rollup
#: (telemetry-core §4.2 / §4.5). The composition root fills each manifest's
#: ``layer`` from this, and ``/lifemodel stats`` (bead 7.7) folds components up to
#: their layer through the same table. ``egress`` is deliberately :data:`INFRA`,
#: NOT ``COGNITION`` — an unknown type maps to nothing (see :func:`layer_for_type`).
LAYER_BY_TYPE: Final[dict[str, ComponentLayer]] = {
    "personality": ComponentLayer.AUTONOMIC,
    "neuron": ComponentLayer.AUTONOMIC,
    "drive": ComponentLayer.AUTONOMIC,
    "aggregation": ComponentLayer.AGGREGATION,
    "launcher": ComponentLayer.COGNITION,
    "cognition": ComponentLayer.COGNITION,
    "proactive": ComponentLayer.INFRA,
    "egress": ComponentLayer.INFRA,
    "trace-writer": ComponentLayer.INFRA,
    "writer": ComponentLayer.INFRA,
}


def layer_for_type(component_type: str) -> ComponentLayer | None:
    """The :class:`ComponentLayer` for a component *type*, or ``None`` if unknown.

    A convenience over :data:`LAYER_BY_TYPE`. Returning ``None`` for an unknown
    type is load-bearing at the composition root: a manifest built with
    ``layer=layer_for_type(<typo>)`` then carries ``None`` and
    :meth:`~lifemodel.core.registry.ComponentRegistry.register` rejects it
    fail-fast (§3) — a new component type can't slip in unlayered.
    """
    return LAYER_BY_TYPE.get(component_type)


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
    #: The async-bridge emit trio (spec §4.4, generic): the SAME tracer + durable
    #: sink + freshness ring the CoreLoop fans in-tick logging onto, exposed so a
    #: component can mint a span under a DIFFERENT (foreign) origin trace than this
    #: tick's — e.g. aggregation emitting the resolution span of a proactive attempt
    #: under its launch's ``origin_traceparent`` (raised from state), not this tick's
    #: trace. ``None``/no-op in a bare unit-test context (no async weave to bridge).
    tracer: TracerPort | None = None
    trace_writer: TraceSink = NULL_TRACE_SINK
    event_ring: EventRing | None = None
    #: The shared metric registry (telemetry-core §4.2), set by the CoreLoop so a
    #: component gate can emit its suppression through the choke-point
    #: (:func:`~lifemodel.core.suppression.emit_suppression_span`) and be counted
    #: into ``lifemodel_suppressions_total``. ``None`` in a bare unit-test context
    #: (no harness) — the choke-point then simply records no metric.
    metrics: MetricRegistry | None = None


@runtime_checkable
class Component(Protocol):
    """A schedulable unit. ``id`` is stable and unique within a registry."""

    id: str

    def step(self, ctx: TickContext) -> Sequence[Intent]: ...

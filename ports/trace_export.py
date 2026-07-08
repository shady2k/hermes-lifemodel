"""``TraceExportPort`` — the boundary that ships a finished tick to a trace backend (lm-27n.10).

Deliberately NOT a decorator over :class:`~lifemodel.ports.tracer.TracerPort`: the
tracer only *mints* context (no span lifecycle to wrap). Export is a separate,
**tick-end** capability — the CoreLoop hands the finished
:class:`~lifemodel.core.coreloop.TickReport` plus the tick's root
:class:`~lifemodel.ports.tracer.TraceContext` to this port after the commit.

The default adapter (:class:`~lifemodel.adapters.trace_export.NoopTraceExporter`)
does nothing, so with no OpenTelemetry in the Hermes venv the whole path is a
behaviour-neutral no-op; the CoreLoop calls it best-effort, so an exporter that
raises never affects the tick outcome.

``TickReport`` is imported only under ``TYPE_CHECKING`` (it is a ``core`` DTO): the
annotation stays a string at runtime (``from __future__ import annotations``), so
this ports module imports no ``core`` at runtime and the layering stays clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .tracer import TraceContext

if TYPE_CHECKING:
    from ..core.coreloop import TickReport


@runtime_checkable
class TraceExportPort(Protocol):
    """Ship one finished tick to a trace backend (injected so it is fake-able)."""

    def export_tick(self, report: TickReport, trace: TraceContext | None) -> None:
        """Export the finished *tick* (its root span + report attributes).

        Best-effort by contract: the CoreLoop wraps the call, but an implementation
        should still avoid raising. ``trace`` is ``None`` for an untraced tick (no
        tracer wired) — the exporter then has no root span to ship.
        """
        ...

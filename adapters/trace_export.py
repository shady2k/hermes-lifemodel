"""Trace-export adapters — the no-op default + an optional OpenTelemetry one (lm-27n.10).

Runtime is **stdlib only**: there is NO hard ``opentelemetry`` import. The factory
:func:`make_trace_exporter` tries to import it *inside* the call (via an injectable
``import_module`` seam) and returns :class:`OtelTraceExporter` only when it is
present, else :class:`NoopTraceExporter` — the common case in the Hermes venv, where
the whole export path is a behaviour-neutral no-op.

:class:`OtelTraceExporter` exports ONLY the tick ROOT span: the W3C ids
(``trace_id``/``span_id``/``parent_span_id``) and the report attributes
(``tick``/``ran``/``failed``/``committed``/``launch_count``) — no per-component
spans. It gets its OTel tracer lazily (on first export) and never imports OTel at
module scope, so importing this adapter off-host stays free.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..ports.tracer import TraceContext

if TYPE_CHECKING:
    from ..core.coreloop import TickReport
    from ..ports.trace_export import TraceExportPort

#: The dotted module the OTel adapter needs; its absence is the no-op signal.
_OTEL_MODULE = "opentelemetry.trace"


class NoopTraceExporter:
    """The default :class:`~lifemodel.ports.trace_export.TraceExportPort`: does nothing.

    Behaviour-neutral by construction — with no trace backend wired, a tick's export
    is a no-op that never raises."""

    def export_tick(self, report: TickReport, trace: TraceContext | None) -> None:
        return None


class OtelTraceExporter:
    """Export the tick root span to OpenTelemetry (only when the dep is present).

    Constructed by :func:`make_trace_exporter` with the imported ``opentelemetry.trace``
    module; the tracer is resolved lazily so construction never touches OTel internals
    (and a partial/odd OTel build fails at export time, which the CoreLoop swallows,
    rather than at wiring time)."""

    def __init__(self, otel_trace: Any) -> None:
        self._otel_trace = otel_trace
        self._tracer: Any | None = None

    def export_tick(self, report: TickReport, trace: TraceContext | None) -> None:
        if self._tracer is None:
            self._tracer = self._otel_trace.get_tracer("lifemodel")
        span = self._tracer.start_span("lifemodel.tick")
        try:
            if trace is not None:
                span.set_attribute("trace_id", trace.trace_id)
                span.set_attribute("span_id", trace.span_id)
                if trace.parent_span_id is not None:
                    span.set_attribute("parent_span_id", trace.parent_span_id)
            span.set_attribute("tick", report.tick)
            span.set_attribute("ran", len(report.ran))
            span.set_attribute("failed", len(report.failed))
            span.set_attribute("committed", report.committed)
            span.set_attribute("launch_count", len(report.launches))
        finally:
            span.end()


def make_trace_exporter(
    *, import_module: Callable[[str], Any] = importlib.import_module
) -> TraceExportPort:
    """Return the OTel exporter iff ``opentelemetry`` imports, else the no-op default.

    The import happens HERE (not at module scope) so the runtime never hard-depends on
    OpenTelemetry. ``import_module`` is injectable so a test can deterministically force
    the absent case (an ``import_module`` that raises :class:`ImportError`)."""
    try:
        otel_trace = import_module(_OTEL_MODULE)
    except ImportError:
        return NoopTraceExporter()
    return OtelTraceExporter(otel_trace)

"""Correlated spans — bind observability to a FOREIGN origin trace (spec §4.4).

The async bridge in one primitive. A proactive attempt is launched inside tick N
(under trace ``T``), but its delivery, the being's async ``[SILENT]``/reply
verdict, and the resolving tick N+k all happen OUTSIDE that tick — with no in-band
W3C channel across the Hermes boundary (§4.4). So each of those out-of-band sites
raises the launch's ``origin_traceparent`` (from ``LaunchProactive`` or the state
anchor) and CONTINUES trace ``T`` here, instead of minting a fresh, disconnected
root. The result: one attempt = one ``trace_id``, readable from the one trace store.

Generic by construction (spec §3 law 4): nothing here is proactive-specific — it
weaves ANY async launch back onto its origin trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..events import EventRing
from ..log import SpanLogger
from ..ports.tracer import MutableActiveSpan, TracerPort, start_span
from ..state.trace_store import TraceSink


@dataclass
class CorrelatedSpan:
    """A span bound to a (possibly foreign) trace + its :class:`SpanLogger`.

    The caller emits events through :attr:`logger` (which self-stamps the origin
    trace's ids), drops decision values onto :attr:`span`, then calls
    :meth:`persist` to upsert the durable ``trace_spans`` row (spec §4.3).
    """

    span: MutableActiveSpan
    logger: SpanLogger
    writer: TraceSink

    def persist(self, *, ended_at: str | None = None) -> None:
        """Upsert this span's durable row through the writer (fail-open, §4.2)."""
        ctx = self.span.context
        self.writer.submit_span(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            component=self.span.component,
            tick=self.span.tick,
            started_at=self.span.started_at,
            ended_at=ended_at if ended_at is not None else self.span.ended_at,
            status=self.span.status,
            attrs=dict(self.span.attrs) or None,
        )


def open_correlated_span(
    *,
    tracer: TracerPort,
    writer: TraceSink,
    ring: EventRing,
    origin_traceparent: str | None,
    component: str,
    tick: int | None = None,
    started_at: str | None = None,
) -> CorrelatedSpan:
    """Open a span under *origin_traceparent*'s trace, or a fresh root if ``None``.

    ``start_root(upstream_traceparent=...)`` keeps the origin ``trace_id`` and mints
    a fresh child span parented on the origin (launch) span — so the emitted
    events/spans land under the ONE attempt trace, not the caller's ambient trace.
    ``origin_traceparent=None`` mints a brand-new root: the DELIBERATE miss path (a
    lost anchor → an ``orphan_async_outcome`` on its own trace, NEVER attached to a
    foreign one — spec §4.4 miss policy).
    """
    origin_ctx = tracer.start_root(upstream_traceparent=origin_traceparent)
    span = start_span(origin_ctx, component=component, tick=tick, started_at=started_at)
    logger = SpanLogger(span, writer=writer, ring=ring)
    return CorrelatedSpan(span=span, logger=logger, writer=writer)

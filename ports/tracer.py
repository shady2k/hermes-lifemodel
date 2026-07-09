"""``TracerPort`` — the boundary that mints/propagates execution trace context (lm-27n.11).

Tracing is a genuine, injectable capability (like the clock, HLA §13): the core
must not reach for a global tracer or mint ids inline, or its execution-correlation
logic becomes untestable and non-deterministic. The **capability** is a
:class:`TracerPort` (injected once at the composition root); the **active context**
is a per-tick :class:`TraceContext` value threaded through
:class:`~lifemodel.core.component.TickContext` — never an ambient contextvar (those
are for *log* decoration only, see :func:`lifemodel.log.bound_log_context`).

Trace ids follow the **W3C Trace Context data model** (not the OpenTelemetry SDK;
stdlib only) — the very model :class:`~lifemodel.domain.objects.Provenance` already
persists — so a durable object's ``trace_id`` and a live log line JOIN on the same
correlation id. The traceparent codec is REUSED from
:mod:`lifemodel.domain.objects.provenance` (one W3C codec, not two).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.objects.provenance import (
    Provenance,
)
from ..domain.objects.provenance import (
    format_traceparent as _format_provenance_traceparent,
)
from ..domain.objects.provenance import (
    parse_traceparent as _parse_provenance_traceparent,
)

#: Default flags when a trace is minted fresh: ``"01"`` = sampled (W3C).
_DEFAULT_TRACE_FLAGS = "01"


@dataclass(frozen=True)
class TraceContext:
    """The ACTIVE execution trace for one execution unit (a tick / a reach-in turn).

    Frozen: the tick's root span is fixed once minted. ``span_id`` is *this* span
    (the tick root); ``parent_span_id`` is the upstream span it continues (``None``
    for a cron root). When it stamps a created object it becomes that object's
    :class:`~lifemodel.domain.objects.Provenance` *creation* context — ``span_id``
    maps to ``creation_span_id`` (never "the object's live span").
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: str = _DEFAULT_TRACE_FLAGS


@runtime_checkable
class TracerPort(Protocol):
    """A source of trace context, injected so tracing is fake-able (§13).

    One implementation mints real random ids (:class:`~lifemodel.adapters.tracer.StdlibTracer`);
    a test one hands out a deterministic sequence
    (:class:`~lifemodel.testing.fakes.FakeTracer`).
    """

    def start_root(self, *, upstream_traceparent: str | None = None) -> TraceContext:
        """Mint THE root span for an execution unit — CONTINUE-OR-MINT.

        If *upstream_traceparent* parses, KEEP its ``trace_id`` and set
        ``parent_span_id`` from it (a reach-in turn continues the caller's trace),
        minting a fresh ``span_id``. Otherwise mint a brand-new trace (a cron tick
        has no upstream): fresh ``trace_id`` + ``span_id``, ``parent_span_id=None``.
        """
        ...

    def child_of(self, parent: TraceContext) -> TraceContext:
        """A child span within *parent*'s trace (same ``trace_id``, new ``span_id``,
        ``parent_span_id = parent.span_id``). The CoreLoop mints one per component,
        parented on the tick's root span, so every component runs in its own span
        (spec §4.2/§5) — the span tree that makes "which component did what / why
        silent" observable. ``ctx.trace`` carries the child; the root stays the
        creation parent for tick-level bookkeeping."""
        ...

    def format_traceparent(self, ctx: TraceContext) -> str:
        """Render *ctx* as a W3C ``traceparent`` header string."""
        ...

    def parse_traceparent(self, value: str) -> TraceContext:
        """Parse a W3C ``traceparent`` string into a :class:`TraceContext`."""
        ...


def format_traceparent(ctx: TraceContext) -> str:
    """Render *ctx* as a W3C ``traceparent`` string, REUSING the durable codec.

    Round-trips exactly what :class:`~lifemodel.domain.objects.Provenance` persists,
    so a formatted live trace and a formatted stored provenance are byte-identical.
    """
    return _format_provenance_traceparent(
        Provenance(
            created_by="tracer",
            component="tracer",
            reason="traceparent",
            trace_id=ctx.trace_id,
            creation_span_id=ctx.span_id,
            trace_flags=ctx.trace_flags,
        )
    )


def parse_traceparent(value: str) -> TraceContext:
    """Parse a W3C ``traceparent`` string into a :class:`TraceContext` (validated).

    Delegates to the provenance codec, so an untrusted upstream header is validated
    (version/hex/non-zero) on the way in; ``parent_span_id`` is ``None`` (a parsed
    header carries only the caller's own trace+span, which becomes our parent).
    """
    trace_id, span_id, flags = _parse_provenance_traceparent(value)
    return TraceContext(trace_id=trace_id, span_id=span_id, parent_span_id=None, trace_flags=flags)

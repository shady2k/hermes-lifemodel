"""``TracerPort`` — the boundary that mints/propagates execution trace context (lm-27n.11).

Tracing is a genuine, injectable capability (like the clock, HLA §13): the core
must not reach for a global tracer or mint ids inline, or its execution-correlation
logic becomes untestable and non-deterministic. The **capability** is a
:class:`TracerPort` (injected once at the composition root); the **active context**
is a per-tick :class:`TraceContext` value threaded through
:class:`~lifemodel.core.component.TickContext` — never an ambient contextvar. A
component's :class:`~lifemodel.log.SpanLogger` self-stamps the span's ids onto every
record, so no ambient bind is needed.

Trace ids follow the **W3C Trace Context data model** (not the OpenTelemetry SDK;
stdlib only) — the very model :class:`~lifemodel.domain.objects.Provenance` already
persists — so a durable object's ``trace_id`` and a live log line JOIN on the same
correlation id. The traceparent codec is REUSED from
:mod:`lifemodel.domain.objects.provenance` (one W3C codec, not two).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

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


#: The closed set of terminal span outcomes recorded in ``trace_spans.status``
#: (spec §4.3). ``ok`` = ran and produced its effect; ``suppressed`` = a
#: deliberate no-act with a closed-enum reason; ``failed`` = raised.
SpanStatus = Literal["ok", "suppressed", "failed"]


@runtime_checkable
class ActiveSpan(Protocol):
    """A MUTABLE, in-flight span handle — the decision-values carrier (spec §4.1).

    Deliberately SEPARATE from the frozen :class:`TraceContext` (which stays a
    pure W3C-id carrier): a component receives an ``ActiveSpan``, drops the
    values that explain its decision onto it with :meth:`set` (``u``,
    ``effective_pressure``, gate outcomes, …), and closes it with :meth:`end`.
    The trace store serializes the accumulated :attr:`attrs` into
    ``trace_spans.attrs_json`` so a span is *self-explaining* — the gap §1.3
    named. The handle wraps a :class:`TraceContext` (its immutable ids) plus the
    mutable ``attrs``/``status``/timing that only settle as the component runs.

    Structural: :class:`~lifemodel.adapters.tracer.MutableActiveSpan` is the live
    implementation and :class:`~lifemodel.testing.fakes.FakeActiveSpan` the test
    double, so callers depend on this seam, not a concrete class.
    """

    @property
    def context(self) -> TraceContext:
        """The immutable W3C ids (trace/span/parent/flags) this span runs under."""
        ...

    @property
    def component(self) -> str | None:
        """The scheduling component that owns this span (``None`` for a bare root)."""
        ...

    @property
    def tick(self) -> int | None:
        """The tick number this span belongs to (``None`` outside a tick)."""
        ...

    @property
    def status(self) -> SpanStatus:
        """The current terminal outcome — ``ok`` until :meth:`end` sets otherwise."""
        ...

    @property
    def started_at(self) -> str | None:
        """ISO-8601 UTC start instant, or ``None`` if unstamped."""
        ...

    @property
    def ended_at(self) -> str | None:
        """ISO-8601 UTC end instant, set by :meth:`end` (``None`` while in-flight)."""
        ...

    @property
    def attrs(self) -> Mapping[str, Any]:
        """The read-only view of decision values accumulated via :meth:`set`."""
        ...

    def set(self, **attrs: Any) -> ActiveSpan:
        """Merge *attrs* into this span's attribute bag; returns ``self`` to chain."""
        ...

    def end(self, *, status: SpanStatus = "ok", ended_at: str | None = None) -> ActiveSpan:
        """Close the span with a terminal *status* (+ optional *ended_at*); chains."""
        ...


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


@dataclass
class MutableActiveSpan:
    """The live :class:`ActiveSpan` — a mutable handle (spec §4.1).

    A component receives one of these, stamps decision values onto it with
    :meth:`set`, and closes it with :meth:`end`. It carries the immutable W3C ids
    in :attr:`context` (never mutated — the frozen :class:`TraceContext`) alongside
    the attribute bag, ``status`` and timing that settle as the component runs. The
    trace store reads these fields to persist one ``trace_spans`` row (spec §4.3).

    Lives beside the :class:`ActiveSpan` protocol (not in ``adapters``) so the
    Hermes-free ``core`` scheduler can mint spans without importing the adapter
    layer; the id-minting :class:`~lifemodel.adapters.tracer.StdlibTracer` stays an
    adapter, this pure value handle does not.
    """

    context: TraceContext
    component: str | None = None
    tick: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    status: SpanStatus = "ok"
    #: The decision-value bag. Private so the public :attr:`attrs` view is
    #: read-only (mutation goes through :meth:`set`, the traced door).
    _attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return self._attrs

    def set(self, **attrs: Any) -> MutableActiveSpan:
        self._attrs.update(attrs)
        return self

    def end(self, *, status: SpanStatus = "ok", ended_at: str | None = None) -> MutableActiveSpan:
        self.status = status
        if ended_at is not None:
            self.ended_at = ended_at
        return self


def start_span(
    context: TraceContext,
    *,
    component: str | None = None,
    tick: int | None = None,
    started_at: str | None = None,
) -> MutableActiveSpan:
    """Open a fresh :class:`MutableActiveSpan` over *context* (spec §4.1).

    A thin, allocation-only factory: the caller (the CoreLoop) supplies the child
    :class:`TraceContext` it minted for the component plus the ``started_at`` it
    stamped from its clock. Kept separate from the tracer's id-minting so opening a
    span never draws randomness or touches a clock here.
    """
    return MutableActiveSpan(context=context, component=component, tick=tick, started_at=started_at)

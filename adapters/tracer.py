"""``StdlibTracer`` — the real :class:`TracerPort`, stdlib-only (lm-27n.11).

Mints W3C trace/span ids with :mod:`secrets` (a CSPRNG) — **no OpenTelemetry**, no
extra dependency, so it runs inside Hermes' own interpreter. This is the DEFAULT the
composition root wires; tests inject
:class:`~lifemodel.testing.fakes.FakeTracer` instead. The traceparent format/parse
methods delegate to the shared codec in :mod:`lifemodel.ports.tracer` (which reuses
the durable provenance codec), so there is exactly one W3C implementation.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..ports.tracer import (
    SpanStatus,
    TraceContext,
)
from ..ports.tracer import (
    format_traceparent as _format_traceparent,
)
from ..ports.tracer import (
    parse_traceparent as _parse_traceparent,
)

#: Default flags for a freshly minted trace: ``"01"`` = sampled (W3C).
_DEFAULT_TRACE_FLAGS = "01"
#: W3C field widths, in *bytes* (``token_hex`` doubles them into hex digits).
_TRACE_ID_BYTES = 16
_SPAN_ID_BYTES = 8


def _nonzero_hex(n_bytes: int) -> str:
    """A fresh lowercase-hex id of *n_bytes* bytes, never the all-zero id the W3C
    spec forbids for a trace/span id.

    ``secrets.token_hex`` already yields lowercase hex of the exact width; the loop
    only rejects the (astronomically improbable) all-zero draw so the minted id is
    always a valid, non-zero W3C id. A durable object re-validates it anyway when the
    id flows into a :class:`~lifemodel.domain.objects.Provenance`.
    """
    while True:
        token = secrets.token_hex(n_bytes)
        if any(ch != "0" for ch in token):
            return token


class StdlibTracer:
    """The default :class:`~lifemodel.ports.tracer.TracerPort` (stdlib CSPRNG)."""

    def start_root(self, *, upstream_traceparent: str | None = None) -> TraceContext:
        if upstream_traceparent is not None:
            # CONTINUE: keep the caller's trace, parent onto its span, mint our span.
            upstream = _parse_traceparent(upstream_traceparent)
            return TraceContext(
                trace_id=upstream.trace_id,
                span_id=_nonzero_hex(_SPAN_ID_BYTES),
                parent_span_id=upstream.span_id,
                trace_flags=upstream.trace_flags,
            )
        # MINT: a cron tick has no upstream — a brand-new trace with no parent.
        return TraceContext(
            trace_id=_nonzero_hex(_TRACE_ID_BYTES),
            span_id=_nonzero_hex(_SPAN_ID_BYTES),
            parent_span_id=None,
            trace_flags=_DEFAULT_TRACE_FLAGS,
        )

    def child_of(self, parent: TraceContext) -> TraceContext:
        return TraceContext(
            trace_id=parent.trace_id,
            span_id=_nonzero_hex(_SPAN_ID_BYTES),
            parent_span_id=parent.span_id,
            trace_flags=parent.trace_flags,
        )

    def format_traceparent(self, ctx: TraceContext) -> str:
        return _format_traceparent(ctx)

    def parse_traceparent(self, value: str) -> TraceContext:
        return _parse_traceparent(value)


@dataclass
class MutableActiveSpan:
    """The live :class:`~lifemodel.ports.tracer.ActiveSpan` — a mutable handle.

    A component receives one of these, stamps decision values onto it with
    :meth:`set`, and closes it with :meth:`end`. It carries the immutable W3C
    ids in :attr:`context` (never mutated — the frozen
    :class:`~lifemodel.ports.tracer.TraceContext`) alongside the attribute bag,
    ``status`` and timing that settle as the component runs. The trace store
    reads these fields to persist one ``trace_spans`` row (spec §4.3).
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

    A thin, allocation-only factory: the caller (the CoreLoop, later phases)
    supplies the child :class:`TraceContext` it minted for the component plus
    the ``started_at`` it stamped from its clock. Kept separate from the tracer's
    id-minting so opening a span never draws randomness or touches a clock here.
    """
    return MutableActiveSpan(context=context, component=component, tick=tick, started_at=started_at)

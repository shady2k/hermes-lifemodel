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

from ..ports.tracer import (
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

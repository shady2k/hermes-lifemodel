"""TracerPort / StdlibTracer / FakeTracer + creation_provenance (lm-27n.11).

The execution-correlation half of the observability invariant: a tracer is an
injected capability that mints W3C trace context (stdlib-only, NO OpenTelemetry),
continuing an upstream trace when present or minting a fresh root otherwise. The
FakeTracer hands out a deterministic sequence so provenance/log assertions are
stable, and ``creation_provenance`` turns a live TraceContext into the durable
Provenance a created object carries.
"""

from __future__ import annotations

import re

import pytest

from lifemodel.adapters.tracer import StdlibTracer
from lifemodel.core.trace import creation_provenance
from lifemodel.domain.objects import InvalidPayload
from lifemodel.ports.tracer import (
    TraceContext,
    TracerPort,
    format_traceparent,
    parse_traceparent,
)
from lifemodel.testing import FakeTracer

# A valid upstream W3C traceparent (the canonical spec example).
UPSTREAM_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
UPSTREAM_SPAN_ID = "00f067aa0ba902b7"
UPSTREAM_TRACEPARENT = f"00-{UPSTREAM_TRACE_ID}-{UPSTREAM_SPAN_ID}-01"

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")


def _tracers() -> list[TracerPort]:
    return [StdlibTracer(), FakeTracer()]


@pytest.mark.parametrize("tracer", _tracers())
def test_start_root_mints_a_valid_w3c_trace(tracer: TracerPort) -> None:
    ctx = tracer.start_root()
    assert _HEX32.match(ctx.trace_id)  # 32 lowercase hex, non-zero
    assert ctx.trace_id != "0" * 32
    assert _HEX16.match(ctx.span_id)
    assert ctx.span_id != "0" * 16
    assert ctx.parent_span_id is None  # a fresh root has no parent
    assert ctx.trace_flags == "01"


@pytest.mark.parametrize("tracer", _tracers())
def test_start_root_continues_an_upstream_trace(tracer: TracerPort) -> None:
    # CONTINUE-OR-MINT: an upstream traceparent -> keep the trace_id, parent onto its
    # span, mint a fresh span (a reach-in turn joins the caller's trace).
    ctx = tracer.start_root(upstream_traceparent=UPSTREAM_TRACEPARENT)
    assert ctx.trace_id == UPSTREAM_TRACE_ID  # same trace
    assert ctx.parent_span_id == UPSTREAM_SPAN_ID  # parent = upstream span
    assert ctx.span_id != UPSTREAM_SPAN_ID  # our own fresh span
    assert _HEX16.match(ctx.span_id)


@pytest.mark.parametrize("tracer", _tracers())
def test_start_root_mints_fresh_when_no_upstream(tracer: TracerPort) -> None:
    a = tracer.start_root()
    b = tracer.start_root()
    assert a.trace_id != b.trace_id  # each execution unit gets its own trace


@pytest.mark.parametrize("tracer", _tracers())
def test_child_of_keeps_trace_new_span_parent_set(tracer: TracerPort) -> None:
    parent = tracer.start_root()
    child = tracer.child_of(parent)
    assert child.trace_id == parent.trace_id  # same trace
    assert child.span_id != parent.span_id  # new span
    assert child.parent_span_id == parent.span_id  # parent = the parent's span


@pytest.mark.parametrize("tracer", _tracers())
def test_format_parse_round_trip(tracer: TracerPort) -> None:
    ctx = tracer.start_root()
    header = tracer.format_traceparent(ctx)
    assert header == f"00-{ctx.trace_id}-{ctx.span_id}-{ctx.trace_flags}"
    back = tracer.parse_traceparent(header)
    assert back.trace_id == ctx.trace_id
    assert back.span_id == ctx.span_id
    assert back.trace_flags == ctx.trace_flags


def test_module_codec_reuses_provenance_validation() -> None:
    # The module-level codec is the shared W3C door: round-trip a value, and reject a
    # malformed header (the provenance validators surface as InvalidPayload).
    ctx = TraceContext(trace_id=UPSTREAM_TRACE_ID, span_id=UPSTREAM_SPAN_ID)
    assert parse_traceparent(format_traceparent(ctx)).trace_id == UPSTREAM_TRACE_ID
    with pytest.raises(InvalidPayload):
        parse_traceparent("not-a-traceparent")
    with pytest.raises(InvalidPayload):
        parse_traceparent(f"00-{'0' * 32}-{UPSTREAM_SPAN_ID}-01")  # all-zero trace


def test_fake_tracer_is_a_deterministic_sequence() -> None:
    # Mirrors FakeClock: a fixed, stable sequence so provenance/log assertions pin.
    tracer = FakeTracer()
    first = tracer.start_root()
    second = tracer.start_root()
    assert first.trace_id == "0" * 31 + "1"
    assert first.span_id == "0" * 15 + "1"
    assert second.trace_id == "0" * 31 + "2"  # a fresh trace per tick
    assert second.span_id == "0" * 15 + "2"  # a fresh span too
    # A brand-new FakeTracer restarts the sequence — reproducible across tests.
    assert FakeTracer().start_root().trace_id == first.trace_id


def test_fake_tracer_child_keeps_trace_consumes_span() -> None:
    tracer = FakeTracer()
    root = tracer.start_root()
    child = tracer.child_of(root)
    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id
    assert child.span_id == "0" * 15 + "2"  # next span in the sequence


def test_creation_provenance_stamps_the_tick_trace() -> None:
    trace = FakeTracer().start_root(upstream_traceparent=UPSTREAM_TRACEPARENT)
    prov = creation_provenance(
        trace, created_by="cognition", component="cognition", reason="crystallized"
    )
    assert prov.trace_id == trace.trace_id
    assert prov.creation_span_id == trace.span_id  # the tick's span is the CREATION span
    assert prov.parent_span_id == trace.parent_span_id
    assert prov.trace_flags == trace.trace_flags


def test_stdlib_and_fake_satisfy_the_protocol() -> None:
    assert isinstance(StdlibTracer(), TracerPort)
    assert isinstance(FakeTracer(), TracerPort)

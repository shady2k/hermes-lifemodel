"""Suppression spans — the closed ``reason`` enum + emission helper (spec §5).

The replacement for the lying egress "busy" outcome: a first-class, reason-tagged
suppression span emitted from inside the deciding component's child span. These
tests pin the CONTRACT — the closed reason dictionary (exactly the spec's set,
no string escape hatch) and the minimum attributes every span carries — so sim
and live answer "why did it stay silent?" identically.
"""

from __future__ import annotations

import pytest

from lifemodel.core.suppression import (
    EVENT_SUPPRESSION,
    SUPPRESSION_MIN_FIELDS,
    SuppressionReason,
    emit_suppression_span,
)
from lifemodel.ports.tracer import TraceContext
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger

#: The exact reason set the spec §5.5 fixes — the closed contract, order-independent.
#: T3 added ``silence_window`` / ``decline_backoff`` (deliberate extension for the
#: aggregation wake-gate suppressions; the coordinator syncs spec §5).
_SPEC_REASONS = frozenset(
    {
        "below_threshold",
        "in_flight",
        "pending_proactive",
        "silence_window",
        "decline_backoff",
        "energy_unaffordable",
        "backstop_rate_limited",
        "repeat_pure_longing",
        "act_gate_silent",
        "egress_unavailable",
        "egress_failed",
        "component_failed",
    }
)


def _ctx() -> TraceContext:
    return TraceContext(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        parent_span_id="0000000000000001",
    )


def _logger(*, tick: int = 7) -> FakeSpanLogger:
    """A span-bound logger over a fresh :class:`FakeActiveSpan` (the new contract)."""
    return FakeSpanLogger(FakeActiveSpan(_ctx(), tick=tick))


def test_reason_enum_is_exactly_the_closed_spec_set() -> None:
    # The wire-format codes are the spec's 10, no more, no less — a new gate is a
    # deliberate member addition, never a free-form string.
    codes = {reason.value for reason in SuppressionReason}
    assert codes == _SPEC_REASONS
    assert len(codes) == 12


def test_reason_enum_has_no_string_escape_hatch() -> None:
    # Closed by construction: an unknown code cannot be minted — there is no
    # ``SuppressionReason("made_up_gate")`` that succeeds.
    with pytest.raises(ValueError):
        SuppressionReason("made_up_gate")


def test_min_fields_constant_matches_the_spec_contract() -> None:
    assert frozenset({"reason", "component", "trace_id", "span_id", "tick"}) == (
        SUPPRESSION_MIN_FIELDS
    )


def test_emit_suppression_span_carries_reason_code_and_min_attrs() -> None:
    logger = _logger(tick=7)
    emit_suppression_span(
        logger,
        reason=SuppressionReason.BELOW_THRESHOLD,
        component="aggregation",
    )
    assert len(logger.events) == 1
    record = logger.events[0]
    assert record["event"] == EVENT_SUPPRESSION  # the canonical, debug-queryable event name
    assert record["reason"] == "below_threshold"  # the code, not the enum member
    assert record["component"] == "aggregation"
    # The SpanLogger SELF-stamps the correlation ids + tick, so the durable record
    # carries the whole minimum contract without the caller passing them.
    assert record["trace_id"] == logger.span.context.trace_id
    assert record["span_id"] == logger.span.context.span_id  # joins the span tree
    assert record["tick"] == 7
    assert set(record) >= SUPPRESSION_MIN_FIELDS
    # The reason lands on the span's attribute bag (self-explaining) + closes it.
    assert logger.span.attrs["reason"] == "below_threshold"
    assert logger.span.status == "suppressed"


@pytest.mark.parametrize("reason", list(SuppressionReason))
def test_every_reason_code_emits_a_well_formed_span(reason: SuppressionReason) -> None:
    # Each member of the closed set round-trips through the helper as its own code —
    # span-tree tests across sim/live assert against these stable codes (spec §5.7).
    logger = _logger(tick=1)
    emit_suppression_span(logger, reason=reason, component="c")
    record = logger.events[0]
    assert record["reason"] == reason.value
    assert logger.span.attrs["reason"] == reason.value
    assert set(record) >= SUPPRESSION_MIN_FIELDS


def test_emit_suppression_span_may_enrich_without_dropping_min_attrs() -> None:
    # A specific gate may add context (a threshold value, a rate-limit window) — the
    # minimum attributes remain, the enriching fields ride the event AND the span.
    logger = _logger(tick=3)
    emit_suppression_span(
        logger,
        reason=SuppressionReason.ENERGY_UNAFFORDABLE,
        component="cognition",
        available_energy=0.1,
        required_energy=0.5,
    )
    record = logger.events[0]
    assert record["available_energy"] == 0.1
    assert record["required_energy"] == 0.5
    assert set(record) >= SUPPRESSION_MIN_FIELDS  # the min contract still holds
    assert logger.span.attrs["available_energy"] == 0.1  # self-explaining span
    assert logger.span.attrs["required_energy"] == 0.5


def test_emit_suppression_span_status_override_marks_a_component_fault() -> None:
    # A component fault reuses the helper with ``status="failed"`` (the CoreLoop path)
    # — the span closes ``failed`` while still carrying the reason code.
    logger = _logger(tick=2)
    emit_suppression_span(
        logger,
        reason=SuppressionReason.COMPONENT_FAILED,
        component="broken",
        status="failed",
        error="RuntimeError('boom')",
        consecutive=1,
    )
    assert logger.span.status == "failed"
    assert logger.span.attrs["reason"] == "component_failed"
    assert logger.events[0]["reason"] == "component_failed"


def test_emit_suppression_span_requires_a_span_by_signature() -> None:
    # spec §5: a suppression span without an active span is impossible. The helper's
    # ``logger: SpanBoundLogger`` parameter makes this structural — a bare logger has
    # no ``.span`` and cannot appear in the tick path.
    logger = _logger(tick=9)
    emit_suppression_span(
        logger,
        reason=SuppressionReason.ACT_GATE_SILENT,
        component="proactive",
    )
    assert logger.events[0]["trace_id"] == logger.span.context.trace_id

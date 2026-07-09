"""Suppression spans — the closed ``reason`` enum + emission helper (spec §5).

The replacement for the lying egress "busy" outcome: a first-class, reason-tagged
suppression span emitted from inside the deciding component's child span. These
tests pin the CONTRACT — the closed reason dictionary (exactly the spec's set,
no string escape hatch) and the minimum attributes every span carries — so sim
and live answer "why did it stay silent?" identically.
"""

from __future__ import annotations

from typing import Any

import pytest

from lifemodel.core.suppression import (
    EVENT_SUPPRESSION,
    SUPPRESSION_MIN_FIELDS,
    SuppressionReason,
    emit_suppression_span,
)
from lifemodel.ports.tracer import TraceContext

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


class _RecordingLogger:
    """Captures the single ``.info`` call ``emit_suppression_span`` makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


def _span() -> TraceContext:
    return TraceContext(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        parent_span_id="0000000000000001",
    )


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
    logger = _RecordingLogger()
    span = _span()
    emit_suppression_span(
        logger=logger,
        reason=SuppressionReason.BELOW_THRESHOLD,
        component="aggregation",
        span=span,
        tick=7,
    )
    assert len(logger.calls) == 1
    event, fields = logger.calls[0]
    assert event == EVENT_SUPPRESSION  # the canonical, debug-queryable event name
    assert fields["reason"] == "below_threshold"  # the code, not the enum member
    assert fields["component"] == "aggregation"
    assert fields["trace_id"] == span.trace_id
    assert fields["span_id"] == span.span_id  # self-contained: joins the span tree
    assert fields["tick"] == 7
    # Exactly the minimum contract keys are present (no missing, no surprise extras).
    assert set(fields) == SUPPRESSION_MIN_FIELDS


@pytest.mark.parametrize("reason", list(SuppressionReason))
def test_every_reason_code_emits_a_well_formed_span(reason: SuppressionReason) -> None:
    # Each member of the closed set round-trips through the helper as its own code —
    # span-tree tests across sim/live assert against these stable codes (spec §5.7).
    logger = _RecordingLogger()
    emit_suppression_span(logger=logger, reason=reason, component="c", span=_span(), tick=1)
    _, fields = logger.calls[0]
    assert fields["reason"] == reason.value
    assert set(fields) >= SUPPRESSION_MIN_FIELDS


def test_emit_suppression_span_may_enrich_without_dropping_min_attrs() -> None:
    # A specific gate may add context (a threshold value, a rate-limit window) — the
    # minimum attributes remain, and the enriching fields are carried verbatim.
    logger = _RecordingLogger()
    emit_suppression_span(
        logger=logger,
        reason=SuppressionReason.ENERGY_UNAFFORDABLE,
        component="cognition",
        span=_span(),
        tick=3,
        available_energy=0.1,
        required_energy=0.5,
    )
    _, fields = logger.calls[0]
    assert fields["available_energy"] == 0.1
    assert fields["required_energy"] == 0.5
    assert set(fields) >= SUPPRESSION_MIN_FIELDS  # the min contract still holds


def test_emit_suppression_span_requires_a_span_by_signature() -> None:
    # spec §5: a suppression span without an active span is impossible. The helper's
    # ``span: TraceContext`` parameter (not ``TraceContext | None``) makes this
    # structural — mypy rejects passing None; a real span is always correlation-bound.
    logger = _RecordingLogger()
    span = _span()
    # The call compiles only with a real span; None would be a type error (and the
    # emitted event is correlation-bound to it).
    emit_suppression_span(
        logger=logger,
        reason=SuppressionReason.ACT_GATE_SILENT,
        component="proactive",
        span=span,
        tick=9,
    )
    assert logger.calls[0][1]["trace_id"] == span.trace_id

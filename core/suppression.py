"""Suppression spans — first-class observability for "why did it NOT happen" (spec §5).

The lying egress outcome that relabeled a quiet tick as a false "session busy" is
replaced by a **suppression-span**: a structured log event, emitted from inside the
deciding component's child span (or minted at the egress boundary), that carries a
CLOSED ``reason`` code naming the gate that silenced the tick. Silence stops being
the absence of a record — a quiet tick is as debuggable as a loud one.

The reason set is a **contract, not an open question** (spec §5.5): closed by
construction — a new silencing gate is a new :class:`SuppressionReason` member,
added deliberately, never a free-form string. Every span carries at minimum
``{reason, component, trace_id, span_id, tick}`` so ``/lifemodel debug`` (whose
source of truth is the structural logs / event sink, §5.6) can always answer
"why did the being stay silent?".

A span here is the LOGICAL correlation unit materialised as a structured event:
it joins the span tree through the deciding component's already-bound
:class:`~lifemodel.log.SpanBoundLogger` (over that component's child
:class:`~lifemodel.ports.tracer.ActiveSpan`), and it is *self-explaining* — the
``reason`` + decision values land on the span's attribute bag and the emitting
logger self-stamps the correlation ids onto the durable record. Best-effort OTel
export of child/suppression spans is a separate, later concern (the current
exporter ships only the root).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from ..log import SpanBoundLogger
from ..ports.tracer import SpanStatus
from .tick_metrics import SUPPRESSIONS_TOTAL

if TYPE_CHECKING:
    from .metrics import MetricRegistry

#: The canonical structured-event name for a suppression span (HLA §13 vocab),
#: shared by emitters and the debug reader so a silent decision is queryable.
EVENT_SUPPRESSION = "suppression"


class SuppressionReason(enum.Enum):
    """The closed dictionary of WHY a tick's decision was silenced (spec §5.5).

    Each member's value is the stable, wire-format code logged on the span and
    asserted on by span-tree tests (sim and live share the same codes, §5.7).
    Extending the set is a deliberate act — add a member here; there is no
    stringly-typed escape hatch, so an undocumented reason cannot leak.

    Members (the gate each names):

    * ``BELOW_THRESHOLD`` — the contact vital ``u`` had not crossed the absolute
      wake threshold (no reason to speak).
    * ``IN_FLIGHT`` — a proactive turn is already outstanding (idempotency).
    * ``PENDING_PROACTIVE`` — a sent turn awaits its act-gate read-back.
    * ``SILENCE_WINDOW`` — a real exchange happened too recently (inside the
      post-contact quiet window ``W``); not lonely enough to reach.
    * ``DECLINE_BACKOFF`` — a prior outreach was rejected and the growing decline
      backoff ``R`` is still active (don't drum).
    * ``ENERGY_UNAFFORDABLE`` — the cognition launch could not reserve energy.
    * ``BACKSTOP_RATE_LIMITED`` — the fail-closed delivery backstop held fire.
    * ``REPEAT_PURE_LONGING`` — the pure-longing anti-repeat contract held fire
      (unobtrusiveness preserved, concern kept, baroque machinery shed).
    * ``ACT_GATE_SILENT`` — the async Hermes turn returned ``[SILENT]`` (a
      conscious verdict, logged — not a random break).
    * ``EGRESS_UNAVAILABLE`` — no delivery channel was wired/available.
    * ``EGRESS_FAILED`` — delivery was attempted and failed at the boundary.
    * ``COMPONENT_FAILED`` — a component fault suppressed the tick's outcome
      (the circuit-breaker path).
    """

    BELOW_THRESHOLD = "below_threshold"
    IN_FLIGHT = "in_flight"
    PENDING_PROACTIVE = "pending_proactive"
    SILENCE_WINDOW = "silence_window"
    DECLINE_BACKOFF = "decline_backoff"
    ENERGY_UNAFFORDABLE = "energy_unaffordable"
    BACKSTOP_RATE_LIMITED = "backstop_rate_limited"
    REPEAT_PURE_LONGING = "repeat_pure_longing"
    ACT_GATE_SILENT = "act_gate_silent"
    EGRESS_UNAVAILABLE = "egress_unavailable"
    EGRESS_FAILED = "egress_failed"
    COMPONENT_FAILED = "component_failed"


#: The minimum attributes every suppression span carries (spec §5.5 contract).
#: Emitters MUST supply exactly these; ``/lifemodel debug`` may rely on them.
SUPPRESSION_MIN_FIELDS: frozenset[str] = frozenset(
    {"reason", "component", "trace_id", "span_id", "tick"}
)


def emit_suppression_span(
    logger: SpanBoundLogger,
    *,
    reason: SuppressionReason,
    component: str,
    status: SpanStatus = "suppressed",
    metrics: MetricRegistry | None = None,
    **extra: Any,
) -> None:
    """Emit a first-class suppression span — a "why NOT" decision becomes a record.

    Call this from inside the deciding component (its *logger* is already bound to
    that component's child :class:`~lifemodel.ports.tracer.ActiveSpan`, so the
    emitted event joins the span tree). It does three things, making the span
    *self-explaining* rather than a bare event:

    1. drops ``reason`` + any *extra* decision values onto ``logger.span`` (the
       ``trace_spans.attrs_json`` bag — spec §4.1);
    2. closes the span with *status* (``"suppressed"`` for a gate that held fire,
       ``"failed"`` for a component fault the CoreLoop converts here);
    3. emits the canonical ``suppression`` event through *logger*, which self-stamps
       the span's ``trace_id``/``span_id``/``tick`` — so the durable record carries
       the whole :data:`SUPPRESSION_MIN_FIELDS` contract without the caller passing
       correlation ids by hand.

    Structural note: *logger* is a :class:`~lifemodel.log.SpanBoundLogger` — a
    suppression without an active span is impossible by signature, matching the §5
    invariant that a log/decision without a span cannot exist.

    **Choke-point metric (telemetry-core §4.2, bead lm-fib.7.4).** This is the ONE
    birthplace of every suppression span — in-tick (a component gate, a component
    fault) and out-of-tick (the proactive/egress backstop, the async act-gate
    verdict). When a *metrics* registry is supplied, bump
    ``lifemodel_suppressions_total{component,reason}`` HERE so the count can never
    diverge from the trace. Emission is fail-open (the registry never raises), so a
    suppression is recorded whether or not metrics are wired; *metrics* is ``None``
    in a bare unit test and off a hand-built graph.
    """
    logger.span.set(reason=reason.value, **extra)
    logger.span.end(status=status)
    logger.info(EVENT_SUPPRESSION, reason=reason.value, component=component, **extra)
    if metrics is not None:
        # Fail-open: an undeclared metric / label is a no-op + emit-error bump, never
        # an exception onto the tick or egress path (§7).
        metrics.inc(SUPPRESSIONS_TOTAL, component=component, reason=reason.value)

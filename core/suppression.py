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
it joins the span tree through the deciding component's already-bound child
:class:`~lifemodel.ports.tracer.TraceContext`, and it is *self-contained* — the
minimum attributes are explicit event fields, so the reason + correlation ids
reach the sink even where structlog's contextvar decoration is unavailable (the
degraded Hermes-host fallback). Best-effort OTel export of child/suppression
spans is a separate, later concern (the current exporter ships only the root).
"""

from __future__ import annotations

import enum
from typing import Any

from ..log import EventLogger
from ..ports.tracer import TraceContext

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
    *,
    logger: EventLogger,
    reason: SuppressionReason,
    component: str,
    span: TraceContext,
    tick: int,
    **extra: Any,
) -> None:
    """Emit a first-class suppression span — a "why NOT" decision becomes a record.

    Call this from inside the deciding component (it is already running in its
    child *span*, so the emitted event joins the span tree). The minimum contract
    attributes are passed as EXPLICIT fields (not left to contextvar decoration),
    so the event sink / ``/lifemodel debug`` sees ``reason`` + the W3C correlation
    ids regardless of the logging backend. *extra* attrs may enrich a specific
    gate's span (e.g. a threshold value, a rate-limit window) — they must not
    shadow the minimum names.

    Structural note: *span* is a :class:`~lifemodel.ports.tracer.TraceContext`
    (never ``None``) — a suppression span without an active span is impossible by
    signature, matching the §5 invariant that a log/decision without a span cannot
    exist.
    """
    logger.info(
        EVENT_SUPPRESSION,
        reason=reason.value,
        component=component,
        trace_id=span.trace_id,
        span_id=span.span_id,
        tick=tick,
        **extra,
    )

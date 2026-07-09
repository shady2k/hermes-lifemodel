"""``creation_provenance`` — stamp a tick-path object's creation lineage + trace (lm-27n.11).

The one helper the tick-path creation sites call to build the
:class:`~lifemodel.domain.objects.Provenance` for a NEW episode's object: it fills
the lineage fields (who/which-layer/why) and, when the tick is traced, the W3C
CREATION context from the active :class:`~lifemodel.ports.tracer.TraceContext`.

**Creation provenance is IMMUTABLE per episode.** This helper mints the provenance
for a *fresh* object only; a creation site facing a same-episode upsert (the live
row is still present in ``ctx.objects``) must PRESERVE the existing object's
provenance instead of calling this — otherwise a delivery-fail retry would rewrite
the birth trace and the trace system would lie about when the object was born. The
site decides from the snapshot::

    existing = live_X(ctx.objects)
    provenance = existing.provenance if existing is not None else creation_provenance(...)
"""

from __future__ import annotations

from ..domain.objects import Provenance
from ..ports.tracer import TraceContext


def creation_provenance(
    trace: TraceContext | None,
    *,
    created_by: str,
    component: str,
    reason: str,
    source_object_ids: tuple[str, ...] = (),
    source_signal_ids: tuple[str, ...] = (),
    turn_id: str | None = None,
) -> Provenance:
    """Build the :class:`Provenance` stamped on a NEW episode's object.

    On the live tick *trace* is always present (spec §5: tracing is mandatory — the
    tracer is a required CoreLoop dependency, and each component runs in a child span
    passed in as ``ctx.trace``); the component's span is recorded as the object's
    creation context (``span_id`` → ``creation_span_id``; never "the object's live
    span"). The ``None`` branch is retained only as a defensive fallback for direct
    unit-test construction of a ``TickContext`` without a span — it omits the W3C
    trace fields while keeping the lineage, so a bare test fixture still builds.
    """
    if trace is None:
        return Provenance(
            created_by=created_by,
            component=component,
            reason=reason,
            turn_id=turn_id,
            source_object_ids=source_object_ids,
            source_signal_ids=source_signal_ids,
        )
    return Provenance(
        created_by=created_by,
        component=component,
        reason=reason,
        turn_id=turn_id,
        source_object_ids=source_object_ids,
        source_signal_ids=source_signal_ids,
        trace_id=trace.trace_id,
        creation_span_id=trace.span_id,
        parent_span_id=trace.parent_span_id,
        trace_flags=trace.trace_flags,
    )

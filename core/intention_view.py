"""The contact-intention view — the single door onto the live ``kind='intention'`` row.

lm-27n.4 inserts a typed :class:`~lifemodel.domain.objects.Intention` (the
Bratman/Rubicon *decision record*) as the gate-owner between the active contact
:class:`~lifemodel.domain.objects.Desire` and the launch. Like the desire
(:mod:`lifemodel.core.desire_view`), its lifecycle lives in a singleton
``kind='intention'`` record ``contact:owner`` (HLA §4.1), mutated only through the
intent bus (:class:`~lifemodel.core.intents.PutRecord` /
:class:`~lifemodel.core.intents.TransitionRecord`) and committed atomically with
``State`` — never a hand-built draft.

This module is the ONE place that reads that row back into a typed
:class:`Intention`, and the sole constructor of the singleton, so every site asks
the SAME question — a **live non-terminal** intention (``pending``/``active``/
``deferred``), never "any intention row" (a ``completed``/``dropped``/``expired``
row is absence). Two readers, one predicate, mirroring the desire view:

* :func:`live_contact_intention` reads the start-of-tick records snapshot
  (:attr:`~lifemodel.core.component.TickContext.objects`) — what cognition and
  aggregation consume in-tick;
* :func:`read_live_contact_intention` reads a
  :class:`~lifemodel.ports.memory.MemoryPort` point-in-time (``get`` by id) — what
  the out-of-band proactive rollback / debug view use.

Crystallization is **0-LLM** and behavior-neutral on send *timing*: the Rubicon
fields (goal, plan, implementation_trigger, constraints, admissibility_filter,
reconsideration_triggers, rationale) are fixed descriptive/auditable strings —
they record *why* the being committed to reach out, they do **not** re-derive the
desire gates. The one varying field, ``commitment_strength``, is stamped from the
crystallizing desire's salience (its effective pressure).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    CONTACT_INTENTION_ID,
    Intention,
    IntentionState,
    Provenance,
    default_registry,
)
from ..ports.memory import MemoryPort

#: The kind of the contact intention record (``kind`` column, HLA §4.1).
INTENTION_KIND = "intention"

#: The non-terminal states — an intention in one of these is *live*; anything else
#: (``completed``/``dropped``/``expired``, or absent) reads as absence. The contact
#: intention is created directly ``active`` (never ``pending``), but ``pending`` is
#: kept live here so a future trigger-awaiting intention reads correctly too.
LIVE_INTENTION_STATES: frozenset[str] = frozenset(
    {
        IntentionState.PENDING.value,
        IntentionState.ACTIVE.value,
        IntentionState.DEFERRED.value,
    }
)

#: The fixed Rubicon decision record (0-LLM, descriptive/auditable — NOT a second
#: gate). Only ``commitment_strength``/``salience``/``source_drive`` vary per
#: crystallization; everything else is a constant that records *why* the being
#: committed to reach out.
_GOAL = "reach out to the owner"
_PLAN = "compose and send one proactive contact turn now"
#: Gollwitzer if-then implementation intention.
_IMPLEMENTATION_TRIGGER = (
    "if effective contact pressure is over threshold and the owner is reachable, then send now"
)
_CONSTRAINTS: tuple[str, ...] = (
    "respect the daily send backstop",
    "hold if a proactive turn is already in flight",
    "hold while inside the post-send inhibition window",
)
#: Descriptive summary of the gates aggregation already cleared — NOT re-evaluated
#: here (silence window / decline backoff / ActionPending inhibition / in-flight all
#: live in aggregation; recomputing them would be a duplicate gate).
_ADMISSIBILITY_FILTER = (
    "admissible: the aggregation gates (silence window, decline backoff, "
    "ActionPending inhibition, in-flight) already cleared upstream"
)
#: Recorded for auditability; NOT yet acted on (robustness lands in a later task).
_RECONSIDERATION_TRIGGERS: tuple[str, ...] = (
    "owner replies",
    "owner rebuffs",
    "fatigue or energy shortfall",
    "new information arrives",
    "commitment times out",
)
_RATIONALE = (
    "the being's contact drive crossed the wake threshold and no turn was in "
    "flight, so cognition committed to reaching out"
)

#: Built once; :func:`default_registry` validates its four-kind catalog on every
#: call, so the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def _decode_live(record: MemoryRecord | None) -> Intention | None:
    """Decode *record* into a live contact :class:`Intention`, or ``None``.

    ``None`` when the record is absent, is not the contact-intention singleton, or
    is terminal (``completed``/``dropped``/``expired``). Decoding goes through the
    registry (the single read door), so a malformed row surfaces as its
    :class:`~lifemodel.domain.objects.InvalidPayload`, never a silent miss.
    """
    if record is None or record.kind != INTENTION_KIND or record.id != CONTACT_INTENTION_ID:
        return None
    if record.state not in LIVE_INTENTION_STATES:
        return None
    intention = _REGISTRY.decode(record)
    return intention if isinstance(intention, Intention) else None


def live_contact_intention(objects: Sequence[MemoryRecord]) -> Intention | None:
    """The live (non-terminal) contact intention in a records snapshot.

    Scans the start-of-tick :attr:`~lifemodel.core.component.TickContext.objects`
    snapshot (``active`` AND ``deferred`` rows) for the ``contact:owner`` intention
    and returns it typed, or ``None`` if there is no live one.
    """
    for record in objects:
        intention = _decode_live(record)
        if intention is not None:
            return intention
    return None


def read_live_contact_intention(memory: MemoryPort) -> Intention | None:
    """The live contact intention read point-in-time from a :class:`MemoryPort`.

    For out-of-band readers (the proactive rollback, the debug view) that hold a
    store rather than a tick snapshot. ``get`` by the singleton id, then the same
    live/terminal predicate as :func:`live_contact_intention`.
    """
    return _decode_live(memory.get(INTENTION_KIND, CONTACT_INTENTION_ID))


def build_contact_intention(
    *,
    state: IntentionState,
    commitment_strength: float = 0.0,
    salience: float = 0.0,
    source_drive: float | None = None,
    source: str = "contact-cognition",
    extra_constraints: tuple[str, ...] = (),
    provenance: Provenance | None = None,
) -> Intention:
    """Construct the singleton contact :class:`Intention` in *state*.

    The one constructor for the ``contact:owner`` intention. ``commitment_strength``
    is stamped deterministically from the crystallizing desire's salience (its
    effective pressure) so the decision record is auditable; ``salience`` mirrors it
    and ``source_drive`` records the latent drive ``u``. Every Rubicon field is a
    fixed description (a decision record, not a gate) — the launch timing is decided
    entirely by cognition's affordability gate + aggregation's upstream gates.

    ``extra_constraints`` appends the receptivity appraisal's composing constraints
    (lm-27n.5 — allowed styles, topic sensitivities) for auditability; empty by
    default, so an unpopulated user-model leaves the intention byte-identical to
    .4 (behaviour-neutral).

    ``provenance`` (lm-27n.11) records the creation lineage + the tick's execution
    trace. Its birth trace is IMMUTABLE per episode: cognition passes the LIVE
    intention's existing provenance on a delivery-fail retry (preserve), and a fresh
    one only on a first crystallize — so a retry never rewrites the birth trace.
    """
    return Intention(
        id=CONTACT_INTENTION_ID,
        state=str(state),
        source=source,
        salience=salience,
        provenance=provenance,
        goal=_GOAL,
        commitment_strength=commitment_strength,
        plan=_PLAN,
        implementation_trigger=_IMPLEMENTATION_TRIGGER,
        constraints=_CONSTRAINTS + extra_constraints,
        admissibility_filter=_ADMISSIBILITY_FILTER,
        reconsideration_triggers=_RECONSIDERATION_TRIGGERS,
        expiry=None,
        rationale=_RATIONALE,
    )


def encode_contact_intention(intention: Intention) -> MemoryDraft:
    """Encode *intention* through the registry (the single write door)."""
    return _REGISTRY.encode(intention)

"""The owner-relationship view — the single door onto the singleton
``kind='relationship'`` row ``owner`` (learned norms about the owner, HLA §4.1).

lm-27n.5 makes the being's interaction norms about its owner a first-class typed
row: good/bad hours, cadence, privacy boundaries, acceptable styles, explicit
prefs, plus a ``confidence`` marking whether the row was EXPLICITLY set by the
owner (its boundaries hard-veto) or is a low-confidence seed/inference (its
weak norms only down-weight). Like the desire (:mod:`lifemodel.core.desire_view`)
and intention (:mod:`lifemodel.core.intention_view`) singletons, its lifecycle
lives in a ``kind='relationship'`` record ``owner`` (HLA §4.1), created/updated
ONLY through the intent bus (a :class:`~lifemodel.core.intents.PutRecord` upsert;
a :class:`~lifemodel.core.intents.TransitionRecord` only for archive) — never a
hand-built draft, never direct SQL/config.

This module is the ONE place that reads that row back into a typed
:class:`~lifemodel.domain.objects.Relationship`, and the sole constructor of the
singleton, so every appraisal site asks the SAME question. Two readers, one
predicate, mirroring the desire/intention views:

* :func:`live_owner_relationship` reads the start-of-tick records snapshot
  (:attr:`~lifemodel.core.component.TickContext.objects`) — what aggregation and
  cognition consume in-tick;
* :func:`read_owner_relationship` reads a
  :class:`~lifemodel.ports.memory.MemoryPort` point-in-time — what the debug view
  uses.

**Behavior-neutrality (lm-27n.5):** when NO owner-relationship row exists (or it
is archived), appraisal uses the permissive :data:`DEFAULT_RELATIONSHIP` — empty
boundaries, neutral valence, low confidence — so the appraisal returns
``allowed=True, pressure_multiplier=1.0`` and the being sends EXACTLY as it did
before this task (.4). A seeded/default row NEVER hard-vetoes.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    OWNER_RELATIONSHIP_ID,
    Relationship,
    RelationshipState,
    default_registry,
)
from ..ports.memory import MemoryPort

#: The kind of the owner-relationship record (``kind`` column, HLA §4.1).
RELATIONSHIP_KIND = "relationship"

#: The non-terminal states — a relationship in one of these is *live*; anything
#: else (``archived``, or absent) reads as absence → the permissive
#: :data:`DEFAULT_RELATIONSHIP` is used by appraisal.
LIVE_RELATIONSHIP_STATES: frozenset[str] = frozenset({RelationshipState.ACTIVE.value})

#: Confidence at/above which a relationship's boundaries are treated as EXPLICIT
#: owner boundaries — they may HARD-veto proactive contact (quiet hours, cadence,
#: no-contact privacy). Below it (seeded default / a future inference), the same
#: fields only SOFT down-weight, so a seeded row can never hard-veto (sovereignty:
#: "the being doesn't disappear").
EXPLICIT_CONFIDENCE = 0.9

#: The seeded/default confidence — low, so the default row never hard-vetoes.
DEFAULT_CONFIDENCE = 0.1

#: Built once; :func:`default_registry` validates its four-kind catalog on every
#: call, so the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def build_owner_relationship(
    *,
    state: RelationshipState = RelationshipState.ACTIVE,
    cadence: str = "",
    good_hours: tuple[int, ...] = (),
    bad_hours: tuple[int, ...] = (),
    response_valence_pattern: str = "neutral",
    privacy_boundaries: tuple[str, ...] = (),
    topic_sensitivity: tuple[str, ...] = (),
    intimacy_depth: float = 0.0,
    reply_latency_norm: str = "",
    known_load: str = "",
    acceptable_styles: tuple[str, ...] = (),
    explicit_preferences: tuple[str, ...] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    source: str = "owner-relationship",
) -> Relationship:
    """Construct the singleton owner :class:`Relationship`.

    The one constructor for the ``owner`` relationship row. Every field defaults
    to its permissive/empty value, so ``build_owner_relationship()`` with no
    arguments IS :data:`DEFAULT_RELATIONSHIP`; the owner's explicit-set path
    passes the populated fields plus ``confidence=EXPLICIT_CONFIDENCE`` so its
    boundaries hard-veto.
    """
    return Relationship(
        id=OWNER_RELATIONSHIP_ID,
        state=str(state),
        source=source,
        confidence=confidence,
        cadence=cadence,
        good_hours=good_hours,
        bad_hours=bad_hours,
        response_valence_pattern=response_valence_pattern,
        privacy_boundaries=privacy_boundaries,
        topic_sensitivity=topic_sensitivity,
        intimacy_depth=intimacy_depth,
        reply_latency_norm=reply_latency_norm,
        known_load=known_load,
        acceptable_styles=acceptable_styles,
        explicit_preferences=explicit_preferences,
    )


#: The permissive default appraisal uses when NO owner-relationship row exists
#: (or it is archived). Empty boundaries + neutral valence + low confidence ⇒
#: :func:`~lifemodel.core.receptivity.appraise_receptivity` returns
#: ``allowed=True, pressure_multiplier=1.0`` ⇒ the being behaves exactly as .4.
DEFAULT_RELATIONSHIP: Relationship = build_owner_relationship()


def _decode_live(record: MemoryRecord | None) -> Relationship | None:
    """Decode *record* into the live owner :class:`Relationship`, or ``None``.

    ``None`` when the record is absent, is not the owner-relationship singleton,
    or is terminal (``archived``). Decoding goes through the registry (the single
    read door), so a malformed row surfaces as its
    :class:`~lifemodel.domain.objects.InvalidPayload`, never a silent miss.
    """
    if record is None or record.kind != RELATIONSHIP_KIND or record.id != OWNER_RELATIONSHIP_ID:
        return None
    if record.state not in LIVE_RELATIONSHIP_STATES:
        return None
    relationship = _REGISTRY.decode(record)
    return relationship if isinstance(relationship, Relationship) else None


def live_owner_relationship(objects: Sequence[MemoryRecord]) -> Relationship | None:
    """The live (``active``) owner relationship in a records snapshot, or ``None``.

    Scans the start-of-tick :attr:`~lifemodel.core.component.TickContext.objects`
    snapshot for the ``owner`` relationship and returns it typed, or ``None`` when
    there is no live one (callers then fall back to :data:`DEFAULT_RELATIONSHIP`).
    """
    for record in objects:
        relationship = _decode_live(record)
        if relationship is not None:
            return relationship
    return None


def read_owner_relationship(memory: MemoryPort) -> Relationship | None:
    """The live owner relationship read point-in-time from a :class:`MemoryPort`.

    For out-of-band readers (the debug view) that hold a store rather than a tick
    snapshot. ``get`` by the singleton id, then the same live/terminal predicate
    as :func:`live_owner_relationship`.
    """
    return _decode_live(memory.get(RELATIONSHIP_KIND, OWNER_RELATIONSHIP_ID))


def encode_owner_relationship(relationship: Relationship) -> MemoryDraft:
    """Encode *relationship* through the registry (the single write door)."""
    return _REGISTRY.encode(relationship)

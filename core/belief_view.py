"""The belief view — the registry door onto ``kind='belief'`` rows
(spec ``docs/superpowers/specs/2026-07-17-fact-track-design.md``, v2).

The ONE place a :class:`~lifemodel.domain.objects.belief.Belief` is
constructed/encoded/read, mirroring :mod:`lifemodel.core.commitment_view`.
Ids are DETERMINISTIC (never random, HLA §4.1): :func:`belief_id` scopes a
stable content fingerprint to the *source thought*, so re-deriving the same
(thought, content) pair upserts ONE row, and distinct episodes (different
source thought) or distinct content never conflate.

``confidence`` is mandatory and validated to ``[0, 1]`` here — the registry
does not range-check numbers, so this is the one place a belief's epistemic
weight is enforced before it is ever persisted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..domain.memory import JsonObject, MemoryDraft, MemoryRecord
from ..domain.objects import Belief, BeliefState, InvalidPayload
from ..domain.objects.base import derive_id
from ..domain.objects.provenance import Provenance, Sensitivity
from ..domain.objects.registry import default_registry
from ..ports.memory import MemoryPort

BELIEF_KIND = "belief"

#: The non-terminal states — a belief in one of these is *live* (held);
#: anything else (``superseded``/``dropped``/``expired``, or absent) reads as gone.
LIVE_BELIEF_STATES: frozenset[str] = frozenset({BeliefState.ACTIVE.value})

#: Built once; :func:`default_registry` validates its catalog on every call, so
#: the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def belief_id(source_thought_id: str, content: str) -> str:
    """A deterministic id scoping a content fingerprint to its source thought
    (never random; never a bare global content hash — distinct episodes must
    not conflate).

    Model-supplied ``content`` may hold a lone Unicode surrogate (not UTF-8-
    encodable); the fingerprint ``.encode()`` then raises ``UnicodeEncodeError``,
    which is translated to :class:`InvalidPayload` so a noticing-apply caller's
    narrow ``except InvalidPayload`` still bounds it as bad model data (mirrors
    ``crystallized_commitment_id``, lm-705.3 review I1c), never an uncaught strand.
    """
    try:
        digest = hashlib.sha256(f"{source_thought_id}\x00{content.strip()}".encode()).hexdigest()[
            :16
        ]
    except UnicodeEncodeError as exc:
        raise InvalidPayload("content is not UTF-8 encodable") from exc
    return derive_id(BELIEF_KIND, "seed", digest)


def _validated_confidence(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise InvalidPayload(f"belief confidence must be a number, got {value!r}")
    number = float(value)
    if not (0.0 <= number <= 1.0):
        raise InvalidPayload(f"belief confidence must be in [0,1], got {number}")
    return number


def _floor_sensitivity(raw: object) -> Sensitivity:
    # Conservative floor (spec §9): a proposition about a person is at least
    # SENSITIVE; the model may escalate to PRIVATE. Anything else (incl. a
    # "normal" model proposal) floors up to SENSITIVE.
    if raw == Sensitivity.PRIVATE.value:
        return Sensitivity.PRIVATE
    return Sensitivity.SENSITIVE


def build_belief(
    *,
    id: str,
    content: str,
    subject: str = "owner",
    source_message_ids: Sequence[str] = (),
    source_thought_ids: Sequence[str] = (),
    confidence: float,
    salience: float = 0.0,
    sensitivity: Sensitivity = Sensitivity.SENSITIVE,
    source: str = "noticing",
    provenance: Provenance | None = None,
) -> Belief:
    """Construct a typed :class:`Belief` (the one constructor). Born ``active``."""
    return Belief(
        id=id,
        state=BeliefState.ACTIVE.value,
        source=source,
        salience=salience,
        confidence=_validated_confidence(confidence),
        sensitivity=sensitivity,
        provenance=provenance,
        content=content,
        subject=subject,
        source_message_ids=tuple(source_message_ids),
        source_thought_ids=tuple(source_thought_ids),
    )


def belief_from_seed_fields(
    *,
    source_thought_id: str,
    fields: JsonObject,
    source_message_ids: Sequence[str],
    salience: float = 0.0,
    provenance: Provenance | None = None,
) -> Belief:
    """Strictly parse a model-supplied ``belief`` noticing seed and build the
    :class:`Belief` — the ONE place a noticing seed's untrusted fields are
    decoded. Every failure mode (wrong type, missing/empty content, bad
    confidence) raises :class:`InvalidPayload` — never a silent coercion — so
    the caller's narrow ``except InvalidPayload`` catches exactly "the model
    sent bad data", nothing else."""
    if not isinstance(fields, dict):
        raise InvalidPayload("belief fields must be an object")
    content = fields.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InvalidPayload("belief content must be a non-empty string")
    stripped = content.strip()
    bid = belief_id(source_thought_id, stripped)
    return build_belief(
        id=bid,
        content=stripped,
        subject=str(fields.get("subject", "owner")),
        source_message_ids=source_message_ids,
        source_thought_ids=(source_thought_id,),
        confidence=_validated_confidence(fields.get("confidence")),
        salience=salience,
        sensitivity=_floor_sensitivity(fields.get("sensitivity")),
        provenance=provenance,
    )


def encode_belief(belief: Belief) -> MemoryDraft:
    """Encode *belief* through the registry (the single write door; validates
    on write)."""
    return _REGISTRY.encode(belief)


def _decode_live(record: MemoryRecord | None) -> Belief | None:
    """Decode *record* into a live :class:`Belief`, or ``None``.

    ``None`` when the record is absent, is not a belief, or is terminal.
    Decoding goes through the registry (the single read door), so a malformed
    row surfaces as its :class:`~lifemodel.domain.objects.InvalidPayload`."""
    if record is None or record.kind != BELIEF_KIND:
        return None
    if record.state not in LIVE_BELIEF_STATES:
        return None
    obj = _REGISTRY.decode(record)
    return obj if isinstance(obj, Belief) else None


def live_beliefs(objects: Sequence[MemoryRecord]) -> tuple[Belief, ...]:
    """The live (``active``) beliefs among *objects*, most-salient first
    (deterministic ``id`` tiebreak) — mirrors
    :func:`~lifemodel.core.commitment_view.read_live_commitments`'s in-memory
    filter/sort shape for callers already holding a batch of records."""
    live = [b for r in objects if (b := _decode_live(r)) is not None]
    return tuple(sorted(live, key=lambda b: (-b.salience, b.id)))


def read_active_beliefs(
    memory: MemoryPort,
    *,
    min_confidence: float = 0.0,
    exclude_private: bool = True,
    limit: int,
) -> list[Belief]:
    """The live (``active``) beliefs read point-in-time from a
    :class:`MemoryPort`, most-recent first, filtered by confidence/sensitivity.

    A BOUNDED store query (never decode-all, spec §6): fetch a small superset
    ordered by recency, then apply the payload-level filters (confidence/
    sensitivity) the SQL columns can't express, and cap at ``limit``."""
    fetch = max(limit * 6, 12)
    records = memory.find(
        kind=BELIEF_KIND, state=BeliefState.ACTIVE.value, order_by="created_desc", limit=fetch
    )
    out: list[Belief] = []
    for record in records:
        belief = _decode_live(record)
        if belief is None or (belief.confidence or 0.0) < min_confidence:
            continue
        if exclude_private and belief.sensitivity == Sensitivity.PRIVATE:
            continue
        out.append(belief)
        if len(out) >= limit:
            break
    return out

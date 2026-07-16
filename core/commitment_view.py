"""The commitment view — the registry door onto ``kind='commitment'`` rows (§4.2, v3).

The ONE place a ``Commitment`` is constructed/encoded/read, mirroring
:mod:`lifemodel.core.thought_view`. Ids are DETERMINISTIC (never random, HLA §4.1):
:func:`crystallized_commitment_id` scopes a stable content fingerprint to the *source
thought*, so re-crystallizing the same thought upserts ONE row, and distinct episodes
(different source thought) or distinct content never conflate.
"""

from __future__ import annotations

import hashlib

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    Commitment,
    CommitmentBasis,
    CommitmentState,
    CommitmentTriggerKind,
    Provenance,
    default_registry,
    derive_id,
)
from ..ports.memory import MemoryPort

COMMITMENT_KIND = "commitment"

#: The non-terminal states — a commitment in one of these is *live* (owed);
#: anything else (``honoured``/``dropped``/``expired``, or absent) reads as gone.
LIVE_COMMITMENT_STATES: frozenset[str] = frozenset(
    {CommitmentState.ACTIVE.value, CommitmentState.DEFERRED.value}
)

#: Built once; :func:`default_registry` validates its catalog on every call, so
#: the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def crystallized_commitment_id(source_thought_id: str, content: str) -> str:
    """A deterministic id scoping a content fingerprint to its source thought (never
    random; never a bare global content hash — distinct episodes must not conflate)."""
    digest = hashlib.sha256(f"{source_thought_id}\x00{content.strip()}".encode()).hexdigest()[:16]
    return derive_id(COMMITMENT_KIND, "seed", digest)


def build_commitment(
    *,
    id: str,
    content: str,
    basis: CommitmentBasis,
    trigger_kind: CommitmentTriggerKind,
    trigger_value: str,
    due_at: str | None = None,
    source_thought_ids: tuple[str, ...],
    other_regarding_value: float = 0.0,
    salience: float = 0.0,
    source: str = "thought-processing-apply",
    provenance: Provenance | None = None,
) -> Commitment:
    """Construct a typed :class:`Commitment` (the one constructor). Born ``active``."""
    return Commitment(
        id=id,
        state=str(CommitmentState.ACTIVE),
        source=source,
        salience=salience,
        provenance=provenance,
        content=content,
        basis=basis,
        trigger_kind=trigger_kind,
        trigger_value=trigger_value,
        due_at=due_at,
        source_thought_ids=source_thought_ids,
        other_regarding_value=other_regarding_value,
    )


def encode_commitment(commitment: Commitment) -> MemoryDraft:
    """Encode *commitment* through the registry (the single write door; validates
    on write)."""
    return _REGISTRY.encode(commitment)


def _decode_live(record: MemoryRecord | None) -> Commitment | None:
    """Decode *record* into a live :class:`Commitment`, or ``None``.

    ``None`` when the record is absent, is not a commitment, or is terminal.
    Decoding goes through the registry (the single read door), so a malformed row
    surfaces as its :class:`~lifemodel.domain.objects.InvalidPayload`."""
    if record is None or record.kind != COMMITMENT_KIND:
        return None
    if record.state not in LIVE_COMMITMENT_STATES:
        return None
    obj = _REGISTRY.decode(record)
    return obj if isinstance(obj, Commitment) else None


def read_live_commitments(
    memory: MemoryPort, *, limit: int | None = None
) -> tuple[Commitment, ...]:
    """The live (``active``/``deferred``) commitments read point-in-time from a
    :class:`MemoryPort`, most-salient first (deterministic ``id`` tiebreak).

    ``limit`` caps the *live* list — applied after the terminal-row filter, so a
    high-salience terminal row can never crowd a live one out of the cap."""
    records = memory.find(kind=COMMITMENT_KIND, order_by="salience_desc")
    live = tuple(
        sorted(
            (c for record in records if (c := _decode_live(record)) is not None),
            key=lambda c: (-c.salience, c.id),
        )
    )
    return live if limit is None else live[:limit]

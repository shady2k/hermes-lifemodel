"""The thought view — the registry door onto the ``kind='thought'`` rows (§4.1).

lm-27n.6 wires the typed :class:`~lifemodel.domain.objects.Thought` end-to-end
*without generation*: it persists as ``kind='thought'`` memory records (state
machine ``active``/``parked`` → terminal ``resolved``/``dropped``/``expired``/
``merged``) and RENDERS the live ones into the being's proactive wake packet.
Unlike the desire/intention/relationship singletons
(:mod:`lifemodel.core.desire_view` et al.), a thought is **NON-singleton** —
there can be many live at once — so this view reads and orders a *set*.

This module is the ONE place that reads those rows back into typed
:class:`Thought`s and the sole constructor of a thought, so every "what am I
turning over" site asks the SAME question — the **live non-terminal** thoughts
(``active``/``parked``), most-salient first. Two readers, one predicate,
mirroring the other views:

* :func:`live_thoughts` reads the start-of-tick records snapshot
  (:attr:`~lifemodel.core.component.TickContext.objects`) — what cognition
  renders into the wake packet in-tick;
* :func:`read_live_thoughts` reads a :class:`~lifemodel.ports.memory.MemoryPort`
  point-in-time — what the debug view's "what am I thinking" audit uses.

Every write goes through :func:`build_thought` → :func:`encode_thought` (the
registry encode door) — NEVER a hand-built draft. Ids are **deterministic**
(:func:`thought_id`/:func:`seed_thought_id`), never random: the real
generation-time id policy is a later task; here a reproducible content-digest
seed keeps re-seeding idempotent (one row, not a pile of duplicates).

**Behavior-neutral until a thought exists (lm-27n.6):** with no live thoughts,
:func:`live_thoughts` returns ``()`` so the wake packet adds no block and the
prompt is byte-identical to before.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import Provenance, Thought, ThoughtState, default_registry, derive_id
from ..ports.memory import MemoryPort
from .thought_score import SCAN_WIDTH, attention_score

#: The kind of a thought record (``kind`` column, HLA §4.1).
THOUGHT_KIND = "thought"

#: The non-terminal states — a thought in one of these is *live* (it is being
#: turned over); anything else (``resolved``/``dropped``/``expired``/``merged``,
#: or absent) reads as gone. Mirrors the object core's terminal/live split.
LIVE_THOUGHT_STATES: frozenset[str] = frozenset(
    {ThoughtState.ACTIVE.value, ThoughtState.PARKED.value}
)

#: Built once; :func:`default_registry` validates its four-kind catalog on every
#: call, so the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def thought_id(*parts: str) -> str:
    """A deterministic thought id from *parts* — NEVER random (HLA §4.1).

    The ``.8`` generation-time id policy supersedes this; ``.6`` uses it for the
    reproducible seed/persist path so a given thought maps to one stable id."""
    return derive_id(THOUGHT_KIND, *parts)


def seed_thought_id(content: str) -> str:
    """The deterministic id of an owner-/debug-seeded thought: a stable digest of
    its content, so re-seeding identical content upserts ONE row (idempotent),
    never a growing pile of duplicates."""
    digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]
    return thought_id("seed", digest)


def build_thought(
    *,
    id: str,
    content: str,
    trigger: str = "seed",
    state: ThoughtState = ThoughtState.ACTIVE,
    parent_id: str | None = None,
    salience: float = 0.0,
    attention_score: float = 0.0,
    no_progress_count: int = 0,
    loop_signature: str = "",
    parked_until: str | None = None,
    park_count: int = 0,
    actionability: float = 0.0,
    other_regarding_value: float = 0.0,
    source: str = "thought-seed",
    provenance: Provenance | None = None,
) -> Thought:
    """Construct a typed :class:`Thought` (the one constructor).

    Every attention/appraisal field defaults to its neutral value, so a minimal
    ``build_thought(id=..., content=...)`` is a plain active thought. ``.6`` does
    NOT drive the attention/loop/park fields — they persist so a later engine
    (attention/selection/parking, ``.7``) reads them — it only creates + renders
    + transitions the object."""
    return Thought(
        id=id,
        state=str(state),
        source=source,
        salience=salience,
        provenance=provenance,
        content=content,
        trigger=trigger,
        parent_id=parent_id,
        attention_score=attention_score,
        no_progress_count=no_progress_count,
        loop_signature=loop_signature,
        parked_until=parked_until,
        park_count=park_count,
        actionability=actionability,
        other_regarding_value=other_regarding_value,
    )


def encode_thought(thought: Thought) -> MemoryDraft:
    """Encode *thought* through the registry (the single write door)."""
    return _REGISTRY.encode(thought)


def _decode_live(record: MemoryRecord | None) -> Thought | None:
    """Decode *record* into a live :class:`Thought`, or ``None``.

    ``None`` when the record is absent, is not a thought, or is terminal.
    Decoding goes through the registry (the single read door), so a malformed row
    surfaces as its :class:`~lifemodel.domain.objects.InvalidPayload`."""
    if record is None or record.kind != THOUGHT_KIND:
        return None
    if record.state not in LIVE_THOUGHT_STATES:
        return None
    thought = _REGISTRY.decode(record)
    return thought if isinstance(thought, Thought) else None


def _ordered(thoughts: list[Thought]) -> tuple[Thought, ...]:
    """Most-salient first, with a deterministic ``id`` tiebreak."""
    return tuple(sorted(thoughts, key=lambda t: (-t.salience, t.id)))


def live_thoughts(objects: Sequence[MemoryRecord]) -> tuple[Thought, ...]:
    """The live (``active``/``parked``) thoughts in a records snapshot.

    Scans the start-of-tick :attr:`~lifemodel.core.component.TickContext.objects`
    snapshot for every non-terminal thought and returns them typed, most-salient
    first (deterministic ``id`` tiebreak). Empty when there are none — so the
    wake-packet render stays behavior-neutral."""
    return _ordered([t for record in objects if (t := _decode_live(record)) is not None])


def live_thought_records(
    objects: Sequence[MemoryRecord],
) -> tuple[tuple[MemoryRecord, Thought], ...]:
    """The live thoughts paired with their source records, most-salient first.

    Like :func:`live_thoughts`, but keeps each :class:`MemoryRecord` next to its
    decoded :class:`Thought` — the attention engine (lm-27n.7) needs the record's
    ``updated_at`` (for elapsed-time salience decay) and ``state`` (the
    transition's ``from_state``), neither of which the typed object carries. Same
    predicate and ordering (salience desc, ``id`` tiebreak) as
    :func:`live_thoughts`, so the two never disagree on what is live."""
    pairs = [(record, t) for record in objects if (t := _decode_live(record)) is not None]
    return tuple(sorted(pairs, key=lambda rt: (-rt[1].salience, rt[1].id)))


def selected_thoughts(
    objects: Sequence[MemoryRecord], now: datetime, *, limit: int = 1
) -> tuple[Thought, ...]:
    """The top-*limit* ATTENDED thoughts — the attention selection, recomputed.

    Recomputes the deterministic
    :func:`~lifemodel.core.thought_score.attention_score` over the live *active*
    thoughts and returns the highest-scoring ``limit`` of them (score desc,
    salience desc, ``id`` asc). No global state: a later consumer (lm-27n.8's
    generation) asks the SAME pure question of the snapshot — the engine having run
    this tick is not a precondition. Parked thoughts are excluded: they are
    suspended, not selected for development.

    Drawn from the SAME ``SCAN_WIDTH`` salience-ranked window the engine attends
    within, so a consumer can never select a thought the engine did not scan/update
    this tick (they stay in lockstep)."""
    candidates = live_thought_records(objects)[:SCAN_WIDTH]
    active = [t for _record, t in candidates if t.state == ThoughtState.ACTIVE.value]
    ranked = sorted(active, key=lambda t: (-attention_score(t, now), -t.salience, t.id))
    return tuple(ranked[: max(0, limit)])


def read_thought(memory: MemoryPort, id: str) -> Thought | None:
    """A single live thought read point-in-time from a :class:`MemoryPort`."""
    return _decode_live(memory.get(THOUGHT_KIND, id))


def read_live_thoughts(memory: MemoryPort, *, limit: int | None = None) -> tuple[Thought, ...]:
    """The live thoughts read point-in-time from a :class:`MemoryPort`.

    For out-of-band readers (the debug "what am I thinking" audit) that hold a
    store rather than a tick snapshot. Reads every ``kind='thought'`` row, keeps
    the live ones, and orders them most-salient first (``limit`` caps the *live*
    list — applied after the terminal-row filter, so a high-salience terminal row
    can never crowd a live one out of the cap)."""
    records = memory.find(kind=THOUGHT_KIND, order_by="salience_desc")
    live = _ordered([t for record in records if (t := _decode_live(record)) is not None])
    return live if limit is None else live[:limit]

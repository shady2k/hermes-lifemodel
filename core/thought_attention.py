"""ThoughtAttention — the 0-LLM attention/selection + decay/parking layer (lm-27n.7).

The anti-rumination brake. Each tick it reads the live thoughts from the
start-of-tick snapshot (``ctx.objects``), scores them with the deterministic,
LLM-free :mod:`~lifemodel.core.thought_score` product, **selects** the single
most-attention-worthy active thought, **decays** the salience of unresolved
thoughts, and **parks** any thought that loops without progress (then eventually
expires a chronic looper). It mirrors :class:`~lifemodel.core.aggregation.ContactAggregation`'s
shape: stateless, reads the snapshot, emits ONLY typed memory mutations.

**No generation.** lm-27n.7 does not *develop* a thought (that is .8); attending a
thought only records the attention (``attention_score``) and — since nothing
resolves it this task — bumps its ``no_progress_count``. That is deliberate: it is
exactly what proves the brake converges. Cognition later CONSUMES the selection
(via :func:`~lifemodel.core.thought_view.selected_thoughts`); it does not own
decay/parking.

**Mutation shape.** Two paths, with different validation guarantees:

* a field update that keeps the state (decay salience, set ``attention_score``,
  bump ``no_progress_count``) is **fully typed**: it decodes the thought,
  ``dataclasses.replace``\\ s the changed fields, and emits
  ``PutRecord(PutOp(encode_thought(updated)))`` — a typed upsert on the same id
  (revision bumps, provenance preserved), validated by the registry on encode;
* a **park/unpark/expire** is a state change carried as ``TransitionRecord`` with a
  :class:`~lifemodel.domain.memory.MemoryPatch` ``payload_merge`` (new salience +
  the ``parked_until``/``no_progress_count``/``park_count``/``loop_signature``
  fields). The ``payload_merge`` is a raw field merge — it is **not** typed-encoded
  on write (it is re-validated on the next :func:`registry.decode`), and the live
  committer does not itself call ``validate_transition`` (edge legality is the
  emitter's contract, proven by tests). The transitions emitted here
  (active↔parked, parked→expired) are all legal edges of ``THOUGHT_TRANSITIONS``.
  A typed transition helper that validates + re-encodes is a cross-core hardening
  (see the follow-up), not required for these internal, statically-legal edges.

At most ONE mutation per thought per tick, all in the tick's atomic batch. Two
width caps are asserted invariants: a tick scans at most ``SCAN_WIDTH`` thoughts
and attends at most ``ATTEND_K``.

**Behavior-neutral with no thoughts:** with no live thoughts in the snapshot,
:meth:`ThoughtAttention.step` returns ``[]`` — no mutation, no prompt diff.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta

from ..domain.memory import JsonObject, MemoryPatch, PutOp, TransitionOp
from ..domain.objects import Thought, ThoughtState
from .component import TickContext
from .intents import Intent, PutRecord, TransitionRecord
from .taxonomy import read_thought_contact_created
from .thought_score import (
    ATTEND_K,
    MAX_PARK_CYCLES,
    PARK_AFTER,
    SALIENCE_FLOOR,
    SCAN_WIDTH,
    THOUGHT_SALIENCE_HALFLIFE_MIN,
    attention_score,
    decay_salience,
    loop_signature,
    park_backoff_hours,
    park_window_elapsed,
)
from .thought_view import encode_thought, live_thought_records
from .timeutil import minutes_between

#: Persistence-increment viability (lm-27n.9): the attended thought only accrues
#: ``sustained_attention_count`` (the top-down Rubicon counter) when it is a genuine
#: contact candidate — salient enough AND meaningfully other-serving/actionable. An
#: idle wandering thought (salience ~0.15, other-regarding ~0.10) never qualifies,
#: so it can never accumulate the persistence crystallization needs (anti-frivolity).
VIABLE_SALIENCE = 0.45
VIABLE_RELEVANCE = 0.3


def is_viable_contact_candidate(thought: Thought) -> bool:
    """Is *thought* a viable contact candidate this tick (worth accruing persistence)?

    Salient enough to matter AND meaningfully serving the owner or actionable —
    deliberately looser than the Rubicon gate (this only decides whether to COUNT a
    tick of attention; :func:`~lifemodel.core.thought_crystallization.should_crystallize`
    is the actual bar), but strict enough that idle mind-wandering never accrues."""
    return thought.salience >= VIABLE_SALIENCE and (
        thought.other_regarding_value >= VIABLE_RELEVANCE
        or thought.actionability >= VIABLE_RELEVANCE
    )


class ThoughtAttention:
    """Scores, selects, decays, and parks the being's live thoughts (0-LLM)."""

    def __init__(
        self,
        *,
        halflife_min: float = THOUGHT_SALIENCE_HALFLIFE_MIN,
        scan_width: int = SCAN_WIDTH,
        attend_k: int = ATTEND_K,
        id: str = "thought-attention",
    ) -> None:
        self.id = id
        self._halflife_min = halflife_min
        self._scan_width = scan_width
        self._attend_k = attend_k

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        now = ctx.now
        # The live thoughts + their records, most-salient first. Behavior-neutral
        # with none: no candidates → no mutation → byte-identical prompt.
        live = live_thought_records(ctx.objects)
        if not live:
            return []

        # Cap 1: scan at most SCAN_WIDTH candidates (top by salience, id tiebreak).
        candidates = live[: self._scan_width]
        scores = {thought.id: attention_score(thought, now) for _record, thought in candidates}

        # Cap 2: attend at most ATTEND_K — the top active candidate(s) by score
        # then id. Parked thoughts are never *attended* (they unpark/expire).
        active_ids = [t.id for _r, t in candidates if t.state == ThoughtState.ACTIVE.value]
        attended = set(sorted(active_ids, key=lambda tid: (-scores[tid], tid))[: self._attend_k])

        # Resolve on genuine CREATION, not on a mere proposal (lm-27n.9; codex): a
        # thought's reason ACTUALLY became a contact desire this tick (aggregation
        # emits ``thought_contact_created`` only when it creates one) → RESOLVE that
        # thought here (attention is the SOLE thought writer, so it commits the
        # active→resolved edge; the thought does not decay/park/re-crystallize). A
        # proposal aggregation SUPPRESSED (silence window / backoff / in-flight) emits
        # no created-signal, so its thought stays live — the reason is not silently
        # spent by timing; it re-competes and is bounded by normal decay/parking.
        crystallized_id = read_thought_contact_created(ctx.signals)

        intents: list[Intent] = []
        touched = 0
        for record, thought in candidates:
            score = scores[thought.id]
            if thought.id == crystallized_id and thought.state == ThoughtState.ACTIVE.value:
                mutation: Intent | None = self._resolve_intent(thought)
            elif thought.state == ThoughtState.PARKED.value:
                mutation = self._parked_intent(thought, score, now)
            else:
                mutation = self._active_intent(
                    thought, record.updated_at, score, thought.id in attended, now
                )
            if mutation is not None:
                intents.append(mutation)
                touched += 1

        # The two width bounds, asserted as invariants (codex): a tick updates at
        # most SCAN_WIDTH thoughts and attends at most ATTEND_K.
        assert touched <= self._scan_width, (touched, self._scan_width)
        assert len(attended) <= self._attend_k, (len(attended), self._attend_k)
        return intents

    def _resolve_intent(self, thought: Thought) -> Intent:
        """Resolve a crystallized thought (active→resolved) — its reason became a
        contact desire this tick (lm-27n.9). Emitted INSTEAD of the decay/park
        mutation, so attention stays the SOLE thought writer (no same-tick conflict
        with crystallization/aggregation) and the resolved thought neither decays,
        parks, nor re-crystallizes. ``active→resolved`` is a legal THOUGHT edge."""
        return TransitionRecord(
            op=TransitionOp(
                kind="thought",
                id=thought.id,
                from_state=ThoughtState.ACTIVE.value,
                to_state=ThoughtState.RESOLVED.value,
            )
        )

    def _parked_intent(self, thought: Thought, score: float, now: datetime) -> Intent | None:
        """Unpark an elapsed parked thought (or expire a chronic looper); a parked
        thought still inside its window is left untouched (``None``)."""
        if not park_window_elapsed(thought.parked_until, now):
            return None  # still suspended — no-op, no churn
        if thought.park_count >= MAX_PARK_CYCLES:
            # Past the backoff cap and still looping → expire (bounded rumination).
            return TransitionRecord(
                op=TransitionOp(
                    kind="thought",
                    id=thought.id,
                    from_state=ThoughtState.PARKED.value,
                    to_state=ThoughtState.EXPIRED.value,
                )
            )
        # Unpark: a fresh chance — reset no_progress so the loop counter restarts.
        return TransitionRecord(
            op=TransitionOp(
                kind="thought",
                id=thought.id,
                from_state=ThoughtState.PARKED.value,
                to_state=ThoughtState.ACTIVE.value,
                patch=MemoryPatch(payload_merge={"no_progress_count": 0, "attention_score": score}),
            )
        )

    def _active_intent(
        self,
        thought: Thought,
        updated_at: str,
        score: float,
        is_attended: bool,
        now: datetime,
    ) -> Intent:
        """The single mutation for an active thought: decay + score + (attended)
        no-progress bump, parking when it loops or fades below the floor."""
        elapsed_min = minutes_between(updated_at, now)
        decayed = decay_salience(thought.salience, elapsed_min, halflife_min=self._halflife_min)
        new_no_progress = thought.no_progress_count + (1 if is_attended else 0)
        sig = thought.loop_signature or loop_signature(thought)

        # Park before expire: a repeatedly-attended never-resolved thought (loop),
        # OR one that has decayed to/below the recoverability floor.
        if new_no_progress >= PARK_AFTER or decayed <= SALIENCE_FLOOR:
            new_park_count = thought.park_count + 1
            parked_until = (now + timedelta(hours=park_backoff_hours(new_park_count))).isoformat()
            merge: JsonObject = {
                "parked_until": parked_until,
                "no_progress_count": new_no_progress,
                "park_count": new_park_count,
                "loop_signature": sig,
                "attention_score": score,
            }
            return TransitionRecord(
                op=TransitionOp(
                    kind="thought",
                    id=thought.id,
                    from_state=ThoughtState.ACTIVE.value,
                    to_state=ThoughtState.PARKED.value,
                    patch=MemoryPatch(salience=decayed, payload_merge=merge),
                )
            )

        # Persistence bump (lm-27n.9): a viable, attended contact candidate accrues
        # one tick of ``sustained_attention_count`` — the top-down Rubicon counter
        # crystallization reads NEXT tick. Only when attended AND viable; an idle
        # wanderer (low salience / weak relevance) never accrues, so it can never
        # persist into contact (anti-frivolity). This is the ONLY writer of the
        # counter, and it is DISTINCT from ``no_progress_count`` (the park brake) —
        # a thought can be persistent without being near the brake, and vice versa.
        new_sustained = thought.sustained_attention_count + (
            1 if (is_attended and is_viable_contact_candidate(thought)) else 0
        )

        # Field-only update (no state change): the typed upsert through the door.
        updated = replace(
            thought,
            salience=decayed,
            attention_score=score,
            no_progress_count=new_no_progress,
            loop_signature=sig,
            sustained_attention_count=new_sustained,
        )
        return PutRecord(op=PutOp(draft=encode_thought(updated)))

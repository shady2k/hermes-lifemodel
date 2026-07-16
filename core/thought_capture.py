"""``ThoughtCapture`` — the 0-LLM component that persists appraised thoughts (§4.1).

Slice 1 of the waking mind (lm-705.1). Consumes ``thought_seed`` signals (seeded by
the ``post_llm`` appraisal seam) and emits a ``PutRecord(thought)`` per DISTINCT seed,
through the intent bus + end-of-tick committer — never a direct store write. Born
``active``; capture only (no processing / no desire / no arbiter — those are later
slices). Idempotent: the thought id is the content digest, so a duplicate appraisal
(a host retry) upserts ONE row.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import PutOp
from .component import TickContext
from .intents import Intent, PutRecord
from .taxonomy import KIND_THOUGHT_SEED, read_thought_seed
from .thought_view import build_thought, encode_thought, seed_thought_id
from .trace import creation_provenance

THOUGHT_CAPTURE_ID = "thought-capture"


class ThoughtCapture:
    """Persist each appraised ``thought_seed`` as an ``active`` thought (0-LLM)."""

    id: str = THOUGHT_CAPTURE_ID

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        seen: set[str] = set()
        intents: list[Intent] = []
        for signal in ctx.signals:
            if signal.kind != KIND_THOUGHT_SEED:
                continue
            seed = read_thought_seed(signal)
            thought_id = seed_thought_id(seed.content)
            if thought_id in seen:
                continue  # collapse duplicate seeds this frame → one upsert
            seen.add(thought_id)
            provenance = creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="aggregation",
                reason="event-seeded thought capture",
                source_signal_ids=(signal.origin_id,),
            )
            thought = build_thought(
                id=thought_id,
                content=seed.content,
                trigger="event",
                salience=seed.salience,
                actionability=seed.actionability,
                other_regarding_value=seed.other_regarding_value,
                source="thought-capture",
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_thought(thought))))
        if intents and ctx.logger is not None:
            ctx.logger.span.set(captured=len(intents))
        return intents

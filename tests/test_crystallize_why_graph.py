"""I2 (lm-705.3 review, codex) — a crystallized commitment's generic causal link
must be a QUALIFIED ``kind:id`` string so :func:`~lifemodel.core.why_graph.build_why_graph`
can resolve it. The typed ``Commitment.source_thought_ids`` stays the bare thought
id (the direct typed link, unaffected by this fix); only ``provenance.source_object_ids``
-- the generic why-graph edge -- needed qualifying. A bare ``thought.id`` fails
:func:`lifemodel.core.why_graph._parse_qid` (no ``":"`` to split a kind off, or a wrong
kind), so the edge would have surfaced as a ``missing_ref`` instead of resolving.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.commitment_view import COMMITMENT_KIND
from lifemodel.core.intents import PutRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import THOUGHT_KIND, build_thought, encode_thought
from lifemodel.core.why_graph import build_why_graph
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeMemoryStore
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


def test_crystallized_commitment_why_graph_resolves_the_source_thought() -> None:
    thought_id = "thought:seed:x"
    thought = build_thought(id=thought_id, content="interview Friday", state=ThoughtState.ACTIVE)

    # The real store the completed commitment (and its source thought) live in --
    # the why-graph reader walks it, mirroring tests/test_why_graph.py.
    store = FakeMemoryStore(clock=FakeClock(NOW))
    store.put(encode_thought(thought))

    sig = internal_result_signal(
        origin_id="r1",
        correlation_id="c1",
        raw="{...}",
        parsed={
            "outcome": "crystallize_commitment",
            "commitment": {
                "content": "ask how their interview went",
                "basis": "follow_up",
                "trigger_kind": "event",
                "trigger_value": "next time we talk",
            },
        },
        timestamp="2026-07-17T12:00:00+00:00",
    )
    ctx = make_tick_context(
        state=State(pending_internal_id="c1", pending_internal_subject_id=thought_id),
        now=NOW,
        objects=[draft_to_record(encode_thought(thought), now=NOW)],
        signals=[sig],
    )

    intents = list(ThoughtProcessingApply().step(ctx))
    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    commitment_draft = puts[0].op.draft
    store.put(commitment_draft)  # commit the crystallized commitment into the real store

    root = build_why_graph(store, COMMITMENT_KIND, commitment_draft.id)
    assert root is not None
    source_edges = [e for e in root.edges if e.label == "source"]
    assert len(source_edges) == 1, root.edges
    edge = source_edges[0]
    assert edge.missing_ref is None  # qualified id parsed cleanly -- not dangling
    assert not edge.cycle
    assert edge.node is not None
    assert (edge.node.kind, edge.node.id) == (THOUGHT_KIND, thought_id)  # resolves the real row

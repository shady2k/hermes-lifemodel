"""The why-graph reader (lm-27n.10) — walk the durable causal chain.

``build_why_graph`` answers "why does this Desire/Intention exist?" by walking,
in a fixed edge priority, the durable links a created object carries:
``provenance.source_object_ids`` (generic), the typed domain edges
(``Desire.source_thought_ids`` / ``Thought.parent_id``), then ``supersedes``.
It is pure, read-only, and HARD-bounded (max_depth / max_nodes / a visited set
so a cycle is a ``cycle`` edge, never an infinite loop; a dangling ref is a
``missing_ref`` edge). It returns DATA (:class:`WhyNode` / :class:`WhyEdge`),
never text — the renderer lives in the command/debug layer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.core.trace import creation_provenance
from lifemodel.core.why_graph import (
    WhyNode,
    build_why_graph,
    why_contact_desire,
    why_contact_intention,
)
from lifemodel.domain.objects import (
    CONTACT_DESIRE_ID,
    CONTACT_INTENTION_ID,
    DesireSpring,
    DesireState,
    IntentionState,
    Provenance,
    qualified_id,
)
from lifemodel.testing import FakeClock, FakeMemoryStore, FakeTracer

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _store() -> FakeMemoryStore:
    return FakeMemoryStore(clock=FakeClock(NOW))


def _prov(reason: str, *, component: str = "test", **kw: object) -> Provenance:
    return Provenance(created_by="test", component=component, reason=reason, **kw)  # type: ignore[arg-type]


def _child(node: WhyNode, label: str) -> WhyNode:
    """The child WhyNode reached by the (single) edge with *label*."""
    edge = next(e for e in node.edges if e.label == label)
    assert edge.node is not None, f"edge {label!r} has no node: {edge}"
    return edge.node


def _count(node: WhyNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(_count(e.node) for e in node.edges if e.node is not None)


def _depth(node: WhyNode) -> int:
    kids = [e.node for e in node.edges if e.node is not None]
    return 0 if not kids else 1 + max(_depth(k) for k in kids)


def _seed_full_chain(store: FakeMemoryStore) -> None:
    """intention --source--> desire --source_thought--> thought --parent--> thought."""
    store.put(
        encode_thought(
            build_thought(
                id="thought:parent",
                content="the owner sounded low last week",
                provenance=_prov("noticed in the last exchange", component="thought-generation"),
            )
        )
    )
    store.put(
        encode_thought(
            build_thought(
                id="thought:child",
                content="I should check in on them",
                parent_id="thought:parent",
                provenance=_prov("developed from a prior thought", component="thought-generation"),
            )
        )
    )
    store.put(
        encode_contact_desire(
            build_contact_desire(
                state=DesireState.ACTIVE,
                salience=2.0,
                spring=DesireSpring.THOUGHT,
                source_thought_ids=("thought:child",),
                provenance=_prov("crystallized a contact desire", component="contact-aggregation"),
            )
        )
    )
    trace = FakeTracer().start_root()
    store.put(
        encode_contact_intention(
            build_contact_intention(
                state=IntentionState.ACTIVE,
                commitment_strength=2.0,
                provenance=creation_provenance(
                    trace,
                    created_by="cognition",
                    component="cognition",
                    reason="crystallized contact intention",
                    source_object_ids=(qualified_id("desire", CONTACT_DESIRE_ID),),
                ),
            )
        )
    )


def test_walks_the_full_intention_to_thought_chain() -> None:
    store = _store()
    _seed_full_chain(store)

    root = build_why_graph(store, "intention", CONTACT_INTENTION_ID)
    assert root is not None
    assert (root.kind, root.id, root.state) == ("intention", CONTACT_INTENTION_ID, "active")
    assert root.reason == "crystallized contact intention"
    assert root.component == "cognition"
    assert root.trace_id is not None  # the birth trace stamped by cognition

    desire = _child(root, "source")
    assert (desire.kind, desire.id) == ("desire", CONTACT_DESIRE_ID)
    assert desire.reason == "crystallized a contact desire"

    child_thought = _child(desire, "source_thought")
    assert (child_thought.kind, child_thought.id) == ("thought", "thought:child")

    parent_thought = _child(child_thought, "parent_thought")
    assert (parent_thought.kind, parent_thought.id) == ("thought", "thought:parent")
    assert parent_thought.edges == ()  # terminal of the chain — no further parent


def test_domain_links_are_not_duplicated_as_source_edges() -> None:
    # The desire's source_thought lives ONLY in the typed edge, never mirrored into
    # source_object_ids — so exactly ONE edge reaches the thought (no drift).
    store = _store()
    _seed_full_chain(store)
    desire = _child(build_why_graph(store, "intention", CONTACT_INTENTION_ID), "source")  # type: ignore[arg-type]
    labels = [e.label for e in desire.edges]
    assert labels.count("source_thought") == 1
    assert "source" not in labels  # the desire carries no source_object_ids


def test_entrypoints_resolve_the_contact_singletons() -> None:
    store = _store()
    _seed_full_chain(store)
    intention = why_contact_intention(store)
    desire = why_contact_desire(store)
    assert intention is not None and intention.kind == "intention"
    assert desire is not None and desire.kind == "desire"
    # the intention chain reaches the same desire the desire entrypoint returns
    assert _child(intention, "source").id == desire.id


def test_absent_root_is_none() -> None:
    assert build_why_graph(_store(), "intention", CONTACT_INTENTION_ID) is None
    assert why_contact_intention(_store()) is None
    assert why_contact_desire(_store()) is None


def test_terminal_node_is_still_walkable() -> None:
    # A resolved/terminal desire is absence to the LIVE view readers, but the why
    # graph reads the row directly (history is walkable) — so the chain includes it.
    store = _store()
    store.put(
        encode_contact_desire(
            build_contact_desire(
                state=DesireState.SATISFIED,  # terminal
                provenance=_prov("the owner reached out first"),
            )
        )
    )
    root = build_why_graph(store, "desire", CONTACT_DESIRE_ID)
    assert root is not None
    assert root.state == "satisfied"
    assert root.reason == "the owner reached out first"


def test_supersedes_edge() -> None:
    store = _store()
    store.put(
        encode_contact_desire(
            build_contact_desire(state=DesireState.DROPPED, provenance=_prov("old drive urge"))
        )
    )
    # A newer desire that supersedes the old one (supersedes is a qualified id).
    draft = encode_contact_desire(
        build_contact_desire(state=DesireState.ACTIVE, provenance=_prov("fresh urge"))
    )
    superseding = build_contact_desire(state=DesireState.ACTIVE, provenance=_prov("fresh urge"))
    # rebuild the draft with a supersedes pointer (the builder does not take one)
    from dataclasses import replace

    superseding = replace(
        superseding, id="new", supersedes=qualified_id("desire", CONTACT_DESIRE_ID)
    )
    store.put(encode_contact_desire(superseding))

    root = build_why_graph(store, "desire", "new")
    assert root is not None
    edge = next(e for e in root.edges if e.label == "supersedes")
    assert edge.node is not None
    assert (edge.node.kind, edge.node.id) == ("desire", CONTACT_DESIRE_ID)
    assert edge.node.reason == "old drive urge"
    _ = draft  # unused builder probe


def test_cycle_becomes_a_cycle_edge_not_an_infinite_loop() -> None:
    store = _store()
    store.put(encode_thought(build_thought(id="thought:a", content="a", parent_id="thought:b")))
    store.put(encode_thought(build_thought(id="thought:b", content="b", parent_id="thought:a")))
    root = build_why_graph(store, "thought", "thought:a")
    assert root is not None
    b = _child(root, "parent_thought")
    back = next(e for e in b.edges if e.label == "parent_thought")
    assert back.cycle is True
    assert back.node is None  # the repeat is reported, not re-expanded
    assert _count(root) == 2  # a and b, each once


def test_missing_ref_becomes_a_missing_edge() -> None:
    store = _store()
    store.put(
        encode_thought(build_thought(id="thought:only", content="only", parent_id="thought:ghost"))
    )
    root = build_why_graph(store, "thought", "thought:only")
    assert root is not None
    edge = next(e for e in root.edges if e.label == "parent_thought")
    assert edge.node is None
    assert edge.cycle is False
    assert edge.missing_ref == "thought:ghost"


def _seed_thought_chain(store: FakeMemoryStore, n: int) -> str:
    prev: str | None = None
    for i in range(n):
        tid = f"thought:t{i}"
        store.put(encode_thought(build_thought(id=tid, content=f"t{i}", parent_id=prev)))
        prev = tid
    return f"thought:t{n - 1}"


def test_max_depth_is_bounded() -> None:
    store = _store()
    deepest = _seed_thought_chain(store, 10)
    root = build_why_graph(store, "thought", deepest, max_depth=3)
    assert root is not None
    assert _depth(root) <= 3


def test_max_nodes_is_bounded() -> None:
    store = _store()
    deepest = _seed_thought_chain(store, 20)
    root = build_why_graph(store, "thought", deepest, max_nodes=5)
    assert root is not None
    assert _count(root) <= 5

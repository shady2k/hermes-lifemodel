"""The why-graph reader — walk the durable causal chain of a typed object (lm-27n.10).

The CONSUMING half of observability: "why does this Desire/Intention exist?" /
"why did I write to the owner?" is answered by walking the durable links a created
object already carries — never a live-only recompute. Pure, Hermes-free, stdlib
only, and read-only: it reads rows through a :class:`~lifemodel.ports.memory.MemoryPort`
and decodes them through the registry, returning **data** (:class:`WhyNode` /
:class:`WhyEdge`), never text — the renderer lives in the command/debug layer.

**Domain links are the truth (no drift).** The walk follows, in a fixed priority:

1. ``provenance.source_object_ids`` — the explicit GENERIC links (label ``"source"``),
   each a ``"kind:id"`` qualified id. The ONLY such link stamped today is the
   Intention→Desire edge (cognition, lm-27n.10); nothing mirrors the typed domain
   edges here, so an edge is never authoritative in two places.
2. the TYPED domain edges: ``Desire.source_thought_ids`` (label ``"source_thought"``)
   and ``Thought.parent_id`` (label ``"parent_thought"``) — the domain's own fields
   stay the single source of that lineage.
3. ``supersedes`` (label ``"supersedes"``) — a ``"kind:id"`` pointer to the row this
   one replaced.

**Bounded HARD.** ``max_depth`` caps how deep the walk expands; ``max_nodes`` caps
the total materialized nodes; a ``visited`` set on ``(kind, id)`` turns any repeat
into a ``WhyEdge(cycle=True)`` (never an infinite loop), and a referenced id absent
from the store into a ``WhyEdge(missing_ref=...)``. A terminal or absent row is still
a walkable node — read via ``memory.get`` directly (not the live-only view readers),
so a resolved desire/thought remains part of the history.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.memory import MemoryRecord
from ..domain.objects import (
    CONTACT_DESIRE_ID,
    CONTACT_INTENTION_ID,
    BaseObject,
    Desire,
    Intention,
    ObjectCoreError,
    Thought,
    default_registry,
)
from ..ports.memory import MemoryPort

#: Default bounds — small enough that a pathological chain can never fan the reader
#: out unbounded, large enough that a real contact lineage fits comfortably.
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_NODES = 64

#: Built once (the registry validates its catalog on every construction); the reader
#: reuses one instance, mirroring the view readers.
_REGISTRY = default_registry()


@dataclass(frozen=True)
class WhyEdge:
    """One causal link out of a :class:`WhyNode`.

    ``node`` is the resolved target (``None`` for a non-expanded edge); ``missing_ref``
    is the referenced id when it is not in the store; ``cycle`` marks a link back to an
    already-visited node (reported, not re-expanded). Exactly one of
    ``node``/``missing_ref``/``cycle`` is meaningful.
    """

    label: str
    node: WhyNode | None = None
    missing_ref: str | None = None
    cycle: bool = False


@dataclass(frozen=True)
class WhyNode:
    """A typed object in the causal graph, plus its outgoing causal edges.

    The descriptive fields (``reason``/``component``/``trace_id``/``creation_span_id``)
    come from the row's :class:`~lifemodel.domain.objects.Provenance`; they are ``None``
    when the row predates provenance (or could not be decoded).
    """

    kind: str
    id: str
    state: str
    reason: str | None
    component: str | None
    trace_id: str | None
    creation_span_id: str | None
    created_at: str
    updated_at: str
    edges: tuple[WhyEdge, ...]


def build_why_graph(
    memory: MemoryPort,
    kind: str,
    id: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> WhyNode | None:
    """Walk the causal chain rooted at ``(kind, id)`` — ``None`` if the row is absent.

    Pure and read-only. See the module docstring for the edge priority and the bounds.
    """
    record = memory.get(kind, id)
    if record is None:
        return None
    return _Walk(memory, max_depth=max_depth, max_nodes=max_nodes).node(record, depth=0)


def why_contact_intention(memory: MemoryPort) -> WhyNode | None:
    """ "Why did I (decide to) write?" — the contact intention's chain, or ``None``."""
    return build_why_graph(memory, Intention.KIND, CONTACT_INTENTION_ID)


def why_contact_desire(memory: MemoryPort) -> WhyNode | None:
    """ "Why do I want to reach out?" — the contact desire's chain, or ``None``."""
    return build_why_graph(memory, Desire.KIND, CONTACT_DESIRE_ID)


class _Walk:
    """One bounded traversal — carries the shared visited set + node budget."""

    def __init__(self, memory: MemoryPort, *, max_depth: int, max_nodes: int) -> None:
        self._memory = memory
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._visited: set[tuple[str, str]] = set()
        self._count = 0

    def node(self, record: MemoryRecord, *, depth: int) -> WhyNode:
        self._visited.add((record.kind, record.id))
        self._count += 1
        obj = self._decode(record)
        prov = obj.provenance if obj is not None else None
        edges = self._edges(obj, depth=depth) if obj is not None else ()
        return WhyNode(
            kind=record.kind,
            id=record.id,
            state=record.state,
            reason=prov.reason if prov is not None else None,
            component=prov.component if prov is not None else None,
            trace_id=prov.trace_id if prov is not None else None,
            creation_span_id=prov.creation_span_id if prov is not None else None,
            created_at=record.created_at,
            updated_at=record.updated_at,
            edges=edges,
        )

    def _decode(self, record: MemoryRecord) -> BaseObject | None:
        """Decode a row, degrading to ``None`` on a malformed/unknown row.

        A read-only audit reader must never crash on a bad row; an un-decodable node
        still shows its ``kind``/``id``/``state`` (from the record), just no lineage."""
        try:
            return _REGISTRY.decode(record)
        except ObjectCoreError:
            return None

    def _edges(self, obj: BaseObject, *, depth: int) -> tuple[WhyEdge, ...]:
        if depth >= self._max_depth:
            return ()
        edges: list[WhyEdge] = []
        for label, ref, raw in _refs(obj):
            if ref is None:  # an unparseable qualified id — surface it as missing
                edges.append(WhyEdge(label=label, missing_ref=raw))
                continue
            if ref in self._visited:
                edges.append(WhyEdge(label=label, cycle=True))
                continue
            if self._count >= self._max_nodes:
                break  # node budget exhausted — stop expanding (bounded)
            child = self._memory.get(ref[0], ref[1])
            if child is None:
                edges.append(WhyEdge(label=label, missing_ref=raw))
                continue
            edges.append(WhyEdge(label=label, node=self.node(child, depth=depth + 1)))
        return tuple(edges)


def _refs(obj: BaseObject) -> list[tuple[str, tuple[str, str] | None, str]]:
    """The outgoing causal refs of *obj*, in walk priority: ``(label, parsed, raw)``.

    ``parsed`` is ``(kind, id)`` or ``None`` when the raw reference cannot be parsed."""
    refs: list[tuple[str, tuple[str, str] | None, str]] = []
    prov = obj.provenance
    if prov is not None:
        for qid in prov.source_object_ids:
            refs.append(("source", _parse_qid(qid), qid))
    if isinstance(obj, Desire):
        for tid in obj.source_thought_ids:
            refs.append(("source_thought", (Thought.KIND, tid), tid))
    elif isinstance(obj, Intention):
        pass  # the intention→desire edge is a source_object_id (handled above)
    if isinstance(obj, Thought) and obj.parent_id is not None:
        refs.append(("parent_thought", (Thought.KIND, obj.parent_id), obj.parent_id))
    if obj.supersedes is not None:
        refs.append(("supersedes", _parse_qid(obj.supersedes), obj.supersedes))
    return refs


def _parse_qid(qid: str) -> tuple[str, str] | None:
    """Split a ``"kind:id"`` qualified id on its FIRST ``":"`` (ids may contain more)."""
    kind, sep, ident = qid.partition(":")
    if not sep or not kind or not ident:
        return None
    return (kind, ident)


def display_id(kind: str, id: str) -> str:
    """A clean ``"kind:id"`` label for rendering — avoids doubling a self-qualified id.

    Thought ids are already ``"thought:..."`` (``derive_id`` prefixes the kind), so a
    naive ``f"{kind}:{id}"`` would read ``"thought:thought:..."``; when the id already
    carries its kind prefix, the id alone is the label."""
    return id if id.startswith(f"{kind}:") else f"{kind}:{id}"

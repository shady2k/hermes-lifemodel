"""``/lifemodel trace <trace_id | last [N]>`` — the read-only trace viewer (spec §4.6/§6.7).

Generic by construction (invariant law 4): it renders ANY trace the same way —
tick → components → decisions (span attrs) → launch → async outcome → resolution
— by rebuilding the span tree from ``parent_span_id`` and hanging each span's
events off it. Nothing here is proactive-specific; a suppression tick, a delivered
launch, and an orphaned async outcome all render through the one path.

Read path (spec §4.2, read-your-writes): the command runs in the gateway process
that holds the singleton :class:`~lifemodel.state.trace_store.TraceWriter`, so we
:meth:`flush` it (drain its queue into ``observability.sqlite``) BEFORE reading,
then read the durable rows. The in-memory :class:`~lifemodel.events.EventRing` is
overlaid on the flushed events and deduped by ``record_id`` (so a record that is
both flushed and still-in-ring is not doubled) — that overlay logic is the pure
:func:`_merge_events` seam, exercised by tests; in the live command the flush
already makes sqlite complete, so an empty overlay is lossless.

Fail-soft (invariant law 3): a missing/locked/corrupt trace DB degrades to a
friendly message, never a crash — the durable trace is disposable, losing it
never changes the being's behaviour.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state.trace_store import connect, observability_db_path, peek_trace_writer

#: Default number of root traces ``trace last`` renders when no N is given.
_DEFAULT_LAST_N = 1
#: Hard cap so ``trace last 999`` can never dump the whole store into one message.
_MAX_LAST_N = 20


@dataclass(frozen=True)
class _Span:
    """One ``trace_spans`` row, decoded."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    component: str | None
    tick: int | None
    started_at: str | None
    ended_at: str | None
    status: str | None
    attrs: dict[str, Any]


@dataclass(frozen=True)
class _Event:
    """One ``trace_events`` row (or ring overlay), decoded."""

    record_id: int
    trace_id: str
    span_id: str | None
    tick: int | None
    event: str
    ts: str
    fields: dict[str, Any]


@dataclass
class _Node:
    """A span plus its child spans and attached events (the render tree)."""

    span: _Span
    children: list[_Node] = field(default_factory=list)
    events: list[_Event] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_spans(conn: sqlite3.Connection, trace_id: str) -> list[_Span]:
    rows = conn.execute(
        "SELECT trace_id, span_id, parent_span_id, component, tick, started_at, "
        "ended_at, status, attrs_json FROM trace_spans WHERE trace_id = ?",
        (trace_id,),
    ).fetchall()
    return [_Span(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], _loads(r[8])) for r in rows]


def _read_events(conn: sqlite3.Connection, trace_id: str) -> list[_Event]:
    rows = conn.execute(
        "SELECT record_id, trace_id, span_id, tick, event, ts, fields_json "
        "FROM trace_events WHERE trace_id = ?",
        (trace_id,),
    ).fetchall()
    return [_Event(r[0], r[1], r[2], r[3], r[4], r[5], _loads(r[6])) for r in rows]


def _ring_event(record: Mapping[str, Any], trace_id: str) -> _Event | None:
    """Decode one EventRing record into an :class:`_Event`, or ``None`` if it is
    not a durable trace event for *trace_id* (missing ids / wrong trace)."""
    if record.get("trace_id") != trace_id:
        return None
    record_id = record.get("record_id")
    event = record.get("event")
    if not isinstance(record_id, int) or not isinstance(event, str):
        return None
    reserved = {"record_id", "trace_id", "span_id", "tick", "event", "ts"}
    fields = {k: v for k, v in record.items() if k not in reserved}
    return _Event(
        record_id=record_id,
        trace_id=trace_id,
        span_id=record.get("span_id"),
        tick=record.get("tick"),
        event=event,
        ts=str(record.get("ts") or ""),
        fields=fields,
    )


def _merge_events(
    flushed: Sequence[_Event], ring: Sequence[Mapping[str, Any]], trace_id: str
) -> list[_Event]:
    """Overlay the in-memory ring on the flushed rows, dedup by ``record_id`` (§4.2).

    The flushed durable rows win; a ring record whose ``record_id`` is already
    flushed is dropped (not doubled), and a ring-only record (not yet flushed) is
    added — read-your-writes without flapping.
    """
    by_id: dict[int, _Event] = {e.record_id: e for e in flushed}
    for record in ring:
        decoded = _ring_event(record, trace_id)
        if decoded is not None and decoded.record_id not in by_id:
            by_id[decoded.record_id] = decoded
    return sorted(by_id.values(), key=lambda e: (e.ts, e.record_id))


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #


def _build_tree(spans: Sequence[_Span], events: Sequence[_Event]) -> list[_Node]:
    """Rebuild the span forest by ``parent_span_id`` and hang events off their span.

    Roots are spans with no parent OR a parent not present in this trace (a
    cross-tick weave whose origin span was pruned) — either way they render at
    top level. Events whose ``span_id`` matches no span are collected under a
    synthetic bucket so nothing is silently dropped.
    """
    nodes = {s.span_id: _Node(span=s) for s in spans}
    for event in events:
        node = nodes.get(event.span_id) if event.span_id is not None else None
        if node is not None:
            node.events.append(event)
    roots: list[_Node] = []
    for span in spans:
        node = nodes[span.span_id]
        parent = nodes.get(span.parent_span_id) if span.parent_span_id is not None else None
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
    _sort_recursive(roots)
    return roots


def _sort_key_span(node: _Node) -> tuple[str, int, str]:
    return (node.span.started_at or "", node.span.tick or 0, node.span.span_id)


def _sort_recursive(nodes: list[_Node]) -> None:
    nodes.sort(key=_sort_key_span)
    for node in nodes:
        node.events.sort(key=lambda e: (e.ts, e.record_id))
        _sort_recursive(node.children)


def _orphan_events(spans: Sequence[_Span], events: Sequence[_Event]) -> list[_Event]:
    """Events whose ``span_id`` matches no span in this trace (never dropped)."""
    span_ids = {s.span_id for s in spans}
    orphans = [e for e in events if e.span_id is None or e.span_id not in span_ids]
    return sorted(orphans, key=lambda e: (e.ts, e.record_id))


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def _fmt_fields(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    parts = [f"{k}={_fmt_value(v)}" for k, v in sorted(fields.items())]
    return " " + " ".join(parts)


def _fmt_value(value: Any) -> str:
    text = str(value)
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _render_span(node: _Node, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    span = node.span
    label = span.component or ("tick" if span.parent_span_id is None else "span")
    head = f"{indent}{label}"
    if span.tick is not None:
        head += f" · tick {span.tick}"
    if span.status:
        head += f" [{span.status}]"
    attrs = _fmt_fields(span.attrs)
    if attrs:
        head += f" ·{attrs}"
    lines.append(head)
    for event in node.events:
        marker = "⚠ " if event.event == "orphan_async_outcome" else ""
        lines.append(f"{indent}  · {marker}{event.event}{_fmt_fields(event.fields)}")
    for child in node.children:
        _render_span(child, depth + 1, lines)


def render_trace(
    trace_id: str,
    spans: Sequence[_Span],
    events: Sequence[_Event],
) -> list[str]:
    """Render one trace's span tree + events as indented lines (no header)."""
    if not spans and not events:
        return [f"  (no spans or events recorded for trace {trace_id})"]
    roots = _build_tree(spans, events)
    lines: list[str] = []
    for root in roots:
        _render_span(root, 1, lines)
    orphans = _orphan_events(spans, events)
    if orphans:
        lines.append("  (events with no span)")
        for event in orphans:
            marker = "⚠ " if event.event == "orphan_async_outcome" else ""
            lines.append(f"    · {marker}{event.event}{_fmt_fields(event.fields)}")
    return lines


# --------------------------------------------------------------------------- #
# Read + dispatch
# --------------------------------------------------------------------------- #


def _root_trace_ids(conn: sqlite3.Connection, limit: int) -> list[str]:
    """The most recent *limit* ROOT traces by earliest ``started_at`` (spec §4.6).

    A root trace has at least one span with ``parent_span_id IS NULL`` (or, for an
    events-only orphan, an ``orphan_async_outcome`` event); we order by the trace's
    earliest known timestamp so "last N" is stable and newest-first.
    """
    rows = conn.execute(
        "SELECT trace_id, MIN(t) AS started FROM ("
        "  SELECT trace_id, started_at AS t FROM trace_spans"
        "    WHERE parent_span_id IS NULL AND started_at IS NOT NULL"
        "  UNION ALL"
        "  SELECT trace_id, ts AS t FROM trace_events"
        ") GROUP BY trace_id ORDER BY started DESC, trace_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def _read_and_render(
    conn: sqlite3.Connection, trace_id: str, ring: Sequence[Mapping[str, Any]]
) -> list[str]:
    spans = _read_spans(conn, trace_id)
    flushed = _read_events(conn, trace_id)
    events = _merge_events(flushed, ring, trace_id)
    lines = [f"trace {trace_id}"]
    lines += render_trace(trace_id, spans, events)
    return lines


def _parse_args(raw_args: str) -> tuple[str, str | int]:
    """Return ``("id", trace_id)`` or ``("last", n)`` or ``("usage", "")``."""
    parts = raw_args.strip().split()
    if not parts:
        return ("usage", "")
    if parts[0].lower() == "last":
        if len(parts) == 1:
            return ("last", _DEFAULT_LAST_N)
        try:
            n = int(parts[1])
        except ValueError:
            return ("usage", "")
        return ("last", max(1, min(n, _MAX_LAST_N)))
    return ("id", parts[0])


def _live_ring_flush(base_dir: Path) -> None:
    """Drain the live singleton writer's queue into sqlite for read-your-writes (§4.2).

    Best-effort: a bare CLI process with no live being has no writer to flush —
    the durable rows are then simply whatever was last committed."""
    writer = peek_trace_writer(observability_db_path(base_dir))
    if writer is not None:
        writer.flush(timeout=2.0)


def trace_for_dir(base_dir: Path, raw_args: str, *, ring: Sequence[Mapping[str, Any]] = ()) -> str:
    """Answer ``/lifemodel trace <trace_id | last [N]>`` — read-only, fail-soft.

    ``*ring*`` is the in-memory freshness overlay (spec §4.2); the live command
    passes nothing (a fresh graph has no live ring — the flush makes sqlite
    complete), while tests inject ring records to prove the dedup.
    """
    kind, arg = _parse_args(raw_args)
    if kind == "usage":
        return (
            "usage: /lifemodel trace <trace_id> | trace last [N]\n"
            "  (renders a durable execution trace: tick → components → decisions "
            "→ launch → async outcome → resolution)\n"
        )

    db_path = observability_db_path(base_dir)
    if not db_path.exists():
        return "lifemodel trace: no trace store yet (observability.sqlite not created).\n"

    _live_ring_flush(base_dir)

    header = ["lifemodel trace  (read-only)", "=" * 30, ""]
    try:
        with closing(connect(db_path, create_parent=False)) as conn:
            if kind == "id":
                assert isinstance(arg, str)
                body = _read_and_render(conn, arg, ring)
                if body == [f"trace {arg}", f"  (no spans or events recorded for trace {arg})"]:
                    return f"lifemodel trace: no trace {arg}\n"
            else:  # last N
                assert isinstance(arg, int)
                trace_ids = _root_trace_ids(conn, arg)
                if not trace_ids:
                    return "lifemodel trace: no traces recorded yet.\n"
                body = []
                for i, trace_id in enumerate(trace_ids):
                    if i:
                        body.append("")
                    body += _read_and_render(conn, trace_id, ring)
    except sqlite3.Error as exc:
        return f"lifemodel trace: trace store unreadable ({exc}).\n"

    return "\n".join(header + body) + "\n"

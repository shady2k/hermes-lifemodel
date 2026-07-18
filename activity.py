"""``python3 -m lifemodel.activity`` — the unified turn/tick timeline reader (lm-hg7 Task 12).

The primary owner/debugging-agent-facing deliverable of the turn-observability
plan (tasks 1-11): a tick already writes a ``turn``-free root span
(``frame_kind="execution"``, ``trigger=...``); a Hermes turn now ALSO writes a
root (``frame_kind="turn"``) plus a ``turn.injector.<component>`` child per
``pre_llm_call`` injector, a ``turn.tool.<tool>`` child per tool call, and a
``turn.completion`` child on close — all into the SAME ``observability.sqlite``
(see :mod:`lifemodel.core.turn_recorder`). Before this reader existed, seeing a
turn's shape meant hand-querying the sqlite file; this module is the shell-runnable
answer.

Two views, mirroring ``/lifemodel trace``'s ``last N`` / ``<trace_id>`` split
(:mod:`lifemodel.trace_view`), reused where cleanly possible rather than
re-implemented:

* **``last [N]`` (the default)** — the interleaved timeline: every ROOT span
  (``parent_span_id IS NULL``), newest first, labelled by its
  ``attrs_json->>'frame_kind'`` (a pre-Task-7 row predates the stamp and has
  none — defaults to ``"execution"`` so an old row never crashes the reader).
  A turn line shows its ``origin`` + a short per-injector outcome summary +
  ``incomplete`` when ``ended_at IS NULL`` (an open/abandoned turn is NEVER
  rendered as a success). A tick line shows its ``trigger``. A long run of
  consecutive ``heartbeat`` ticks collapses to one summary line so a turn or a
  non-heartbeat tick — the interesting occasions — are never buried in noise.
* **``turn <trace_id>``** — that one turn's full child tree, rendered through
  :func:`lifemodel.trace_view.render_trace` (the SAME tree renderer
  ``/lifemodel trace`` uses — not reimplemented here), with any ``belief:``/
  ``commitment:`` id riding a span's attrs enriched by a bounded, read-only
  lookup against ``lifemodel.sqlite``'s ``memory_records`` table (state only —
  a belief/commitment's ``content`` never rides an observability surface,
  matching the redaction discipline the injectors themselves already hold to;
  see ``hooks.py``'s belief/commitment injectors). A missing/unresolvable ref
  just leaves the bare id showing (no enrichment line for it) — never a crash.

The :func:`~lifemodel.debug.render_dump_for_dir` state header is prepended to
both views (the being's vitals are always useful context for "what was
happening").

**Read-only + fail-soft, like every other reader in this module family**
(:mod:`lifemodel.trace_view`, :mod:`lifemodel.stats_view`): ``observability.sqlite``
is opened ``?mode=ro`` (the live gateway's :class:`~lifemodel.state.trace_store.TraceWriter`
writes it concurrently under WAL — a plain read-write open here would be an
unnecessary second writer on a file that already has one); a missing/locked/
corrupt store, or an unresolvable ref lookup, degrades to a friendly line, never
a crash — mirroring ``stats_view``'s ``_safe_now``/``_safe_window`` discipline.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .debug import local_time, render_dump_for_dir
from .state.trace_store import observability_db_path
from .trace_view import _loads, _read_spans, _Span, render_trace

#: Default number of root units ``activity last`` renders when no N is given —
#: bigger than ``trace``'s (1): this is a scan-the-timeline view, not a single
#: trace deep-dive, so a useful default shows a real slice of recent activity.
_DEFAULT_LAST_N = 10
#: Hard cap so ``activity last 999999`` can never dump the whole store into one call.
_MAX_LAST_N = 100

#: A run of at least this many consecutive ``frame_kind="execution"``/
#: ``trigger="heartbeat"`` root ticks collapses into one summary line — short
#: runs (1-2) still render individually, since collapsing those buys nothing.
_HEARTBEAT_COLLAPSE_MIN_RUN = 3

#: Self-qualified id prefixes this reader knows how to enrich (see
#: ``domain/objects/base.py:derive_id`` — a Belief/Commitment's OWN id already
#: carries its kind, e.g. ``"belief:seed:<digest>"``).
_REF_PREFIXES = ("belief:", "commitment:")
#: Bounded scan: a turn's child spans are few, but this caps the enrichment
#: lookups regardless, so a pathological attrs bag can never turn one command
#: into an unbounded scan of ``lifemodel.sqlite``.
_MAX_REFS = 32

_USAGE = (
    "usage: python3 -m lifemodel.activity [last [N]] | turn <trace_id>\n"
    "  (last [N]: the tick/turn timeline, newest first, default N="
    f"{_DEFAULT_LAST_N} · turn <id>: that turn's child span tree)\n"
)


# --------------------------------------------------------------------------- #
# Root span decode
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Root:
    """One ROOT ``trace_spans`` row (``parent_span_id IS NULL``), decoded."""

    trace_id: str
    span_id: str
    component: str | None
    tick: int | None
    started_at: str | None
    ended_at: str | None
    status: str | None
    attrs: dict[str, Any]


def _decode_root(row: tuple[Any, ...]) -> _Root:
    trace_id, span_id, component, tick, started_at, ended_at, status, attrs_json = row
    return _Root(
        trace_id=trace_id,
        span_id=span_id,
        component=component,
        tick=tick,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        attrs=_loads(attrs_json),
    )


def _root_rows(conn: sqlite3.Connection, limit: int) -> list[_Root]:
    """The most recent *limit* ROOT spans, newest ``started_at`` first."""
    rows = conn.execute(
        "SELECT trace_id, span_id, component, tick, started_at, ended_at, status, attrs_json "
        "FROM trace_spans WHERE parent_span_id IS NULL AND started_at IS NOT NULL "
        "ORDER BY started_at DESC, trace_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_decode_root(row) for row in rows]


def _str_attr(attrs: Mapping[str, Any], key: str, default: str) -> str:
    """A string attr, or *default* — absent/wrong-typed never raises (old rows)."""
    value = attrs.get(key)
    return value if isinstance(value, str) and value else default


def _frame_kind_of(attrs: Mapping[str, Any]) -> str:
    """``attrs["frame_kind"]``, defaulting to ``"execution"`` when absent.

    A span written before Task 7 stamped no ``frame_kind`` at all; treating that
    absence as ``"execution"`` (a tick, the pre-existing occasion) is what lets
    this reader render an old store without crashing
    (``test_reader_tolerates_old_span_without_frame_kind``)."""
    return _str_attr(attrs, "frame_kind", "execution")


def _status_label(root: _Root) -> str:
    """The one-word status a timeline/tree line shows for *root*.

    ``ended_at IS NULL`` means the root was persisted OPEN (a turn mid-flight,
    or one abandoned by a crash) — that is ALWAYS rendered ``"incomplete"``,
    never the closed vocabulary's ``"ok"`` (an open span must never read as a
    success, see ``test_open_turn_renders_incomplete_not_success``)."""
    if root.ended_at is None:
        return "incomplete"
    return root.status if isinstance(root.status, str) and root.status else "ok"


# --------------------------------------------------------------------------- #
# Timeline (``last [N]``)
# --------------------------------------------------------------------------- #


def _injector_summary(conn: sqlite3.Connection, trace_id: str) -> str:
    """A short ``name=outcome`` summary of *trace_id*'s ``turn.injector.*`` children."""
    rows = conn.execute(
        "SELECT component, attrs_json FROM trace_spans "
        "WHERE trace_id = ? AND component LIKE 'turn.injector.%' ORDER BY component",
        (trace_id,),
    ).fetchall()
    parts: list[str] = []
    for component, attrs_json in rows:
        if not isinstance(component, str):
            continue
        outcome = _str_attr(_loads(attrs_json), "outcome", "unknown")
        parts.append(f"{component.rsplit('.', 1)[-1]}={outcome}")
    return " ".join(parts)


def _render_turn_line(root: _Root, summary: str) -> str:
    origin = _str_attr(root.attrs, "origin", "?")
    status = _status_label(root)
    line = f"{local_time(root.started_at)}  turn {root.trace_id}  origin={origin} status={status}"
    return f"{line}  [{summary}]" if summary else line


def _render_tick_line(root: _Root, frame_kind: str) -> str:
    trigger = _str_attr(root.attrs, "trigger", "?")
    tick_label = f"#{root.tick}" if root.tick is not None else "#?"
    return (
        f"{local_time(root.started_at)}  {frame_kind} {tick_label}  "
        f"trigger={trigger} status={_status_label(root)}"
    )


def _render_timeline_body(conn: sqlite3.Connection, roots: Sequence[_Root]) -> list[str]:
    """Render *roots* (newest-first) as one line each, turns always visible.

    Consecutive ``execution``/``heartbeat`` ticks collapse into one summary
    line once a run reaches :data:`_HEARTBEAT_COLLAPSE_MIN_RUN` — a turn (or
    any non-heartbeat tick) always flushes and breaks the run first, so it is
    never folded into a collapsed summary
    (``test_timeline_interleaves_and_labels_frame_kind_newest_first``)."""
    lines: list[str] = []
    run: list[_Root] = []

    def flush_run() -> None:
        if not run:
            return
        if len(run) < _HEARTBEAT_COLLAPSE_MIN_RUN:
            lines.extend(_render_tick_line(r, "execution") for r in run)
        else:
            lines.append(
                f"  … {len(run)} heartbeat ticks collapsed (#{run[-1].tick}–#{run[0].tick}) …"
            )
        run.clear()

    for root in roots:
        frame_kind = _frame_kind_of(root.attrs)
        if frame_kind == "turn":
            flush_run()
            lines.append(_render_turn_line(root, _injector_summary(conn, root.trace_id)))
            continue
        if frame_kind == "execution" and _str_attr(root.attrs, "trigger", "?") == "heartbeat":
            run.append(root)
            continue
        flush_run()
        lines.append(_render_tick_line(root, frame_kind))
    flush_run()
    return lines


# --------------------------------------------------------------------------- #
# Turn detail (``turn <trace_id>``) — reuses trace_view's tree renderer
# --------------------------------------------------------------------------- #


def _collect_ref_ids(spans: Sequence[_Span]) -> list[str]:
    """Every ``belief:``/``commitment:`` self-qualified id riding *spans*' attrs.

    Bounded to :data:`_MAX_REFS` regardless of how many spans/attrs a turn
    carries — a turn's child spans are few in practice, but this keeps the
    later ``lifemodel.sqlite`` lookup a fixed-size scan on principle."""
    ids: list[str] = []
    seen: set[str] = set()
    for span in spans:
        for value in span.attrs.values():
            candidates: Sequence[Any] = value if isinstance(value, list) else (value,)
            for candidate in candidates:
                if (
                    isinstance(candidate, str)
                    and candidate.startswith(_REF_PREFIXES)
                    and candidate not in seen
                ):
                    seen.add(candidate)
                    ids.append(candidate)
                    if len(ids) >= _MAX_REFS:
                        return ids
    return ids


def _lifemodel_db_path(base_dir: Path) -> Path:
    """``lifemodel.sqlite``'s path under *base_dir* (sibling of ``observability.sqlite``).

    Mirrors :func:`~lifemodel.state.trace_store.observability_db_path`; the
    filename itself is a private constant over in ``state.sqlite_store`` (not
    exported), so this reader — a read-only, best-effort peek, never a writer —
    names it directly rather than reaching into that module's internals."""
    return base_dir / "lifemodel.sqlite"


def _lookup_ref_states(base_dir: Path, ref_ids: Sequence[str]) -> dict[str, str]:
    """``{ref_id: state}`` for every *ref_id* actually found in ``memory_records``.

    Read-only (``?mode=ro``) and fail-soft: a missing store, a locked/corrupt
    file, or any per-row surprise degrades to an empty mapping — the caller
    then simply shows the bare id with no enrichment, never a crash. Looks up
    ``state`` only (never ``payload_json``/``content``) — a belief/commitment's
    content must never ride an observability surface, mirroring the redaction
    discipline the injectors themselves already hold to (see ``hooks.py``)."""
    if not ref_ids:
        return {}
    db_path = _lifemodel_db_path(base_dir)
    if not db_path.exists():
        return {}
    found: dict[str, str] = {}
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            for ref_id in ref_ids:
                kind = ref_id.split(":", 1)[0]
                row = conn.execute(
                    "SELECT state FROM memory_records WHERE kind = ? AND id = ?",
                    (kind, ref_id),
                ).fetchone()
                if row is not None and isinstance(row[0], str):
                    found[ref_id] = row[0]
    except sqlite3.Error:
        return {}
    return found


def _turn_detail(conn: sqlite3.Connection, trace_id: str, base_dir: Path) -> str:
    spans = _read_spans(conn, trace_id)
    if not spans:
        return f"lifemodel activity: no turn {trace_id}\n"
    lines = [f"turn {trace_id}"]
    lines += render_trace(trace_id, spans, ())
    enriched = _lookup_ref_states(base_dir, _collect_ref_ids(spans))
    if enriched:
        lines.append("")
        lines.append("refs:")
        lines.extend(f"  {ref_id} -> {enriched[ref_id]}" for ref_id in sorted(enriched))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Command entrypoint (read-only, fail-soft on every source)
# --------------------------------------------------------------------------- #


def _parse_args(raw_args: str) -> tuple[str, str | int]:
    """Return ``("last", N)``, ``("turn", trace_id)``, or ``("usage", "")``.

    Unlike ``/lifemodel trace``'s parser, bare/empty args are NOT usage here —
    they mean the default timeline (spec: ``""``/``"last [N]"`` both render
    the timeline)."""
    parts = raw_args.strip().split()
    if not parts:
        return ("last", _DEFAULT_LAST_N)
    head = parts[0].lower()
    if head == "last":
        if len(parts) == 1:
            return ("last", _DEFAULT_LAST_N)
        try:
            n = int(parts[1])
        except ValueError:
            return ("usage", "")
        return ("last", max(1, min(n, _MAX_LAST_N)))
    if head == "turn" and len(parts) == 2:
        return ("turn", parts[1])
    return ("usage", "")


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open *db_path* read-only — never a second writer on the live WAL file."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _safe_header(base_dir: Path) -> str:
    """The ``/lifemodel debug`` state block, degrading to a friendly line on any
    hiccup — this reader's own state must never crash the whole command."""
    try:
        return render_dump_for_dir(base_dir)
    except Exception as exc:  # noqa: BLE001 - a read-only view must never crash (§7)
        return f"🫀 **lifemodel debug** (read-only)\n\n<unavailable: {exc}>\n"


def activity_for_dir(base_dir: Path, raw_args: str) -> str:
    """Answer ``python3 -m lifemodel.activity [last [N] | turn <trace_id>]``.

    Read-only + fail-soft throughout (see the module docstring): a missing/
    locked/corrupt ``observability.sqlite`` degrades to a friendly line, never
    raises. ``""``/``"last [N]"`` render the interleaved tick/turn timeline
    (state header first); ``"turn <trace_id>"`` renders that turn's child
    tree, ids enriched from ``lifemodel.sqlite``; anything else is usage.
    """
    kind, arg = _parse_args(raw_args)
    if kind == "usage":
        return _USAGE

    header = _safe_header(base_dir)
    db_path = observability_db_path(base_dir)
    if not db_path.exists():
        return (
            header
            + "\nlifemodel activity: no trace store yet (observability.sqlite not created).\n"
        )

    try:
        with closing(_connect_ro(db_path)) as conn:
            if kind == "turn":
                assert isinstance(arg, str)
                body = _turn_detail(conn, arg, base_dir)
            else:
                assert isinstance(arg, int)
                roots = _root_rows(conn, arg)
                if not roots:
                    body = "lifemodel activity: no activity recorded yet.\n"
                else:
                    body = "\n".join(_render_timeline_body(conn, roots)) + "\n"
    except sqlite3.Error as exc:
        return header + f"\nlifemodel activity: trace store unreadable ({exc}).\n"

    return header + "\n" + body


# --------------------------------------------------------------------------- #
# ``python3 -m lifemodel.activity`` CLI
# --------------------------------------------------------------------------- #


def _default_base_dir() -> Path:
    """``$LIFEMODEL_BASE_DIR``, else the live being's default workspace dir."""
    override = os.environ.get("LIFEMODEL_BASE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "workspace" / "lifemodel"


def _main(argv: list[str]) -> int:
    # Direct stream write, NOT print(): this is a standalone CLI entry (not a
    # tick component) — see ``smoke.py``'s ``_main`` for the same discipline.
    sys.stdout.write(activity_for_dir(_default_base_dir(), " ".join(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

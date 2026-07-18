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

A COMPACT state header is prepended to both views (the being's vitals are
always useful context for "what was happening"): plain columns read straight
off ``lifemodel.sqlite``'s ``runtime_state`` singleton row (``tick_count``,
``last_tick_at``, ``energy``, ``fatigue``, ``u``, the affect axes,
``last_exchange_at``) via a ``?mode=ro`` connection — **never**
:func:`~lifemodel.debug.render_dump_for_dir`. That path builds a full
``LifeModel`` graph, which constructs
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` — a READ-WRITE
constructor (creates dirs, switches WAL, runs migrations, and can
quarantine/rebootstrap the file on a failed ``quick_check``). This reader's
whole premise is running from the working tree against the LIVE being's db
*before* deploying a change (spec workflow) — a newer build's migrations must
never run against that file out from under the running gateway. So NO code
path in this module may ever construct ``SQLiteRuntimeStore``; the header
shows only plain fields already sitting in the row, never the derived
``u``/gates/phase ``render_dump`` computes (that stays ``/lifemodel debug``'s
job).

The header also carries a COMPACT **BDI** section (C-I2): the live contact
desire (state + spring), the live contact intention (state), and the top few
live thoughts (most-salient first) — all bounded ``?mode=ro`` queries against
``lifemodel.sqlite``'s ``memory_records`` table (never the registry's full
decode, never ``SQLiteRuntimeStore``), "live" meaning the same non-terminal
states :mod:`~lifemodel.core.desire_view`/:mod:`~lifemodel.core.intention_view`/
:mod:`~lifemodel.core.thought_view` already define.

The ``last [N]`` timeline also shows a **writer-drop health** line —
``lifemodel_trace_writer_dropped_records``/``_write_errors``, the durable
gauges the trace-writer snapshots into ``metrics.sqlite`` — so a MISSING turn
span reads as "the writer silently dropped/erred", not "nothing happened",
when those counters are non-zero. And root-span selection for the timeline
BACKFILLS past a long run of heartbeat roots (bounded by a scan cap) rather
than taking a flat ``LIMIT N``, so a turn parked behind more than ``N``
heartbeats is never dropped off the end of the page before the heartbeat
collapse ever sees it.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from .core.desire_view import DESIRE_KIND, LIVE_DESIRE_STATES
from .core.intention_view import INTENTION_KIND, LIVE_INTENTION_STATES
from .core.thought_view import LIVE_THOUGHT_STATES, THOUGHT_KIND
from .debug import local_time
from .domain.objects import CONTACT_DESIRE_ID, CONTACT_INTENTION_ID
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

#: I3 fix — a flat ``LIMIT N`` root fetch can return N roots that are ALL
#: heartbeats, silently dropping a turn parked just past them (the exact
#: drowning this reader exists to prevent). :func:`_timeline_lines` instead
#: backfills: fetch progressively more roots (each retry multiplying the
#: fetch size by this factor) until N RENDERED units (a collapsed heartbeat
#: run counts as one) are in hand, or the scan cap below is hit.
_ROOT_FETCH_MULTIPLIER: Final = 4
#: Hard cap on how many roots one ``last N`` call will ever scan while
#: backfilling past a heartbeat run — "a few thousand", per the fix brief —
#: so a pathological store (a years-long heartbeat run with no other
#: activity) can never turn one command into an unbounded table scan.
_ROOT_SCAN_CAP: Final = 2000

#: I5 — the durable trace-writer health gauges (``core/tick_metrics.py``),
#: snapshotted into ``metrics.sqlite`` by the metrics sampler (bead 7.6).
#: Surfaced in the ``last [N]`` timeline header: a MISSING turn span can be a
#: silent queue-drop/write-error, not "nothing happened" (spec §9).
_WRITER_DROPPED_METRIC: Final = "lifemodel_trace_writer_dropped_records"
_WRITER_ERRORS_METRIC: Final = "lifemodel_trace_writer_write_errors"

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


def _completion_attr(spans: Sequence[_Span], attr: str) -> str | None:
    """The ``turn.completion`` child's full *attr* value (``final_output`` /
    ``reasoning``), or ``None``.

    ``turn_recorder.py`` stores up to ``_MAX_TEXT`` (4000) chars there;
    :func:`~lifemodel.trace_view.render_trace`'s generic attr formatter (``_fmt_value``)
    truncates every value to 200 — fine for the tree's other attrs, but not for the two
    places a turn's actual words + the being's own reasoning live (M2 + the reasoning
    follow-up). This reads the value directly from the span, bypassing that truncation."""
    for span in spans:
        if span.component == "turn.completion":
            value = span.attrs.get(attr)
            return value if isinstance(value, str) else None
    return None


def _strip_attr(spans: Sequence[_Span], component: str, attr: str) -> list[_Span]:
    """*spans* with *attr* removed from the ONE *component* span's attrs.

    So the generic tree renderer never ALSO prints a truncated duplicate of a
    value this reader is about to show in full separately (M2)."""
    return [
        replace(span, attrs={k: v for k, v in span.attrs.items() if k != attr})
        if span.component == component and attr in span.attrs
        else span
        for span in spans
    ]


def _turn_detail(conn: sqlite3.Connection, trace_id: str, base_dir: Path) -> str:
    spans = _read_spans(conn, trace_id)
    if not spans:
        return f"lifemodel activity: no turn {trace_id}\n"
    lines = [f"turn {trace_id}"]
    full_output = _completion_attr(spans, "final_output")
    reasoning = _completion_attr(spans, "reasoning")
    # Strip the two long values from the tree so render_trace never ALSO prints a
    # 200-char-truncated duplicate — they're shown in full below.
    tree_spans = list(spans)
    if full_output is not None:
        tree_spans = _strip_attr(tree_spans, "turn.completion", "final_output")
    if reasoning is not None:
        tree_spans = _strip_attr(tree_spans, "turn.completion", "reasoning")
    lines += render_trace(trace_id, tree_spans, ())
    if reasoning is not None:  # the "why did it answer that" — first, it leads to the words
        lines.append("")
        lines.append("turn.completion reasoning (full):")
        lines.append(reasoning)
    if full_output is not None:
        lines.append("")
        lines.append("turn.completion final_output (full):")
        lines.append(full_output)
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


# --------------------------------------------------------------------------- #
# State header (C1) — a GENUINELY read-only projection of runtime_state.
#
# Never :func:`~lifemodel.debug.render_dump_for_dir`: that path builds a full
# ``LifeModel`` graph, which constructs
# :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` — a READ-WRITE
# constructor (creates dirs, switches WAL, runs migrations, can
# quarantine/rebootstrap a corrupt file). This reader's whole premise is
# running from the working tree against the LIVE being's db before deploying —
# a newer build's migrations must never run against that file out from under
# the running gateway. So nothing below may EVER construct
# ``SQLiteRuntimeStore``; only plain columns already sitting in the row are
# shown, never the derived u/gates/phase ``render_dump`` computes.
# --------------------------------------------------------------------------- #

_STATE_HEADER_TITLE = "🫀 **lifemodel activity** (read-only state)"


def _read_runtime_state_row(base_dir: Path) -> dict[str, Any] | None:
    """The raw ``runtime_state.state_json`` blob, parsed — or ``None``.

    ``None`` covers every way this can come back empty: no ``lifemodel.sqlite``
    yet, no ``runtime_state`` row yet (a being that has never ticked), a
    locked/corrupt file, or a malformed/non-dict JSON body (:func:`_loads`
    already swallows that to ``{}``, folded to ``None`` here too — a real row
    is never empty, it always carries at least ``schema_version``).

    Opened via :func:`_connect_ro` (``?mode=ro``) — a plain ``SELECT`` against
    a row that, if present, already exists. This is the one function standing
    between the C1 fix and a regression: it must never construct
    :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`.
    """
    db_path = _lifemodel_db_path(base_dir)
    if not db_path.exists():
        return None
    try:
        with closing(_connect_ro(db_path)) as conn:
            row = conn.execute("SELECT state_json FROM runtime_state WHERE id = 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None or not isinstance(row[0], str):
        return None
    data = _loads(row[0])
    return data or None


def _fmt_state_float(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return f"{value:.2f}" if is_number else "n/a"


def _fmt_state_int(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "n/a"


def _fmt_state_ts(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    return local_time(value) if isinstance(value, str) else "n/a"


# --------------------------------------------------------------------------- #
# BDI section (C-I2) — the compact desire/intention/thoughts read.
#
# The spec calls for "current vitals/BDI" in this unified reader; C1 only
# restored the vitals half. This queries ``lifemodel.sqlite``'s
# ``memory_records`` table directly through :func:`_connect_ro` — same rule as
# everywhere else in this module: NEVER the registry's full decode (that would
# need a complete :class:`~lifemodel.domain.memory.MemoryRecord`, dragging in
# more machinery than a compact header needs) and NEVER
# :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`. "Live" states are
# the desire/intention/thought VIEWS' own frozensets
# (:data:`~lifemodel.core.desire_view.LIVE_DESIRE_STATES` et al.) — imported,
# never hand-copied, so a state added there can never silently drift out of
# sync here (the exact class of vocab-drift bug C-M3 fixes elsewhere in this
# same pass). ``content``/``spring`` come out of ``payload_json`` via
# :func:`~lifemodel.trace_view._loads` (already fail-soft — a malformed/absent
# blob decodes to ``{}``, never raises), so one bad row never sinks the rest.
# --------------------------------------------------------------------------- #

#: How many live thoughts the compact header shows, most-salient first.
_MAX_THOUGHTS_SHOWN: Final = 5


def _desire_line(conn: sqlite3.Connection) -> str | None:
    """``desire: <state> (spring=<spring>)`` for the live contact desire, or
    ``None`` when absent/terminal (never shows a satisfied/dropped/expired row)."""
    row = conn.execute(
        "SELECT state, payload_json FROM memory_records WHERE kind = ? AND id = ?",
        (DESIRE_KIND, CONTACT_DESIRE_ID),
    ).fetchone()
    if row is None:
        return None
    state, payload_json = row
    if not isinstance(state, str) or state not in LIVE_DESIRE_STATES:
        return None
    spring = _loads(payload_json if isinstance(payload_json, str) else None).get("spring")
    spring_s = spring if isinstance(spring, str) and spring else "?"
    return f"**desire:** {state} (spring={spring_s})"


def _intention_line(conn: sqlite3.Connection) -> str | None:
    """``intention: <state>`` for the live contact intention, or ``None`` when
    absent/terminal."""
    row = conn.execute(
        "SELECT state FROM memory_records WHERE kind = ? AND id = ?",
        (INTENTION_KIND, CONTACT_INTENTION_ID),
    ).fetchone()
    if row is None:
        return None
    state = row[0]
    if not isinstance(state, str) or state not in LIVE_INTENTION_STATES:
        return None
    return f"**intention:** {state}"


def _thought_lines(conn: sqlite3.Connection) -> list[str]:
    """Up to :data:`_MAX_THOUGHTS_SHOWN` live thoughts, most-salient first — a
    malformed row (bad JSON, non-string ``content``) is skipped, never a crash,
    and never stops the rest of the rows from rendering."""
    placeholders = ", ".join("?" for _ in LIVE_THOUGHT_STATES)
    rows = conn.execute(
        "SELECT payload_json FROM memory_records WHERE kind = ? AND state IN "
        f"({placeholders}) ORDER BY salience DESC LIMIT ?",
        (THOUGHT_KIND, *LIVE_THOUGHT_STATES, _MAX_THOUGHTS_SHOWN),
    ).fetchall()
    lines: list[str] = []
    for (payload_json,) in rows:
        content = _loads(payload_json if isinstance(payload_json, str) else None).get("content")
        if isinstance(content, str) and content:
            lines.append(f"- {content}")
    return lines


def _bdi_lines(base_dir: Path) -> list[str]:
    """The compact BDI section's lines (C-I2): live desire/intention/top
    thoughts, read straight off ``lifemodel.sqlite``'s ``memory_records`` table
    via :func:`_connect_ro`. Fail-soft: a missing db or a locked/corrupt/
    schema-less file degrades to an empty list — the header simply omits the
    section (never crashes, and never takes down the vitals half above it)."""
    db_path = _lifemodel_db_path(base_dir)
    if not db_path.exists():
        return []
    try:
        with closing(_connect_ro(db_path)) as conn:
            desire = _desire_line(conn)
            intention = _intention_line(conn)
            thoughts = _thought_lines(conn)
    except sqlite3.Error:
        return []
    lines: list[str] = []
    if desire is not None:
        lines.append(desire)
    if intention is not None:
        lines.append(intention)
    if thoughts:
        lines.append("**thoughts:**")
        lines.extend(thoughts)
    return lines


def _render_state_header(base_dir: Path) -> str:
    """The compact read-only vitals + BDI header (C1 + C-I2): plain
    ``runtime_state`` columns (``tick_count``, ``last_tick_at``, ``energy``,
    ``fatigue``, ``u``, the affect axes, ``last_exchange_at``) — never the
    derived u/gates/phase ``/lifemodel debug`` computes (out of scope for this
    fix; that reconstruction also needs the read-write ``SQLiteRuntimeStore``
    this reader must never touch, see the section banner above) — plus the
    compact live desire/intention/thoughts section (C-I2, see the BDI section
    banner above). The BDI section is independent of the vitals row's own
    availability (:func:`_bdi_lines` is always attempted) — a being could in
    principle have live BDI rows without (yet) a ``runtime_state`` row, and the
    two sections read two different tables/failure surfaces, so one's absence
    should never suppress the other."""
    data = _read_runtime_state_row(base_dir)
    if data is None:
        lines = [_STATE_HEADER_TITLE, "", "<unavailable: no readable runtime_state row>"]
    else:
        lines = [
            _STATE_HEADER_TITLE,
            "",
            f"**tick_count:** {_fmt_state_int(data, 'tick_count')}",
            f"**last_tick_at:** {_fmt_state_ts(data, 'last_tick_at')}",
            f"**energy:** {_fmt_state_float(data, 'energy')}",
            f"**fatigue:** {_fmt_state_float(data, 'fatigue')}",
            f"**u:** {_fmt_state_float(data, 'u')}",
            f"**affect:** v={_fmt_state_float(data, 'affect_valence')} "
            f"a={_fmt_state_float(data, 'affect_arousal')}",
            f"**last_exchange_at:** {_fmt_state_ts(data, 'last_exchange_at')}",
        ]
    lines.extend(_bdi_lines(base_dir))
    return "\n".join(lines) + "\n"


def _safe_header(base_dir: Path) -> str:
    """:func:`_render_state_header`, degrading to a friendly line on any
    hiccup — this reader's own state must never crash the whole command.
    Deliberately never ``debug.render_dump_for_dir`` — see the C1 section
    banner above."""
    try:
        return _render_state_header(base_dir)
    except Exception as exc:  # noqa: BLE001 - a read-only view must never crash (§7)
        return f"{_STATE_HEADER_TITLE}\n\n<unavailable: {exc}>\n"


# --------------------------------------------------------------------------- #
# Timeline backfill (I3) + writer-drop health (I5)
# --------------------------------------------------------------------------- #


def _trailing_heartbeat_run_len(roots: Sequence[_Root]) -> int:
    """The length of the run of consecutive heartbeat roots ending *roots*
    (newest-first) — i.e. the OLDEST roots in the fetched page, which may be
    only a PREFIX of a longer run that continues further back than this page
    reaches (C-M2). ``0`` when the oldest fetched root is not itself a
    heartbeat."""
    count = 0
    for root in reversed(roots):
        is_heartbeat = _frame_kind_of(root.attrs) == "execution" and (
            _str_attr(root.attrs, "trigger", "?") == "heartbeat"
        )
        if not is_heartbeat:
            break
        count += 1
    return count


def _timeline_lines(conn: sqlite3.Connection, n: int) -> list[str]:
    """Up to *n* newest activity units (turns + non-heartbeat ticks + collapsed
    heartbeat runs), newest first — never silently dropping a turn parked
    behind more than *n* heartbeat roots (I3).

    A flat ``LIMIT n`` root fetch can return *n* roots that are ALL
    heartbeats, so a turn just past them is never even fetched, let alone
    rendered — the exact drowning this reader exists to prevent. This instead
    backfills: fetch progressively more roots (each retry multiplying by
    :data:`_ROOT_FETCH_MULTIPLIER`) until *n* rendered units are in hand, the
    store is exhausted (fewer roots came back than were asked for), or
    :data:`_ROOT_SCAN_CAP` is hit — whichever comes first, so a pathological
    store can never turn one call into an unbounded scan.

    C-M2 fix: "*n* rendered units are in hand" is not, by itself, enough to
    stop. If the fetched page ENDS with the first 1-2 roots of a longer
    heartbeat run (the rest of that run sits just past this page's edge,
    unfetched), :func:`_render_timeline_body` cannot yet tell it is short of
    :data:`_HEARTBEAT_COLLAPSE_MIN_RUN` and renders those 1-2 roots as
    individual lines rather than folding them into the single collapsed line
    they would become once the rest of the run is in hand — inflating the
    rendered-unit count and stopping the backfill one fetch too early. So this
    keeps fetching past a nominally-sufficient count while the store is not yet
    exhausted AND the trailing (oldest-fetched) heartbeat run in hand is still
    shorter than the collapse threshold — it might still grow into a full
    collapse on the next, larger fetch.
    """
    fetch = min(max(n * _ROOT_FETCH_MULTIPLIER, n), _ROOT_SCAN_CAP)
    while True:
        roots = _root_rows(conn, fetch)
        exhausted = len(roots) < fetch or fetch >= _ROOT_SCAN_CAP
        trailing_run = _trailing_heartbeat_run_len(roots)
        incomplete_trailing_run = 0 < trailing_run < _HEARTBEAT_COLLAPSE_MIN_RUN
        lines = _render_timeline_body(conn, roots)
        if exhausted or (len(lines) >= n and not incomplete_trailing_run):
            return lines[:n]
        fetch = min(fetch * _ROOT_FETCH_MULTIPLIER, _ROOT_SCAN_CAP)


def _latest_gauge_value(conn: sqlite3.Connection, name: str) -> float | None:
    """The most recent ``metric_samples`` value for gauge *name* within the
    newest ``run_id``, or ``None`` when nothing has ever been sampled for it.

    C-I1 fix: a direct, bounded ``SELECT`` against the already-open (``?mode=ro``,
    see :func:`_connect_ro`) *conn* — deliberately NEVER
    :func:`~lifemodel.state.metrics_store.read_samples`, which opens its OWN
    plain ``sqlite3.connect(str(db_path))`` (state/metrics_store.py — no
    ``mode=ro``, no ``uri=True``). That is a genuine read-WRITE handle: on a
    file that does not exist yet it silently creates an empty one (the exact
    exists-check/connect race this reader must never risk against the live
    being's db), and it is a second, unnecessary writer on a file the metrics
    sampler thread already owns. Mirrors the newest-sample-of-the-newest-run
    shape ``read_samples(name=name, latest_run=True, limit=1)`` computes, just
    inlined as one read-only query.
    """
    row = conn.execute(
        "SELECT value FROM metric_samples WHERE name = ? AND run_id = "
        "(SELECT run_id FROM metric_samples ORDER BY ts DESC, rowid DESC LIMIT 1) "
        "ORDER BY ts DESC, rowid DESC LIMIT 1",
        (name,),
    ).fetchone()
    return float(row[0]) if row is not None else None


def _writer_health_line(base_dir: Path) -> str | None:
    """``trace-writer: dropped=<N> write_errors=<N>`` from ``metrics.sqlite``'s
    durable gauges (I5, spec §9) — a MISSING turn span can be a silent
    queue-drop/write-error, not "nothing happened"; this makes that visible in
    the ``last [N]`` timeline.

    Opens ``metrics.sqlite`` itself via :func:`_connect_ro` (C-I1 fix) — the
    SAME read-only discipline every other source this reader touches already
    holds to (``observability.sqlite``, ``lifemodel.sqlite``); the store's
    filename comes from the public :func:`~lifemodel.state.metrics_store.metrics_db_path`
    helper, imported lazily so an unimportable/absent metrics subtree degrades
    this ONE optional line rather than the whole reader.

    Fail-soft: an unimportable ``state.metrics_store``, a missing
    ``metrics.sqlite``, or any read hiccup returns ``None`` — the caller simply
    omits the line (this one line is optional, unlike a whole section, so
    there is no friendly placeholder to show in its place)."""
    try:
        from .state.metrics_store import metrics_db_path
    except Exception:  # noqa: BLE001 - optional subtree; omit the line, never crash
        return None
    db_path = metrics_db_path(base_dir)
    if not db_path.exists():
        return None
    try:
        with closing(_connect_ro(db_path)) as conn:
            dropped = _latest_gauge_value(conn, _WRITER_DROPPED_METRIC)
            errors = _latest_gauge_value(conn, _WRITER_ERRORS_METRIC)
    except sqlite3.Error:
        return None
    if dropped is None and errors is None:
        return None  # nothing sampled yet — omit rather than show a bare n/a pair
    dropped_s = "n/a" if dropped is None else f"{dropped:.0f}"
    errors_s = "n/a" if errors is None else f"{errors:.0f}"
    return f"trace-writer: dropped={dropped_s} write_errors={errors_s}"


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
                lines = _timeline_lines(conn, arg)
                body_lines: list[str] = []
                health = _writer_health_line(base_dir)
                if health is not None:
                    body_lines.append(health)
                    body_lines.append("")
                body_lines.extend(
                    lines if lines else ["lifemodel activity: no activity recorded yet."]
                )
                body = "\n".join(body_lines) + "\n"
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

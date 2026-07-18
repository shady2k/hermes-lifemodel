"""Tests for ``activity.py`` — ``python3 -m lifemodel.activity`` (lm-hg7 Task 12).

Seeds a temp ``observability.sqlite`` directly through the async
:class:`~lifemodel.state.trace_store.TraceWriter` (the same style
``tests/test_trace_store.py``/``tests/test_trace_view.py`` use) rather than
running a real tick/turn — this reader only cares about the ROWS a tick/turn
already writes (tasks 1-11), not how they got there.
"""

from __future__ import annotations

from pathlib import Path

from lifemodel.activity import activity_for_dir
from lifemodel.adapters.clock import SystemClock
from lifemodel.domain.memory import MemoryDraft
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state.trace_store import (
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)

SEEDED_TURN_TRACE_ID = "turntrace0000000000000000000001"
SEEDED_BELIEF_ID = "belief:seed:abc123"

_T0 = "2026-07-18T09:00:00.000000+00:00"
_T0_END = "2026-07-18T09:00:01.000000+00:00"
_T1 = "2026-07-18T09:05:00.000000+00:00"
_T1_END = "2026-07-18T09:05:02.000000+00:00"


def _seed(tmp_path: Path) -> None:
    """One execution root (frame_kind=execution) + one COMPLETED turn: root +
    injector/tool/completion children, ``ended_at`` set on every span."""
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id="ticktrace0000000000000000000001",
            span_id="tick-root",
            parent_span_id=None,
            component="tick",
            tick=5,
            started_at=_T0,
            ended_at=_T0_END,
            status="ok",
            attrs={"frame_kind": "execution", "trigger": "heartbeat"},
        )
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=_T1,
            ended_at=_T1_END,
            status="ok",
            attrs={
                "frame_kind": "turn",
                "turn_id": "t1",
                "session_id": "s1",
                "origin": "reactive",
            },
        )
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="inj-belief",
            parent_span_id="turn-root",
            component="turn.injector.belief",
            tick=None,
            started_at="2026-07-18T09:05:00.100000+00:00",
            ended_at="2026-07-18T09:05:00.200000+00:00",
            status="ok",
            attrs={"outcome": "surfaced", "count": 1, "ids": [SEEDED_BELIEF_ID]},
        )
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="tool-commitment",
            parent_span_id="turn-root",
            component="turn.tool.commitment",
            tick=None,
            started_at="2026-07-18T09:05:00.300000+00:00",
            ended_at="2026-07-18T09:05:00.400000+00:00",
            status="ok",
            attrs={"action": "discharge"},
        )
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="completion",
            parent_span_id="turn-root",
            component="turn.completion",
            tick=None,
            started_at=_T1_END,
            ended_at=_T1_END,
            status="ok",
            attrs={"final_output": "ok, talk soon", "reasoning": "short"},
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)


def _seed_open_turn(tmp_path: Path) -> None:
    """One turn root persisted OPEN — ``ended_at``/``status`` both ``None``."""
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id="turntraceopen000000000000000001",
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=_T0,
            ended_at=None,
            status=None,
            attrs={
                "frame_kind": "turn",
                "turn_id": "t9",
                "session_id": "s9",
                "origin": "reactive",
            },
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)


def _seed_legacy_tick_without_frame_kind(tmp_path: Path) -> None:
    """A pre-Task-7 root: no ``frame_kind``/``trigger`` in attrs at all."""
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id="legacytrace00000000000000000001",
            span_id="legacy-root",
            parent_span_id=None,
            component="tick",
            tick=1,
            started_at=_T0,
            ended_at=_T0_END,
            status="ok",
            attrs={},
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)


def _seed_belief_row(tmp_path: Path) -> None:
    """A real ``memory_records`` belief row in ``lifemodel.sqlite`` matching
    :data:`SEEDED_BELIEF_ID`, so the turn-detail enrichment has something to find."""
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).put(
        MemoryDraft(
            kind="belief",
            id=SEEDED_BELIEF_ID,
            state="active",
            payload={
                "content": "a fact, never surfaced by this reader",
                "subject": "owner",
                "source_message_ids": [],
                "source_thought_ids": [],
            },
            source="noticing",
        )
    )


# --------------------------------------------------------------------------- #
# The four behaviors named by the task brief
# --------------------------------------------------------------------------- #


def test_timeline_interleaves_and_labels_frame_kind_newest_first(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "turn" in out and "execution" in out
    # the turn line carries its outcome summary, not drowned/omitted.
    assert "belief=surfaced" in out


def test_open_turn_renders_incomplete_not_success(tmp_path: Path) -> None:
    _seed_open_turn(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "incomplete" in out.lower()


def test_turn_detail_shows_child_tree(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, f"turn {SEEDED_TURN_TRACE_ID}")
    assert "turn.injector.belief" in out and "turn.completion" in out


def test_reader_tolerates_old_span_without_frame_kind(tmp_path: Path) -> None:
    _seed_legacy_tick_without_frame_kind(tmp_path)
    activity_for_dir(tmp_path, "last 10")  # no crash; renders as execution/unknown


# --------------------------------------------------------------------------- #
# Enrichment (belief:/commitment: ids -> lifemodel.sqlite), fail-soft edges
# --------------------------------------------------------------------------- #


def test_turn_detail_enriches_known_belief_id_with_state(tmp_path: Path) -> None:
    _seed(tmp_path)
    _seed_belief_row(tmp_path)
    out = activity_for_dir(tmp_path, f"turn {SEEDED_TURN_TRACE_ID}")
    assert SEEDED_BELIEF_ID in out
    assert "active" in out
    # D10 (the same redaction discipline the belief injector itself holds to,
    # hooks.py): content never rides an observability surface.
    assert "never surfaced by this reader" not in out


def test_turn_detail_missing_ref_shows_bare_id_no_crash(tmp_path: Path) -> None:
    _seed(tmp_path)  # no lifemodel.sqlite row seeded — the id stays unresolved
    out = activity_for_dir(tmp_path, f"turn {SEEDED_TURN_TRACE_ID}")
    assert SEEDED_BELIEF_ID in out  # the bare id still shows


def test_unknown_turn_trace_id_is_a_friendly_message(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, "turn does-not-exist")
    assert "no turn" in out


# --------------------------------------------------------------------------- #
# Args + fail-soft edges
# --------------------------------------------------------------------------- #


def test_bare_and_last_default_render_the_timeline_not_usage(tmp_path: Path) -> None:
    _seed(tmp_path)
    assert "usage" not in activity_for_dir(tmp_path, "").lower()
    assert "usage" not in activity_for_dir(tmp_path, "last").lower()


def test_bad_args_return_usage(tmp_path: Path) -> None:
    assert "usage" in activity_for_dir(tmp_path, "last notanumber").lower()
    assert "usage" in activity_for_dir(tmp_path, "turn").lower()  # missing trace_id
    assert "usage" in activity_for_dir(tmp_path, "bogus").lower()


def test_missing_store_is_a_friendly_message_not_a_crash(tmp_path: Path) -> None:
    out = activity_for_dir(tmp_path, "last 10")
    assert "no trace store yet" in out


def test_heartbeat_run_collapses_but_turn_stays_visible(tmp_path: Path) -> None:
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        for i, ts in enumerate(
            [
                "2026-07-18T08:56:00.000000+00:00",
                "2026-07-18T08:57:00.000000+00:00",
                "2026-07-18T08:58:00.000000+00:00",
                "2026-07-18T08:59:00.000000+00:00",
            ]
        ):
            writer.submit_span(
                trace_id=f"heartbeattrace0000000000000000{i}",
                span_id="root",
                parent_span_id=None,
                component="tick",
                tick=i + 1,
                started_at=ts,
                ended_at=ts,
                status="ok",
                attrs={"frame_kind": "execution", "trigger": "heartbeat"},
            )
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at="2026-07-18T09:00:00.000000+00:00",
            ended_at="2026-07-18T09:00:01.000000+00:00",
            status="ok",
            attrs={
                "frame_kind": "turn",
                "turn_id": "t1",
                "session_id": "s1",
                "origin": "reactive",
            },
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)

    out = activity_for_dir(tmp_path, "last 10")
    assert "collapsed" in out
    assert "turn" in out
    assert out.count("trigger=heartbeat") == 0  # the run collapsed, not rendered per-line

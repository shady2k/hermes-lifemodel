"""Tests for ``activity.py`` — ``python3 -m lifemodel.activity`` (lm-hg7 Task 12).

Seeds a temp ``observability.sqlite`` directly through the async
:class:`~lifemodel.state.trace_store.TraceWriter` (the same style
``tests/test_trace_store.py``/``tests/test_trace_view.py`` use) rather than
running a real tick/turn — this reader only cares about the ROWS a tick/turn
already writes (tasks 1-11), not how they got there.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from lifemodel.activity import activity_for_dir
from lifemodel.adapters.clock import SystemClock
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.thought_view import build_thought, encode_thought, seed_thought_id
from lifemodel.domain.memory import MemoryDraft
from lifemodel.domain.objects import DesireSpring, DesireState, IntentionState, ThoughtState
from lifemodel.state.metrics_store import MetricsSampler, metrics_db_path
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state.trace_store import (
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)

SEEDED_TURN_TRACE_ID = "turntrace0000000000000000000001"
SEEDED_BELIEF_ID = "belief:seed:abc123"
LONG_COMPLETION_TRACE_ID = "turntracelong00000000000000001"

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


def _seed_long_completion(tmp_path: Path) -> str:
    """A turn whose ``turn.completion`` ``final_output`` is > 200 chars (M2)."""
    long_output = "the quick brown fox jumps over the lazy dog. " * 6  # > 200 chars
    assert len(long_output) > 200
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id=LONG_COMPLETION_TRACE_ID,
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=_T1,
            ended_at=_T1_END,
            status="ok",
            attrs={"frame_kind": "turn", "turn_id": "t2", "session_id": "s2", "origin": "reactive"},
        )
        writer.submit_span(
            trace_id=LONG_COMPLETION_TRACE_ID,
            span_id="completion",
            parent_span_id="turn-root",
            component="turn.completion",
            tick=None,
            started_at=_T1_END,
            ended_at=_T1_END,
            status="ok",
            attrs={"final_output": long_output, "reasoning": "short"},
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)
    return long_output


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


# --------------------------------------------------------------------------- #
# codex review wave B (lm-hg7): C1 read-only header, I3 un-drowned timeline,
# I5 writer-drop health, M2 full completion output
# --------------------------------------------------------------------------- #


def test_state_header_is_read_only_never_constructs_sqlite_runtime_store(
    tmp_path: Path, monkeypatch
) -> None:
    """C1: the state header must NEVER construct ``SQLiteRuntimeStore`` — that
    constructor is a read-write path (dir creation, WAL switch, migrations,
    quarantine-on-corrupt) that must not run against a live being's db from a
    bystander reader. Seeds a real ``runtime_state`` row through the actual
    store BEFORE patching (so the read-only header has real data to show),
    then patches ``__init__`` to explode and asserts the header still renders
    the real value — not merely "didn't crash" (a crash here would be
    swallowed by the header's own broad fail-soft ``except``, which would
    mask a regression as an ``<unavailable: ...>`` line instead of failing
    the test)."""
    SQLiteRuntimeStore(tmp_path, clock=SystemClock()).commit(
        State(tick_count=42, energy=0.75, last_tick_at=_T1)
    )

    def _boom(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("activity's state header must never construct SQLiteRuntimeStore (C1)")

    monkeypatch.setattr(SQLiteRuntimeStore, "__init__", _boom)

    out = activity_for_dir(tmp_path, "last 10")

    assert "42" in out  # the real tick_count, read directly off runtime_state
    assert "unavailable" not in out.lower()  # not the broad-except fallback masking a raise


def test_state_header_missing_db_is_a_friendly_line_not_a_crash(tmp_path: Path) -> None:
    out = activity_for_dir(tmp_path, "last 10")
    assert "unavailable" in out.lower()


def test_timeline_last_n_still_shows_turn_behind_many_heartbeats(tmp_path: Path) -> None:
    """I3: a flat ``LIMIT N`` root fetch used to return N roots that were ALL
    heartbeats, so a turn parked just past them was never even fetched. Seeds
    30 heartbeat roots newer than one turn root and asks for ``last 5`` —
    the turn must still surface."""
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id=SEEDED_TURN_TRACE_ID,
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at="2026-07-18T08:00:00.000000+00:00",
            ended_at="2026-07-18T08:00:01.000000+00:00",
            status="ok",
            attrs={
                "frame_kind": "turn",
                "turn_id": "t1",
                "session_id": "s1",
                "origin": "reactive",
            },
        )
        for i in range(30):
            writer.submit_span(
                trace_id=f"heartbeattrace000000000000{i:04d}",
                span_id="root",
                parent_span_id=None,
                component="tick",
                tick=i + 1,
                # all newer than the turn above
                started_at=f"2026-07-18T08:{i + 1:02d}:00.000000+00:00",
                ended_at=f"2026-07-18T08:{i + 1:02d}:00.000000+00:00",
                status="ok",
                attrs={"frame_kind": "execution", "trigger": "heartbeat"},
            )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)

    out = activity_for_dir(tmp_path, "last 5")
    assert SEEDED_TURN_TRACE_ID in out  # not drowned by the 30 newer heartbeats
    assert "collapsed" in out  # the heartbeat run still collapses


def test_timeline_shows_writer_drop_health_from_metrics_sqlite(tmp_path: Path) -> None:
    """I5: dropped/write_errors gauges snapshotted into ``metrics.sqlite`` are
    surfaced in the ``last N`` timeline — a missing turn span can be a silent
    writer drop, not "nothing happened"."""
    _seed(tmp_path)
    reg = MetricRegistry()
    reg.gauge("lifemodel_trace_writer_dropped_records").set(3.0)
    reg.gauge("lifemodel_trace_writer_write_errors").set(1.0)
    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), run_id="R", heartbeat_every=1)
    sampler.sample_once(now=_T1)
    sampler.close()

    out = activity_for_dir(tmp_path, "last 10")
    assert "dropped=3" in out
    assert "write_errors=1" in out


def test_timeline_omits_writer_health_line_when_metrics_store_absent(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "trace-writer" not in out  # no metrics.sqlite at all — the line is omitted


def test_turn_detail_shows_completion_final_output_in_full(tmp_path: Path) -> None:
    """M2: ``turn.completion``'s ``final_output`` must show in full, not
    truncated to ``render_trace``'s generic 200-char attr limit."""
    long_output = _seed_long_completion(tmp_path)
    out = activity_for_dir(tmp_path, f"turn {LONG_COMPLETION_TRACE_ID}")
    assert long_output in out


def test_turn_detail_shows_completion_reasoning_in_full(tmp_path: Path) -> None:
    """lm-hg7: the being's own reasoning (the "why did it answer that") must show
    in full in the turn deep-dive — the same untruncated treatment as final_output."""
    long_reasoning = (
        "they seem tired tonight, so a short warm reply is kinder than a question. " * 4
    )
    assert len(long_reasoning) > 200
    trace_id = "aa11bb22cc33dd44ee55ff6600778899"
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_span(
            trace_id=trace_id,
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=_T1,
            ended_at=_T1_END,
            status="ok",
            attrs={"frame_kind": "turn", "turn_id": "t9", "session_id": "s9", "origin": "reactive"},
        )
        writer.submit_span(
            trace_id=trace_id,
            span_id="completion",
            parent_span_id="turn-root",
            component="turn.completion",
            tick=None,
            started_at=_T1_END,
            ended_at=_T1_END,
            status="ok",
            attrs={"final_output": "Привет.", "reasoning": long_reasoning},
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)
    out = activity_for_dir(tmp_path, f"turn {trace_id}")
    assert "reasoning (full):" in out
    assert long_reasoning in out  # untruncated, not clipped to 200 chars


# --------------------------------------------------------------------------- #
# codex re-review wave C (lm-hg7): C-I1 read-only metrics read, C-I2 BDI
# header, C-M2 backfill pagination edge
# --------------------------------------------------------------------------- #


def test_writer_health_read_never_opens_a_non_read_only_connection_to_metrics_db(
    tmp_path: Path, monkeypatch
) -> None:
    """C-I1: the writer-health read used to go through
    ``state.metrics_store.read_samples``, which opens ``metrics.sqlite`` with a
    plain ``sqlite3.connect(str(db_path))`` — NOT ``?mode=ro`` — violating this
    reader's "every durable store is read-only" contract. Spies on the REAL
    ``sqlite3.connect`` (patched at the module level, so any regression that
    reintroduces a plain-connect path is caught regardless of which function
    does it) and asserts every connect touching ``metrics.sqlite`` carries
    ``uri=True`` and ``mode=ro`` — while also proving the health line still
    renders the real sampled values."""
    _seed(tmp_path)
    reg = MetricRegistry()
    reg.gauge("lifemodel_trace_writer_dropped_records").set(3.0)
    reg.gauge("lifemodel_trace_writer_write_errors").set(1.0)
    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), run_id="R", heartbeat_every=1)
    sampler.sample_once(now=_T1)
    sampler.close()

    real_connect = sqlite3.connect

    def _spying_connect(database, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if "metrics.sqlite" in str(database):
            assert kwargs.get("uri") is True and "mode=ro" in str(database), (
                f"non-read-only connect to metrics.sqlite: {database!r} {kwargs!r}"
            )
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _spying_connect)

    out = activity_for_dir(tmp_path, "last 10")

    assert "dropped=3" in out
    assert "write_errors=1" in out


def test_writer_health_read_never_calls_read_samples(tmp_path: Path, monkeypatch) -> None:
    """C-I1 (belt-and-suspenders): even if some future refactor reaches back
    for ``state.metrics_store.read_samples`` (the plain-connect function this
    fix removes from the call path), it must not actually be called — patched
    to explode; the health line must still render the real values, proving the
    explosion never fires."""
    _seed(tmp_path)
    reg = MetricRegistry()
    reg.gauge("lifemodel_trace_writer_dropped_records").set(3.0)
    reg.gauge("lifemodel_trace_writer_write_errors").set(1.0)
    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), run_id="R", heartbeat_every=1)
    sampler.sample_once(now=_T1)
    sampler.close()

    from lifemodel.state import metrics_store

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("activity's writer-health read must never call read_samples (C-I1)")

    monkeypatch.setattr(metrics_store, "read_samples", _boom)

    out = activity_for_dir(tmp_path, "last 10")

    assert "dropped=3" in out
    assert "write_errors=1" in out


def _seed_bdi_rows(tmp_path: Path) -> None:
    """A live desire + intention + two thoughts (one active, one parked) in
    ``lifemodel.sqlite`` — through the real registry encode doors (never a
    hand-built payload), matching how the being's own tick would persist them."""
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.put(
        encode_contact_desire(
            build_contact_desire(state=DesireState.ACTIVE, salience=0.5, spring=DesireSpring.DRIVE)
        )
    )
    store.put(
        encode_contact_intention(
            build_contact_intention(
                state=IntentionState.ACTIVE, commitment_strength=0.7, salience=0.6
            )
        )
    )
    store.put(
        encode_thought(
            build_thought(
                id=seed_thought_id("the most salient test thought"),
                content="the most salient test thought",
                state=ThoughtState.ACTIVE,
                salience=0.9,
            )
        )
    )
    store.put(
        encode_thought(
            build_thought(
                id=seed_thought_id("a less salient parked thought"),
                content="a less salient parked thought",
                state=ThoughtState.PARKED,
                salience=0.1,
            )
        )
    )


def test_state_header_shows_compact_bdi_section(tmp_path: Path) -> None:
    """C-I2: the unified reader's header must carry desire/intention/thoughts,
    not just runtime vitals."""
    _seed_bdi_rows(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "**desire:** active" in out
    assert "spring=drive" in out
    assert "**intention:** active" in out
    assert "the most salient test thought" in out
    assert "a less salient parked thought" in out


def test_bdi_section_never_constructs_sqlite_runtime_store(tmp_path: Path, monkeypatch) -> None:
    """C-I2: like the C1 vitals header, the BDI section must NEVER construct
    ``SQLiteRuntimeStore`` — patches ``__init__`` to explode and asserts the
    section still renders the real seeded data (not merely "didn't crash",
    which a broad fail-soft except could mask as an empty section instead of
    failing the test)."""
    _seed_bdi_rows(tmp_path)

    def _boom(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            "activity's BDI section must never construct SQLiteRuntimeStore (C-I2)"
        )

    monkeypatch.setattr(SQLiteRuntimeStore, "__init__", _boom)

    out = activity_for_dir(tmp_path, "last 10")

    assert "the most salient test thought" in out


def test_bdi_section_omits_terminal_desire_intention_and_thought(tmp_path: Path) -> None:
    """A satisfied desire / completed intention / resolved thought are
    terminal — absence, not a live row — so none of them should render."""
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.SATISFIED)))
    store.put(encode_contact_intention(build_contact_intention(state=IntentionState.COMPLETED)))
    store.put(
        encode_thought(
            build_thought(
                id=seed_thought_id("a resolved thought"),
                content="a resolved thought",
                state=ThoughtState.RESOLVED,
            )
        )
    )
    out = activity_for_dir(tmp_path, "last 10")
    assert "**desire:**" not in out
    assert "**intention:**" not in out
    assert "a resolved thought" not in out


def test_bdi_section_skips_a_malformed_thought_row_without_crashing(tmp_path: Path) -> None:
    """A row with unparseable ``payload_json`` must be skipped, never crash the
    whole BDI section (or the reader) — the other, well-formed row still shows."""
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.put(
        encode_thought(
            build_thought(
                id=seed_thought_id("good thought"),
                content="good thought",
                state=ThoughtState.ACTIVE,
                salience=0.9,
            )
        )
    )
    db_path = tmp_path / "lifemodel.sqlite"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "INSERT INTO memory_records (kind, id, state, recipient_id, payload_json, salience, "
            "confidence, expires_at, source, created_at, updated_at, revision, schema_version) "
            "VALUES ('thought', 'thought:bad', 'active', 'owner', 'not valid json{{', 0.95, "
            "NULL, NULL, 'test', '2026-07-18T00:00:00.000000+00:00', "
            "'2026-07-18T00:00:00.000000+00:00', 0, 1)"
        )
        conn.commit()

    out = activity_for_dir(tmp_path, "last 10")
    assert "good thought" in out  # the malformed row didn't sink the section


def test_bdi_section_absent_when_no_lifemodel_db(tmp_path: Path) -> None:
    """No ``lifemodel.sqlite`` at all — no BDI section, no crash (the vitals
    header already covers this path with its own friendly message)."""
    out = activity_for_dir(tmp_path, "last 10")
    assert "**desire:**" not in out
    assert "**intention:**" not in out


def test_backfill_collapses_heartbeat_run_split_across_a_page_boundary(tmp_path: Path) -> None:
    """C-M2: the backfill re-fetches with a LARGER limit each retry — the SAME
    query, not an offset — so a fetched page's OLDEST rows can be only the
    first 1-2 members of a longer heartbeat run whose rest lies just past this
    page's edge. Rendered individually (too short to collapse YET), those 1-2
    lines can push the rendered-unit count to N and stop the backfill one
    fetch too early — even though one more fetch would collapse them and free
    enough budget to reveal a turn sitting right after the run.

    Seeds (newest -> oldest): RUN_A (13 heartbeats, collapses fully within the
    first fetch) + turnX + RUN_B (4 heartbeats total, but only its first 2
    fall inside the ``last 4`` first-fetch window of 16 roots) + turnC (only
    reachable once RUN_B is fully fetched and collapses to free the 4th slot).
    """
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)

    def _ts(i: int) -> str:
        return f"2026-07-18T08:{59 - i:02d}:00.000000+00:00"

    try:
        i = 0
        for _ in range(13):  # RUN_A: fully inside page 1, collapses there already
            ts = _ts(i)
            writer.submit_span(
                trace_id=f"runatrace000000000000000000{i:02d}",
                span_id="root",
                parent_span_id=None,
                component="tick",
                tick=1000 - i,
                started_at=ts,
                ended_at=ts,
                status="ok",
                attrs={"frame_kind": "execution", "trigger": "heartbeat"},
            )
            i += 1
        ts = _ts(i)
        writer.submit_span(  # turnX — the run-breaker between RUN_A and RUN_B
            trace_id="turnxtrace0000000000000000001",
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=ts,
            ended_at=ts,
            status="ok",
            attrs={"frame_kind": "turn", "turn_id": "tx", "session_id": "sx", "origin": "reactive"},
        )
        i += 1
        for _ in range(4):  # RUN_B: only its first 2 fall inside page 1 (fetch=16)
            ts = _ts(i)
            writer.submit_span(
                trace_id=f"runbtrace000000000000000000{i:02d}",
                span_id="root",
                parent_span_id=None,
                component="tick",
                tick=500 - i,
                started_at=ts,
                ended_at=ts,
                status="ok",
                attrs={"frame_kind": "execution", "trigger": "heartbeat"},
            )
            i += 1
        ts = _ts(i)
        writer.submit_span(  # turnC — must surface once RUN_B fully collapses
            trace_id="turnctrace0000000000000000001",
            span_id="turn-root",
            parent_span_id=None,
            component="turn",
            tick=None,
            started_at=ts,
            ended_at=ts,
            status="ok",
            attrs={"frame_kind": "turn", "turn_id": "tc", "session_id": "sc", "origin": "reactive"},
        )
        writer.flush(timeout=5.0)
    finally:
        release_trace_writer(db)

    out = activity_for_dir(tmp_path, "last 4")
    assert "turnctrace0000000000000000001" in out  # the turn past the split run surfaces
    assert out.count("trigger=heartbeat") == 0  # both runs collapsed, never per-line

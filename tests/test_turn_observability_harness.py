"""End-to-end real-code harness for turn observability (lm-hg7 Task 13 — final).

Tasks 1-12 each proved ONE piece of the turn-observability plan in isolation
(the metric, the recorder's methods, each injector's child span, the tool
spans, the activity reader) — always against a fake tracer/sink. This is the
"does it actually fit together" proof: a LIVE-SHAPED :class:`TurnRecorder` —
the real :class:`~lifemodel.adapters.tracer.StdlibTracer`, a real
:class:`~lifemodel.state.trace_store.TraceWriter` over a temp
``observability.sqlite``, a shared :class:`~lifemodel.core.metrics.MetricRegistry`,
and the real :class:`~lifemodel.adapters.clock.SystemClock` — drives one REACTIVE
turn through the REAL belief injector (:func:`~lifemodel.hooks.make_belief_injector`,
seeded to genuinely surface a belief) and a real tool span
(:meth:`~lifemodel.core.turn_recorder.TurnRecorder.tool_open`/:meth:`tool_close`),
then closes it.

After a ``flush`` (read-your-writes, mirroring ``trace_view``/``activity``), the
spans are read back from the DURABLE store two ways — direct SQL, and
:func:`~lifemodel.activity.activity_for_dir` (the exact reader the debugging
agent runs) — and the shared :class:`MetricRegistry` is checked to carry the
matching ``turn_injector_total{component,outcome}`` series. This is the proof
that "write a turn, read it back from the durable store" holds end-to-end.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from lifemodel.activity import activity_for_dir
from lifemodel.adapters.clock import SystemClock
from lifemodel.adapters.tracer import StdlibTracer
from lifemodel.core.belief_view import belief_id, build_belief, encode_belief
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_metrics import TURN_INJECTOR_TOTAL
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.hooks import make_belief_injector
from lifemodel.state.model import State
from lifemodel.state.trace_store import (
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)

_SESSION_ID = "e2e-session-1"
_TURN_ID = "e2e-turn-1"
_TOOL_CALL_ID = "e2e-call-1"
_CONFIDENT_BELIEF = "They go quiet before big decisions."


def _read(db_path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(sql, params).fetchall()


def test_real_turn_is_written_and_read_back_from_the_durable_store(
    tmp_path: Path, build_lm
) -> None:
    db_path = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db_path)
    try:
        registry = MetricRegistry()
        rec = TurnRecorder(
            tracer=StdlibTracer(), writer=writer, metrics=registry, clock=SystemClock()
        )

        # Seed the store with a confident, non-private belief the REAL injector
        # will genuinely surface (mirrors tests/test_belief_injector.py's own seeding).
        build_lm().state.commit(State())
        bid = belief_id("t1", _CONFIDENT_BELIEF)
        build_lm().state.put(
            encode_belief(
                build_belief(
                    id=bid,
                    content=_CONFIDENT_BELIEF,
                    confidence=0.8,
                    sensitivity=Sensitivity.SENSITIVE,
                    source_thought_ids=("t1",),
                )
            )
        )

        # ---- drive one REACTIVE turn through the real code paths ----
        rec.ensure_turn(_SESSION_ID, _TURN_ID)

        injector = make_belief_injector(build_lm, recorder=rec, metrics=registry)
        result = injector(
            session_id=_SESSION_ID, turn_id=_TURN_ID, user_message="how have I seemed lately?"
        )
        assert result is not None and _CONFIDENT_BELIEF in result["context"]

        rec.tool_open(_SESSION_ID, _TURN_ID, tool="commitment", tool_call_id=_TOOL_CALL_ID)
        rec.tool_close(_TOOL_CALL_ID, status="ok", action="discharge")

        rec.close_turn(
            _SESSION_ID,
            _TURN_ID,
            final_output="Noted — here's what I'm carrying for you.",
            reasoning="surfaced a belief, discharged a commitment",
        )

        assert writer.flush(timeout=5.0)  # read-your-writes, mirroring trace_view/activity
    finally:
        release_trace_writer(db_path)

    # --------------------------------------------------------------------- #
    # Read the DURABLE store back — exactly the shape the debugging agent will.
    # --------------------------------------------------------------------- #
    root_rows = _read(
        db_path,
        "SELECT trace_id, span_id, parent_span_id, status, ended_at "
        "FROM trace_spans WHERE component = 'turn'",
    )
    assert len(root_rows) == 1  # the ONE turn root this harness wrote
    trace_id, root_span_id, root_parent, root_status, root_ended = root_rows[0]
    assert root_parent is None
    assert root_status == "ok"
    assert root_ended is not None  # closed, never left open

    components = {
        row[0]
        for row in _read(
            db_path, "SELECT component FROM trace_spans WHERE trace_id = ?", (trace_id,)
        )
    }
    assert components == {
        "turn",
        "turn.injector.belief",
        "turn.tool.commitment",
        "turn.completion",
    }

    # Every child actually parents onto the ONE root span — real correlation via
    # parent_span_id, not just a shared trace_id.
    child_parents = {
        row[0]
        for row in _read(
            db_path,
            "SELECT parent_span_id FROM trace_spans WHERE trace_id = ? AND component != 'turn'",
            (trace_id,),
        )
    }
    assert child_parents == {root_span_id}

    (belief_attrs_json,) = _read(
        db_path,
        "SELECT attrs_json FROM trace_spans "
        "WHERE trace_id = ? AND component = 'turn.injector.belief'",
        (trace_id,),
    )[0]
    belief_attrs = json.loads(belief_attrs_json)
    assert belief_attrs["outcome"] == "surfaced"
    assert belief_attrs["ids"] == [bid]
    assert _CONFIDENT_BELIEF not in belief_attrs_json  # D10: content never rides the span

    (tool_status, tool_attrs_json) = _read(
        db_path,
        "SELECT status, attrs_json FROM trace_spans "
        "WHERE trace_id = ? AND component = 'turn.tool.commitment'",
        (trace_id,),
    )[0]
    assert tool_status == "ok"
    assert json.loads(tool_attrs_json)["action"] == "discharge"

    (completion_attrs_json,) = _read(
        db_path,
        "SELECT attrs_json FROM trace_spans WHERE trace_id = ? AND component = 'turn.completion'",
        (trace_id,),
    )[0]
    assert "here's what I'm carrying" in json.loads(completion_attrs_json)["final_output"]

    # --------------------------------------------------------------------- #
    # The reader the debugging agent actually runs: `python3 -m lifemodel.activity`.
    # --------------------------------------------------------------------- #
    out = activity_for_dir(tmp_path, f"turn {trace_id}")
    assert f"turn {trace_id}" in out
    assert "turn.injector.belief" in out and "outcome=surfaced" in out
    assert "turn.tool.commitment" in out
    assert "turn.completion" in out
    assert _CONFIDENT_BELIEF not in out  # D10 redaction holds through the reader too

    # The timeline view also surfaces this turn (newest-first, origin/status shown).
    timeline = activity_for_dir(tmp_path, "last 10")
    assert trace_id in timeline
    assert "origin=reactive" in timeline and "status=ok" in timeline

    # --------------------------------------------------------------------- #
    # The shared MetricRegistry carries the matching turn_injector_total series.
    # --------------------------------------------------------------------- #
    assert registry.get(TURN_INJECTOR_TOTAL).value(component="belief", outcome="surfaced") == 1.0

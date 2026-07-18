"""Tests for ``core/turn_recorder.py`` — the full ``TurnRecorder`` service
(construction, ``ensure_turn``, ``injector_span``, ``tool_open``/``tool_close``,
``close_turn``; lm-hg7).

Contract under test (tasks 3-6 of the turn-observability plan):

* ``ensure_turn`` persists an OPEN root span (``ended_at``/``status`` both
  ``None``) tagged ``frame_kind="turn"`` — outside the internal ledger lock;
* it is idempotent per ``(session_id, turn_id)`` — a second call for the same
  key opens no second root;
* opening a NEW turn for a session reconciles any OTHER still-open turn of
  that same session to a closed ``status="failed"`` / ``outcome="abandoned"``
  span — the only reliable "the prior turn died" signal;
* ``origin="reactive"`` mints a fresh trace; ``origin="proactive"`` with an
  ``upstream_traceparent`` CONTINUES the parsed upstream trace id;
* the ledger is bounded (TTL + max entries, lazy) and ``ensure_turn`` never
  raises even when the underlying sink's ``submit_span`` blows up (fail-soft);
* ``injector_span`` persists a ``turn.injector.<component>`` CHILD span of the
  open turn root and increments ``lifemodel_turn_injector_total`` — ``status="ok"``
  / the injector's own ``outcome`` on a clean exit, ``status="failed"`` /
  ``outcome="error"`` (and a RE-RAISE) on a body exception;
* with no open turn for the key, the span degrades to a bare parentless root
  rather than raising;
* ``tool_open``/``tool_close`` persist a ``turn.tool.<tool>`` CHILD span keyed
  by ``tool_call_id``, independent of the single per-turn ledger; an unknown
  ``tool_call_id`` is a best-effort no-op;
* ``close_turn`` persists a ``turn.completion`` CHILD (``final_output``/
  ``reasoning`` sliced to a bounded length) then closes the root
  (``ended_at``=now, ``status`` clamped to the closed vocabulary) and drops the
  ledger entry — idempotent (a second close / an unknown key is a no-op) and
  fail-soft even when the sink blows up.

Stdlib only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_metrics import TURN_INJECTOR_TOTAL
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.testing.fakes import FakeClock, FakeTracer

_NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


class CapturingSink:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def submit_span(self, **kw: Any) -> bool:
        self.spans.append(kw)
        return True

    def submit_event(self, **kw: Any) -> bool:
        return True

    def submit_correlation(self, **kw: Any) -> bool:
        return True


def _recorder() -> TurnRecorder:
    return TurnRecorder(
        tracer=FakeTracer(),
        writer=CapturingSink(),
        metrics=MetricRegistry(),
        clock=FakeClock(_NOW),
    )


def test_ensure_turn_persists_open_root_with_frame_kind_and_no_end() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1", model="opus", platform="telegram", origin="reactive")
    (root,) = [s for s in rec._writer.spans if s["component"] == "turn"]
    assert root["tick"] is None
    assert root["ended_at"] is None and root["status"] is None
    assert root["attrs"]["frame_kind"] == "turn"
    assert root["attrs"]["turn_id"] == "t1" and root["attrs"]["origin"] == "reactive"


def test_ensure_turn_is_idempotent_per_key() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.ensure_turn("s1", "t1")  # same turn — no second root
    assert len([s for s in rec._writer.spans if s["component"] == "turn"]) == 1


def test_a_new_turn_reconciles_the_prior_open_turn_of_the_same_session() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.ensure_turn("s1", "t2")  # t1 never closed — abandoned
    closed = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "failed"]
    assert len(closed) == 1
    assert closed[0]["attrs"]["turn_id"] == "t1"
    assert closed[0]["attrs"]["outcome"] == "abandoned"


def test_reactive_mints_fresh_trace_proactive_continues_upstream() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1", origin="reactive")
    rec.ensure_turn(
        "s2",
        "t9",
        origin="proactive",
        upstream_traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01",
    )
    roots = {
        s["attrs"]["turn_id"]: s
        for s in rec._writer.spans
        if s["component"] == "turn" and s["ended_at"] is None
    }
    assert roots["t9"]["trace_id"] == "a" * 32  # continued the upstream trace id
    assert roots["t1"]["trace_id"] != "a" * 32


def test_ledger_is_bounded_and_never_raises_on_a_broken_sink() -> None:
    class BoomSink(CapturingSink):
        def submit_span(self, **kw: Any) -> bool:
            raise RuntimeError("disk gone")

    rec = TurnRecorder(
        tracer=FakeTracer(),
        writer=BoomSink(),
        metrics=MetricRegistry(),
        clock=FakeClock(_NOW),
        max_entries=2,
    )
    for i in range(5):
        rec.ensure_turn("s1", f"t{i}")  # must not raise despite the sink blowing up


def test_injector_span_success_persists_ok_child_and_increments_outcome() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    with rec.injector_span("s1", "t1", "belief") as span:
        span.set(outcome="surfaced", count=2, ids=["belief:ab", "belief:cd"])
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"][0]
    assert child["status"] == "ok" and child["attrs"]["outcome"] == "surfaced"
    assert child["attrs"]["count"] == 2
    assert (
        rec._metrics.get(TURN_INJECTOR_TOTAL).value(component="belief", outcome="surfaced") == 1.0
    )


def test_injector_span_reraises_and_marks_failed_with_error_outcome() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    with pytest.raises(RuntimeError), rec.injector_span("s1", "t1", "belief") as span:
        span.set(outcome="surfaced")  # then the body blows up before completing
        raise RuntimeError("boom")
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"][0]
    assert child["status"] == "failed" and child["attrs"]["outcome"] == "error"
    assert rec._metrics.get(TURN_INJECTOR_TOTAL).value(component="belief", outcome="error") == 1.0


def test_injector_span_never_called_set_closes_with_unknown_outcome() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    with rec.injector_span("s1", "t1", "genesis"):
        pass  # the injector never called span.set — the default outcome ships
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.genesis"][0]
    assert child["status"] == "ok" and child["attrs"]["outcome"] == "unknown"
    assert (
        rec._metrics.get(TURN_INJECTOR_TOTAL).value(component="genesis", outcome="unknown") == 1.0
    )


def test_injector_span_with_no_open_turn_degrades_to_a_bare_parentless_child() -> None:
    rec = _recorder()  # no ensure_turn call — the ledger has no entry for this key
    with rec.injector_span("s1", "t1", "belief") as span:
        span.set(outcome="empty")
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"][0]
    assert child["parent_span_id"] is None  # a fresh root, not a crash
    assert child["status"] == "ok" and child["attrs"]["outcome"] == "empty"


def test_tool_open_close_persists_child_keyed_by_call_id() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_7")
    rec.tool_open("s1", "t1", tool="check_in", tool_call_id="call_8")  # concurrent, distinct id
    rec.tool_close("call_7", status="ok", action="discharge")
    child = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"][0]
    assert child["status"] == "ok" and child["attrs"]["action"] == "discharge"
    rec.tool_close("nope")  # unknown id — best-effort no-op, no raise


def test_close_turn_writes_completion_and_closes_root() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="ok, talk soon", reasoning="short and warm")
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert "talk soon" in completion["attrs"]["final_output"]
    assert completion["attrs"]["reasoning"] == "short and warm"
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root and closed_root[-1]["ended_at"] is not None
    assert ("s1", "t1") not in rec._ledger  # entry removed
    rec.close_turn("s1", "t1")  # second close — no raise, no duplicate root close
    closed_ok = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert len(closed_ok) == 1  # the second close persisted nothing more


def test_close_turn_truncates_oversized_text_and_clamps_bad_status() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="x" * 5000, status="bogus")
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert len(completion["attrs"]["final_output"]) == 4000
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["ended_at"]]
    assert closed_root[-1]["status"] == "ok"  # out-of-vocabulary status clamped, not persisted raw


def test_close_turn_on_unknown_key_is_a_no_op() -> None:
    rec = _recorder()
    rec.close_turn("nope", "nope")  # never opened — no raise, nothing persisted
    assert rec._writer.spans == []


def test_close_turn_never_raises_on_a_broken_sink() -> None:
    class BoomSink(CapturingSink):
        def submit_span(self, **kw: Any) -> bool:
            raise RuntimeError("disk gone")

    rec = TurnRecorder(
        tracer=FakeTracer(), writer=BoomSink(), metrics=MetricRegistry(), clock=FakeClock(_NOW)
    )
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="hi")  # must not raise despite the sink blowing up
    assert ("s1", "t1") not in rec._ledger  # still removed even though persistence failed

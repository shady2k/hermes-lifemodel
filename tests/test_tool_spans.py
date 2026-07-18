"""Tests for ``make_tool_span_open``/``make_tool_span_close`` (lm-hg7 Task 11) —
the ``pre_tool_call``/``post_tool_call`` observers that open/close a
``turn.tool.<tool_name>`` CHILD span through a live :class:`TurnRecorder`.

The kwarg NAMES exercised here are the ones actually verified against the host
(``~/.hermes/hermes-agent``):

* ``pre_tool_call`` (``hermes_cli/plugins.py``'s
  ``_get_pre_tool_call_directive_details`` → ``invoke_hook("pre_tool_call", ...)``):
  ``tool_name``, ``args``, ``task_id``, ``session_id``, ``tool_call_id``,
  ``turn_id``, ``api_request_id``, ``middleware_trace``.
* ``post_tool_call`` (``model_tools.py``'s ``_emit_post_tool_call_hook`` →
  ``invoke_hook("post_tool_call", ...)``): ``tool_name``, ``args``, ``result``,
  ``task_id``, ``session_id``, ``tool_call_id``, ``turn_id``, ``api_request_id``,
  ``duration_ms``, ``status``, ``error_type``, ``error_message``,
  ``middleware_trace``.

Both callbacks are pure OBSERVERS (they always return ``None`` — a
``pre_tool_call`` callback is only a block/approve DIRECTIVE when it returns a
dict with ``action`` in ``{"block", "approve"}``), so a wiring test elsewhere
(``tests/test_plugin.py``) also checks they never veto a tool call.

Stdlib only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.hooks import make_tool_span_close, make_tool_span_open
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


def test_open_then_close_persists_a_turn_tool_child_keyed_by_call_id() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    pre = make_tool_span_open(rec)
    post = make_tool_span_close(rec)

    # Exactly the host's pre_tool_call kwarg names (args/task_id/api_request_id/
    # middleware_trace included, and ignored via **_kw) — never function_name.
    result = pre(
        tool_name="commitment",
        args={"action": "discharge", "id": "c1"},
        task_id="task-1",
        session_id="s1",
        tool_call_id="call_7",
        turn_id="t1",
        api_request_id="req-1",
        middleware_trace=[],
    )
    assert result is None  # a pure observer — never a block/approve directive

    close_result = post(
        tool_name="commitment",
        args={"action": "discharge", "id": "c1"},
        result='{"status": "ok"}',
        task_id="task-1",
        session_id="s1",
        tool_call_id="call_7",
        turn_id="t1",
        api_request_id="req-1",
        duration_ms=42,
        status="ok",
        error_type=None,
        error_message=None,
        middleware_trace=[],
    )
    assert close_result is None

    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["status"] == "ok"
    assert child["attrs"]["duration_ms"] == 42
    # The child is parented to the open turn's root, not a bare fresh root.
    assert child["parent_span_id"] is not None


def test_two_concurrent_tool_calls_stay_distinct_by_call_id() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    pre = make_tool_span_open(rec)
    post = make_tool_span_close(rec)

    pre(tool_name="commitment", tool_call_id="call_a", session_id="s1", turn_id="t1")
    pre(tool_name="check_in", tool_call_id="call_b", session_id="s1", turn_id="t1")

    post(tool_call_id="call_a", status="ok", duration_ms=10)

    tool_spans = [s for s in rec._writer.spans if s["component"].startswith("turn.tool.")]
    assert {s["component"] for s in tool_spans} == {"turn.tool.commitment"}
    # call_b was opened but never closed — no premature span for it yet.
    post(tool_call_id="call_b", status="ok", duration_ms=5)
    tool_spans = [s for s in rec._writer.spans if s["component"].startswith("turn.tool.")]
    assert {s["component"] for s in tool_spans} == {"turn.tool.commitment", "turn.tool.check_in"}


def test_close_propagates_a_non_ok_status() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    pre = make_tool_span_open(rec)
    post = make_tool_span_close(rec)

    pre(tool_name="commitment", tool_call_id="call_9", session_id="s1", turn_id="t1")
    post(tool_call_id="call_9", status="failed", duration_ms=3, error_type="ValueError")

    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["status"] == "failed"


def test_close_with_unknown_call_id_is_a_no_op_and_never_raises() -> None:
    rec = _recorder()
    post = make_tool_span_close(rec)
    assert post(tool_call_id="never-opened", status="ok") is None
    assert rec._writer.spans == []


def test_open_with_no_turn_recorder_raises_never_reaches_the_caller() -> None:
    """A broken sink must never surface out of these observer callbacks — they
    sit directly on the host's tool-dispatch hot path (``model_tools.py``'s
    ``handle_function_call``), and :class:`TurnRecorder`'s own methods are
    fail-soft by construction; this just confirms the thin wrapper adds no new
    way to raise."""

    class BoomSink(CapturingSink):
        def submit_span(self, **kw: Any) -> bool:
            raise RuntimeError("disk gone")

    rec = TurnRecorder(
        tracer=FakeTracer(), writer=BoomSink(), metrics=MetricRegistry(), clock=FakeClock(_NOW)
    )
    pre = make_tool_span_open(rec)
    post = make_tool_span_close(rec)

    pre(tool_name="commitment", tool_call_id="call_1", session_id="s1", turn_id="t1")
    post(tool_call_id="call_1", status="ok", duration_ms=1)  # must not raise

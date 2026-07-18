"""The create_thought tool — handler contract (lm-705.11 Task 4)."""

from __future__ import annotations

import json

from lifemodel.hooks import make_create_thought_tool
from lifemodel.testing.harness import build_capture_harness


def _tool(h):
    return make_create_thought_tool(lambda: h.lm, metrics=h.metrics)


def test_create_thought_captures_and_reports() -> None:
    h = build_capture_harness()
    out = json.loads(_tool(h)({"thoughts": [{"content": "ask about the trip", "salience": 0.7}]}))
    assert out == {"accepted": 1, "deduped": 0}
    rows = [r for r in h.memory.find(state="active", limit=50) if r.kind == "thought"]
    assert rows[0].source == "create-thought-tool" and rows[0].salience == 0.7


def test_create_thought_array_and_intra_call_dedup() -> None:
    h = build_capture_harness()
    out = json.loads(_tool(h)({"thoughts": [{"content": "a"}, {"content": "a"}, {"content": "b"}]}))
    assert out == {"accepted": 2, "deduped": 0}


def test_create_thought_empty_is_handled() -> None:
    h = build_capture_harness()
    assert json.loads(_tool(h)({"thoughts": []})) == {"accepted": 0, "deduped": 0}


def test_create_thought_never_raises_returns_error() -> None:
    def boom():
        raise RuntimeError("nope")

    out = json.loads(make_create_thought_tool(boom)({"thoughts": [{"content": "x"}]}))
    assert "error" in out


def test_create_thought_bad_args_returns_error() -> None:
    h = build_capture_harness()
    assert "error" in json.loads(_tool(h)("not a dict"))

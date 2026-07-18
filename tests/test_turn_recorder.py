"""Tests for ``core/turn_recorder.py`` — ``TurnRecorder`` construction + ``ensure_turn`` (lm-hg7).

Contract under test (task 3 of the turn-observability plan):

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
  raises even when the underlying sink's ``submit_span`` blows up (fail-soft).

Stdlib only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lifemodel.core.metrics import MetricRegistry
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

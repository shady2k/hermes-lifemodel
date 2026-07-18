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
* ``close_turn`` persists a ``turn.completion`` CHILD (``final_output`` sliced to
  a bounded length) then closes the root (``ended_at``=now, ``status`` mapped to
  the closed vocabulary) and drops the ledger entry — idempotent (a second close
  / an unknown key is a no-op) and fail-soft even when the sink blows up.

Codex diff-review wave A (lm-hg7) additions:

* ``tool_close``/``close_turn`` map a raw host ``status`` FAIL-CLOSED —
  ``"ok"``→``"ok"``, ``"blocked"``→``"suppressed"``, ``"error"``/anything else→
  ``"failed"`` (never ``"ok"`` for an unrecognized value) — and ``tool_close``
  preserves the raw value as a ``host_status`` attr;
* ``_reconcile_session_locked`` re-emits the FULL opening attr set
  (``session_id``/``model``/``platform`` alongside ``frame_kind``/``turn_id``/
  ``origin``), not just four of them — the store's wholesale attrs upsert was
  silently erasing the rest;
* ``injector_span``'s SETUP (ledger lookup, ``child_of``/``start_root``, the
  clock read) is wrapped separately from the body: a setup failure degrades to
  a no-op span (the injector's body still runs normally) rather than being
  mistaken for a body failure and suppressing the whole injection;
* ``ensure_turn`` submits the open root BEFORE the ledger entry becomes visible
  (and both under the same lock ``close_turn`` acquires), so a racing close can
  never have its write land before the open write; the ledger is evicted AFTER
  inserting the new entry, so its post-insert size is bounded to
  ``max_entries``, never ``max_entries + 1``;
* ``close_turn`` also takes ``model``/``platform`` (``post_llm_call`` always
  carries them, unlike ``pre_llm_call``) and fills them in at close if the
  entry's own stashed values are blank; the dead ``reasoning`` param (the host
  never had one to give it) is removed.

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


class _ChildOfBoomTracer(FakeTracer):
    """A tracer whose ``child_of`` blows up — simulates a tracing-layer hiccup
    happening in ``injector_span``'s SETUP (I4), distinct from a body exception."""

    def child_of(self, parent: Any) -> Any:
        raise RuntimeError("tracer exploded")


class _StartRootBoomTracer(FakeTracer):
    """A tracer whose ``start_root`` blows up — the no-open-turn setup path."""

    def start_root(self, *, upstream_traceparent: str | None = None) -> Any:
        raise RuntimeError("tracer exploded")


def test_injector_span_setup_failure_yields_noop_span_and_body_still_runs() -> None:
    # I4: injector_span's setup (ledger lookup + child_of/start_root + clock read)
    # used to run BEFORE the try/yield, so a broken tracer raised straight out of
    # `with recorder.injector_span(...)`, which the injector's own outer fail-soft
    # except would catch — silently discarding the injection body (it never even
    # ran). Now a setup failure degrades to a no-op span and the body still runs.
    rec = TurnRecorder(
        tracer=_ChildOfBoomTracer(),
        writer=CapturingSink(),
        metrics=MetricRegistry(),
        clock=FakeClock(_NOW),
    )
    rec.ensure_turn("s1", "t1")  # succeeds: start_root() is not the boomed method here
    body_ran = False
    with rec.injector_span("s1", "t1", "belief") as span:
        body_ran = True
        span.set(outcome="surfaced")  # the injector's normal work — no exception
    assert body_ran  # NOT suppressed by the broken tracer
    # setup never got a context to persist under, so no child span is written —
    # but crucially, nothing raised and the body still executed normally above.
    assert not [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"]


def test_injector_span_setup_failure_with_no_open_turn_still_yields_noop() -> None:
    rec = TurnRecorder(
        tracer=_StartRootBoomTracer(),
        writer=CapturingSink(),
        metrics=MetricRegistry(),
        clock=FakeClock(_NOW),
    )
    # no ensure_turn call — _ledger_ctx returns None, so injector_span's setup
    # falls to start_root(), which is the boomed method here.
    body_ran = False
    with rec.injector_span("s1", "t1", "belief") as span:
        body_ran = True
        span.set(outcome="empty")
    assert body_ran


def test_tool_open_close_persists_child_keyed_by_call_id() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_7")
    rec.tool_open("s1", "t1", tool="check_in", tool_call_id="call_8")  # concurrent, distinct id
    rec.tool_close("call_7", status="ok", action="discharge")
    child = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"][0]
    assert child["status"] == "ok" and child["attrs"]["action"] == "discharge"
    assert child["attrs"]["host_status"] == "ok"
    rec.tool_close("nope")  # unknown id — best-effort no-op, no raise


def test_tool_close_maps_host_error_to_failed_and_keeps_host_status() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_e")
    rec.tool_close("call_e", status="error")
    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["status"] == "failed"
    assert child["attrs"]["host_status"] == "error"


def test_tool_close_maps_host_blocked_to_suppressed_and_keeps_host_status() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_b")
    rec.tool_close("call_b", status="blocked")
    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["status"] == "suppressed"
    assert child["attrs"]["host_status"] == "blocked"


def test_tool_close_maps_unknown_host_status_to_failed_never_ok() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_u")
    rec.tool_close("call_u", status="whatever-this-is")
    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["status"] == "failed"
    assert child["attrs"]["host_status"] == "whatever-this-is"


def test_close_turn_writes_completion_and_closes_root() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="ok, talk soon")
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert "talk soon" in completion["attrs"]["final_output"]
    assert "reasoning" not in completion["attrs"]  # omitted when empty (this call passes none)
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root and closed_root[-1]["ended_at"] is not None
    assert ("s1", "t1") not in rec._ledger  # entry removed
    rec.close_turn("s1", "t1")  # second close — no raise, no duplicate root close
    closed_ok = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert len(closed_ok) == 1  # the second close persisted nothing more


def test_close_turn_preserves_the_open_root_attrs_not_just_ended_at() -> None:
    # submit_span UPSERTS attrs_json wholesale (no partial merge at the store
    # layer, state/trace_store.py's ON CONFLICT clause), so a close that
    # persisted only ITS OWN new attrs would silently erase what ensure_turn
    # already wrote — origin/model/platform are otherwise unrecoverable once
    # closed, and activity.py's timeline line reads origin back from exactly
    # this closed row.
    rec = _recorder()
    rec.ensure_turn("s1", "t1", model="opus", platform="telegram", origin="reactive")
    rec.close_turn("s1", "t1", final_output="ok, talk soon")
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root[-1]["attrs"]["origin"] == "reactive"
    assert closed_root[-1]["attrs"]["model"] == "opus"
    assert closed_root[-1]["attrs"]["platform"] == "telegram"


def test_close_turn_persists_reasoning_on_the_completion_child_when_present() -> None:
    # The being's own "why did I answer that" (lm-hg7): when the caller extracts a
    # non-empty reasoning from conversation_history, it rides the turn.completion span
    # (bounded) — the one place a turn's decision, not just its words, is answerable.
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn(
        "s1", "t1", final_output="Привет.", reasoning="they greeted me — keep it warm and short"
    )
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert completion["attrs"]["reasoning"] == "they greeted me — keep it warm and short"
    assert completion["attrs"]["final_output"] == "Привет."


def test_close_turn_fills_model_platform_supplied_only_at_close() -> None:
    # M5: pre_llm_call (where ensure_turn runs) does not reliably carry model/
    # platform, but post_llm_call (where close_turn runs) always does — so a
    # non-empty value passed to close_turn must win even though ensure_turn only
    # ever stashed "".
    rec = _recorder()
    rec.ensure_turn("s1", "t1")  # no model/platform known yet
    rec.close_turn("s1", "t1", final_output="hi", model="opus", platform="telegram")
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root[-1]["attrs"]["model"] == "opus"
    assert closed_root[-1]["attrs"]["platform"] == "telegram"


def test_close_turn_falls_back_to_the_entrys_stashed_model_platform_when_blank() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1", model="opus", platform="telegram")
    rec.close_turn("s1", "t1", final_output="hi")  # no model/platform at close
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root[-1]["attrs"]["model"] == "opus"
    assert closed_root[-1]["attrs"]["platform"] == "telegram"


def test_reconcile_abandoned_turn_preserves_origin_session_model_platform() -> None:
    # I2: _reconcile_session_locked used to re-emit only frame_kind/turn_id/
    # origin/outcome, so the store's wholesale attrs upsert erased session_id/
    # model/platform that ensure_turn had already written — an abandoned root
    # would read back with no session/model/platform at all.
    rec = _recorder()
    rec.ensure_turn("s1", "t1", model="opus", platform="telegram", origin="proactive")
    rec.ensure_turn("s1", "t2")  # t1 never closed — reconciled abandoned
    closed = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "failed"]
    assert closed[0]["attrs"]["origin"] == "proactive"
    assert closed[0]["attrs"]["session_id"] == "s1"
    assert closed[0]["attrs"]["model"] == "opus"
    assert closed[0]["attrs"]["platform"] == "telegram"
    assert closed[0]["attrs"]["outcome"] == "abandoned"


def test_close_turn_truncates_oversized_text_and_fails_closed_on_bad_status() -> None:
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="x" * 5000, status="bogus")
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert len(completion["attrs"]["final_output"]) == 4000
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["ended_at"]]
    # I1: an out-of-vocabulary status now fails CLOSED — never "ok".
    assert closed_root[-1]["status"] == "failed"
    # C-M1: the raw pre-map value survives as host_status even on the ROOT close
    # (tool_close already kept this; close_turn used to map-and-discard it).
    assert closed_root[-1]["attrs"]["host_status"] == "bogus"


def test_close_turn_root_carries_host_status_on_a_recognized_value_too() -> None:
    # C-M1: host_status is the RAW pre-map value on every close, not just the
    # out-of-vocabulary case above.
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="hi", status="blocked")
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["ended_at"]]
    assert closed_root[-1]["status"] == "suppressed"
    assert closed_root[-1]["attrs"]["host_status"] == "blocked"


def test_tool_close_real_host_status_wins_over_a_caller_supplied_attr() -> None:
    # C-M1: attrs assembly used to be {"host_status": status, **attrs}, letting a
    # caller-supplied attrs["host_status"] silently override the real mapped-from
    # value. It must now be the other way around: **attrs first, host_status=status
    # last, so the real raw status always wins.
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_x")
    rec.tool_close("call_x", status="ok", host_status="forged-by-caller")
    (child,) = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"]
    assert child["attrs"]["host_status"] == "ok"


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


def test_ensure_turn_submits_the_open_write_before_the_ledger_entry_is_visible() -> None:
    # M1: the open root's write must be enqueued BEFORE a racing close_turn could
    # ever see this key in the ledger — otherwise a close that runs concurrently
    # could have ITS write applied by the writer's single FIFO consumer before
    # this (delayed) open write, resetting ended_at/status back to open.
    rec = _recorder()
    visible_when_open_write_went_out: bool | None = None

    real_submit_span = rec._writer.submit_span

    def _spy_submit_span(**kw: Any) -> bool:
        nonlocal visible_when_open_write_went_out
        if kw.get("component") == "turn" and kw.get("ended_at") is None:
            visible_when_open_write_went_out = ("s1", "t1") in rec._ledger
        return real_submit_span(**kw)

    rec._writer.submit_span = _spy_submit_span  # type: ignore[method-assign]
    rec.ensure_turn("s1", "t1")
    assert visible_when_open_write_went_out is False  # not yet in the ledger at submit time


def test_ledger_post_insert_size_is_bounded_to_max_entries() -> None:
    # M4: _evict_locked used to run BEFORE the new entry was inserted, so
    # steady-state size was max_entries + 1. Distinct SESSIONS are required here:
    # ensure_turn reconciles (and drops) any OTHER open turn of the SAME session
    # before minting a new one, so same-session calls never accumulate.
    rec = TurnRecorder(
        tracer=FakeTracer(),
        writer=CapturingSink(),
        metrics=MetricRegistry(),
        clock=FakeClock(_NOW),
        max_entries=2,
    )
    rec.ensure_turn("s1", "t1")
    rec.ensure_turn("s2", "t1")
    rec.ensure_turn("s3", "t1")  # a third distinct session — must evict, not grow to 3
    assert len(rec._ledger) <= 2

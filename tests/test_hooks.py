"""Tests for :mod:`lifemodel.hooks` — the afferent frame boundary (spec §3/§4/§5).

The hooks no longer publish to a durable bus: each starts an ExecutionFrame that
folds the reading into the being's durable state and commits IMMEDIATELY (spec §3).
``pre_gateway_dispatch`` starts an ``EVENT`` frame carrying ``contact_observed``;
``post_llm_call`` starts an ``ASYNC_COMPLETION`` frame carrying ``proactive_outcome``.
Each observer takes a builder that yields a fresh ``LifeModel`` per call so the
frame loads state fresh under the one state-actor lock.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lifemodel.adapters.clock import SystemClock
from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.objects import DesireState
from lifemodel.hooks import _is_no_reply, make_inbound_observer, make_post_llm_observer
from lifemodel.ports.memory import MemoryPort
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock
from lifemodel.testing.fakes import FakeTracer

#: A being that has been BORN — the precondition of the contact drive. ``u`` models a
#: contact deficit inside an EXISTING relationship, so an UNBORN being's drive does not
#: accrue at all (``core/solitude_drive.py``: birth is not longing). Every scenario that
#: exercises the drive is therefore about a being that has someone to miss.
_BORN = "2026-07-01T10:00:00+00:00"

#: The fixed frame clock so seeded timestamps (last_tick_at, pending_since) stay
#: consistent — a real SystemClock would make ``dt`` huge and trip deadline-staleness.
_NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
_T0 = _NOW

#: A valid W3C traceparent standing in for a launch span's origin anchor — the
#: async bridge (§4.4) re-binds the outcome under THIS trace_id.
_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_ORIGIN_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"


def _ring_events(lm: Any, event: str) -> list[dict[str, Any]]:
    """Every ring record for *event* — the async outcome fans onto the origin
    trace's freshness ring (spec §4.4), self-stamped with its ``trace_id``."""
    return [rec for rec in lm.event_ring.read() if rec.get("event") == event]


def _seed_active_desire(store: MemoryPort) -> None:
    """Persist a live active contact-desire row."""
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))


def _lm_with_pending(tmp_path: Path, corr: str = "p-1", *, origin: str | None = _ORIGIN_TP) -> Any:
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm.state.commit(
        State(
            genesis_completed_at=_BORN,
            u=1.5,
            pending_proactive_id=corr,
            pending_proactive_since="2026-07-06T11:55:00+00:00",  # 5 min before now
            pending_proactive_origin_traceparent=origin,
            last_tick_at=_NOW.isoformat(),  # == now → dt=0, no spurious rise
        )
    )
    _seed_active_desire(lm.state)
    return lm


# --- _is_no_reply -----------------------------------------------------------


def test_is_no_reply_matches_all_markers_case_insensitively() -> None:
    for marker in (
        "NO_REPLY",
        "no_reply",
        "NO REPLY",
        "no reply",
        "[SILENT]",
        "[silent]",
        "SILENT",
        "silent",
    ):
        assert _is_no_reply(f"  {marker}  ") is True


def test_is_no_reply_rejects_prose_mentioning_a_marker() -> None:
    assert _is_no_reply("I considered NO_REPLY but decided to say hi!") is False
    assert _is_no_reply("") is False
    assert _is_no_reply("Hi! How are you?") is False


def test_decline_marker_is_a_member_of_substring_declines() -> None:
    from lifemodel.core.wake_packet import DECLINE_MARKER
    from lifemodel.hooks import _SUBSTRING_DECLINE_MARKERS

    assert DECLINE_MARKER in _SUBSTRING_DECLINE_MARKERS
    assert _is_no_reply(DECLINE_MARKER) is True


def test_is_no_reply_failclosed_on_bracketed_marker_wrapped_in_prose() -> None:
    assert _is_no_reply("Not going to nudge them right now. [SILENT]") is True
    assert _is_no_reply("I think I will hold back for now. [silent]") is True
    assert _is_no_reply("[SILENT] — staying quiet this time.") is True


def test_is_no_reply_does_not_substring_match_bare_words() -> None:
    assert _is_no_reply("we sat in silent comfort tonight") is False
    assert _is_no_reply("No reply is needed here, I just wanted to say hi.") is False
    assert _is_no_reply("It was a silent, easy kind of evening.") is False


def test_is_no_reply_bare_marker_whole_response_still_rejects() -> None:
    assert _is_no_reply("[SILENT]") is True
    assert _is_no_reply("SILENT") is True
    assert _is_no_reply("NO_REPLY") is True
    assert _is_no_reply("no reply") is True


# --- make_post_llm_observer — the ASYNC_COMPLETION frame (spec §3/§5) ---------


def test_post_llm_sent_resolves_pending_and_starts_action_pending(tmp_path: Path) -> None:
    # Scenario (5) sent + scenario (7) immediacy: a real text turn is a SENT outcome —
    # its own async-completion frame commits IMMEDIATELY (no heartbeat): the desire
    # resolves to satisfied, action_pending/backoff are set, u is NOT satiated.
    lm = _lm_with_pending(tmp_path, corr="p-1")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi, I miss you!",
    )
    final = lm.state.load()
    assert read_live_contact_desire(lm.state) is None  # desire resolved (satisfied)
    assert lm.state.get("desire", "contact:owner").state == "satisfied"
    assert final.pending_proactive_id is None  # turn resolved in its own frame
    assert final.action_pending_since is not None  # ActionPending inhibition started
    assert final.proactive_send_log  # backstop counter recorded the send


class _CapturingSink:
    """A trace sink that records every submitted span (for the wiring assertion)."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def submit_span(self, **kw: Any) -> bool:
        self.spans.append(kw)
        return True

    def submit_event(self, **kw: Any) -> bool:
        return True

    def submit_correlation(self, **kw: Any) -> bool:
        return True


def test_post_llm_extracts_turn_reasoning_onto_the_completion_span(tmp_path: Path) -> None:
    # lm-hg7: the observer pulls the being's reasoning off conversation_history
    # (msg["reasoning"], the way agent/turn_finalizer.py does) and hands it to
    # close_turn, so the turn.completion span answers "why did it reply that", not
    # just "what". This is the seam between _extract_turn_reasoning and close_turn.
    sink = _CapturingSink()
    rec = TurnRecorder(
        tracer=FakeTracer(),
        writer=sink,
        metrics=MetricRegistry(),
        clock=FakeClock(datetime(2026, 7, 18, tzinfo=UTC)),
    )
    rec.ensure_turn("s1", "t1")
    make_post_llm_observer(lambda: build_lifemodel(base_dir=tmp_path), recorder=rec)(
        session_id="s1",
        turn_id="t1",
        user_message="привет",
        assistant_response="Привет.",
        conversation_history=[
            {"role": "user", "content": "привет"},
            {
                "role": "assistant",
                "reasoning": "they greeted me; keep it warm",
                "content": "Привет.",
            },
        ],
    )
    completion = [s for s in sink.spans if s["component"] == "turn.completion"][0]
    assert completion["attrs"]["reasoning"] == "they greeted me; keep it warm"
    assert completion["attrs"]["final_output"] == "Привет."


def test_post_llm_silent_drops_desire_with_decline_backoff(tmp_path: Path) -> None:
    # Scenario (5) silent: a [SILENT] turn is a SILENT outcome → the desire is dropped
    # and a decline backoff is recorded; pending is cleaned; u is not touched.
    lm = _lm_with_pending(tmp_path, corr="p-2")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...", assistant_response="[SILENT]"
    )
    final = lm.state.load()
    assert lm.state.get("desire", "contact:owner").state == "dropped"
    assert final.decline_count >= 1  # decline backoff applied
    assert final.declined_at is not None
    assert final.pending_proactive_id is None


def test_post_llm_prose_wrapped_silent_marker_is_silent(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="Not going to nudge them right now. [SILENT]",
    )
    assert lm.state.get("desire", "contact:owner").state == "dropped"


def test_post_llm_message_mentioning_silent_word_is_sent(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="Hey — it's been a silent kind of evening, I miss you.",
    )
    assert lm.state.get("desire", "contact:owner").state == "satisfied"


def test_post_llm_ignores_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    make_post_llm_observer(lambda: lm)(
        user_message="just a normal user message", assistant_response="hi"
    )
    # not our impulse → no frame → the pending turn + desire are untouched
    assert lm.state.load().pending_proactive_id == "p-1"
    assert lm.state.get("desire", "contact:owner").state == "active"


def test_post_llm_ignores_when_desire_not_active(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    # pending turn but NO live desire row
    lm.state.commit(
        State(genesis_completed_at=_BORN, pending_proactive_id="p-1", last_tick_at=_NOW.isoformat())
    )
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} x", assistant_response="hi"
    )
    assert lm.state.load().pending_proactive_id == "p-1"  # untouched — no active desire


# --- §4.4: the resolved outcome, WOVEN UNDER THE LAUNCH'S ORIGIN TRACE -------


def test_post_llm_weaves_delivered_outcome_under_origin_trace(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-delivered")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi, I miss you!",
    )
    outcomes = _ring_events(lm, "proactive_outcome")
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "delivered"
    assert outcomes[0]["correlation_id"] == "p-delivered"
    assert outcomes[0]["trace_id"] == _ORIGIN_TRACE_ID  # under the launch's trace


def test_post_llm_weaves_silent_outcome_and_act_gate_silent_suppression(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-silent")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="[SILENT]",
    )
    outcomes = _ring_events(lm, "proactive_outcome")
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "silent"
    assert outcomes[0]["trace_id"] == _ORIGIN_TRACE_ID
    suppressions = [
        rec for rec in _ring_events(lm, "suppression") if rec.get("reason") == "act_gate_silent"
    ]
    assert len(suppressions) == 1
    assert suppressions[0]["trace_id"] == _ORIGIN_TRACE_ID  # suppression under origin too


def test_post_llm_does_not_emit_outcome_for_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    make_post_llm_observer(lambda: lm)(
        user_message="just a normal user message", assistant_response="hi"
    )
    assert _ring_events(lm, "proactive_outcome") == []


def test_post_llm_orphan_async_outcome_when_origin_anchor_missing(tmp_path: Path) -> None:
    # Miss policy (§4.4, load-bearing): a pending turn whose origin anchor is GONE emits
    # an explicit ``orphan_async_outcome`` on its OWN trace — NEVER attaching the outcome
    # to a fresh/foreign trace — while the resolution still commits.
    lm = _lm_with_pending(tmp_path, corr="p-orphan", origin=None)
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="[SILENT]",
    )
    orphans = _ring_events(lm, "orphan_async_outcome")
    assert len(orphans) == 1
    assert orphans[0]["correlation_id"] == "p-orphan"
    # The outcome is NOT attached to any (foreign) trace — no proactive_outcome at all.
    assert _ring_events(lm, "proactive_outcome") == []
    # Control flow is intact: the outcome still resolved the desire in its frame.
    assert lm.state.get("desire", "contact:owner").state == "dropped"


# --- proactive_outcome_detail — durable under the origin trace (bead lm-otq) --


def test_post_llm_emits_outcome_detail_under_origin_with_extra_fields(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-detail")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi, I miss you!",
        reasoning="because X",
        model="claude",
    )
    events = _ring_events(lm, "proactive_outcome_detail")
    assert len(events) == 1
    fields = events[0]
    assert fields["correlation_id"] == "p-detail"
    assert fields["outcome"] == "sent"
    assert fields["assistant_response"] == "Hey, hi, I miss you!"
    assert fields["extra_fields"]["reasoning"] == "because X"
    assert fields["extra_fields"]["model"] == "claude"
    assert fields["trace_id"] == _ORIGIN_TRACE_ID  # under the launch's trace


def test_post_llm_does_not_emit_outcome_detail_for_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated")
    make_post_llm_observer(lambda: lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        reasoning="because X",
    )
    assert _ring_events(lm, "proactive_outcome_detail") == []
    assert _ring_events(lm, "proactive_outcome") == []


# --- proactive_reasoning — durable under the origin trace (bead lm-otq step 2) --


def test_post_llm_emits_proactive_reasoning_with_untruncated_reasoning(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-reason")
    long_reasoning = "Sasha asks about a markdown editor... " + ("x" * 2000)
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search_session_history"}}],
            "finish_reason": "tool_calls",
            "reasoning": "first pass reasoning",
        },
        {
            "role": "assistant",
            "content": "Hey, hi!",
            "finish_reason": "stop",
            "reasoning": long_reasoning,
        },
    ]
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi!",
        conversation_history=history,
    )
    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    fields = events[0]
    assert fields["correlation_id"] == "p-reason"
    assert fields["message_count"] == 3
    assert fields["trace_id"] == _ORIGIN_TRACE_ID
    messages = fields["messages"]
    assert len(messages) == 3
    last = messages[-1]
    assert last["role"] == "assistant"
    assert last["finish_reason"] == "stop"
    assert last["reasoning"] == long_reasoning
    assert len(last["reasoning"]) > 800
    tool_msg = messages[1]
    assert tool_msg["has_tool_calls"] is True
    assert "search_session_history" in tool_msg["tool_call_names"]
    assert tool_msg["reasoning"] == "first pass reasoning"


def test_post_llm_proactive_reasoning_unavailable_when_history_missing(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-nohistory")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi!",
    )
    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    assert events[0]["available"] is False


def test_post_llm_proactive_reasoning_unavailable_when_history_not_a_list(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-badtype")
    make_post_llm_observer(lambda: lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} a pull inside...",
        assistant_response="Hey, hi!",
        conversation_history="not-a-list",
    )
    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    assert events[0]["available"] is False


def test_post_llm_does_not_emit_proactive_reasoning_for_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated2")
    make_post_llm_observer(lambda: lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        conversation_history=[{"role": "assistant", "reasoning": "irrelevant"}],
    )
    assert _ring_events(lm, "proactive_reasoning") == []


# --- make_inbound_observer — the EVENT frame (spec §3/§4) --------------------


def test_inbound_contact_satiates_drive_and_stamps_last_exchange(tmp_path: Path) -> None:
    # Scenario (1): a real inbound contact_observed satiates u, sets last_exchange_at,
    # and resolves any pending desire → SATISFIED — committed in its own EVENT frame.
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm.state.commit(State(genesis_completed_at=_BORN, u=2.0, last_tick_at=_NOW.isoformat()))
    _seed_active_desire(lm.state)
    event = SimpleNamespace(text="hi!", internal=False, id="m-42")
    make_inbound_observer(lambda: lm)(event=event)
    final = lm.state.load()
    assert final.u < 2.0  # drive satiated by the genuine two_way contact
    assert final.last_exchange_at is not None
    assert lm.state.get("desire", "contact:owner").state == "satisfied"


def test_inbound_ignores_internal_and_own_impulse(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm.state.commit(State(genesis_completed_at=_BORN, u=2.0, last_tick_at=_NOW.isoformat()))
    make_inbound_observer(lambda: lm)(event=SimpleNamespace(text="x", internal=True, id="a"))
    make_inbound_observer(lambda: lm)(
        event=SimpleNamespace(text=f"{IMPULSE_LABEL_PREFIX} own", internal=False, id="b")
    )
    # neither started a frame → u untouched, no exchange recorded
    final = lm.state.load()
    assert final.u == 2.0
    assert final.last_exchange_at is None


@pytest.mark.parametrize("text", ["/new", "/model", "/commands", "/lifemodel force-wake"])
def test_inbound_control_command_is_not_contact_sensor_bandpass(tmp_path: Path, text: str) -> None:
    # Scenario (2): a slash/control command is filtered at the sensor band-pass (spec §4)
    # — it must NOT count as contact, so no frame runs and u is untouched.
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm.state.commit(State(genesis_completed_at=_BORN, u=2.0, last_tick_at=_NOW.isoformat()))
    make_inbound_observer(lambda: lm)(event=SimpleNamespace(text=text, internal=False, id="m-4"))
    final = lm.state.load()
    assert final.u == 2.0  # unchanged — the command was not contact
    assert final.last_exchange_at is None


def test_inbound_normal_chat_message_is_contact(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm.state.commit(State(genesis_completed_at=_BORN, u=2.0, last_tick_at=_NOW.isoformat()))
    make_inbound_observer(lambda: lm)(event=SimpleNamespace(text="hi", internal=False, id="m-3"))
    assert lm.state.load().last_exchange_at is not None  # a genuine exchange was recorded


# --- register(ctx) wiring smoke test ------------------------------------------


class _FakeCtx:
    profile_name = "test-being"

    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}
        self.hooks: list[tuple[str, Any]] = []
        self.tools: dict[str, Any] = {}
        self.platforms: dict[str, Any] = {}

    def register_command(
        self, name: str, handler: Any, description: str = "", args_hint: str = ""
    ) -> None:
        self.commands[name] = handler

    def register_hook(self, hook_name: str, callback: Any) -> None:
        self.hooks.append((hook_name, callback))

    def register_tool(self, name: str, **kwargs: Any) -> None:
        self.tools[name] = kwargs

    def register_platform(self, name: str, **kwargs: Any) -> None:
        self.platforms[name] = kwargs


@pytest.fixture(autouse=True)
def _stub_gateway_for_register() -> None:
    """register() wires the being platform as a REQUIRED step (spec §4.3) whose
    ``being_platform`` import needs ``gateway.*`` — stub it so register() completes
    off-host (in prod, where gateway is present, that step is genuinely required)."""
    from gateway_stubs import install_gateway_stubs

    install_gateway_stubs()


def test_register_wires_post_llm_call_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import lifemodel

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # completes with gateway stubbed (autouse fixture)

    matches = [cb for name, cb in ctx.hooks if name == "post_llm_call"]
    assert len(matches) == 1

    # The registered callback starts a frame that RESOLVES the pending desire.
    sdir = tmp_path / "workspace" / "lifemodel"
    store = SQLiteRuntimeStore(sdir, clock=SystemClock())
    store.commit(
        State(genesis_completed_at=_BORN, pending_proactive_id="p1", last_tick_at=_T0.isoformat())
    )
    _seed_active_desire(store)  # a live desire so the outcome gate passes

    matches[0](user_message=f"{IMPULSE_LABEL_PREFIX} impulse text", assistant_response="NO_REPLY")

    # The desire was resolved (dropped, silent) — the hook committed via its frame.
    desire = read_live_contact_desire(store)
    assert desire is None  # resolved
    assert store.get("desire", "contact:owner").state == "dropped"


def test_register_wires_pre_gateway_dispatch_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lifemodel

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # completes with gateway stubbed (autouse fixture)

    matches = [cb for name, cb in ctx.hooks if name == "pre_gateway_dispatch"]
    assert len(matches) == 1

    # Genuinely wired: a real inbound message starts a frame that satiates the drive.
    # The live path uses the real SystemClock, so anchor last_tick_at at ~now (dt≈0)
    # to keep the drive's elapsed-silence rise negligible against the satiation.
    sdir = tmp_path / "workspace" / "lifemodel"
    store = SQLiteRuntimeStore(sdir, clock=SystemClock())
    store.commit(
        State(genesis_completed_at=_BORN, u=50.0, last_tick_at=datetime.now(UTC).isoformat())
    )

    matches[0](event=SimpleNamespace(text="hey there", internal=False, id="m-1"))

    persisted = store.load()
    assert persisted.last_exchange_at is not None  # the exchange was recorded
    assert persisted.u < 50.0  # the drive was satiated by the contact

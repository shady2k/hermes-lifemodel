"""Tests for :mod:`lifemodel.hooks` — verdict/exchange signal publishing (spec §7.1).

Phase E3: the hooks no longer mutate ``State`` directly — they **publish signals**
(verdict / exchange) to ``lm.bus``. Producers only enqueue (spec §7.1); the
aggregation layer (inside ``coreloop.tick()``) consumes them on the next tick.
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
from lifemodel.core.taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    read_verdict,
    read_verdict_correlation,
)
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.domain.egress import Verdict
from lifemodel.domain.objects import DesireState
from lifemodel.hooks import _is_no_reply, make_inbound_observer, make_post_llm_observer
from lifemodel.ports.memory import MemoryPort
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore

_T0 = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)

#: A valid W3C traceparent standing in for a launch span's origin anchor — the
#: async bridge (§4.4) re-binds the outcome under THIS trace_id.
_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_ORIGIN_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"


def _ring_events(lm: Any, event: str) -> list[dict[str, Any]]:
    """Every ring record for *event* — the async outcome now fans onto the origin
    trace's freshness ring (spec §4.4), self-stamped with its ``trace_id``."""
    return [rec for rec in lm.event_ring.read() if rec.get("event") == event]


def _seed_active_desire(store: MemoryPort) -> None:
    """Persist a live active contact-desire row (the old desire_status="active")."""
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))


def _lm_with_pending(tmp_path: Path, corr: str = "p-1", *, origin: str | None = _ORIGIN_TP) -> Any:
    lm = build_lifemodel(base_dir=tmp_path)
    lm.state.commit(
        State(
            pending_proactive_id=corr,
            pending_proactive_since="2026-07-06T00:00:00+00:00",
            pending_proactive_origin_traceparent=origin,
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
    assert _is_no_reply("Привет! Как ты?") is False


# --- make_post_llm_observer — signal publishing (spec §7.1) ------------------


def test_post_llm_publishes_fulfill_verdict_signal(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-1")
    obs = make_post_llm_observer(lm)
    obs(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
    )
    signals = lm.bus.peek_unprocessed()
    verdicts = [s for s in signals if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.FULFILL
    assert read_verdict_correlation(verdicts[0]) == "p-1"


def test_post_llm_publishes_reject_on_silent(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    obs = make_post_llm_observer(lm)
    obs(user_message=f"{IMPULSE_LABEL_PREFIX} ...", assistant_response="[SILENT]")
    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert read_verdict(verdicts[0]) is Verdict.REJECT


def test_post_llm_ignores_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path)
    obs = make_post_llm_observer(lm)
    obs(user_message="just a normal user message", assistant_response="hi")  # not our impulse
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT] == []


def test_post_llm_ignores_when_desire_not_active(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    # pending turn but NO live desire row (the old desire_status="none")
    lm.state.commit(State(pending_proactive_id="p-1"))
    make_post_llm_observer(lm)(user_message=f"{IMPULSE_LABEL_PREFIX} x", assistant_response="hi")
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT] == []


# --- §4.4: the resolved outcome, WOVEN UNDER THE LAUNCH'S ORIGIN TRACE -------


def test_post_llm_weaves_delivered_outcome_under_origin_trace(tmp_path: Path) -> None:
    # FULFILL (real text): the outcome lands on the origin trace's ring, self-stamped
    # with the ORIGIN trace_id (§4.4) — one attempt, one trace_id, not a fresh root.
    lm = _lm_with_pending(tmp_path, corr="p-delivered")
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
    )
    outcomes = _ring_events(lm, "proactive_outcome")
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "delivered"
    assert outcomes[0]["correlation_id"] == "p-delivered"
    assert outcomes[0]["trace_id"] == _ORIGIN_TRACE_ID  # under the launch's trace


def test_post_llm_weaves_silent_outcome_and_act_gate_silent_suppression(tmp_path: Path) -> None:
    # REJECT ([SILENT]): the outcome is "silent" AND the 4th suppression reason
    # (ACT_GATE_SILENT) is emitted UNDER THE ORIGIN TRACE (phase-2 carryover, §4.4).
    lm = _lm_with_pending(tmp_path, corr="p-silent")
    make_post_llm_observer(lm)(
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
    make_post_llm_observer(lm)(user_message="just a normal user message", assistant_response="hi")
    assert _ring_events(lm, "proactive_outcome") == []


def test_post_llm_orphan_async_outcome_when_origin_anchor_missing(tmp_path: Path) -> None:
    # Miss policy (§4.4, load-bearing): a pending turn whose origin anchor is GONE
    # emits an explicit ``orphan_async_outcome`` on its OWN trace — NEVER attaching the
    # outcome to a fresh/foreign trace — while the verdict signal still publishes.
    lm = _lm_with_pending(tmp_path, corr="p-orphan", origin=None)
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="[SILENT]",
    )
    orphans = _ring_events(lm, "orphan_async_outcome")
    assert len(orphans) == 1
    assert orphans[0]["correlation_id"] == "p-orphan"
    # The outcome is NOT attached to any (foreign) trace — no proactive_outcome at all.
    assert _ring_events(lm, "proactive_outcome") == []
    # Control flow is intact: the verdict still resolves the desire next tick.
    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.REJECT


# --- proactive_verdict_detail — durable under the origin trace (bead lm-otq) --


def test_post_llm_emits_verdict_detail_under_origin_with_extra_fields(tmp_path: Path) -> None:
    # The DEBUG discovery detail is now DURABLE in the trace store under the origin
    # trace regardless of log level (§4.3, 5th-source collapse) — no ambient gating.
    lm = _lm_with_pending(tmp_path, corr="p-detail")

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        reasoning="because X",
        model="claude",
    )

    events = _ring_events(lm, "proactive_verdict_detail")
    assert len(events) == 1
    fields = events[0]
    assert fields["correlation_id"] == "p-detail"
    assert fields["verdict"] == "fulfill"
    assert fields["assistant_response"] == "Саш, привет, скучаю!"
    assert fields["extra_fields"]["reasoning"] == "because X"
    assert fields["extra_fields"]["model"] == "claude"
    assert fields["trace_id"] == _ORIGIN_TRACE_ID  # under the launch's trace


def test_post_llm_does_not_emit_verdict_detail_for_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated")

    make_post_llm_observer(lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        reasoning="because X",
    )

    assert _ring_events(lm, "proactive_verdict_detail") == []
    assert _ring_events(lm, "proactive_outcome") == []


def test_post_llm_verdict_detail_does_not_change_signal_or_outcome(tmp_path: Path) -> None:
    # Regression: the discovery detail must not disturb the verdict signal publish
    # or the proactive_outcome record.
    lm = _lm_with_pending(tmp_path, corr="p-regress")

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        reasoning="because X",
    )

    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.FULFILL
    assert read_verdict_correlation(verdicts[0]) == "p-regress"

    outcomes = _ring_events(lm, "proactive_outcome")
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "delivered"
    assert outcomes[0]["correlation_id"] == "p-regress"


# --- proactive_reasoning — durable under the origin trace (bead lm-otq step 2) --


def test_post_llm_emits_proactive_reasoning_with_untruncated_reasoning(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-reason")

    # Longer than the old 800-char truncation cap, but within the new 4000-char
    # generous cap — proves this event is untruncated relative to the old one.
    long_reasoning = "Саша спрашивает про markdown редактор... " + ("x" * 2000)
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
            "content": "Саш, привет!",
            "finish_reason": "stop",
            "reasoning": long_reasoning,
        },
    ]

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
        conversation_history=history,
    )

    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    fields = events[0]
    assert fields["correlation_id"] == "p-reason"
    assert fields["message_count"] == 3
    assert fields["trace_id"] == _ORIGIN_TRACE_ID  # under the launch's trace
    messages = fields["messages"]
    assert len(messages) == 3

    last = messages[-1]
    assert last["role"] == "assistant"
    assert last["finish_reason"] == "stop"
    # untruncated (well beyond any short-preview cap like 800 chars)
    assert last["reasoning"] == long_reasoning
    assert len(last["reasoning"]) > 800

    tool_msg = messages[1]
    assert tool_msg["has_tool_calls"] is True
    assert "search_session_history" in tool_msg["tool_call_names"]
    assert tool_msg["reasoning"] == "first pass reasoning"


def test_post_llm_proactive_reasoning_unavailable_when_history_missing(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-nohistory")

    # No conversation_history kwarg at all — must not raise.
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
    )

    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    assert events[0]["available"] is False


def test_post_llm_proactive_reasoning_unavailable_when_history_not_a_list(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-badtype")

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
        conversation_history="not-a-list",
    )

    events = _ring_events(lm, "proactive_reasoning")
    assert len(events) == 1
    assert events[0]["available"] is False


def test_post_llm_does_not_emit_proactive_reasoning_for_uncorrelated_turn(tmp_path: Path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated2")

    make_post_llm_observer(lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        conversation_history=[{"role": "assistant", "reasoning": "irrelevant"}],
    )

    assert _ring_events(lm, "proactive_reasoning") == []


# --- make_inbound_observer — signal publishing (spec §7.1) ------------------


def test_inbound_publishes_exchange_signal(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text="привет!", internal=False, id="m-42")
    make_inbound_observer(lm)(event=event)
    exchanges = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE]
    assert len(exchanges) == 1


def test_inbound_ignores_internal_and_own_impulse(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    make_inbound_observer(lm)(event=SimpleNamespace(text="x", internal=True, id="a"))
    make_inbound_observer(lm)(
        event=SimpleNamespace(text=f"{IMPULSE_LABEL_PREFIX} own", internal=False, id="b")
    )
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE] == []


def test_inbound_ignores_own_lifemodel_force_wake_command(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text="/lifemodel force-wake", internal=False, id="m-1")
    make_inbound_observer(lm)(event=event)
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE] == []


def test_inbound_ignores_own_lifemodel_debug_command(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text="/lifemodel debug", internal=False, id="m-2")
    make_inbound_observer(lm)(event=event)
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE] == []


def test_inbound_still_publishes_exchange_for_normal_chat_message(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text="привет", internal=False, id="m-3")
    make_inbound_observer(lm)(event=event)
    exchanges = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE]
    assert len(exchanges) == 1


@pytest.mark.parametrize(
    "text",
    ["/new", "/model", "/commands", "/lifemodel debug"],
)
def test_inbound_ignores_any_slash_command(tmp_path: Path, text: str) -> None:
    # Owner's decision (lm-ia3): operating the tool via a slash/control command
    # is not conversing with the being — no slash-prefixed message, regardless
    # of which command, counts as a genuine two-way exchange.
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text=text, internal=False, id="m-4")
    make_inbound_observer(lm)(event=event)
    exchanges = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE]
    assert exchanges == []


# --- register(ctx) wiring smoke test ------------------------------------------


class _FakeCtx:
    profile_name = "test-being"

    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}
        self.hooks: list[tuple[str, Any]] = []

    def register_command(
        self, name: str, handler: Any, description: str = "", args_hint: str = ""
    ) -> None:
        self.commands[name] = handler

    def register_hook(self, hook_name: str, callback: Any) -> None:
        self.hooks.append((hook_name, callback))


def test_register_wires_post_llm_call_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import lifemodel

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # must not raise even without a real Hermes host

    matches = [cb for name, cb in ctx.hooks if name == "post_llm_call"]
    assert len(matches) == 1

    # The registered callback publishes a verdict signal (not state mutation).
    sdir = tmp_path / "workspace" / "lifemodel"
    store = SQLiteRuntimeStore(sdir, clock=SystemClock())
    pending_state = State(pending_proactive_id="p1", last_tick_at=_T0.isoformat())
    store.commit(pending_state)
    _seed_active_desire(store)  # a live desire so the verdict gate passes

    matches[0](
        user_message=f"{IMPULSE_LABEL_PREFIX} impulse text",
        assistant_response="NO_REPLY",
    )

    # Neither State nor the desire row is mutated — the hook only publishes a signal.
    desire = read_live_contact_desire(store)
    assert desire is not None and desire.state == "active"  # unchanged — published, not applied


def test_register_wires_pre_gateway_dispatch_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lifemodel

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # must not raise even without a real Hermes host

    matches = [cb for name, cb in ctx.hooks if name == "pre_gateway_dispatch"]
    assert len(matches) == 1

    # Genuinely wired: a real inbound message publishes an exchange signal.
    sdir = tmp_path / "workspace" / "lifemodel"
    store = SQLiteRuntimeStore(sdir, clock=SystemClock())
    store.commit(State(u=50.0, last_tick_at=_T0.isoformat()))

    matches[0](event=SimpleNamespace(text="hey there", internal=False, id="m-1"))

    # State is NOT mutated — the hook only publishes an exchange signal.
    persisted = store.load()
    assert persisted.last_exchange_at is None  # unchanged
    assert persisted.u == 50.0  # unchanged

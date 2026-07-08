"""Tests for :mod:`lifemodel.hooks` — verdict/exchange signal publishing (spec §7.1).

Phase E3: the hooks no longer mutate ``State`` directly — they **publish signals**
(verdict / exchange) to ``lm.bus``. Producers only enqueue (spec §7.1); the
aggregation layer (inside ``coreloop.tick()``) consumes them on the next tick.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lifemodel.log as lm_logging
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
from lifemodel.domain.objects import DesireState
from lifemodel.events import EventSink
from lifemodel.hooks import _is_no_reply, make_inbound_observer, make_post_llm_observer
from lifemodel.log import EventTee
from lifemodel.ports.memory import MemoryPort
from lifemodel.sim.aggregation import Verdict
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore

_T0 = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)


class _RecordingLogger:
    """A minimal :class:`~lifemodel.log.EventLogger` that records calls per level."""

    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict[str, Any]]] = []
        self.info_calls: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **fields: Any) -> None:
        self.debug_calls.append((event, dict(fields)))

    def info(self, event: str, **fields: Any) -> None:
        self.info_calls.append((event, dict(fields)))

    def warning(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass

    def error(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass

    def critical(self, event: str, **fields: Any) -> None:  # pragma: no cover - unused here
        pass


def _seed_active_desire(store: MemoryPort) -> None:
    """Persist a live active contact-desire row (the old desire_status="active")."""
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))


def _lm_with_pending(tmp_path: Path, corr: str = "p-1", *, logger: Any = None) -> Any:
    lm = build_lifemodel(base_dir=tmp_path, logger=logger)
    lm.state.commit(
        State(
            pending_proactive_id=corr,
            pending_proactive_since="2026-07-06T00:00:00+00:00",
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


# --- B3 (lm-j2w): the resolved outcome, logged at INFO (always visible) -----


def test_post_llm_logs_delivered_outcome_at_info_on_real_text(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    lm = _lm_with_pending(tmp_path, corr="p-delivered", logger=logger)
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
    )
    outcomes = [c for c in logger.info_calls if c[0] == "proactive_outcome"]
    assert len(outcomes) == 1
    _, fields = outcomes[0]
    assert fields["outcome"] == "delivered"
    assert fields["correlation_id"] == "p-delivered"


def test_post_llm_logs_silent_outcome_at_info_on_no_reply(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    lm = _lm_with_pending(tmp_path, corr="p-silent", logger=logger)
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} ...",
        assistant_response="[SILENT]",
    )
    outcomes = [c for c in logger.info_calls if c[0] == "proactive_outcome"]
    assert len(outcomes) == 1
    _, fields = outcomes[0]
    assert fields["outcome"] == "silent"
    assert fields["correlation_id"] == "p-silent"


def test_post_llm_does_not_log_outcome_for_uncorrelated_turn(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    lm = _lm_with_pending(tmp_path, logger=logger)
    make_post_llm_observer(lm)(user_message="just a normal user message", assistant_response="hi")
    assert [c for c in logger.info_calls if c[0] == "proactive_outcome"] == []


# --- proactive_verdict_detail — DEBUG discovery log (bead lm-otq) -----------


def test_post_llm_emits_verdict_detail_at_debug_with_extra_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-detail", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        reasoning="because X",
        model="claude",
    )

    events = [c for c in base.debug_calls if c[0] == "proactive_verdict_detail"]
    assert len(events) == 1
    _, fields = events[0]
    assert fields["correlation_id"] == "p-detail"
    assert fields["verdict"] == "fulfill"
    assert fields["assistant_response"] == "Саш, привет, скучаю!"
    assert fields["extra_fields"]["reasoning"] == "because X"
    assert fields["extra_fields"]["model"] == "claude"


def test_post_llm_does_not_emit_verdict_detail_when_effective_level_is_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "info-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-info", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        reasoning="because X",
    )

    assert base.debug_calls == []  # gated — DEBUG never reaches the base logger/sink


def test_post_llm_does_not_emit_verdict_detail_for_uncorrelated_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated", logger=logger)

    make_post_llm_observer(lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        reasoning="because X",
    )

    assert base.debug_calls == []
    assert [c for c in base.info_calls if c[0] == "proactive_outcome"] == []


def test_post_llm_verdict_detail_does_not_change_signal_or_outcome_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: adding the DEBUG discovery log must not disturb the existing
    # verdict signal publish or the INFO proactive_outcome log.
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-regress", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        reasoning="because X",
    )

    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.FULFILL
    assert read_verdict_correlation(verdicts[0]) == "p-regress"

    outcomes = [c for c in base.info_calls if c[0] == "proactive_outcome"]
    assert len(outcomes) == 1
    _, fields = outcomes[0]
    assert fields["outcome"] == "delivered"
    assert fields["correlation_id"] == "p-regress"


# --- proactive_reasoning — DEBUG discovery log (bead lm-otq step 2) --------


def test_post_llm_emits_proactive_reasoning_with_untruncated_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-reason", logger=logger)

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

    events = [c for c in base.debug_calls if c[0] == "proactive_reasoning"]
    assert len(events) == 1
    _, fields = events[0]
    assert fields["correlation_id"] == "p-reason"
    assert fields["message_count"] == 3
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


def test_post_llm_proactive_reasoning_unavailable_when_history_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-nohistory", logger=logger)

    # No conversation_history kwarg at all — must not raise.
    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
    )

    events = [c for c in base.debug_calls if c[0] == "proactive_reasoning"]
    assert len(events) == 1
    _, fields = events[0]
    assert fields["available"] is False


def test_post_llm_proactive_reasoning_unavailable_when_history_not_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-badtype", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
        conversation_history="not-a-list",
    )

    events = [c for c in base.debug_calls if c[0] == "proactive_reasoning"]
    assert len(events) == 1
    _, fields = events[0]
    assert fields["available"] is False


def test_post_llm_does_not_emit_proactive_reasoning_when_effective_level_is_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "info-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-info2", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет!",
        conversation_history=[{"role": "assistant", "reasoning": "should not log"}],
    )

    assert base.debug_calls == []  # gated — DEBUG never reaches the base logger/sink


def test_post_llm_does_not_emit_proactive_reasoning_for_uncorrelated_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-uncorrelated2", logger=logger)

    make_post_llm_observer(lm)(
        user_message="just a normal user message",  # not our impulse -> gate short-circuits
        assistant_response="hi",
        conversation_history=[{"role": "assistant", "reasoning": "irrelevant"}],
    )

    assert [c for c in base.debug_calls if c[0] == "proactive_reasoning"] == []


def test_post_llm_proactive_reasoning_does_not_change_signal_or_outcome_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: adding proactive_reasoning must not disturb the existing
    # verdict signal publish or the INFO proactive_outcome log.
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)
    base = _RecordingLogger()
    logger = EventTee(base, EventSink(tmp_path / "debug-events.jsonl"))
    lm = _lm_with_pending(tmp_path, corr="p-regress2", logger=logger)

    make_post_llm_observer(lm)(
        user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...",
        assistant_response="Саш, привет, скучаю!",
        conversation_history=[{"role": "assistant", "reasoning": "because X"}],
    )

    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.FULFILL
    assert read_verdict_correlation(verdicts[0]) == "p-regress2"

    outcomes = [c for c in base.info_calls if c[0] == "proactive_outcome"]
    assert len(outcomes) == 1
    _, fields = outcomes[0]
    assert fields["outcome"] == "delivered"
    assert fields["correlation_id"] == "p-regress2"


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

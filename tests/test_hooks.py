"""Tests for :mod:`lifemodel.hooks` — verdict/exchange signal publishing (spec §7.1).

Phase E3: the hooks no longer mutate ``State`` directly — they **publish signals**
(verdict / exchange) to ``lm.bus``. Producers only enqueue (spec §7.1); the
aggregation layer (inside ``coreloop.tick()``) consumes them on the next tick.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    read_verdict,
    read_verdict_correlation,
)
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.hooks import _is_no_reply, make_inbound_observer, make_post_llm_observer
from lifemodel.sim.aggregation import Verdict
from lifemodel.state.model import State

_T0 = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)


def _lm_with_pending(tmp_path: Path, corr: str = "p-1") -> Any:
    lm = build_lifemodel(base_dir=tmp_path)
    lm.state.commit(
        State(
            desire_status="active",
            pending_proactive_id=corr,
            pending_proactive_since="2026-07-06T00:00:00+00:00",
        )
    )
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
    lm.state.commit(State(desire_status="none", pending_proactive_id="p-1"))
    make_post_llm_observer(lm)(user_message=f"{IMPULSE_LABEL_PREFIX} x", assistant_response="hi")
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT] == []


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
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    pending_state = State(
        desire_status="active", pending_proactive_id="p1", last_tick_at=_T0.isoformat()
    )
    state_file.write_text(json.dumps(pending_state.to_dict()))

    matches[0](
        user_message=f"{IMPULSE_LABEL_PREFIX} impulse text",
        assistant_response="NO_REPLY",
    )

    # State is NOT mutated — the hook only publishes a signal.
    persisted = json.loads(state_file.read_text())
    assert persisted["desire_status"] == "active"  # unchanged — signal published, not applied


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
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(State(u=50.0, last_tick_at=_T0.isoformat()).to_dict()))

    matches[0](event=SimpleNamespace(text="hey there", internal=False, id="m-1"))

    # State is NOT mutated — the hook only publishes an exchange signal.
    persisted = json.loads(state_file.read_text())
    assert persisted["last_exchange_at"] is None  # unchanged
    assert persisted["u"] == 50.0  # unchanged

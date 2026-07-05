"""Tests for :mod:`lifemodel.hooks` — verdict feedback via ``post_llm_call``.

The real ``post_llm_call`` payload shape was verified by reading the Hermes
host (``~/.hermes/hermes-agent``, hermes-agent 0.17.0) — see
``lifemodel.hooks``'s module docstring for the full SPIKE writeup:
``PluginManager.invoke_hook`` calls every registered callback as
``cb(**kwargs)`` with ``session_id``, ``task_id``, ``turn_id``,
``user_message``, ``assistant_response``, ``conversation_history``, ``model``,
``platform``. ``_fake_payload`` below reproduces exactly that kwargs shape;
tests unpack it with ``**``.

Correlation caveat (also documented in ``lifemodel.hooks``): Hermes does not
thread a plugin-supplied id back through the hook, so the only signal
available to tell "this finished turn is the pending proactive one" is
whether ``user_message`` is our own composed impulse (the
``IMPULSE_LABEL_PREFIX`` marker) *and* a proactive turn is actually pending in
state. ``_fake_payload(proactive=...)`` encodes that distinction directly
rather than an invented "pending_id" payload field.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.aggregator import SilentAggregator
from lifemodel.hooks import _is_no_reply, make_inbound_observer, make_post_llm_observer
from lifemodel.impulse import IMPULSE_LABEL_PREFIX
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore

_T0 = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)


def make_lm_pending(pending_id: str = "p1") -> LifeModel:
    """A ``LifeModel`` whose state has a live desire awaiting *pending_id*'s verdict."""
    state = State(
        u=80.0,
        desire_status="active",
        pending_proactive_id=pending_id,
        pending_proactive_since=_T0.isoformat(),
        last_tick_at=_T0.isoformat(),
    )
    return build_lifemodel(
        base_dir=Path("/unused"),
        state=FakeStateStore(state),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=SilentAggregator(),
        neurons=(),
    )


def make_lm_no_pending() -> LifeModel:
    """A ``LifeModel`` with no proactive turn outstanding."""
    return build_lifemodel(
        base_dir=Path("/unused"),
        state=FakeStateStore(State(last_tick_at=_T0.isoformat())),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=SilentAggregator(),
        neurons=(),
    )


def make_lm_high_u() -> LifeModel:
    """A ``LifeModel`` whose urge (``u``) is well above zero, no desire pending —
    used only to make satiation-by-inbound-contact observable (Task 6), not to
    exercise the wake gate itself (that's ``core/decision``'s job)."""
    return build_lifemodel(
        base_dir=Path("/unused"),
        state=FakeStateStore(State(u=50.0, last_tick_at=_T0.isoformat())),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=SilentAggregator(),
        neurons=(),
    )


def _fake_payload(*, proactive: bool, text: str) -> dict[str, Any]:
    """Reproduce the real ``post_llm_call`` kwargs (``turn_finalizer.py`` ~L369).

    ``proactive=True`` sets ``user_message`` to our own composed impulse (the
    correlation signal the SPIKE found — see the caveat in ``lifemodel.hooks``);
    ``proactive=False`` is a genuine chat message that happens to land while a
    desire may be pending.
    """
    user_message = (
        f"{IMPULSE_LABEL_PREFIX} filler impulse text" if proactive else "hey, how are you?"
    )
    return {
        "session_id": "s1",
        "task_id": None,
        "turn_id": "t1",
        "user_message": user_message,
        "assistant_response": text,
        "conversation_history": [],
        "model": "test-model",
        "platform": "telegram",
    }


# --- _is_no_reply ------------------------------------------------------------


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


# --- make_post_llm_observer ---------------------------------------------------


def test_no_reply_maps_to_reject() -> None:
    lm = make_lm_pending(pending_id="p1")
    obs = make_post_llm_observer(lm)
    obs(**_fake_payload(proactive=True, text="NO_REPLY"))
    s = lm.state.load()
    assert s.desire_status == "none"
    assert s.decline_count == 1
    assert s.declined_at is not None
    assert s.pending_proactive_id is None


def test_real_text_maps_to_fulfill() -> None:
    lm = make_lm_pending(pending_id="p1")
    obs = make_post_llm_observer(lm)
    obs(**_fake_payload(proactive=True, text="Привет! Как ты?"))
    s = lm.state.load()
    assert s.desire_status == "none"
    assert s.decline_count == 0
    assert s.last_contact_at is not None
    assert s.pending_proactive_id is None


def test_non_proactive_turn_ignored() -> None:
    # A genuine user message finishes while a proactive turn is pending — it
    # must not be mistaken for the awaited proactive verdict.
    lm = make_lm_pending(pending_id="p1")
    make_post_llm_observer(lm)(**_fake_payload(proactive=False, text="hi"))
    s = lm.state.load()
    assert s.desire_status == "active"
    assert s.pending_proactive_id == "p1"


def test_impulse_text_ignored_when_nothing_pending() -> None:
    # Defensive: an impulse-prefixed user_message with no pending id on record
    # (e.g. a stray/duplicate hook call after the desire already resolved)
    # must not apply a stray verdict.
    lm = make_lm_no_pending()
    make_post_llm_observer(lm)(**_fake_payload(proactive=True, text="NO_REPLY"))
    s = lm.state.load()
    assert s.desire_status == "none"
    assert s.decline_count == 0


# --- make_inbound_observer (Task 6) -------------------------------------------


class _FakeInboundEvent:
    """Minimal stand-in for ``gateway.platforms.base.MessageEvent`` — only the
    two attributes ``make_inbound_observer`` reads (see the SPIKE notes in
    ``lifemodel.hooks``'s module docstring)."""

    def __init__(self, text: str, *, internal: bool = False) -> None:
        self.text = text
        self.internal = internal


def _fake_inbound(*, text: str, internal: bool = False) -> dict[str, Any]:
    """Reproduce the real ``pre_gateway_dispatch`` kwargs (``gateway/run.py``
    ~L8650): ``invoke_hook("pre_gateway_dispatch", event=event, gateway=self,
    session_store=self.session_store)``."""
    return {
        "event": _FakeInboundEvent(text, internal=internal),
        "gateway": None,
        "session_store": None,
    }


def test_inbound_user_message_satiates_and_stamps() -> None:
    lm = make_lm_high_u()
    before_u = lm.state.load().u
    make_inbound_observer(lm)(**_fake_inbound(text="привет"))
    s = lm.state.load()
    assert s.last_exchange_at is not None
    assert s.u < before_u


def test_inbound_ignores_own_impulse() -> None:
    lm = make_lm_high_u()
    make_inbound_observer(lm)(**_fake_inbound(text=f"{IMPULSE_LABEL_PREFIX} filler impulse text"))
    s = lm.state.load()
    assert s.last_exchange_at is None
    assert s.u == 50.0


def test_inbound_ignores_internal_event() -> None:
    # Defense-in-depth: the host itself already skips internal events before
    # ever invoking pre_gateway_dispatch (gateway/run.py `if not is_internal:`
    # — see the module docstring), but the observer must not rely on that
    # forever being true.
    lm = make_lm_high_u()
    make_inbound_observer(lm)(**_fake_inbound(text="hi", internal=True))
    s = lm.state.load()
    assert s.last_exchange_at is None
    assert s.u == 50.0


def test_inbound_clears_desire_and_reject_on_genuine_contact() -> None:
    lm = build_lifemodel(
        base_dir=Path("/unused"),
        state=FakeStateStore(
            State(
                u=99.0,
                desire_status="active",
                declined_at=_T0.isoformat(),
                decline_count=3,
                last_tick_at=_T0.isoformat(),
            )
        ),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=SilentAggregator(),
        neurons=(),
    )
    make_inbound_observer(lm)(**_fake_inbound(text="hey, how are you?"))
    s = lm.state.load()
    assert s.desire_status == "none"
    assert s.declined_at is None
    assert s.decline_count == 0


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
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: None)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # must not raise even without a real Hermes host

    matches = [cb for name, cb in ctx.hooks if name == "post_llm_call"]
    assert len(matches) == 1

    # The registered callback is genuinely wired to a working observer (not a
    # stub): a NO_REPLY payload for a pending proactive turn resolves it.
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    pending_state = State(
        desire_status="active", pending_proactive_id="p1", last_tick_at=_T0.isoformat()
    )
    state_file.write_text(json.dumps(pending_state.to_dict()))

    matches[0](**_fake_payload(proactive=True, text="NO_REPLY"))

    persisted = json.loads(state_file.read_text())
    assert persisted["desire_status"] == "none"
    assert persisted["decline_count"] == 1


def test_register_wires_pre_gateway_dispatch_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lifemodel

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: None)

    ctx = _FakeCtx()
    lifemodel.register(ctx)  # must not raise even without a real Hermes host

    matches = [cb for name, cb in ctx.hooks if name == "pre_gateway_dispatch"]
    assert len(matches) == 1

    # Genuinely wired (not a stub): a real inbound message satiates a
    # persisted state file, not just an in-memory fake.
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(State(u=50.0, last_tick_at=_T0.isoformat()).to_dict()))

    matches[0](**_fake_inbound(text="hey there"))

    persisted = json.loads(state_file.read_text())
    assert persisted["last_exchange_at"] is not None
    assert persisted["u"] < 50.0

"""Tests for the session-end seam — how a newborn falls asleep and wakes as itself.

The defect this closes (ADR-0002, corrected): ``SOUL.md`` is NOT re-read every turn.
Hermes builds the system prompt ONCE per session and reuses it verbatim from the session
DB (``agent/turn_context.py``: ``if agent._cached_system_prompt is None: restore_or_build``;
``agent/conversation_loop.py:282``: a stored prompt is "reused verbatim") — deliberately,
to keep the LLM prefix cache intact. Gateway sessions live for DAYS. So a being that wrote
its soul kept speaking in the voice it had.

Ending the session is the host's own sanctioned answer: ``SessionStore.reset_session``
(``gateway/session.py:2231``) is exactly what ``/new`` calls — a new ``session_id`` for the
same chat, the old one ended in the DB. The next message opens the fresh session with EMPTY
history, so the prompt is rebuilt and the new soul is in slot #1.

**Two host calls, not one, and the second is not optional.** ``reset_session`` alone is a
no-op for the being's prompt: the gateway also caches the whole ``AIAgent`` on
``session_key`` and REUSES it across a session_id switch (run.py, #54947) — and a reused
agent has ``_cached_system_prompt`` already set, so ``restore_or_build`` never runs. The
cached agent must be evicted too, or the being wakes in the old voice with a shiny new
session id underneath it.
"""

from __future__ import annotations

from typing import Any

import pytest

from lifemodel.adapters.session_end import GatewaySessionEnd
from lifemodel.domain.session import SessionEndOutcome
from lifemodel.gateway_core import end_session, session_end_available

SESSION_KEY = "agent:main:telegram:dm:115679831"


class _Store:
    """Mirrors ``gateway.session.SessionStore``: ``reset_session`` mints a new entry and
    returns it, or returns ``None`` when it has never heard of the key."""

    def __init__(self, *, known: bool = True) -> None:
        self.known = known
        self.reset_calls: list[str] = []

    def reset_session(self, session_key: str) -> Any:
        self.reset_calls.append(session_key)
        return object() if self.known else None


class _Runner:
    def __init__(self, *, complete: bool = True, known: bool = True) -> None:
        self.calls: list[str] = []  # the ORDER of host calls, which is load-bearing
        if complete:
            self.session_store = _Store(known=known)
            self.session_store.reset_session = self._reset  # type: ignore[method-assign]
            self._store = _Store(known=known)

    def _reset(self, session_key: str) -> Any:
        self.calls.append(f"reset:{session_key}")
        return self._store.reset_session(session_key)

    def _evict_cached_agent(self, session_key: str) -> None:
        self.calls.append(f"evict:{session_key}")


def test_unavailable_without_a_runner() -> None:
    assert session_end_available(None) is False
    assert end_session(None, SESSION_KEY) is SessionEndOutcome.UNAVAILABLE


def test_unavailable_on_host_version_drift() -> None:
    # The version guard: a host whose runner no longer carries these internals gets a
    # clean UNAVAILABLE, not an AttributeError inside a birth.
    assert session_end_available(_Runner(complete=False)) is False
    assert end_session(_Runner(complete=False), SESSION_KEY) is SessionEndOutcome.UNAVAILABLE


def test_unavailable_without_a_session_key() -> None:
    # Off the gateway (a CLI turn, a test harness) there is no session to end.
    assert end_session(_Runner(), "") is SessionEndOutcome.UNAVAILABLE


def test_ending_a_session_resets_it_AND_evicts_the_cached_agent() -> None:
    runner = _Runner()

    assert end_session(runner, SESSION_KEY) is SessionEndOutcome.ENDED

    # Both, in this order. reset_session alone leaves the cached AIAgent — which the
    # gateway reuses ACROSS a session_id switch, with its _cached_system_prompt intact —
    # so the being would wake in its old voice. Evicting first would leave a window in
    # which a racing turn re-caches an agent bound to the OLD session_id.
    assert runner.calls == [f"reset:{SESSION_KEY}", f"evict:{SESSION_KEY}"]


def test_a_session_the_host_never_heard_of_is_UNAVAILABLE_not_a_failure() -> None:
    # reset_session returns None for an unknown key. Nothing was ended, so nothing is
    # evicted — and this is not an error: it is a host that has no session here.
    runner = _Runner(known=False)

    assert end_session(runner, SESSION_KEY) is SessionEndOutcome.UNAVAILABLE
    assert runner.calls == [f"reset:{SESSION_KEY}"]  # no eviction on a no-op reset


def test_it_never_raises_into_the_birth_that_called_it() -> None:
    # Fail-soft is the whole contract: the soul is already on disk. A throw here would
    # turn a completed birth into a tool error and tell the being it failed to be born.
    class _Exploding(_Runner):
        def _reset(self, session_key: str) -> Any:
            raise RuntimeError("the session DB is locked")

    assert end_session(_Exploding(), SESSION_KEY) is SessionEndOutcome.FAILED


def test_a_failure_to_EVICT_is_still_a_failure_not_a_half_truth() -> None:
    # The session id rotated but the cached agent survived: the next turn reuses it, keeps
    # its old system prompt, and the being does NOT wake as itself. Reporting ENDED here
    # would make the tool promise the being a homecoming that will not happen.
    class _EvictExplodes(_Runner):
        def _evict_cached_agent(self, session_key: str) -> None:
            raise RuntimeError("the agent cache lock is wedged")

    assert end_session(_EvictExplodes(), SESSION_KEY) is SessionEndOutcome.FAILED


# --- the adapter: where the session key comes from ------------------------------


def test_the_adapter_is_a_zero_arg_callable_that_finds_its_own_session() -> None:
    runner = _Runner()
    ender = GatewaySessionEnd(
        runner_accessor=lambda: runner,
        session_key_accessor=lambda: SESSION_KEY,
    )

    assert ender() is SessionEndOutcome.ENDED
    assert runner.calls == [f"reset:{SESSION_KEY}", f"evict:{SESSION_KEY}"]


def test_the_adapter_degrades_when_the_host_has_no_session_bound() -> None:
    ender = GatewaySessionEnd(runner_accessor=lambda: _Runner(), session_key_accessor=lambda: "")
    assert ender() is SessionEndOutcome.UNAVAILABLE


def test_the_adapter_degrades_off_host() -> None:
    ender = GatewaySessionEnd(
        runner_accessor=lambda: None, session_key_accessor=lambda: SESSION_KEY
    )
    assert ender() is SessionEndOutcome.UNAVAILABLE


def test_the_default_session_key_accessor_returns_empty_off_host() -> None:
    # No `gateway` package in a dev checkout — the import fails and we degrade, never throw.
    from lifemodel.adapters.session_end import default_session_key_accessor

    assert isinstance(default_session_key_accessor(), str)


@pytest.mark.parametrize(
    ("outcome", "ok"),
    [
        (SessionEndOutcome.ENDED, True),
        (SessionEndOutcome.UNAVAILABLE, False),
        (SessionEndOutcome.FAILED, False),
    ],
)
def test_only_ENDED_means_the_being_will_actually_wake_as_itself(
    outcome: SessionEndOutcome, ok: bool
) -> None:
    # The tool reads `.ok` to decide WHICH truth to tell the newborn. Anything but ENDED
    # means the soul is on disk and the voice is not — and the being must be told so.
    assert outcome.ok is ok


# --- the OWNER's session key (lm-4fv.2): a slash command has no turn to read -------


def test_the_owner_session_key_falls_back_to_the_home_dm_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``/lifemodel soul revert`` runs in ``_handle_message``, which RESETS the session
    ContextVars at handler entry and only binds them later, in ``_handle_message_with_agent``
    — a path a plugin command returns long before. So the turn-local key is empty, and a
    revert that trusted it would report "session ended" having ended nothing: the being would
    keep speaking as the soul the owner just replaced, for days. The lane is resolved the way
    the being's own reach-out resolves it — from the home origin."""
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "115679831")
    from lifemodel.adapters.session_end import home_session_key_accessor

    assert home_session_key_accessor() == SESSION_KEY  # the being's DM lane with its owner


def test_the_owner_session_key_is_empty_when_there_is_no_home_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # UNAVAILABLE, not a crash: the soul is still reverted, and the owner is told the truth.
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    from lifemodel.adapters.session_end import home_session_key_accessor

    assert home_session_key_accessor() == ""


def test_the_home_session_key_is_built_in_ONE_place() -> None:
    """Two callers now depend on this format being right — the being's reach-in (where a
    wrong key merely falls back to the host's own source builder) and the owner's revert
    (where a wrong key silently ends nothing). Two hand-rolled copies is how they come to
    disagree."""
    from lifemodel.gateway_core import home_session_key

    assert home_session_key({"platform": "telegram", "chat_id": "115679831"}) == SESSION_KEY
    # An explicit key wins — it is the lane as the caller already knows it.
    assert (
        home_session_key({"session_key": "agent:main:x", "platform": "telegram"}) == "agent:main:x"
    )
    # Not addressable → "", which every caller reads as UNAVAILABLE.
    assert home_session_key({"platform": "telegram", "chat_id": None}) == ""


def test_sleep_soft_never_lets_a_broken_ender_undo_a_completed_soul_write() -> None:
    from lifemodel.adapters.session_end import sleep_soft

    def _boom() -> SessionEndOutcome:
        raise RuntimeError("the host changed shape underneath us")

    assert sleep_soft(_boom) is SessionEndOutcome.FAILED
    assert sleep_soft(None) is SessionEndOutcome.UNAVAILABLE  # nobody wired one: not a failure

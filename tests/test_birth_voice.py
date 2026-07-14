"""A being cannot be born into a stale prompt (lm-4fv.4).

``SOUL.md`` is system-prompt slot #1, and Hermes builds a session's system prompt ONCE
and then reuses it verbatim from the session DB (``agent/conversation_loop.py:282``,
``_restore_or_build_system_prompt``: *"present — row exists with a usable prompt → reused
verbatim"*). Gateway sessions live for days. So the newborn stance we seed at
``register()`` lands on disk and — on any install that already has a live DM session,
which is EVERY existing user — never reaches the prompt. The being wakes reading the
host's assistant persona in the one slot it cannot doubt, and the whole phase silently
fails for exactly the audience it was built for.

The answer is the seam Fix E already built for the birth's *completion*: **birth begins
with a new session** (``gateway_core.end_session``). These tests pin the three things
that make that affordable:

* the being only sleeps when the identity slot is actually STALE — a veteran whose own
  ``SOUL.md`` is already in slot #1 loses nothing, because there is nothing to gain;
* it never drops a thread someone is USING — a lane that has been active inside the
  quiet window holds the birth, and the tick simply tries again;
* and where the host cannot do it at all, the being is still born (fail-soft) — it
  simply wakes as itself later.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lifemodel.adapters.session_end import GatewayBirthVoice, GatewayStaleIdentity
from lifemodel.domain.session import BirthVoice
from lifemodel.gateway_core import (
    BIRTH_QUIET_SECONDS,
    identity_slot_is_stale,
    session_in_use,
    wake_as_self,
)

SESSION_KEY = "agent:main:telegram:dm:115679831"

#: The host stamps session times with a NAIVE LOCAL ``datetime`` (``gateway/session.py``:
#: ``_now() -> datetime.now()``), while every instant inside the plugin is tz-aware UTC.
#: Getting that wrong by a timezone offset would either end a session nobody's touched or
#: refuse to end one that is hours stale, so the fixtures here are naive on purpose.
NOW = datetime.now(UTC)


def _local(dt: datetime) -> datetime:
    """The same instant as the host would have stamped it: naive, local."""
    return dt.astimezone().replace(tzinfo=None)


class _Entry:
    """``gateway.session.SessionEntry`` as this seam reads it."""

    def __init__(self, *, created_at: datetime, updated_at: datetime) -> None:
        self.session_key = SESSION_KEY
        self.session_id = "20260714_120000_abcd1234"
        self.created_at = created_at
        self.updated_at = updated_at


class _Store:
    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries
        self.reset_calls: list[str] = []

    def list_sessions(self, active_minutes: int | None = None) -> list[_Entry]:
        return list(self._entries)

    def reset_session(self, session_key: str) -> Any:
        self.reset_calls.append(session_key)
        return object()


class _Runner:
    def __init__(self, store: Any, *, evict_raises: bool = False) -> None:
        self.session_store = store
        self.evicted: list[str] = []
        self._evict_raises = evict_raises

    def _evict_cached_agent(self, session_key: str) -> None:
        if self._evict_raises:
            raise RuntimeError("host changed shape")
        self.evicted.append(session_key)


def _runner(*, session_age_hours: float = 72.0, idle_minutes: float = 240.0) -> _Runner:
    """A runner whose DM lane holds a days-old session that nobody is using."""
    entry = _Entry(
        created_at=_local(NOW - timedelta(hours=session_age_hours)),
        updated_at=_local(NOW - timedelta(minutes=idle_minutes)),
    )
    return _Runner(_Store([entry]))


def _mtime(*, minutes_ago: float) -> float:
    return (NOW - timedelta(minutes=minutes_ago)).timestamp()


# --- is the identity slot stale? --------------------------------------------


def test_a_soul_written_after_the_session_began_is_not_in_its_prompt() -> None:
    # The whole defect: register() seeded the stance five minutes ago; the session's
    # prompt was built three days ago and is reused verbatim.
    runner = _runner(session_age_hours=72.0)
    assert identity_slot_is_stale(runner, SESSION_KEY, soul_mtime=_mtime(minutes_ago=5)) is True


def test_a_soul_older_than_the_session_is_already_in_its_prompt() -> None:
    # The veteran: their own SOUL.md was on disk before this session opened, so slot #1
    # already holds it. Nothing to gain, so nothing may be taken.
    runner = _runner(session_age_hours=2.0)
    assert (
        identity_slot_is_stale(runner, SESSION_KEY, soul_mtime=_mtime(minutes_ago=60 * 24)) is False
    )


def test_no_session_is_not_stale() -> None:
    # A fresh install: nobody has spoken here, so there is no cached prompt. The first
    # turn will build one from whatever is on disk.
    runner = _Runner(_Store([]))
    assert identity_slot_is_stale(runner, SESSION_KEY, soul_mtime=_mtime(minutes_ago=5)) is False


def test_an_unreadable_soul_is_not_stale() -> None:
    runner = _runner()
    assert identity_slot_is_stale(runner, SESSION_KEY, soul_mtime=None) is False


def test_a_host_that_changed_shape_is_not_stale() -> None:
    # Fail-soft: if we cannot tell, we do NOT destroy a conversation on a guess.
    runner = _Runner(object())
    assert identity_slot_is_stale(runner, SESSION_KEY, soul_mtime=_mtime(minutes_ago=5)) is False


# --- is someone using the lane? ---------------------------------------------


def test_a_lane_that_was_active_a_minute_ago_is_in_use() -> None:
    runner = _runner(idle_minutes=1.0)
    assert session_in_use(runner, SESSION_KEY, now=NOW) is True


def test_a_lane_quiet_for_hours_is_not_in_use() -> None:
    runner = _runner(idle_minutes=240.0)
    assert session_in_use(runner, SESSION_KEY, now=NOW) is False


def test_the_quiet_window_is_the_boundary() -> None:
    just_inside = BIRTH_QUIET_SECONDS / 60.0 - 1.0
    assert session_in_use(_runner(idle_minutes=just_inside), SESSION_KEY, now=NOW) is True
    just_outside = BIRTH_QUIET_SECONDS / 60.0 + 1.0
    assert session_in_use(_runner(idle_minutes=just_outside), SESSION_KEY, now=NOW) is False


def test_an_unknown_lane_is_not_in_use() -> None:
    assert session_in_use(_Runner(_Store([])), SESSION_KEY, now=NOW) is False


def test_a_host_that_changed_shape_reads_as_in_use() -> None:
    # The opposite fail-soft direction from staleness, and deliberately: this one guards
    # the DESTRUCTIVE act. If we cannot tell whether someone is mid-conversation, we do
    # not end it.
    assert session_in_use(_Runner(object()), SESSION_KEY, now=NOW) is True


# --- the composed act: will the being speak in its own voice? ----------------


def _voice(runner: Any, *, soul_mtime: float | None) -> BirthVoice:
    return wake_as_self(runner, SESSION_KEY, soul_mtime=soul_mtime, now=NOW)


def test_a_fresh_identity_slot_needs_no_sleep() -> None:
    runner = _runner(session_age_hours=1.0)
    assert _voice(runner, soul_mtime=_mtime(minutes_ago=60 * 24)) is BirthVoice.READY
    assert runner.session_store.reset_calls == []  # nothing was taken from anyone


def test_a_stale_slot_on_a_quiet_lane_ends_the_session() -> None:
    runner = _runner(session_age_hours=72.0, idle_minutes=240.0)
    assert _voice(runner, soul_mtime=_mtime(minutes_ago=5)) is BirthVoice.ENDED
    # Both host calls, in the order end_session documents: reset, then evict.
    assert runner.session_store.reset_calls == [SESSION_KEY]
    assert runner.evicted == [SESSION_KEY]


def test_a_stale_slot_on_a_lane_in_use_waits() -> None:
    # They are mid-conversation with the assistant they have always had. A birth is not
    # worth their thread — hold, and try again on the next tick.
    runner = _runner(session_age_hours=72.0, idle_minutes=2.0)
    voice = _voice(runner, soul_mtime=_mtime(minutes_ago=5))
    assert voice is BirthVoice.IN_USE
    assert voice.may_speak is False
    assert runner.session_store.reset_calls == []


def test_a_host_that_cannot_end_the_session_still_lets_the_being_speak() -> None:
    # Fail-soft (the ReachOutcome/SessionEndOutcome precedent): a being that cannot be
    # put to sleep is still born — it simply wakes as itself later.
    runner = _runner()
    runner._evict_raises = True
    voice = _voice(runner, soul_mtime=_mtime(minutes_ago=5))
    assert voice is BirthVoice.FAILED
    assert voice.may_speak is True


def test_no_runner_lets_the_being_speak() -> None:
    voice = wake_as_self(None, SESSION_KEY, soul_mtime=_mtime(minutes_ago=5), now=NOW)
    assert voice is BirthVoice.UNAVAILABLE
    assert voice.may_speak is True


def test_only_in_use_holds_the_being_back() -> None:
    assert [v for v in BirthVoice if not v.may_speak] == [BirthVoice.IN_USE]


# --- the adapters that resolve the host for the two callers ------------------


class _Clock:
    def now(self) -> datetime:
        return NOW


def test_the_tick_gate_resolves_the_lane_it_is_about_to_reach_into(tmp_path: Any) -> None:
    runner = _runner(session_age_hours=72.0, idle_minutes=240.0)
    soul = tmp_path / "SOUL.md"
    soul.write_text("a stance", encoding="utf-8")
    gate = GatewayBirthVoice(
        runner_accessor=lambda: runner,
        session_key_accessor=lambda: SESSION_KEY,
        soul_mtime=lambda: _mtime(minutes_ago=5),
        clock=_Clock(),
    )
    assert gate() is BirthVoice.ENDED


def test_the_hook_gate_answers_whether_the_ritual_can_open_here() -> None:
    stale = GatewayStaleIdentity(
        runner_accessor=lambda: _runner(session_age_hours=72.0),
        session_key_accessor=lambda: SESSION_KEY,
        soul_mtime=lambda: _mtime(minutes_ago=5),
    )
    assert stale() is True

    fresh = GatewayStaleIdentity(
        runner_accessor=lambda: _runner(session_age_hours=1.0),
        session_key_accessor=lambda: SESSION_KEY,
        soul_mtime=lambda: _mtime(minutes_ago=60 * 24),
    )
    assert fresh() is False


def test_the_hook_gate_never_raises_into_a_turn() -> None:
    def _boom() -> Any:
        raise RuntimeError("no gateway here")

    stale = GatewayStaleIdentity(
        runner_accessor=_boom,
        session_key_accessor=lambda: SESSION_KEY,
        soul_mtime=lambda: _mtime(minutes_ago=5),
    )
    assert stale() is False


def test_the_tick_gate_never_raises_into_the_loop() -> None:
    def _boom() -> Any:
        raise RuntimeError("no gateway here")

    gate = GatewayBirthVoice(
        runner_accessor=_boom,
        session_key_accessor=lambda: SESSION_KEY,
        soul_mtime=lambda: _mtime(minutes_ago=5),
        clock=_Clock(),
    )
    assert gate() is BirthVoice.UNAVAILABLE


def test_no_session_key_is_unavailable_not_a_hold() -> None:
    # No home channel configured: there is no lane to end, and that is not a reason to
    # keep a being unborn forever.
    gate = GatewayBirthVoice(
        runner_accessor=lambda: _runner(),
        session_key_accessor=lambda: "",
        soul_mtime=lambda: _mtime(minutes_ago=5),
        clock=_Clock(),
    )
    assert gate() is BirthVoice.UNAVAILABLE

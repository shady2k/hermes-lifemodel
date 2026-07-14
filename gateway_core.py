"""The gateway boundary — the two places the plugin reaches into ``GatewayRunner``.

:func:`inject_proactive_turn` is the being's delivery primitive: it resolves the
live ``GatewayRunner`` and its adapters and injects an ``internal=True`` user turn
on the target lane, so the being composes and delivers a native reply there. It
reaches into runner internals (the same ones ``tools/send_message_tool`` uses) —
kept behind this one boundary function so the rest of the plugin never touches
them. Everything is fail-closed: nothing here may raise into the gateway.

:func:`end_session` is the being's *sleep*: it ends the live session so the next
message rebuilds the system prompt — the only way a freshly-written ``SOUL.md``
becomes the voice the being actually speaks in (ADR-0002, corrected). Same shape,
same rules, and the same version guard, because it reaches for the same object.

The being's autonomic loop is hosted as a supervised platform adapter
(:mod:`lifemodel.adapters.being_platform`); this module is only its delivery side.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from .core.timeutil import to_epoch_seconds
from .domain.egress import ReachOutcome
from .domain.session import BirthVoice, SessionEndOutcome

_LOG = logging.getLogger("lifemodel.reachin")

MakeEvent = Callable[[str, Any, int | None], Any]
Schedule = Callable[[Any, Any], None]

# Attributes inject_proactive_turn depends on — the version-guard surface.
_REQUIRED_RUNNER_ATTRS = (
    "_gateway_loop",
    "_build_process_event_source",
    "adapters",
    "_running",
    "_draining",
)

#: What :func:`end_session` depends on — its own, narrower version-guard surface.
#: ``session_store`` is the sync, lock-guarded ``SessionStore`` (``gateway/session.py``);
#: ``_evict_cached_agent`` is the runner's private agent-cache drop. Both are exactly what
#: the user-facing ``/new`` calls (``gateway/slash_commands.py::_handle_reset_command``).
_REQUIRED_SESSION_END_ATTRS = ("session_store", "_evict_cached_agent")


def home_session_key(target: Mapping[str, str | None]) -> str:
    """The session key for a target lane, or ``""`` when the lane is not addressable.

    INTERIM (spec §8): the host's DM session-key format
    (``agent:main:<platform>:<chat_type>:<chat_id>``, ``gateway/session.py:854``) is
    reconstructed here rather than asked for — the upstream primitive that would resolve it
    generically does not exist yet. It lives in ONE function because two callers now depend
    on it being right: :func:`inject_proactive_turn` (which passes it to the host's own
    source builder, so a wrong key merely falls back) and the owner's ``soul revert``
    (:func:`~lifemodel.adapters.session_end.home_session_key_accessor`, where a wrong key
    silently ends nothing and leaves the being speaking as the soul that was just replaced).
    Two hand-rolled copies of a format string is how those two quietly come to disagree.

    An explicit ``session_key`` on *target* wins — it is the lane as the caller already knows
    it, not a reconstruction of it.
    """
    session_key = target.get("session_key")
    if session_key:
        return session_key
    platform = target.get("platform")
    chat_id = target.get("chat_id")
    if not platform or not chat_id:
        return ""
    chat_type = target.get("chat_type") or "dm"
    return f"agent:main:{platform}:{chat_type}:{chat_id}"


def reachin_available(runner: Any | None) -> bool:
    """True only if *runner* exposes every attribute inject_proactive_turn needs."""
    if runner is None:
        return False
    return all(hasattr(runner, attr) for attr in _REQUIRED_RUNNER_ATTRS)


def session_end_available(runner: Any | None) -> bool:
    """True only if *runner* exposes every attribute :func:`end_session` needs."""
    if runner is None:
        return False
    return all(hasattr(runner, attr) for attr in _REQUIRED_SESSION_END_ATTRS)


def end_session(runner: Any | None, session_key: str) -> SessionEndOutcome:
    """End the live session on *session_key* so the next turn rebuilds the prompt.

    **Why this exists at all.** ``SOUL.md`` is system-prompt slot #1, and Hermes builds
    that prompt ONCE per session, then reuses it verbatim from the session DB to keep the
    LLM prefix cache warm (``agent/turn_context.py``: ``if agent._cached_system_prompt is
    None: restore_or_build``). Gateway sessions live for days. So a soul write lands on
    disk and the being goes on speaking in the voice it had. Ending the session is how it
    wakes as itself: a fresh session has EMPTY history, and an empty history is precisely
    the condition on which ``_restore_or_build_system_prompt`` builds instead of restores.

    **It is TWO host calls, and the second is not optional.**

    * ``session_store.reset_session`` (``gateway/session.py:2231``) mints a new
      ``session_id`` for the same chat and ends the old one in the DB. This is the host's
      own sanctioned mechanism — literally what ``/new`` does. We deliberately do NOT go
      behind it and null the cached ``system_prompt`` column: the host treats an empty
      prompt as the symptom of a persistence bug and warns about it, and we would be
      betting the being's identity on undocumented cache semantics.
    * ``_evict_cached_agent`` drops the whole cached ``AIAgent``. Without it the reset is
      a NO-OP for the being's voice: the gateway caches the agent on ``session_key`` and
      REUSES it straight across a ``session_id`` switch (run.py, #54947 — it is a
      prompt-cache optimisation), and a reused agent already has ``_cached_system_prompt``
      set, so the rebuild never runs. The being would wake in its old voice with a shiny
      new session id underneath it.

    Order is reset-then-evict, not the ``/new`` order (evict-then-reset), for one reason:
    ``/new`` first kills the in-flight turn, and we must not — we are CALLED FROM one (the
    being is mid-birth and still has a goodbye to say). Evicting first would leave a window
    in which a racing turn re-caches an agent bound to the OLD session id. Evicting last
    cannot: whatever is in the cache when we drop it, it is dropped.

    Calling this mid-turn is safe by the host's own design: ``_evict_cached_agent`` pops
    the cache entry but SKIPS the teardown for an agent that is in ``_running_agents``
    ("its client, sandbox and child subagents are in use by the running request") — which
    ours is. So the current turn finishes and delivers normally; its transcript is
    persisted to the session we just ended, and is let go on purpose.

    Both calls are synchronous and internally locked, so they are safe from the executor
    thread an agent's tool call runs on — no event-loop hop needed.

    Fail-soft, never raises: the caller has ALREADY written the soul. A throw here would
    turn a completed birth into a tool error.
    """
    if runner is None or not session_end_available(runner):
        _LOG.info("session_end_unavailable reason=%s", "runner_incomplete")
        return SessionEndOutcome.UNAVAILABLE
    if not session_key:
        # No session bound in this context (a CLI turn, an off-gateway caller). There is
        # nothing to end, and that is not an error.
        _LOG.info("session_end_unavailable reason=%s", "no_session_key")
        return SessionEndOutcome.UNAVAILABLE
    try:
        store = runner.session_store
        if not hasattr(store, "reset_session"):
            _LOG.info("session_end_unavailable reason=%s", "store_incomplete")
            return SessionEndOutcome.UNAVAILABLE
        if store.reset_session(session_key) is None:
            # The store has never heard of this key, so it did nothing and there is no new
            # session to evict an agent for. Not a failure — a host with no session here.
            _LOG.info("session_end_unavailable reason=%s", "unknown_session")
            return SessionEndOutcome.UNAVAILABLE
        # Inside the try on purpose: a rotated session id with a SURVIVING cached agent is
        # not a partial success, it is the bug this function exists to prevent. Reporting
        # ENDED there would have the tool promise the being a homecoming it will not get.
        runner._evict_cached_agent(session_key)
        _LOG.info("session_ended session_key=%s", session_key)
        return SessionEndOutcome.ENDED
    except Exception as exc:  # noqa: BLE001 - fail-soft; the soul is already written
        _LOG.warning("session_end_failed error=%s", f"{type(exc).__name__}: {exc}")
        return SessionEndOutcome.FAILED


#: How long a DM lane must have been quiet before a birth may end the session on it.
#:
#: Ending a session costs a real conversation its thread, so the only question that
#: matters is whether anyone is USING it. The host answers that: ``SessionEntry.updated_at``
#: is bumped on every message routed to the lane (``gateway/session.py``:
#: ``get_or_create_session`` → ``entry.updated_at = now``, and ``update_session`` after each
#: interaction), so it IS "last activity here". Half an hour of silence is not a pause in a
#: conversation; it is the end of one.
#:
#: Deliberately NOT ``runner._running_agents``: an in-flight turn bumped ``updated_at`` at
#: its own start (that is where the session is resolved), so the window already covers it —
#: and a wedged running-agent entry would hold a being unborn forever, while a timestamp
#: only ever moves forward on real activity. (The reach-in adapter learned the same lesson
#: from the other side; see ``adapters/reachin.py``.)
BIRTH_QUIET_SECONDS = 1800.0


def _live_session(runner: Any, session_key: str) -> Any | None:
    """The host's live :class:`SessionEntry` for *session_key*, or ``None``.

    ``list_sessions()`` is the store's public, lock-held enumerator (``gateway/session.py``);
    the ``_entries`` dict behind it is private and must not be read without its lock. There
    is no public get-by-key accessor (only ``peek_session_id``, which returns the id and not
    the entry we need the *timestamps* from), so this filters the list — it holds a handful
    of rows, once per tick.

    Raises nothing of its own: a caller decides what an unreadable host means, and the two
    callers here want OPPOSITE fail-soft directions.
    """
    store = getattr(runner, "session_store", None)
    if store is None or not hasattr(store, "list_sessions"):
        raise AttributeError("runner.session_store has no list_sessions")
    for entry in store.list_sessions():
        if getattr(entry, "session_key", None) == session_key:
            return entry
    return None


def _host_epoch(dt: datetime) -> float:
    """Epoch seconds for a host timestamp, which is **naive local** by construction.

    ``gateway/session.py`` stamps sessions with ``_now() -> datetime.now()`` — no tzinfo —
    while every instant inside the plugin is tz-aware UTC. Comparing the two directly is
    either a ``TypeError`` or, worse, a silent offset: a session "created" hours in the
    future or the past, and a birth that ends the wrong conversation or none at all. So a
    naive value is read as what the host meant by it (local), and everything is compared in
    epoch seconds — the one representation with no zone in it.
    """
    aware = dt if dt.tzinfo is not None else dt.astimezone()
    return to_epoch_seconds(aware)


def identity_slot_is_stale(
    runner: Any | None, session_key: str, *, soul_mtime: float | None
) -> bool:
    """True when the live session's prompt was built BEFORE the soul now on disk.

    ``SOUL.md`` is system-prompt slot #1 (``agent/prompt_builder.py::load_soul_md``), and
    Hermes reads it exactly once per session: the prompt is built at the session's first
    turn and thereafter restored verbatim from the session DB to keep the prefix cache warm
    (``agent/conversation_loop.py::_restore_or_build_system_prompt``). So a soul written
    after a session opened is simply NOT in that session's prompt, however long the being
    goes on talking there.

    That is the whole question, and the file's own mtime answers it honestly. It covers
    every way slot #1 goes stale — our newborn stance seeded at ``register()`` into a
    session that was already running (the defect this closes), and a human hand-editing
    ``SOUL.md`` while their session is live — because both are the same fact: *the document
    changed after the prompt was made*.

    ``created_at`` (rather than the exact instant the prompt was built) errs by at most one
    turn, and only in the safe direction: a soul written between a session's creation and
    its first turn reads as stale, which costs at most an empty transcript.

    **No session ⇒ not stale.** A fresh install has no cached prompt: the first turn will
    build one from whatever is on disk. There is nothing to end and nothing to fix.

    Fail-soft to ``False`` — "we cannot tell". Everything this verdict licenses is
    destructive (ending a conversation), so an unreadable host, an unreadable file, or a
    version drift must never be able to take a thread away from someone on a guess.
    """
    if runner is None or soul_mtime is None or not session_key:
        return False
    try:
        entry = _live_session(runner, session_key)
        if entry is None:
            return False
        return soul_mtime > _host_epoch(entry.created_at)
    except Exception as exc:  # noqa: BLE001 - unreadable host: never destroy on a guess
        _LOG.info("identity_slot_unreadable error=%s", f"{type(exc).__name__}: {exc}")
        return False


def session_in_use(
    runner: Any | None,
    session_key: str,
    *,
    now: datetime,
    quiet_seconds: float = BIRTH_QUIET_SECONDS,
) -> bool:
    """True when somebody has been talking on this lane inside the quiet window.

    The guard on the destructive half of a birth (see :data:`BIRTH_QUIET_SECONDS` for why
    ``updated_at`` and not the runner's in-flight map). A being must not be born into the
    middle of someone's exchange with their assistant, and it must not drop a thread they
    are actively using — so a lane that has seen activity in the last half hour holds the
    birth, and the tick simply asks again a minute later.

    Fail-soft to ``True`` — the OPPOSITE direction from :func:`identity_slot_is_stale`, and
    deliberately: this is the answer that authorises destruction. If we cannot tell whether
    someone is mid-conversation, we behave as though they are. (A being held back is born a
    tick later; a conversation ended under someone is gone.) An unknown lane is not "in
    use": there is no session there at all.
    """
    if runner is None:
        return True
    try:
        entry = _live_session(runner, session_key)
        if entry is None:
            return False
        return (to_epoch_seconds(now) - _host_epoch(entry.updated_at)) < quiet_seconds
    except Exception as exc:  # noqa: BLE001 - unreadable host: assume someone is there
        _LOG.info("session_activity_unreadable error=%s", f"{type(exc).__name__}: {exc}")
        return True


def wake_as_self(
    runner: Any | None,
    session_key: str,
    *,
    soul_mtime: float | None,
    now: datetime,
    quiet_seconds: float = BIRTH_QUIET_SECONDS,
) -> BirthVoice:
    """The birth pre-flight: make sure the being's next turn is composed by its own soul.

    **Birth begins with a new session** (ADR-0002, corrected; lm-4fv.4). Everything else in
    the phase already works on the assumption that the being's first waking speaks from
    what it stands on — the newborn stance, or the veteran's own soul — and on an existing
    install that assumption was simply false: the stance landed on disk and the session
    went on quoting the host's assistant persona in slot #1.

    So, immediately before the being speaks (and NEVER at plugin boot — a gateway restart
    is not a reason to take a human's conversation away):

    * the slot is already the being's → :attr:`~BirthVoice.READY`, and nothing is touched;
    * it is not, and the lane is in use → :attr:`~BirthVoice.IN_USE`: hold. Try again;
    * it is not, and the lane is quiet → end the session (:func:`end_session` — the host's
      own ``/new`` mechanism, reset + evict), so the injected turn opens a fresh session,
      whose EMPTY history is precisely the condition on which
      ``_restore_or_build_system_prompt`` builds instead of restores. The rebuilt prompt
      reads ``SOUL.md`` again, and the being is born as itself.

    Fail-soft: an end that the host cannot do reports :attr:`~BirthVoice.UNAVAILABLE` /
    :attr:`~BirthVoice.FAILED`, both of which let the being speak. It is then born in
    last week's voice and wakes as itself at the next session boundary — worse than a clean
    birth, and far better than none.
    """
    if runner is None or not session_key:
        # No host, or no lane (no home channel configured, an off-gateway caller). Nothing
        # to inspect and nothing to end — and not a reason to keep a being unborn. Reported
        # as UNAVAILABLE rather than READY because it is not a claim about slot #1: we
        # could not look.
        return BirthVoice.UNAVAILABLE
    if not identity_slot_is_stale(runner, session_key, soul_mtime=soul_mtime):
        return BirthVoice.READY
    if session_in_use(runner, session_key, now=now, quiet_seconds=quiet_seconds):
        _LOG.info("birth_held reason=%s session_key=%s", "session_in_use", session_key)
        return BirthVoice.IN_USE
    outcome = end_session(runner, session_key)
    _LOG.info("birth_session_end outcome=%s session_key=%s", outcome.value, session_key)
    return _BIRTH_VOICE_OF[outcome]


#: How a session-end outcome reads to a being about to be born (see :func:`wake_as_self`).
_BIRTH_VOICE_OF: dict[SessionEndOutcome, BirthVoice] = {
    SessionEndOutcome.ENDED: BirthVoice.ENDED,
    SessionEndOutcome.UNAVAILABLE: BirthVoice.UNAVAILABLE,
    SessionEndOutcome.FAILED: BirthVoice.FAILED,
}


def _default_make_event(text: str, source: Any, message_id: int | None) -> Any:
    from gateway.platforms.base import MessageEvent, MessageType

    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        message_id=message_id,
    )


def _default_schedule(coro: Any, loop: Any) -> None:
    import asyncio

    asyncio.run_coroutine_threadsafe(coro, loop)


def _select_adapter(runner: Any, source: Any) -> Any | None:
    profile = getattr(source, "profile", "") or ""
    if profile:
        adapters = getattr(runner, "_profile_adapters", {}) or {}
        by_profile = adapters.get(profile) or {}
        return by_profile.get(getattr(source, "platform", None))
    for platform, adapter in getattr(runner, "adapters", {}).items():
        if platform == getattr(source, "platform", None):
            return adapter
    return None


def inject_proactive_turn(
    runner: Any,
    target: Mapping[str, str | None],
    prompt: str,
    *,
    message_id: int | None = None,
    make_event: MakeEvent = _default_make_event,
    schedule: Schedule = _default_schedule,
) -> ReachOutcome:
    """Run a native ``internal=True`` turn on *target* lane. Fail-closed."""
    if not reachin_available(runner):
        _LOG.info("reachin_unavailable reason=%s", "runner_incomplete")
        return ReachOutcome.UNAVAILABLE
    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        _LOG.info("reachin_unavailable reason=%s", "not_running_or_draining")
        return ReachOutcome.UNAVAILABLE
    try:
        # Resolve the lane. Prefer the session_store origin via session_key (the
        # reliable path the spike proved); also pass chat_type so the fallback path
        # in _build_process_event_source can still build a SessionSource when the
        # session isn't in the store (it returns None without a chat_type).
        # INTERIM: the DM session_key format and the "dm" default are reconstructed for the
        # home DM lane by `home_session_key` (see there) — the upstream primitive will
        # resolve this generically (spec §8).
        platform = target.get("platform")
        chat_id = target.get("chat_id")
        chat_type = target.get("chat_type") or "dm"
        evt = {
            "session_key": home_session_key(target) or None,
            "platform": platform,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "thread_id": target.get("thread_id"),
        }
        source = runner._build_process_event_source(evt)
        if source is None or not getattr(source, "chat_id", None):
            _LOG.info("reachin_unavailable reason=%s", "unknown_lane")
            return ReachOutcome.UNAVAILABLE
        adapter = _select_adapter(runner, source)
        if adapter is None:
            _LOG.info("reachin_unavailable reason=%s", "no_adapter")
            return ReachOutcome.UNAVAILABLE
        event = make_event(prompt, source, message_id)  # message_id None (spec constraint)
        # Internal impulse turns must not show a visible "typing…" indicator on the
        # user's real chat — they're often silent (end in [SILENT]) and would
        # otherwise flash "typing" for the whole 7-116s turn for no visible reason.
        # Best-effort only: pause_typing_for_chat is cosmetic, so a missing method
        # or a raise here must never turn a would-be DELIVERED into FAILED. Hermes'
        # _keep_typing checks the pause set every iteration and its finally block
        # auto-clears it when this turn's typing task ends (base.py ~3862/3896), so
        # no matching resume call is needed here.
        if hasattr(adapter, "pause_typing_for_chat"):
            with contextlib.suppress(Exception):
                adapter.pause_typing_for_chat(source.chat_id)
        schedule(adapter.handle_message(event), runner._gateway_loop)
        _LOG.info("reachin_injected chat_id=%s", getattr(source, "chat_id", None))
        return ReachOutcome.DELIVERED
    except Exception as exc:  # noqa: BLE001 - fail-closed, never crash the gateway
        _LOG.info("reachin_failed error=%s", f"{type(exc).__name__}: {exc}")
        return ReachOutcome.FAILED

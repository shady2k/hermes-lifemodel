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
from typing import Any

from .domain.egress import ReachOutcome
from .domain.session import SessionEndOutcome

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
        # INTERIM: the DM session_key format ("agent:main:<platform>:<chat_type>:<chat_id>")
        # and the "dm" default are hardcoded for the home DM lane — the upstream
        # primitive will resolve this generically (spec §8).
        platform = target.get("platform")
        chat_id = target.get("chat_id")
        chat_type = target.get("chat_type") or "dm"
        session_key = target.get("session_key")
        if not session_key and platform and chat_id:
            session_key = f"agent:main:{platform}:{chat_type}:{chat_id}"
        evt = {
            "session_key": session_key,
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

"""Interim monkey-patch of two upstream-shaped Hermes core primitives (spec §3).

These functions have the exact signatures we intend to upstream as GatewayRunner
methods (with a PluginContext facade). For now the plugin calls them directly with
an explicitly-resolved runner + injected seams (so they unit-test without Hermes).
Everything is fail-closed: nothing here may raise into the gateway.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .domain.egress import ReachOutcome
from .logging import EventLogger, get_logger

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


def reachin_available(runner: Any | None) -> bool:
    """True only if *runner* exposes every attribute inject_proactive_turn needs."""
    if runner is None:
        return False
    return all(hasattr(runner, attr) for attr in _REQUIRED_RUNNER_ATTRS)


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
    logger: EventLogger | None = None,
) -> ReachOutcome:
    """Run a native ``internal=True`` turn on *target* lane. Fail-closed."""
    log = logger or get_logger("lifemodel.reachin")
    if not reachin_available(runner):
        log.info("reachin_unavailable", reason="runner_incomplete")
        return ReachOutcome.UNAVAILABLE
    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        log.info("reachin_unavailable", reason="not_running_or_draining")
        return ReachOutcome.UNAVAILABLE
    try:
        evt = {
            "platform": target.get("platform"),
            "chat_id": target.get("chat_id"),
            "thread_id": target.get("thread_id"),
        }
        source = runner._build_process_event_source(evt)
        if source is None or not getattr(source, "chat_id", None):
            log.info("reachin_unavailable", reason="unknown_lane")
            return ReachOutcome.UNAVAILABLE
        adapter = _select_adapter(runner, source)
        if adapter is None:
            log.info("reachin_unavailable", reason="no_adapter")
            return ReachOutcome.UNAVAILABLE
        event = make_event(prompt, source, message_id)  # message_id None (spec constraint)
        schedule(adapter.handle_message(event), runner._gateway_loop)
        log.info("reachin_injected", chat_id=getattr(source, "chat_id", None))
        return ReachOutcome.DELIVERED
    except Exception as exc:  # noqa: BLE001 - fail-closed, never crash the gateway
        log.info("reachin_failed", error=f"{type(exc).__name__}: {exc}")
        return ReachOutcome.FAILED

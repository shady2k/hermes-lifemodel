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
from .log import EventLogger, get_logger

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


def _spawn_on_loop(loop: Any, coro: Any) -> Any:
    """Schedule *coro* on *loop* whether we're on it or calling from another thread.

    If the calling code is itself running on *loop*, ``create_task`` is the cheap
    in-loop path; otherwise (the common plugin-registration case, which runs off the
    gateway loop) ``run_coroutine_threadsafe`` hands it to the gateway thread. Both
    return a cancellable object the runner can track.
    """
    import asyncio

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None and running is loop:
        return loop.create_task(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop)


def register_gateway_service(
    runner: Any,
    key: str,
    coro_factory: Callable[[], Any],
    *,
    logger: EventLogger | None = None,
) -> bool:
    """Spawn a gateway-owned supervised task. Fail-closed.

    The runner owns the lifecycle: the task is tracked in ``runner._background_tasks``
    so the gateway cancels it on shutdown, and runs on ``runner._gateway_loop``. Any
    spawn failure is logged and degrades to ``False`` rather than raising into the
    gateway (spec §3.2 — a plugin bug must never crash the host).
    """
    log = logger or get_logger("lifemodel.service")
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None:
        # At register()/discovery time the gateway startup is synchronous — no
        # running loop yet, and runner._gateway_loop may be unset. Fall back to
        # the currently running loop (works when this is called in-loop, e.g. from
        # an on_session_start hook after startup completes).
        try:
            import asyncio

            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
    if loop is None:
        log.info("gateway_service_unavailable", key=key, reason="no_loop")
        return False
    try:
        task = _spawn_on_loop(loop, coro_factory())
        bucket = getattr(runner, "_background_tasks", None)
        if isinstance(bucket, set):
            bucket.add(task)
            done = getattr(task, "add_done_callback", None)
            if callable(done):
                done(bucket.discard)
        log.info("gateway_service_started", key=key)
        return True
    except Exception as exc:  # noqa: BLE001 - fail-closed
        log.info("gateway_service_failed", key=key, error=f"{type(exc).__name__}: {exc}")
        return False


def install_core_shim(ctx: Any, *, logger: EventLogger | None = None) -> None:
    """Best-effort: expose the two primitives as PluginContext methods (reusable).

    Monkey-patches ``inject_proactive_turn`` / ``register_gateway_service`` onto
    ``type(ctx)`` so any plugin can call ``ctx.inject_proactive_turn(...)`` with the
    runner resolved via :func:`~lifemodel.adapters.reachin.default_runner_accessor`.
    Purely decorative — never blocks plugin load (spec §7, best-effort shim).
    """
    log = logger or get_logger("lifemodel.shim")
    try:
        from .adapters.reachin import default_runner_accessor

        cls = type(ctx)

        def _ctx_inject(
            self: Any, target: Mapping[str, str | None], prompt: str, **kw: Any
        ) -> ReachOutcome:
            return inject_proactive_turn(default_runner_accessor(), target, prompt, **kw)

        def _ctx_register(self: Any, key: str, coro_factory: Callable[[], Any], **kw: Any) -> bool:
            return register_gateway_service(default_runner_accessor(), key, coro_factory, **kw)

        if not hasattr(cls, "inject_proactive_turn"):
            cls.inject_proactive_turn = _ctx_inject
        if not hasattr(cls, "register_gateway_service"):
            cls.register_gateway_service = _ctx_register
        log.info("core_shim_installed", cls=cls.__name__)
    except Exception as exc:  # noqa: BLE001 - decorative; never block load
        log.info("core_shim_skipped", error=f"{type(exc).__name__}: {exc}")

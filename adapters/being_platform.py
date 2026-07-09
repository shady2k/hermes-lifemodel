"""The being as a gateway-supervised platform adapter (the Hermes boundary).

This is the ONLY module that hosts the autonomic loop, and the only new place
that imports Hermes' adapter surface. It subclasses ``BasePlatformAdapter`` so
the gateway owns its lifecycle: it calls ``connect()`` at startup and — when the
adapter signals a fatal error — its reconnect watcher re-dials ``connect()``.

``connect()`` starts a :class:`SupervisedLoop` that drives the Hermes-free
decision path (``build_lifemodel`` → :func:`proactive_tick`) every interval and
delivers a surfaced launch into the user's real Telegram lane via reach-in. If
the loop dies, :meth:`_on_loop_death` converts that into
``_set_fatal_error(retryable=True)`` + ``_notify_fatal_error()`` — the load-bearing
detail: gateway supervision is notification-based, so a silently-dying task is
invisible without this (the previous failure mode; cf. IRC ``_receive_loop``).

Because it imports ``gateway.*`` at module load, this file is NOT importable
off-host; it is exercised at runtime in the gateway, never by the unit suite.
All of its logic lives in tested Hermes-free units (``core/supervised_loop``,
``core/proactive``).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult

from ..composition import build_lifemodel
from ..core.proactive import proactive_tick
from ..core.supervised_loop import SupervisedLoop
from ..log import EventLogger, get_logger
from .owner_tz import resolve_owner_tz
from .reachin import ReachInEgress, default_runner_accessor

PLATFORM_NAME = "lifemodel"
LOOP_INTERVAL_SEC = 60.0


class BeingAdapter(BasePlatformAdapter):  # type: ignore[misc]  # base is Any (gateway untyped)
    """Hosts the being's autonomic brain as a supervised gateway loop."""

    def __init__(
        self,
        config: Any,
        *,
        base_dir: Path,
        target: dict[str, str | None] | None,
        logger: EventLogger | None = None,
        interval_sec: float = LOOP_INTERVAL_SEC,
    ) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        self._base_dir = base_dir
        self._target: dict[str, str | None] = target or {}
        self._log = logger or get_logger("lifemodel.being")
        self._interval = interval_sec
        self._egress = ReachInEgress(runner_accessor=default_runner_accessor, logger=self._log)
        self._loop: SupervisedLoop | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._shutting_down = False

    def _tick(self) -> None:
        """One brain tick: fresh graph per tick (matches the per-tick invariant)."""
        # Resolve the owner's display timezone from Hermes at the boundary and inject
        # it as a plain stdlib tzinfo (the core stays Hermes-free). Fail-open to
        # None → server-local, so a timezone quirk never drops a tick (HLA §11).
        lm = build_lifemodel(
            base_dir=self._base_dir, logger=self._log, display_tz=resolve_owner_tz()
        )
        proactive_tick(lm, self._egress, self._target, logger=self._log)

    def _on_loop_death(self, exc: BaseException | None) -> None:
        """Convert an unexpected loop death into a gateway-visible fatal error."""
        if self._shutting_down:
            return
        self._log.info("being_loop_died", error=repr(exc))
        self._set_fatal_error("brain_loop_exited", f"proactive loop died: {exc!r}", retryable=True)
        # always on the gateway loop in practice; suppress if somehow off-loop.
        # Track the notify task so a failure to notify (which would strand the
        # reconnect) is at least logged rather than a silent event-loop warning.
        with contextlib.suppress(RuntimeError):
            task = asyncio.get_running_loop().create_task(self._notify_fatal_error())
            task.add_done_callback(self._on_notify_done)

    def _on_notify_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._log.info("being_notify_fatal_failed", error=repr(exc))

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._shutting_down = False
        self._loop = SupervisedLoop(
            tick=self._tick, interval_sec=self._interval, on_death=self._on_loop_death
        )
        self._loop_task = asyncio.create_task(self._loop.run())
        self._mark_connected()
        self._log.info("being_connected", is_reconnect=is_reconnect, interval=self._interval)
        return True

    async def disconnect(self) -> None:
        self._shutting_down = True
        if self._loop is not None:
            self._loop.stop()
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        self._mark_disconnected()  # keep status accurate on a clean stop
        self._log.info("being_disconnected")

    async def send(
        self, chat_id: str, content: str, reply_to: Any = None, metadata: Any = None
    ) -> SendResult:
        # The being's own lane is not a message sink: proactive delivery goes into
        # the user's Telegram lane via reach-in, so a reply routed back here is a
        # no-op. Fail clearly rather than pretend success.
        return SendResult(success=False, error="lifemodel is an internal drive, not a message sink")

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        # The being has no real chats; return a minimal synthetic descriptor.
        return {"id": chat_id, "platform": PLATFORM_NAME, "type": "internal"}


def register_being_platform(
    ctx: Any,
    *,
    base_dir: Path,
    target: dict[str, str | None] | None,
    logger: EventLogger | None = None,
) -> None:
    """Register the being as a gateway platform (call from ``register(ctx)``)."""
    log = logger or get_logger("lifemodel.being")
    ctx.register_platform(
        PLATFORM_NAME,
        label="Life Model",
        adapter_factory=lambda cfg: BeingAdapter(cfg, base_dir=base_dir, target=target, logger=log),
        check_fn=lambda: True,
        emoji="🫀",
    )

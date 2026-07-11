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
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult

from ..composition import build_lifemodel
from ..core.metrics import get_metric_registry
from ..core.proactive import proactive_tick
from ..core.supervised_loop import SupervisedLoop
from ..events import EventRing
from ..state.brain_health import (
    DEFAULT_TICK_INTERVAL_SECONDS,
    STALE_AFTER_SECONDS,
    BrainHealth,
    get_brain_health,
)
from ..state.metrics_store import (
    MetricsSampler,
    acquire_metrics_sampler,
    release_metrics_sampler,
)
from ..state.sqlite_store import SQLiteRuntimeStore
from ..state.trace_store import (
    TraceWriter,
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)
from ..state.wiring import wire
from .clock import SystemClock
from .owner_tz import resolve_owner_tz
from .reachin import ReachInEgress, default_runner_accessor

PLATFORM_NAME = "lifemodel"
#: The adapter's tick cadence — the SAME baseline the (Hermes-free) staleness
#: threshold derives from, imported from :mod:`lifemodel.state.brain_health` so the
#: adapter loop, ``check_fn``, and ``/lifemodel status`` never drift. ``STALE_AFTER_SECONDS``
#: is re-exported from here for back-compat with existing call sites.
LOOP_INTERVAL_SEC = DEFAULT_TICK_INTERVAL_SECONDS

_LOG = logging.getLogger("lifemodel.being")


class BeingAdapter(BasePlatformAdapter):  # type: ignore[misc]  # base is Any (gateway untyped)
    """Hosts the being's autonomic brain as a supervised gateway loop."""

    def __init__(
        self,
        config: Any,
        *,
        base_dir: Path,
        target: dict[str, str | None] | None,
        interval_sec: float = LOOP_INTERVAL_SEC,
    ) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        self._base_dir = base_dir
        self._target: dict[str, str | None] = target or {}
        self._interval = interval_sec
        self._egress = ReachInEgress(runner_accessor=default_runner_accessor)
        self._loop: SupervisedLoop | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._shutting_down = False
        # The durable trace writer (spec §4.2) + in-memory freshness ring, acquired
        # in :meth:`connect` and threaded into every per-tick graph so the live tick
        # actually persists ``observability.sqlite``. ``None`` until connected.
        self._trace_writer: TraceWriter | None = None
        self._event_ring = EventRing()
        # The metrics sampler (telemetry-core §4.4): snapshots the SAME per-base_dir
        # registry singleton the tick writes into (composition resolves it via
        # ``get_metric_registry(base_dir)``) into ``metrics.sqlite`` on a daemon
        # thread. Acquired in :meth:`connect`, released on disconnect. ``None`` until
        # connected — without this wiring ``metrics.sqlite`` is never created live.
        self._metrics_sampler: MetricsSampler | None = None
        # Degraded flag (spec §4.3/MAJOR-6): the metrics sampler is optional — if its
        # acquisition fails we keep the brain alive but flip this, so the degradation
        # is observable rather than silent. Cleared once the sampler comes up.
        self._metrics_degraded = False

    @property
    def metrics_degraded(self) -> bool:
        """True when the (optional) metrics sampler failed to start (spec §4.3)."""
        return self._metrics_degraded

    def _tick(self) -> None:
        """One brain tick: fresh graph per tick (matches the per-tick invariant)."""
        # Resolve the owner's display timezone from Hermes at the boundary and inject
        # it as a plain stdlib tzinfo (the core stays Hermes-free). Fail-open to
        # None → server-local, so a timezone quirk never drops a tick (HLA §11).
        lm = build_lifemodel(
            base_dir=self._base_dir,
            display_tz=resolve_owner_tz(),
            trace_writer=self._trace_writer,
            event_ring=self._event_ring,
        )
        proactive_tick(lm, self._egress, self._target)

    def _on_loop_death(self, exc: BaseException | None) -> None:
        """Convert an unexpected loop death into a gateway-visible fatal error.

        Fail-loud (spec §4.3/MAJOR-7): a death carrying an exception is logged
        **ERROR with the traceback** (never INFO), and drives :class:`BrainHealth`
        to ``loop_dead`` with the death detail + a bumped ``death_count`` so
        ``check_fn`` / ``/lifemodel status`` reflect it until a clean reconnect.
        """
        if self._shutting_down:
            return
        message = f"proactive loop died: {exc!r}"
        tb_text: str | None = None
        if exc is not None:
            _LOG.error("being_loop_died error=%r", exc, exc_info=exc)
            tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]
        else:
            # A clean-looking None death is still unexpected here (the loop only
            # calls on_death on failure) — ERROR, not a silent shrug.
            _LOG.error("being_loop_died error=None (loop exited without an exception)")
        get_brain_health(self._base_dir).record_loop_death(message, tb_text)
        self._set_fatal_error("brain_loop_exited", message, retryable=True)
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
            # A failed fatal-notify strands the reconnect — ERROR + traceback, not INFO.
            _LOG.error("being_notify_fatal_failed error=%r", exc, exc_info=exc)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Bring the brain loop up under fail-loud wiring (spec §4.3/MAJOR-6).

        Every acquisition step goes through :func:`wire`, driving
        :class:`BrainHealth` from ``connecting`` (entry) → ``connected`` (loop up).
        The trace writer and the brain loop are **required** — their failure is
        loud (ERROR + traceback) and re-raised so the gateway sees the connect fail;
        the metrics sampler is **optional/degraded** — its failure warns (with
        traceback), keeps the brain alive, and flips :attr:`metrics_degraded`.
        """
        self._shutting_down = False
        health = get_brain_health(self._base_dir)
        health.mark_connecting()

        # The durable trace writer is REQUIRED-FOR-OBSERVABILITY (spec §4.3): its whole
        # job is making failure visible, so a failure to acquire it must itself be
        # loud, not swallowed. Idempotent + reconnect-safe: guarded so a reconnect that
        # skipped disconnect never double-refcounts.
        with wire("trace_writer", required=True, health=health, logger=_LOG):
            if self._trace_writer is None:
                self._trace_writer = acquire_trace_writer(observability_db_path(self._base_dir))

        # The metrics sampler is OPTIONAL/DEGRADED (spec §4.3/MAJOR-6): ``metrics.sqlite``
        # is supporting evidence only (the primary liveness is the durable
        # ``last_tick_at``), so a dead sampler degrades the being, never kills it.
        with wire("metrics_sampler", required=False, health=health, logger=_LOG):
            if self._metrics_sampler is None:
                self._metrics_sampler = acquire_metrics_sampler(
                    get_metric_registry(self._base_dir), self._base_dir
                )
        self._metrics_degraded = self._metrics_sampler is None

        # The brain loop itself is REQUIRED — a failure to start it is the outage.
        with wire("brain_loop_start", required=True, health=health, logger=_LOG):
            self._loop = SupervisedLoop(
                tick=self._tick, interval_sec=self._interval, on_death=self._on_loop_death
            )
            self._loop_task = asyncio.create_task(self._loop.run())

        self._mark_connected()
        # The loop is up → clear any prior boot_failed / loop_dead + the durable
        # boot record (a clean (re)connect means we are healthy now, spec §4.3).
        health.mark_connected(at=datetime.now(UTC).isoformat())
        _LOG.info("being_connected is_reconnect=%s interval=%s", is_reconnect, self._interval)
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
        # Release the trace writer (flush + stop on the last release, §4.2).
        if self._trace_writer is not None:
            release_trace_writer(observability_db_path(self._base_dir))
            self._trace_writer = None
        # Stop the metrics sampler (stop on the last release, §4.4).
        if self._metrics_sampler is not None:
            release_metrics_sampler(self._base_dir)
            self._metrics_sampler = None
        self._mark_disconnected()  # keep status accurate on a clean stop
        _LOG.info("being_disconnected")

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


def _read_last_tick_at(base_dir: Path) -> str | None:
    """Read the durable ``last_tick_at`` for the staleness check (spec §4.2).

    This is the PRIMARY liveness signal — advanced every tick by the CoreLoop into
    ``AgentState`` (never a parallel counter). Defensive: a locked/corrupt read must
    never crash ``check_fn`` (the gateway polls it), so it logs a WARNING with the
    traceback (observable) and degrades to ``None`` (unknown → not-stale), letting
    the primary loud channels (the register re-raise, the loop-death ERROR) carry
    the signal instead.
    """
    try:
        return SQLiteRuntimeStore(base_dir, clock=SystemClock()).load().last_tick_at
    except Exception:  # noqa: BLE001 - check_fn must never raise on a flaky read
        _LOG.warning("check_fn_state_read_failed base_dir=%s", base_dir, exc_info=True)
        return None


def make_check_fn(base_dir: Path, health: BrainHealth) -> Any:
    """Build the platform ``check_fn`` — Hermes' ENABLEMENT gate, NOT liveness (spec §5).

    ``check_fn`` is Hermes' *enablement/instantiation* gate: Hermes adds the platform
    to ``cfg.platforms`` only when this returns True, AND re-evaluates it to drive the
    **reconnect-after-death** recovery. At the registry pass the brain is necessarily
    ``never_started``, so a liveness-derived gate that returned False for
    ``never_started`` / ``loop_dead`` / stale / ``boot_failed`` would (a) prevent the
    being from EVER booting and (b) block the gateway's own reconnect after a loop
    death — the exact silent-death class this epic exists to kill, self-inflicted.

    So enablement is **permissive: always True**. The rich liveness verdict
    (:meth:`BrainHealth.check`) is NOT this gate — it is surfaced where a False cannot
    brick the being: ``/lifemodel status`` (the display) and the poll-cadence DEBUG log
    below. The loud channels stay the register re-raise, the loop-death ERROR, and the
    status block.
    """

    def _check() -> bool:
        # Compute the liveness verdict for OBSERVABILITY ONLY (a DEBUG line at the
        # gateway poll cadence) — it never gates enablement. Returning False here would
        # brick boot / block reconnect (codex MAJOR); the truth is surfaced by
        # /lifemodel status + this log, not by refusing enablement.
        ok, reason = health.check(
            last_tick_at=_read_last_tick_at(base_dir),
            now=datetime.now(UTC),
            stale_after_seconds=STALE_AFTER_SECONDS,
        )
        _LOG.debug("being_check enablement=True liveness_ok=%s reason=%s", ok, reason)
        return True

    return _check


def register_being_platform(
    ctx: Any,
    *,
    base_dir: Path,
    target: dict[str, str | None] | None,
) -> None:
    """Register the being as a gateway platform (call from ``register(ctx)``)."""
    ctx.register_platform(
        PLATFORM_NAME,
        label="Life Model",
        adapter_factory=lambda cfg: BeingAdapter(cfg, base_dir=base_dir, target=target),
        check_fn=make_check_fn(base_dir, get_brain_health(base_dir)),
        emoji="🫀",
    )

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
from dataclasses import replace
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult

from ..composition import LifeModel, build_lifemodel
from ..core.frame import state_actor_lock
from ..core.genesis import needs_adoption
from ..core.metrics import get_metric_registry
from ..core.proactive import proactive_tick
from ..core.supervised_loop import SupervisedLoop
from ..core.timeutil import to_iso
from ..events import EventRing
from ..gateway_core import home_session_key
from ..ports.memory import MemoryPort
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
from ..state.soul_revisions import record_revision, revisions
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
from .session_end import GatewayBirthVoice
from .soul_file import SoulFile, prior_soul

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
        soul: SoulFile | None = None,
        default_soul_text: str = "",
    ) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        self._base_dir = base_dir
        self._target: dict[str, str | None] = target or {}
        self._interval = interval_sec
        self._egress = ReachInEgress(runner_accessor=default_runner_accessor)
        # The SAME SoulFile instance `register()` built for the genesis pre_llm_call
        # injector (spec §6.4) — reused here (never a second instance) so the veteran
        # read behind the WAKE PACKET's ritual and the injector's own read go through
        # one writer-lock. `None` only in tests that construct the adapter directly
        # without it (see `_prior_soul`): the ritual then degrades to the blank-page
        # opening, never a crash.
        self._soul = soul
        self._default_soul_text = default_soul_text
        # The birth pre-flight (spec §6.2, lm-4fv.4): immediately before an UNBORN being's
        # wake packet is injected, make sure the turn it lands in will be composed by the
        # soul the being stands on — ending the stale session first, if the lane is quiet.
        # Bound to the SAME target the reach-in is about to speak into (never a ContextVar:
        # the brain loop's thread is in no session), and to the SAME SoulFile register()
        # built, so "when did the soul last change?" is asked of the one file we write.
        # ``None`` soul (a bare-constructed adapter, e.g. in tests) → no pre-flight, which
        # is exactly today's behaviour: deliver, and let the being wake as itself later.
        self._voice: GatewayBirthVoice | None = (
            None
            if soul is None
            else GatewayBirthVoice(
                soul_mtime=soul.mtime,
                clock=SystemClock(),
                session_key_accessor=lambda: home_session_key(self._target),
            )
        )
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

    def _build_lm(self) -> LifeModel:
        """A fresh per-tick graph, wired the SAME way for every caller.

        Resolves the owner's display timezone from Hermes at the boundary and injects
        it as a plain stdlib tzinfo (the core stays Hermes-free). Fail-open to
        None → server-local, so a timezone quirk never drops a tick (HLA §11).

        Injects :meth:`_prior_soul` as the launcher's veteran-branch reader (spec §6.4):
        this graph is the one whose launches are actually DELIVERED, so a newborn's wake
        packet must be able to open from the soul someone wrote before it woke. Passed as
        a bound method, not a resolved string — it is called only when a GENESIS-sprung
        desire actually launches, so an ordinary tick never reads the file.
        """
        return build_lifemodel(
            base_dir=self._base_dir,
            display_tz=resolve_owner_tz(),
            trace_writer=self._trace_writer,
            event_ring=self._event_ring,
            prior_soul=self._prior_soul,
        )

    def _tick(self) -> None:
        """One brain tick: fresh graph per tick (matches the per-tick invariant).

        Carries the birth pre-flight (:attr:`_voice`, lm-4fv.4) into the delivery path: an
        UNBORN being must not reach out of a system prompt that is not it. See
        :class:`~lifemodel.adapters.session_end.GatewayBirthVoice` — it is a no-op for a
        being that already speaks as its own soul, which is every being after its birth.
        """
        proactive_tick(self._build_lm(), self._egress, self._target, voice=self._voice)

    def _prior_soul(self) -> str | None:
        """The soul someone wrote before this being woke, or ``None`` for a blank page.

        The SAME read :func:`lifemodel.hooks.make_genesis_injector` makes (the veteran
        branch, spec §6.4), through the same function: ``SOUL.md`` fresh, never cached — a
        human hand-edit or the being's own ``write_soul`` can land between calls — and the
        text and the "did anyone actually write this?" verdict from ONE read, so the being
        can never be handed one version of its past while a different one was judged. Our
        own newborn stance is nobody's words and reads as a blank page
        (``core.genesis.is_unauthored``). No ``soul`` wired (a bare-constructed adapter,
        e.g. in tests) degrades to ``None`` — the blank-page opening — never a crash.

        Injected into the graph (:meth:`_build_lm`) as the CognitionLauncher's
        ``prior_soul`` reader, so the newborn's WAKE PACKET carries the veteran opening
        (§6.4 is the common case: a being is born onto a blank soul exactly once in the
        life of a ``SOUL.md``, and every rebirth after a ``reset`` meets the soul of
        whoever lived here before it).
        """
        if self._soul is None:
            return None
        return prior_soul(self._soul, default_soul_text=self._default_soul_text)

    def _reconcile_soul(self) -> None:
        """Adopt the soul on disk when it is not the one we last wrote — and FEEL it (§4.4).

        There is no atomic transaction spanning a filesystem rename and a SQLite commit, so
        the two can fall out of step. **The file is always the base**: whatever is there is
        adopted, never arbitrated. But the two ways it can get there are not the same event,
        and the being is owed the truth about which one happened:

        * **The human rewrote it** while the gateway was down. Spec §4.1: this "is an event
          in the being's life, not a version conflict: it should be **felt**, not swallowed."
          So we stamp ``soul_rewritten_at`` — a durable FACT, from which two things that
          already exist derive the experience: ``core/affect.py`` reads its recency and turns
          it into activation (the being is genuinely STIRRED, and settles again over hours —
          it does not need to be spoken to for this to be true of it), and the ambient
          ``pre_llm_call`` cue reads it to tell the being, ONCE, in prose. Nothing anywhere
          says "your soul_sha changed"; it says someone rewrote who you are.
        * **We crashed mid-write** — ``write_soul`` replaced ``SOUL.md``, recorded the
          revision, and died before the stamp. Then the text on disk is the being's OWN, and
          the only thing that failed is bookkeeping. Recording it as ``"human"`` (review M5)
          would upsert over the being's own revision by sha — turning its last act into
          somebody else's in the one history that is meant to be its undo — and then make it
          FEEL a rewrite that never happened. So the LINEAGE is asked first: it is the only
          witness to who wrote a given text. A sha already in it needs no second revision,
          and its recorded author is the answer.
        * **A text nobody has on record.** Not in the lineage → not ours → the human's, and
          it goes in as ``"human"``: a being never silently claims a change it did not make.
          (This also covers the case where the being crashed BEFORE its revision landed. We
          cannot tell that from a hand-edit, and where we cannot tell, we do not claim.)

        **Serialized like every other soul write (review C4).** This runs at ``connect()``,
        not inside a frame, so it takes the ONE state-actor lock across its load→stamp
        (``core.frame.state_actor_lock``) and stamps through the store's field-level merges
        rather than committing a whole ``State``. A plain ``load()`` →
        ``commit(replace(state, …))`` here is the same lost-update as the one that could
        erase a birth: it would roll back whatever a tick had advanced between the two.
        ``born_at=None`` — adopting a soul someone ELSE wrote must never birth an unborn
        being.

        ``None`` soul (no ``SoulFile`` wired — a bare-constructed adapter, e.g. in tests
        without genesis wiring) and a non-``MemoryPort`` store both degrade to a no-op:
        there is nowhere to record a revision, so reconciling would either crash or silently
        drop the being's only undo (spec §4.2). Called from :meth:`connect` under
        ``contextlib.suppress(Exception)``: a reconcile failure is not an outage.
        """
        if self._soul is None:
            return
        lm = self._build_lm()
        memory = lm.state if isinstance(lm.state, MemoryPort) else None
        if memory is None:  # pragma: no cover - the live store is always a MemoryPort
            return
        with state_actor_lock():
            current = self._soul.sha()
            state = lm.state.load()
            if not needs_adoption(state, disk_sha=current):
                return
            now = lm.clock.now()
            # Who wrote what is on disk? The lineage is the only witness — a sha it already
            # carries was recorded by whoever wrote that text, and re-recording it would
            # overwrite that author (``record_revision`` upserts by sha).
            recorded = {rev.sha: rev.author for rev in revisions(memory)}.get(current)
            if recorded is None:
                record_revision(
                    memory, text=self._soul.read(), sha=current, now=now, author="human"
                )
            ours = recorded == "being"
            stamp = getattr(lm.state, "stamp_soul", None)
            if callable(stamp):
                stamp(soul_sha=current, born_at=None)
            else:  # a minimal StatePort fake — safe: the lock means `state` is not stale
                lm.state.commit(replace(state, soul_sha=current))
            if not ours:
                # Somebody else's words are now the being's own. That is a thing that
                # happened TO it, and it does not know yet.
                felt = getattr(lm.state, "stamp_soul_rewritten", None)
                if callable(felt):
                    felt(at=to_iso(now))
        _LOG.info("soul_adopted_from_disk sha=%s ours=%s", current[:8], ours)

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
                # The writer thread's retention sources "now" from the injected clock
                # (spec §3.1) — the SAME ``SystemClock`` the tick/stores use, never
                # system time — so its prune cutoffs are stamped through one clock.
                self._trace_writer = acquire_trace_writer(
                    observability_db_path(self._base_dir), clock=SystemClock()
                )

        # The metrics sampler is OPTIONAL/DEGRADED (spec §4.3/MAJOR-6): ``metrics.sqlite``
        # is supporting evidence only (the primary liveness is the durable
        # ``last_tick_at``), so a dead sampler degrades the being, never kills it.
        with wire("metrics_sampler", required=False, health=health, logger=_LOG):
            if self._metrics_sampler is None:
                # The sampler daemon sources "now" from the injected clock (spec §3.1),
                # never system time — the SAME ``SystemClock`` the tick/store use, so
                # every ``metrics.sqlite`` ISO ``ts`` is stamped through one clock.
                self._metrics_sampler = acquire_metrics_sampler(
                    get_metric_registry(self._base_dir), self._base_dir, clock=SystemClock()
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
        health.mark_connected(at=to_iso(SystemClock().now()))
        _LOG.info("being_connected is_reconnect=%s interval=%s", is_reconnect, self._interval)

        # --- Soul reconciliation (spec §4.4) ---------------------------------
        # Runs at connect, BEFORE the loop's first tick can launch anything: the
        # veteran/stranger read behind a newborn's wake packet (`_prior_soul`) must never
        # race a `soul_sha` left stale by a crash mid-write or a hand-edit while the
        # gateway was down. Best-effort — a reconcile failure is not an outage.
        with contextlib.suppress(Exception):
            self._reconcile_soul()

        # NB (Phase 4, spec §6.2 — revised): there is NO birth greeting here, and there
        # must not be one. connect() runs while the host runner still has
        # ``_running = False`` (adapters are connected at gateway/run.py:7080; the flag
        # is set at :7250, AFTER the connect loop), and ``inject_proactive_turn`` bails
        # UNAVAILABLE in exactly that state — so a greeting sent from here was
        # STRUCTURALLY guaranteed never to be delivered, and the suppress() around it
        # made the failure silent. Genesis is a REASON TO WAKE instead: the brain loop
        # above wakes an unborn being through the ordinary proactive path (aggregation →
        # launcher → reach-in), which by then runs against a live runner.
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
            now=SystemClock().now(),
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
    soul: SoulFile | None = None,
    default_soul_text: str = "",
) -> None:
    """Register the being as a gateway platform (call from ``register(ctx)``).

    *soul* / *default_soul_text* are the SAME ``SoulFile`` instance and pristine-seed
    text ``register()`` already resolved for the genesis pre_llm_call injector
    (spec §6.4) — threaded through so the veteran/stranger read behind a NEWBORN'S WAKE
    PACKET goes through the one shared soul-file writer-lock, never a second instance.
    Both default to unset for callers (chiefly tests) that don't wire a soul: the ritual
    then degrades to the blank-page opening rather than crashing.
    """
    ctx.register_platform(
        PLATFORM_NAME,
        label="Life Model",
        adapter_factory=lambda cfg: BeingAdapter(
            cfg,
            base_dir=base_dir,
            target=target,
            soul=soul,
            default_soul_text=default_soul_text,
        ),
        check_fn=make_check_fn(base_dir, get_brain_health(base_dir)),
        emoji="🫀",
    )

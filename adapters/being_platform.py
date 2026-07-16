"""The being as a gateway-supervised platform adapter (the Hermes boundary).

This is the ONLY module that hosts the autonomic loop, and the only new place
that imports Hermes' adapter surface. It subclasses ``BasePlatformAdapter`` so
the gateway owns its lifecycle: it calls ``connect()`` at startup and — when the
adapter signals a fatal error — its reconnect watcher re-dials ``connect()``.

``connect()`` starts a :class:`SupervisedLoop` that drives the Hermes-free
decision path (``build_lifemodel`` → one ``ExecutionFrame`` →
:func:`~lifemodel.core.proactive.dispatch_launches`) every interval and
delivers a surfaced launch into the user's real Telegram lane via reach-in. If
the loop dies, :meth:`_on_loop_death` converts that into
``_set_fatal_error(retryable=True)`` + ``_notify_fatal_error()`` — the load-bearing
detail: gateway supervision is notification-based, so a silently-dying task is
invisible without this (the previous failure mode; cf. IRC ``_receive_loop``).

The SAME tick also owns the non-delivered internal-cognition seam (lm-705.6,
design §3.1): any ``LaunchInternalCognition`` the frame surfaced is handed to
the adapter-owned :class:`~lifemodel.adapters.internal_runner.InternalCognitionRunner`,
built here (on the gateway loop `connect()` itself already runs on) and torn
down in :meth:`disconnect`. This bead ships no live emitter of that intent yet
(noticing/processing do, later) — the wiring exists so the seam is exercised
end-to-end without waiting for its first real consumer.

Because it imports ``gateway.*`` at module load, this file is NOT importable
off-host; it is exercised at runtime in the gateway, never by the unit suite.
All of its logic lives in tested Hermes-free units (``core/supervised_loop``,
``core/proactive``, ``core/internal_cognition``, ``adapters/internal_runner``).
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
from ..core.frame import FrameTrigger, run_frame, state_actor_lock
from ..core.genesis import needs_adoption
from ..core.internal_cognition import NullInternalApply
from ..core.llm_port import InternalCognitionRequest, LlmPort
from ..core.metrics import get_metric_registry
from ..core.proactive import dispatch_launches
from ..core.supervised_loop import SupervisedLoop
from ..core.timeutil import to_iso
from ..events import EventRing
from ..gateway_core import home_session_key
from ..ports.memory import MemoryPort
from ..state.brain_health import DEFAULT_TICK_INTERVAL_SECONDS, get_brain_health
from ..state.metrics_store import (
    MetricsSampler,
    acquire_metrics_sampler,
    release_metrics_sampler,
)
from ..state.soul_revisions import record_revision, revisions
from ..state.trace_store import (
    TraceWriter,
    acquire_trace_writer,
    observability_db_path,
    release_trace_writer,
)
from ..state.wiring import wire
from .clock import SystemClock
from .internal_runner import InternalCognitionRunner
from .owner_tz import resolve_owner_tz
from .reachin import ReachInEgress, default_runner_accessor
from .session_end import GatewayBirthVoice
from .soul_file import SoulFile, prior_soul

PLATFORM_NAME = "lifemodel"
#: The adapter's tick cadence — imported from :mod:`lifemodel.state.brain_health` (the
#: single source the loop interval and the ``/lifemodel status`` staleness threshold both
#: derive from) so the two never drift.
LOOP_INTERVAL_SEC = DEFAULT_TICK_INTERVAL_SECONDS

#: FR20's v1 default (design §3.4 — "simplest reliable form", no product-specified
#: number yet). A generous-but-real ceiling: idle ticks stay 0-LLM (S5) since
#: nothing emits ``LaunchInternalCognition`` without a real trigger (lm-705.5), so
#: this only bounds the SPIKE once a trigger exists. Tune once noticing/processing
#: are live and the actual call cadence is known.
DEFAULT_DAILY_INTERNAL_CALL_CEILING = 50

#: The internal-cognition pass's fixed system framing (design §3.3) — content-free
#: on purpose: THIS bead's own consumer is
#: :class:`~lifemodel.core.internal_cognition.NullInternalApply` (ignores the
#: result), so the instructions only need to keep a real call honest about
#: non-delivery; noticing/processing (lm-705.5/.2) will pass their own.
_INTERNAL_COGNITION_INSTRUCTIONS = (
    "This is the being's own private, non-delivered internal thinking. Nothing you "
    "produce here is shown to anyone."
)

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
        llm: LlmPort | None = None,
    ) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        self._base_dir = base_dir
        self._target: dict[str, str | None] = target or {}
        self._interval = interval_sec
        self._egress = ReachInEgress(runner_accessor=default_runner_accessor)
        # The internal-cognition seam's LlmPort (lm-705.6) — the real adapter
        # (`PluginLlmPort` over `ctx.llm`) wired by `register()`. `None` degrades
        # the seam off cleanly (no runner built in `connect()`, see there): a
        # bare-constructed adapter (every existing test) or a host build without
        # `ctx.llm` still boots and ticks exactly as before this bead.
        self._llm = llm
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
        # The internal-cognition seam's runner (lm-705.6) — built in :meth:`connect`
        # ONLY when a LlmPort was injected (``self._llm is not None``); ``None``
        # otherwise, and :meth:`_tick`/:meth:`disconnect` both guard on that so the
        # seam degrades off cleanly rather than ever blocking the proactive brain loop.
        self._internal_runner: InternalCognitionRunner | None = None

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

        Runs ONE ``ExecutionFrame`` (never :func:`~lifemodel.core.proactive.proactive_tick`
        directly, lm-705.6): the SAME frame's report feeds BOTH the proactive delivery
        path (:func:`~lifemodel.core.proactive.dispatch_launches`, behavior-identical to
        the old ``proactive_tick`` call) and the internal-cognition seam
        (``report.internal_launches`` →
        :meth:`~lifemodel.adapters.internal_runner.InternalCognitionRunner.launch`) —
        a second ``run_frame`` call here would double-tick (a second bookkeeping bump,
        a second energy/fatigue recovery pass).
        """
        lm = self._build_lm()
        assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
        report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
        dispatch_launches(lm, report, self._egress, self._target, voice=self._voice)
        if self._internal_runner is not None:
            for launch in report.internal_launches:
                self._internal_runner.launch(
                    InternalCognitionRequest(
                        instructions=_INTERNAL_COGNITION_INSTRUCTIONS,
                        input_text=launch.prompt,
                    ),
                    launch.correlation_id,
                )

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
        ``/lifemodel status`` reflects it until a clean reconnect.
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

        # The internal-cognition seam's runner (lm-705.6) is OPTIONAL/DEGRADED, like
        # the metrics sampler above: it is not yet exercised by any live emitter
        # (noticing/processing land later) and must never be able to block boot or
        # the proactive brain loop. Built + recovered BEFORE `brain_loop_start`
        # below creates the loop's task — `connect()` has no `await` before that
        # point, so the loop's first tick genuinely cannot run until this whole
        # method returns (mirrors the soul-reconciliation ordering note further
        # down). `self._llm is None` (no LlmPort injected — an off-host/bare test
        # construction, or a host build with no `ctx.llm`) skips it entirely: the
        # being ticks exactly as it did before this bead.
        with wire("internal_cognition_runner", required=False, health=health, logger=_LOG):
            if self._internal_runner is None and self._llm is not None:
                self._internal_runner = InternalCognitionRunner(
                    self._build_lm,
                    self._llm,
                    self._egress,
                    self._target,
                    daily_ceiling=DEFAULT_DAILY_INTERNAL_CALL_CEILING,
                    gateway_loop=asyncio.get_running_loop(),
                    apply=NullInternalApply(),
                )
                self._internal_runner.recover_stale(self._build_lm())

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
        # Cancel + await every in-flight internal-cognition call (lm-705.6) — AFTER
        # the main loop is stopped (no new launches) and BEFORE the trace
        # writer/metrics sampler release below, so a task's fail-loud logging still
        # has a live sink for however briefly it runs after cancellation.
        if self._internal_runner is not None:
            await self._internal_runner.cancel_all()
            # Null it (mirroring the trace_writer / metrics_sampler releases below) so a
            # reconnect-WITH-disconnect rebuilds a fresh runner bound to the new gateway
            # loop, while the connect-side ``is None`` guard makes a reconnect that SKIPS
            # disconnect keep the EXISTING runner + its still-live tasks — never orphaning
            # them (they were never cancel_all'd on that path).
            self._internal_runner = None
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


def make_check_fn() -> Any:
    """The platform ``check_fn`` — Hermes' ENABLEMENT gate, always True, reads NOTHING.

    ``check_fn`` is Hermes' *enablement/instantiation* predicate ("are my dependencies
    available?"), re-evaluated to drive **reconnect-after-death** recovery — NOT a liveness
    probe. Ours is unconditionally True: a False would (a) prevent the being from EVER
    booting — at the registry pass the brain is necessarily ``never_started`` — and (b)
    block the gateway's own reconnect after a loop death, the exact silent-death class this
    epic exists to kill, self-inflicted.

    **It reads no state, on purpose (lm-54i).** An earlier version read ``last_tick_at`` for
    a DEBUG liveness line. But ``register()`` runs this gate into EVERY plugin-hosting
    process — including a ``hermes serve`` that never ticks — and Hermes polls it there too.
    ``make deploy`` restarts only the gateway, so a stale ``serve`` kept polling; once the
    fresh gateway had migrated the store forward, its ``.load()`` raised ``StateSchemaError``
    every ~15s (813 ``check_fn_state_read_failed`` WARNINGs in one incident) against a DB it
    does not own. An enablement gate has no business reading the being's state. The rich
    liveness verdict (:meth:`BrainHealth.check`) lives where a stale reader cannot reach it
    and a False cannot brick the being: ``/lifemodel status``, ``metrics.sqlite``, and the
    gateway-owned loop-death ERROR.
    """

    def _check() -> bool:
        return True

    return _check


def register_being_platform(
    ctx: Any,
    *,
    base_dir: Path,
    target: dict[str, str | None] | None,
    soul: SoulFile | None = None,
    default_soul_text: str = "",
    llm: LlmPort | None = None,
) -> None:
    """Register the being as a gateway platform (call from ``register(ctx)``).

    *soul* / *default_soul_text* are the SAME ``SoulFile`` instance and pristine-seed
    text ``register()`` already resolved for the genesis pre_llm_call injector
    (spec §6.4) — threaded through so the veteran/stranger read behind a NEWBORN'S WAKE
    PACKET goes through the one shared soul-file writer-lock, never a second instance.
    Both default to unset for callers (chiefly tests) that don't wire a soul: the ritual
    then degrades to the blank-page opening rather than crashing.

    *llm* (lm-705.6) is the internal-cognition seam's :class:`~lifemodel.core.llm_port.LlmPort`
    — ``register()`` resolves the real :class:`~lifemodel.adapters.plugin_llm_adapter.PluginLlmPort`
    over ``ctx.llm`` and passes it through; ``None`` (the default, and every caller
    that doesn't wire one) leaves the seam degraded off in :meth:`BeingAdapter.connect`.
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
            llm=llm,
        ),
        check_fn=make_check_fn(),
        emoji="🫀",
    )

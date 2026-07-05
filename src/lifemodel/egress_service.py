"""In-process proactive tick — the sole decision brain (spec §3.2/§5/§6).

:func:`run_proactive_tick` is the delivery-aware wake path: each tick it
reconstructs the certified desire model from persisted ``State`` via
:func:`~lifemodel.core.decision.decide_reachout` (the live adapter over
``lifemodel.sim``) and, on a clean wake, launches a native proactive turn
through the injected :class:`~lifemodel.ports.proactive.ProactiveEgressPort`.

**The verdict (fulfill/reject) is deliberately NOT decided here.** A wake only
means "cognition should compose a turn now" — whether that turn actually says
something (FULFILL) or has nothing to add (REJECT) can only be known once the
LLM's final output comes back, which arrives later via the ``post_llm_call``
observer (a future task) calling
:func:`~lifemodel.core.decision.apply_verdict`. So a successful launch here
records a *pending* proactive id + timestamp and leaves the desire ``active`` —
neither satiated nor rejected — until that observer resolves it. A launch that
does **not** reach ``DELIVERED`` (busy / unavailable / failed) is rolled back
immediately: the desire returns to ``none`` and the pending id is cleared, with
no reject recorded, so the next tick's urge is free to retry rather than being
stranded mid-flight or wrongly penalized by the growing backoff.

The supervised ``proactive_service_loop`` that drives this on a timer landed in
task 7 (busy centralization) — see its own docstring below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from .composition import LifeModel
from .core.decision import THETA, decide_reachout
from .domain.egress import ReachOutcome
from .domain.wake import WakePacket
from .gateway_core import reachin_available
from .impulse import compose_impulse
from .logging import EventLogger
from .ports.proactive import ProactiveEgressPort


def run_proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
    busy: bool = False,
) -> ReachOutcome:
    """One in-process proactive tick — decide via ``core/decision``, launch on wake.

    1. Load ``state`` + ``now`` and hand them to :func:`decide_reachout`, which
       rises the drive by elapsed silence, evaluates the wake gates, and mutates
       ``state`` in place (it is the single source of truth for the drive/gate
       logic — this function never reimplements it).
    2. On a wake: stamp a fresh ``pending_proactive_id``/``pending_proactive_since``
       and call ``egress.reach_out``. Anything short of ``DELIVERED`` rolls the
       desire back to ``none`` and clears the pending id — the turn never
       launched, so there is nothing pending to await a verdict for, and no
       reject is recorded (that would wrongly arm the growing backoff for a
       launch failure that was never cognition's call to make).
    3. Regardless of wake/outcome: stamp the liveness marker and commit once.

    ``outcome`` communicates only whether a turn was launched this tick — the
    ``ReachOutcome`` returned by ``egress.reach_out`` on a wake, or
    ``SKIPPED_BUSY`` as the generic "no reach-out attempted" sentinel when
    ``decide_reachout`` did not wake (whatever its reason — below threshold,
    in flight, silence window, or decline backoff).
    """
    state = lm.state.load()
    now = lm.clock.now()

    decision = decide_reachout(state, now=now, busy=busy)

    outcome = ReachOutcome.SKIPPED_BUSY  # default: no reach-out attempted this tick
    if decision.wake:
        state.pending_proactive_id = f"p-{state.tick_count}-{now.isoformat()}"
        state.pending_proactive_since = now.isoformat()

        last_contact = (
            datetime.fromisoformat(state.last_contact_at)
            if state.last_contact_at is not None
            else None
        )
        packet = WakePacket(
            reason=decision.reason, pressure_kind="urge", pressure=state.u, threshold=THETA
        )
        impulse = compose_impulse(packet, now=now, last_contact_at=last_contact)
        outcome = egress.reach_out(target, impulse)

        if outcome is not ReachOutcome.DELIVERED:
            # The turn never launched — roll back so next tick's urge can retry.
            # No reject: REJECT is cognition's verdict on a turn that DID run and
            # had nothing to say, not a launch failure at the egress layer.
            state.desire_status = "none"
            state.pending_proactive_id = None
            state.pending_proactive_since = None
            logger.info("proactive_launch_failed", outcome=outcome.value)

    state.tick_count += 1
    state.egress_service_alive_at = now.isoformat()  # liveness stamp (lm-64s, spec §6)
    lm.state.commit(state)
    logger.info("proactive_tick", wake=decision.wake, reason=decision.reason, outcome=outcome.value)
    return outcome


async def proactive_service_loop(
    *,
    build_lm: Callable[[], LifeModel],
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    runner_accessor: Callable[[], Any | None],
    logger: EventLogger,
    interval_seconds: float = 60.0,
) -> None:
    """Supervised in-process brain: tick every interval until shutdown.

    Waits for the gateway to finish starting (``_running`` True / adapters wired),
    then loops: each *interval_seconds* it runs one :func:`run_proactive_tick`
    (marking the session busy if the gateway has an active turn) and stamps
    liveness. It self-guards on ``_running``/``_draining`` and exits cleanly on
    shutdown; a tick error is logged and swallowed so one bad tick can't kill the
    loop (spec §3.2 — runner-owned, isolated, cancellable).
    """
    import asyncio

    # Wait for the gateway to finish starting (adapters wired, _running True).
    for _ in range(600):  # ~5 min cap; then proceed best-effort
        runner = runner_accessor()
        if runner is not None and getattr(runner, "_running", False):
            break
        if runner is not None and getattr(runner, "_draining", False):
            return
        await asyncio.sleep(0.5)

    logger.info("proactive_service_loop_started", interval=interval_seconds)
    while True:
        runner = runner_accessor()
        if (
            runner is None
            or getattr(runner, "_draining", False)
            or not getattr(runner, "_running", False)
        ):
            logger.info("proactive_service_loop_stop")
            return
        if not reachin_available(runner):
            # Reach-in not usable right now (version drift / adapters not wired):
            # yield to the cron fallback by NOT ticking and NOT stamping liveness,
            # so its freshness check sees us as absent and it takes over as the
            # brain (spec §6 — avoids the deadlock where we defer cron but can't
            # deliver ourselves).
            logger.info("proactive_yield_to_cron")
            await asyncio.sleep(interval_seconds)
            continue
        # INTERIM: no busy-skip. runner._running_agents stays truthy while a session
        # is merely OPEN (not actively mid-turn), so it wrongly reported "busy" on
        # every tick and blocked delivery entirely. The reach-in primitive is robust
        # to an active turn (Hermes merge/FIFO semantics), so we inject regardless.
        # A precise per-session in-flight check belongs to the upstream primitive
        # (spec §5/§8).
        busy = False
        try:
            run_proactive_tick(build_lm(), egress, target, logger=logger, busy=busy)
        except Exception as exc:  # noqa: BLE001 - a tick error must not kill the loop
            logger.info("proactive_tick_error", error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(interval_seconds)

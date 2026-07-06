"""In-process proactive tick — drives the layered pipeline (spec §13/§14, model A).

:func:`run_proactive_tick` is the delivery-aware wake path: each tick it calls
``lm.coreloop.tick()`` (which runs personality→neuron→aggregation→cognition and
commits state via the single state-actor). If the pipeline surfaces a
``LaunchProactive`` intent, the tick applies the **global backstop**
(``core.backstop.allow_send``) — a fail-closed rate limit (spec §14) — and, if
allowed, injects the being's native proactive turn via
``egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)``. A blocked
launch holds the desire (``deferred``); a failed launch rolls pending back
(``active`` to retry). Liveness is always stamped in one reconciliation commit.

The supervised ``proactive_service_loop`` that drives this on a timer owns the
``busy`` computation (now inert — in-flight is a signal concern).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .composition import LifeModel
from .core.backstop import allow_send
from .core.wake_packet import IMPULSE_LABEL_PREFIX
from .domain.egress import ReachOutcome
from .gateway_core import reachin_available
from .log import EventLogger
from .ports.proactive import ProactiveEgressPort

#: The supervised proactive loop's tick cadence (spec §3.2/§6). The single
#: source of truth for "~once a minute" — the liveness watchdog in
#: :mod:`lifemodel.tick` sizes ``SERVICE_LIVENESS_MAX_AGE`` against this, and the
#: debug dump (spec §3.3 drift-owner table) imports it so the displayed interval
#: can never drift from the real cadence.
PROACTIVE_LOOP_INTERVAL_SEC = 60.0


def run_proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
    busy: bool = False,  # retained for the loop's call shape; in-flight is a signal now
) -> ReachOutcome:
    """One in-process proactive tick — run the layered pipeline, launch on a
    surfaced desire, gated by the global backstop (spec §13/§14, model A).

    Assumes a fresh ``LifeModel`` per call (the supervised loop builds one each
    tick); the reconciliation commit is a host concern outside the state-actor,
    safe under that invariant.
    """
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    report = lm.coreloop.tick()  # pipeline runs + state committed by the state-actor
    now = lm.clock.now()

    outcome = ReachOutcome.SKIPPED_BUSY
    rollback_status: str | None = None
    refund_energy = 0.0
    if report.launches:
        state = lm.state.load()
        launch = report.launches[0]
        if not allow_send(state.proactive_send_log, now):
            rollback_status = "deferred"  # backstop: hold the desire, send nothing (spec §14)
            refund_energy = launch.reserved_energy  # no turn ran -> refund the reservation
            logger.info("proactive_backstop_blocked")
        else:
            outcome = egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)
            if outcome is not ReachOutcome.DELIVERED:
                rollback_status = "active"  # launch failed — keep active to retry
                refund_energy = launch.reserved_energy  # no turn ran -> refund
                logger.info("proactive_launch_failed", outcome=outcome.value)

    # one reconciliation commit: liveness stamp + optional pending rollback + reservation refund
    state = lm.state.load()
    state.egress_service_alive_at = now.isoformat()
    if rollback_status is not None:
        state.pending_proactive_id = None
        state.pending_proactive_since = None
        state.desire_status = rollback_status
        state.energy += refund_energy
    lm.state.commit(state)
    logger.info("proactive_tick", launches=len(report.launches), outcome=outcome.value)
    return outcome


async def proactive_service_loop(
    *,
    build_lm: Callable[[], LifeModel],
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    runner_accessor: Callable[[], Any | None],
    logger: EventLogger,
    interval_seconds: float = PROACTIVE_LOOP_INTERVAL_SEC,
) -> None:
    """Supervised in-process brain: tick every interval until shutdown.

    Waits for the gateway to finish starting (``_running`` True / adapters wired),
    then loops: each *interval_seconds* it computes ``busy`` ONCE from the
    runner — the single accurate source for the no-wake-in-flight gate (task 7;
    the delivery adapter no longer second-guesses it) — and runs one
    :func:`run_proactive_tick` with it, then stamps liveness. It self-guards on
    ``_running``/``_draining`` and exits cleanly on shutdown; a tick error is
    logged and swallowed so one bad tick can't kill the loop (spec §3.2 —
    runner-owned, isolated, cancellable).
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
        # The ONE busy computation (task 7, HLA/spec RC2): decide_reachout's
        # no-wake-in-flight gate needs exactly one accurate signal, computed
        # here and threaded down — the delivery adapter (ReachInEgress) no
        # longer second-guesses it. `runner._running_agents` was tried and
        # rejected: it stays truthy while a session is merely OPEN (not
        # actively mid-turn), so it wrongly reported "busy" on every tick and
        # blocked delivery entirely. Until the upstream primitive exposes a
        # precise per-session in-flight signal (spec §5/§8), the accurate
        # answer is "never veto on this basis" — hence always False.
        busy = False
        try:
            run_proactive_tick(build_lm(), egress, target, logger=logger, busy=busy)
        except Exception as exc:  # noqa: BLE001 - a tick error must not kill the loop
            logger.info("proactive_tick_error", error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(interval_seconds)

"""The Hermes-free proactive tick — decide, then deliver through a port (§13/§14).

:func:`proactive_tick` is the being's delivery-aware wake path, called once per
loop tick by the platform adapter. It drives ``lm.coreloop.tick()`` (the layered
pipeline: personality → neuron → aggregation → cognition, committed by the single
state-actor). If the pipeline surfaces a ``LaunchProactive`` intent, it applies the
global backstop (:func:`core.backstop.allow_send`, fail-closed rate limit) and, if
allowed, injects the being's native proactive turn via
``egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)``. A blocked
launch holds the desire (``deferred``); a failed launch rolls pending back to
``active`` to retry. Either rollback refunds the reserved energy in one
reconciliation commit.

Hermes-free: it talks only to the injected ``ProactiveEgressPort`` and the core
graph, so it unit-tests with fakes off-host. It stamps NO liveness — ``last_tick_at``
(the dt clock) is stamped by ``coreloop.tick()``, and its freshness IS the liveness
signal the debug HEALTH view reads.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..composition import LifeModel
from ..domain.egress import ReachOutcome
from ..log import EventLogger
from ..ports.proactive import ProactiveEgressPort
from .backstop import allow_send
from .wake_packet import IMPULSE_LABEL_PREFIX


def proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
) -> ReachOutcome:
    """Run one proactive tick: pipeline → backstop → deliver. Never raises past
    the injected collaborators. Assumes a fresh ``LifeModel`` per call (the loop
    builds one each tick), so the rollback reconciliation commit is safe."""
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    report = lm.coreloop.tick()  # pipeline runs + state committed by the state-actor
    now = lm.clock.now()

    if not report.launches:
        logger.info("proactive_tick", launches=0, outcome=ReachOutcome.SKIPPED_BUSY.value)
        return ReachOutcome.SKIPPED_BUSY

    launch = report.launches[0]
    state = lm.state.load()
    outcome = ReachOutcome.SKIPPED_BUSY
    rollback_status: str | None = None

    if not allow_send(state.proactive_send_log, now):
        rollback_status = "deferred"  # backstop: hold the desire, send nothing (spec §14)
        logger.info("proactive_backstop_blocked")
    else:
        outcome = egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)
        if outcome is not ReachOutcome.DELIVERED:
            rollback_status = "active"  # launch failed — keep active to retry
            logger.info("proactive_launch_failed", outcome=outcome.value)

    if rollback_status is not None:
        # No turn ran -> refund the energy reservation and roll pending back.
        state = lm.state.load()
        state.pending_proactive_id = None
        state.pending_proactive_since = None
        state.desire_status = rollback_status
        state.energy += launch.reserved_energy
        lm.state.commit(state)

    logger.info("proactive_tick", launches=len(report.launches), outcome=outcome.value)
    return outcome

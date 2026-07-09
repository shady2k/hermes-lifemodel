"""The Hermes-free proactive tick — decide, then deliver through a port (§13/§14).

:func:`proactive_tick` is the being's delivery-aware wake path, called once per
loop tick by the platform adapter. It drives ``lm.coreloop.tick()`` (the layered
pipeline: personality → neuron → aggregation → cognition, committed by the single
state-actor). If the pipeline surfaces a ``LaunchProactive`` intent, it applies the
global backstop (:func:`core.backstop.allow_send`, fail-closed rate limit) and, if
allowed, injects the being's native proactive turn via
``egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)``.

Launch ≠ fulfilment: a delivered launch leaves the contact desire ``active`` with
``pending_proactive_id`` set (the FULFILL/REJECT verdict resolves it next tick). A
**blocked** launch holds the desire (``active → deferred``); a **failed** launch
keeps it ``active`` to retry. Either rollback clears ``pending`` and refunds the
reserved energy in ONE atomic commit through the state-actor (a bus
``TransitionRecord`` + ``UpdateState`` committed by ``commit_tick``) — never an
out-of-band ``state.commit`` that could split State from the desire row. Both
rollback edges are legal (``active→deferred`` is non-terminal; delivery-fail makes
no transition), so the terminal-state guard is never tripped.

Hermes-free: it talks only to the injected ``ProactiveEgressPort`` and the core
graph, so it unit-tests with fakes off-host. It stamps NO liveness — ``last_tick_at``
(the dt clock) is stamped by ``coreloop.tick()``, and its freshness IS the liveness
signal the debug HEALTH view reads.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..composition import LifeModel
from ..domain.egress import ReachOutcome
from ..domain.memory import TransitionOp
from ..domain.objects import CONTACT_DESIRE_ID, CONTACT_INTENTION_ID, DesireState, IntentionState
from ..log import EventLogger
from ..ports.memory import MemoryPort
from ..ports.proactive import ProactiveEgressPort
from .backstop import allow_send
from .desire_view import read_live_contact_desire
from .intention_view import read_live_contact_intention
from .intents import Intent, TransitionRecord, UpdateState
from .state_actor import StateActor
from .suppression import SuppressionReason, emit_suppression_span
from .wake_packet import IMPULSE_LABEL_PREFIX


def proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
) -> ReachOutcome | None:
    """Run one proactive tick: pipeline → backstop → deliver. Never raises past
    the injected collaborators. Assumes a fresh ``LifeModel`` per call (the loop
    builds one each tick), so the rollback reconciliation commit is safe.

    Returns a :class:`ReachOutcome` ONLY for a real delivery attempt (delivered /
    failed / unavailable — the egress boundary, spec §9). A QUIET tick returns
    ``None``: either no launch was produced (the core stayed quiet — its reason is
    already a suppression span logged in-tick by aggregation/CognitionLauncher) or
    the backstop held the launch (logged here as a ``backstop_rate_limited``
    suppression span). ``DELIVERED`` means the turn was queued — whether the being
    actually spoke is the async ``proactive_outcome`` read-back, not this return."""
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
    report = lm.coreloop.tick()  # pipeline runs + state committed by the state-actor

    # No reach attempted — the core stayed quiet. This is a core DECISION, not a
    # delivery outcome: aggregation/CognitionLauncher already logged the reason as a
    # suppression span during the tick (spec §5). No egress outcome to report.
    if not report.launches:
        logger.info("proactive_tick", launches=0, outcome=None)
        return None

    launch = report.launches[0]
    actor = lm.state_actor
    state = actor.state  # post-tick committed state (energy already reserved by cognition)
    now = lm.clock.now()

    if not allow_send(state.proactive_send_log, now):
        # Backstop held fire (fail-closed rate limit): a core decision to HOLD, not a
        # delivery attempt. Log the reason as a suppression span, defer the desire,
        # and return None (quiet — no egress outcome).
        _emit_backstop_suppression(lm, logger=logger, tick=state.tick_count)
        logger.info("proactive_backstop_blocked")
        _rollback(lm, actor, launch.reserved_energy, defer=True)  # hold: active -> deferred
        logger.info("proactive_tick", launches=len(report.launches), outcome=None)
        return None

    full_prompt = IMPULSE_LABEL_PREFIX + launch.prompt
    # The owner's core observability ask (lm-j2w B3): "I want to know what
    # we handed the agent." DEBUG-only (a no-op unless the owner has set
    # loglevel=debug) and carries the COMPLETE, untruncated text — byte-
    # identical to what egress.reach_out is about to receive — plus the
    # correlation_id shared with the outcome verdict logged in hooks.py.
    logger.debug("proactive_prompt", correlation_id=launch.correlation_id, prompt=full_prompt)
    outcome = egress.reach_out(target, full_prompt)
    if outcome is not ReachOutcome.DELIVERED:
        logger.info("proactive_launch_failed", outcome=outcome.value)
        _rollback(lm, actor, launch.reserved_energy, defer=False)  # keep active, retry

    logger.info("proactive_tick", launches=len(report.launches), outcome=outcome.value)
    return outcome


def _emit_backstop_suppression(lm: LifeModel, *, logger: EventLogger, tick: int) -> None:
    """Log a backstop-held launch as a ``backstop_rate_limited`` suppression span.

    The egress runs out-of-tick (after ``coreloop.tick()`` returns), so there is no
    in-tick span to bind to; mint a fresh root span on the graph's tracer so the
    suppression still carries the §5 minimum (``reason``/``component``/``trace_id``/
    ``span_id``/``tick``). Best-effort: a graph without a tracer (a hand-built
    test ``LifeModel``) skips the span — the ``proactive_backstop_blocked`` log line
    still records the hold."""
    tracer = lm.tracer
    if tracer is None:
        return
    span = tracer.start_root()
    emit_suppression_span(
        logger=logger,
        reason=SuppressionReason.BACKSTOP_RATE_LIMITED,
        component="proactive",
        span=span,
        tick=tick,
    )


def _rollback(lm: LifeModel, actor: StateActor, reserved_energy: float, *, defer: bool) -> None:
    """Atomically undo a launch that did not deliver — clear pending + refund
    energy, and (on a backstop block) hold BOTH the desire AND the intention
    ``active → deferred`` in lockstep.

    One atomic ``commit_tick`` via the state-actor, so State, the desire row and
    the intention row never split. Each ``active → deferred`` edge is only emitted
    when that row is still ``active`` — a same-tick exchange may have terminalized
    it, in which case holding is moot and would be an illegal transition out of a
    terminal state, so it is skipped (the pending-clear + refund still apply). A
    delivery-fail (``defer=False``) keeps both rows ``active`` to retry — no
    transition, just the pending-clear + refund."""
    state = actor.state
    intents: list[Intent] = [
        UpdateState(
            {
                "pending_proactive_id": None,
                "pending_proactive_since": None,
                "energy": state.energy + reserved_energy,
            }
        )
    ]
    if defer and isinstance(lm.state, MemoryPort):
        desire = read_live_contact_desire(lm.state)
        if desire is not None and desire.state == DesireState.ACTIVE:
            intents.insert(
                0,
                TransitionRecord(
                    op=TransitionOp(
                        kind="desire",
                        id=CONTACT_DESIRE_ID,
                        from_state=DesireState.ACTIVE,
                        to_state=DesireState.DEFERRED,
                    )
                ),
            )
        intention = read_live_contact_intention(lm.state)
        if intention is not None and intention.state == IntentionState.ACTIVE:
            intents.insert(
                0,
                TransitionRecord(
                    op=TransitionOp(
                        kind="intention",
                        id=CONTACT_INTENTION_ID,
                        from_state=IntentionState.ACTIVE,
                        to_state=IntentionState.DEFERRED,
                    )
                ),
            )
    actor.apply(intents)

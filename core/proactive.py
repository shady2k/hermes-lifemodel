"""The Hermes-free proactive tick — decide, then deliver through a port (§13/§14).

:func:`proactive_tick` is the being's delivery-aware wake path, called once per
loop tick by the platform adapter. It drives ``lm.coreloop.tick()`` (the layered
pipeline: personality → neuron → aggregation → cognition, committed by the single
state-actor). If the pipeline surfaces a ``LaunchProactive`` intent, it applies the
global backstop (:func:`core.backstop.allow_send`, fail-closed rate limit) and, if
allowed, injects the being's native proactive turn via
``egress.reach_out(target, launch.prompt)`` — the prompt already opens with the
self-attribution line that doubles as the being's self-exclusion marker.

Launch ≠ fulfilment: a delivered launch leaves the contact desire ``active`` with
``pending_proactive_id`` set (the ``sent``/``silent`` proactive outcome resolves it in
its own async-completion frame, spec §3/§5). A
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

import contextlib
from collections.abc import Mapping, Sequence

from ..composition import LifeModel
from ..domain.egress import ReachOutcome
from ..domain.memory import TransitionOp
from ..domain.objects import CONTACT_DESIRE_ID, CONTACT_INTENTION_ID, DesireState, IntentionState
from ..domain.signal import Signal
from ..ports.memory import MemoryPort
from ..ports.proactive import ProactiveEgressPort
from ..ports.tracer import parse_traceparent
from .backstop import allow_send
from .correlate import open_correlated_span
from .desire_view import read_live_contact_desire
from .frame import FrameTrigger, run_frame
from .intention_view import read_live_contact_intention
from .intents import Intent, LaunchProactive, TransitionRecord, UpdateState
from .state_actor import StateActor
from .suppression import SuppressionReason, emit_suppression_span
from .timeutil import to_iso


def proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    initial_signals: Sequence[Signal] = (),
    trigger: FrameTrigger = FrameTrigger.HEARTBEAT,
) -> ReachOutcome | None:
    """Run one ExecutionFrame: pipeline → backstop → deliver. Never raises past
    the injected collaborators. Assumes a fresh ``LifeModel`` per call (the loop
    builds one each tick), so the rollback reconciliation commit is safe.

    *initial_signals* seed the frame (spec §3): the live heartbeat passes none
    (empty world), while a test driver may inject a ``contact_observed`` /
    ``proactive_outcome`` reading to exercise the whole pipeline in one frame.

    Every "why" is a span under the launch's ORIGIN TRACE (§4.4), never a bare
    log line: the core stayed quiet → its suppression span was logged in-tick by
    aggregation/CognitionLauncher; the backstop held → a ``BACKSTOP_RATE_LIMITED``
    suppression span here; the egress held/failed → an ``EGRESS_*`` suppression
    span; a delivery → a ``proactive_delivery`` span carrying the exact prompt.

    Returns a :class:`ReachOutcome` ONLY for a real delivery attempt (delivered /
    failed / unavailable — the egress boundary, spec §9). A QUIET tick (no launch,
    or the backstop held) returns ``None``. ``DELIVERED`` means the turn was queued
    — whether the being actually spoke is the async ``proactive_outcome`` read-back,
    not this return."""
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
    # One ExecutionFrame, serialized through the one process-wide state-actor lock
    # (spec §3): pipeline runs + state committed by the state-actor.
    report = run_frame(lm.coreloop, initial_signals, trigger=trigger)

    # No reach attempted — the core stayed quiet. This is a core DECISION, not a
    # delivery outcome: aggregation/CognitionLauncher already logged the reason as a
    # suppression span during the tick (spec §5). No egress outcome to report.
    if not report.launches:
        return None

    launch = report.launches[0]
    actor = lm.state_actor
    state = actor.state  # post-tick committed state (energy already reserved by cognition)
    now = lm.clock.now()

    if not allow_send(state.proactive_send_log, now):
        # Backstop held fire (fail-closed rate limit): a core decision to HOLD, not a
        # delivery attempt. Log the reason as a suppression span UNDER THE LAUNCH'S
        # ORIGIN TRACE (§4.4 — not a fresh root), defer the desire, and return None.
        _emit_egress_suppression(
            lm, SuppressionReason.BACKSTOP_RATE_LIMITED, launch=launch, tick=state.tick_count
        )
        _rollback(lm, actor, launch, defer=True)  # hold: active -> deferred
        return None

    # The prompt already opens with the self-attribution line (build_wake_packet),
    # which is also the marker the being's own hooks self-exclude on — so nothing
    # is prepended here; a machine label would only re-invite the meta-analysis the
    # phenomenological self-state is meant to cure ([SILENT] regression).
    full_prompt = launch.prompt
    outcome = egress.reach_out(target, full_prompt)
    if outcome is not ReachOutcome.DELIVERED:
        # The delivery boundary held/failed: a first-class suppression span naming the
        # gate (egress_unavailable / egress_failed) UNDER THE ORIGIN TRACE (§4.4), not
        # a bare INFO line — so the "why nothing went out" is in the one trace store.
        # The exact prompt we handed the egress rides the span (5th-source collapse).
        _emit_egress_suppression(
            lm,
            _EGRESS_SUPPRESSION[outcome],
            launch=launch,
            tick=state.tick_count,
            prompt=full_prompt,
            outcome=outcome.value,
        )
        _rollback(lm, actor, launch, defer=False)  # keep active, retry
    else:
        # A delivered launch is a span UNDER THE ORIGIN TRACE too (§4.4 / §5 step 3),
        # so the weave carries launch → delivery → async outcome → resolution under one
        # trace_id — not just the "why NOT" edges. The exact prompt rides the span.
        _emit_delivery_span(lm, launch=launch, tick=state.tick_count, prompt=full_prompt)

    return outcome


#: The non-DELIVERED egress boundary outcomes → their suppression reason (spec §5).
_EGRESS_SUPPRESSION: dict[ReachOutcome, SuppressionReason] = {
    ReachOutcome.UNAVAILABLE: SuppressionReason.EGRESS_UNAVAILABLE,
    ReachOutcome.FAILED: SuppressionReason.EGRESS_FAILED,
}


def _emit_egress_suppression(
    lm: LifeModel, reason: SuppressionReason, *, launch: LaunchProactive, tick: int, **extra: object
) -> None:
    """Emit an OUT-OF-TICK ``proactive`` suppression span UNDER THE ORIGIN TRACE (§4.4).

    The egress runs after ``coreloop.tick()`` returns, so there is no in-tick span to
    bind to — but the launch carries its ``origin_traceparent``, so instead of a
    disconnected fresh root we CONTINUE the attempt's trace (one attempt = one
    ``trace_id``). Bind a :class:`~lifemodel.log.SpanLogger` over the graph's SAME
    durable writer + ring (the one trace store), emit the suppression, persist the
    span row. Best-effort: a graph without a tracer (a hand-built test ``LifeModel``)
    skips it."""
    tracer = lm.tracer
    if tracer is None:
        return
    now = to_iso(lm.clock.now())
    bridge = open_correlated_span(
        tracer=tracer,
        writer=lm.trace_writer,
        ring=lm.event_ring,
        origin_traceparent=launch.origin_traceparent,
        component="proactive",
        tick=tick,
        started_at=now,
    )
    bridge.span.set(**extra)  # decision values (e.g. the egress outcome) onto the span
    # Choke-point count (§4.2): this OUT-OF-TICK egress suppression lands in
    # ``lifemodel_suppressions_total`` through the same door as in-tick ones.
    emit_suppression_span(bridge.logger, reason=reason, component="proactive", metrics=lm.metrics)
    bridge.persist(ended_at=now)


def _emit_delivery_span(lm: LifeModel, *, launch: LaunchProactive, tick: int, prompt: str) -> None:
    """Emit a DELIVERED ``proactive`` span under the launch's origin trace (§5 step 3).

    The exact *prompt* handed to the egress rides the span as an attr AND a
    ``proactive_prompt`` DEBUG event, so "what we handed the agent" (the owner's
    core observability ask) is durable under the attempt's ``trace_id`` — the
    5th-source collapse (§4.3), not a hindsight/DEBUG-log side channel."""
    tracer = lm.tracer
    if tracer is None:
        return
    now = to_iso(lm.clock.now())
    bridge = open_correlated_span(
        tracer=tracer,
        writer=lm.trace_writer,
        ring=lm.event_ring,
        origin_traceparent=launch.origin_traceparent,
        component="proactive",
        tick=tick,
        started_at=now,
    )
    bridge.span.set(
        correlation_id=launch.correlation_id, outcome=ReachOutcome.DELIVERED.value, prompt=prompt
    )
    bridge.logger.debug("proactive_prompt", correlation_id=launch.correlation_id, prompt=prompt)
    bridge.logger.info(
        "proactive_delivery",
        correlation_id=launch.correlation_id,
        outcome=ReachOutcome.DELIVERED.value,
    )
    bridge.span.end(status="ok", ended_at=now)
    bridge.persist(ended_at=now)


def _rollback(lm: LifeModel, actor: StateActor, launch: LaunchProactive, *, defer: bool) -> None:
    """Atomically undo a launch that did not deliver — clear pending (+ its async
    anchor, §4.4) + refund energy, and (on a backstop block) hold BOTH the desire AND
    the intention ``active → deferred`` in lockstep.

    One atomic ``commit_tick`` via the state-actor, so State, the desire row and
    the intention row never split. Each ``active → deferred`` edge is only emitted
    when that row is still ``active`` — a same-tick exchange may have terminalized
    it, in which case holding is moot and would be an illegal transition out of a
    terminal state, so it is skipped (the pending-clear + refund still apply). A
    delivery-fail (``defer=False``) keeps both rows ``active`` to retry — no
    transition, just the pending-clear + refund. The correlation anchor is cleared in
    lockstep with ``pending_proactive_id`` (§4.4 clear-site), and the disposable index
    correlation is stamped ``resolved_at`` so retention can reclaim the origin trace."""
    state = actor.state
    intents: list[Intent] = [
        UpdateState(
            {
                "pending_proactive_id": None,
                "pending_proactive_since": None,
                "pending_proactive_origin_traceparent": None,
                "energy": state.energy + launch.reserved_energy,
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
    # Retire the disposable index correlation (§4.4): this attempt is aborted (its
    # state anchor was just cleared above), so stamp ``resolved_at`` — otherwise the
    # unresolved index row would protect the origin trace from retention forever.
    # Best-effort/fail-open like all trace writes.
    with contextlib.suppress(ValueError):
        origin_trace_id = parse_traceparent(launch.origin_traceparent).trace_id
        resolved_at = to_iso(lm.clock.now())
        lm.trace_writer.submit_correlation(
            correlation_id=launch.correlation_id,
            origin_trace_id=origin_trace_id,
            created_at=resolved_at,
            resolved_at=resolved_at,
        )

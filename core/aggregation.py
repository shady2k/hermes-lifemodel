"""ContactAggregation — the AGGREGATION layer for the contact desire (spec §3/§7).

Stateless: every frame it reads the live contact desire from the start-of-frame
records snapshot (``ctx.objects``, via :func:`live_contact_desire`) plus the drive's
transient ``contact_pressure`` value (the FRESH ``u``) and the ephemeral
``contact_observed``/``proactive_outcome``/``in_flight`` signals, applies them in the
order contact → outcome → wake (threaded through locals, like ``core/decision.py``'s
functions), and emits at most ONE desire-row mutation (a :class:`PutRecord` to
birth an ``active`` desire, or a :class:`TransitionRecord` to advance a live one)
plus the residual-field :class:`UpdateState`.

The desire *lifecycle* is a first-class typed row (lm-27n.3): ``urge →
PutRecord(active)``; ``SENT → active→satisfied``; ``SILENT → active→dropped``;
``FAILED``/``STALE → active→dropped`` (attempt ended, nothing to reinforce); a real
``contact_observed`` terminalizes the live desire (``→satisfied``) and **dominates a
same-frame proactive_outcome**. ``satisfied``/``dropped``/``expired`` are terminal —
aggregation never transitions out of them (it only ever births a fresh ``active``
desire, an upsert on the singleton id). This layer keeps its safety gates (effective
pressure / ActionPending inhibition, silence window, decline backoff, in-flight, the
certified ``evaluate_wake`` threshold) and the atomic desire/intention interlock.

**T3 re-cut (spec §3/§8):** drive-only. The baroque gates that did NOT survive the
rebuild are gone — appropriateness is the async act-gate's job (Hermes turn), not
aggregation's, so the receptivity appraisal is cut entirely; the top-down
thought-proposal spring is cut entirely (``spring`` is always ``DRIVE``; thoughts
return in a later phase). The pure-longing anti-repeat CONCERN is kept but shed of
its machinery: a second drive-only bid while one is unanswered HOLDS. ``u`` comes
from the drive's same-tick ``contact_pressure`` signal (``UpdateState`` is only
visible after commit); this layer never writes ``u`` (send ≠ contact). And a quiet
tick is no longer the absence of a record: on a non-wake it emits a suppression
span (spec §5) naming the gate that held fire.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.egress import ProactiveOutcome
from ..domain.memory import PutOp, TransitionOp
from ..domain.objects import DesireSpring, DesireState, IntentionState
from ..sim.wake import GateParams, LaneState, WakeOutcome, evaluate_wake
from .backstop import record_send
from .component import TickContext
from .correlate import open_correlated_span
from .desire_view import build_contact_desire, encode_contact_desire, live_contact_desire
from .intake import apply_backpressure
from .intention_view import live_contact_intention
from .intents import Intent, PutRecord, TransitionRecord, UpdateState
from .invalidation import is_proactive_outcome_stale
from .pressure import effective_pressure, inhibition_at
from .suppression import SuppressionReason, emit_suppression_span
from .taxonomy import (
    KIND_CONTACT_OBSERVED,
    KIND_PROACTIVE_OUTCOME,
    contact_pressure_value,
    is_in_flight,
    read_contact_observed,
    read_proactive_outcome,
    read_proactive_outcome_correlation,
)
from .tick_metrics import INTAKE_COALESCED, INTAKE_SHED_SENSOR, SIGNALS_INTAKE
from .timeutil import minutes_between, to_iso
from .trace import creation_provenance

#: The logical "no live desire" sentinel — the old ``desire_status == "none"``.
_NONE = "none"

#: The atomic lifecycle interlock (lm-27n.4): when a desire resolution transitions
#: the desire, the live intention (the decision record) is transitioned in lockstep
#: — in the SAME frame commit — so the pair can never split-brain. Maps the desire's
#: resolution target to the intention's. SENT/contact → ``completed``; SILENT/failed/
#: stale → ``dropped``; a held (deferred) desire → ``deferred`` (each legal from both
#: ``active`` and, for the terminal targets, ``deferred``).
_INTENTION_TARGET: dict[str, str] = {
    DesireState.SATISFIED.value: IntentionState.COMPLETED.value,
    DesireState.DROPPED.value: IntentionState.DROPPED.value,
    DesireState.DEFERRED.value: IntentionState.DEFERRED.value,
}

#: Maps a non-URGE wake outcome to its suppression reason (spec §5). An URGE maps to
#: nothing — it wakes (or is held by the anti-repeat gate, handled separately).
_WAKE_SUPPRESSION: dict[WakeOutcome, SuppressionReason] = {
    WakeOutcome.BELOW_THRESHOLD: SuppressionReason.BELOW_THRESHOLD,
    WakeOutcome.IN_FLIGHT: SuppressionReason.IN_FLIGHT,
    WakeOutcome.SILENCE_WINDOW: SuppressionReason.SILENCE_WINDOW,
    WakeOutcome.DECLINE_BACKOFF: SuppressionReason.DECLINE_BACKOFF,
}


class ContactAggregation:
    """Owns the contact-desire lifecycle (one desire per lane). Drive-only (T3)."""

    def __init__(
        self,
        *,
        params: GateParams,
        theta: float,
        beta: float,
        u_max: float,
        i0: float = 1.0,
        grace_min: float = 45.0,
        halflife_min: float = 60.0,
        verdict_deadline_min: float = 30.0,
        id: str = "contact-aggregation",
    ) -> None:
        self.id = id
        self._params = params
        self._theta = theta
        self._beta = beta
        self._u_max = u_max
        self._i0 = i0
        self._grace_min = grace_min
        self._halflife_min = halflife_min
        self._verdict_deadline_min = verdict_deadline_min

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        now = ctx.now
        # Priority-class backpressure (spec §7): the gate is the DEFENSIVE layer. The
        # ephemeral bus carried every seed through (frame is a dumb blackboard); HERE
        # the thalamic gate classifies must_process vs best_effort and coalesces the
        # best_effort sensor noise to a bounded count BEFORE reducing — a flood can't
        # each drive a step. must_process (contact_observed / proactive_outcome /
        # in_flight / the drive's contact_pressure) is NEVER shed. Every read below is
        # off ``signals`` (the gated view), so the load-bearing signals are intact and
        # only the noise is bounded. A real backpressure event (overflow) is counted
        # on the intake counter (shed/coalesced), so the shedding is observable.
        intake = apply_backpressure(ctx.signals)
        signals = intake.signals
        if ctx.metrics is not None and intake.overflowed:
            ctx.metrics.inc(SIGNALS_INTAKE, intake.best_effort_shed, outcome=INTAKE_SHED_SENSOR)
            ctx.metrics.inc(SIGNALS_INTAKE, intake.best_effort_kept, outcome=INTAKE_COALESCED)
        # The drive's FRESH u from its transient contact_pressure signal (T3): the
        # drive's UpdateState is only visible AFTER commit, so aggregation reads the
        # same-tick u here, falling back to the start-of-tick ctx.state.u baseline.
        u_now = contact_pressure_value(signals, default=state.u)

        # The live desire from the start-of-tick snapshot (active/deferred), and
        # its logical state threaded through the reducer — the old ``agg.status``.
        live = live_contact_desire(ctx.objects)
        desire_state = live.state if live is not None else _NONE

        # The live intention (the decision record cognition crystallized) from the
        # same snapshot — resolved atomically with the desire below. ``None`` when
        # the desire resolved before it ever crystallized (e.g. an inbound exchange
        # terminalizes a never-launched desire): only the desire transitions then.
        live_intention = live_contact_intention(ctx.objects)

        # working copies of the residual policy fields (threaded like decision.py)
        last_exchange_at = state.last_exchange_at
        # The silence-window gate's anchor, DECOUPLED from the exchange record
        # (lm-md6.1): an admin override (force_wake backdates it, satiate sets it to
        # now) or ``None`` to fall back to the real last_exchange_at below. A genuine
        # exchange this tick clears it so the gate re-tracks the fresh exchange.
        silence_anchor_at = state.silence_anchor_at
        declined_at = state.declined_at
        decline_count = state.decline_count
        last_contact_at = state.last_contact_at
        action_pending_since = state.action_pending_since
        pending_id = state.pending_proactive_id
        pending_since = state.pending_proactive_since
        # The async-correlation anchor (spec §4.4), threaded in lockstep with
        # ``pending_id``: cleared exactly when the verdict resolves the attempt below.
        pending_origin = state.pending_proactive_origin_traceparent
        send_log = state.proactive_send_log
        unanswered_outbound_count = state.unanswered_outbound_count

        # Captured when a proactive outcome resolves an in-flight attempt this frame,
        # so the resolution span is woven UNDER THE ORIGIN TRACE (§4.4) after the reducer.
        resolved_correlation: str | None = None
        resolved_origin: str | None = None
        resolved_outcome: ProactiveOutcome | None = None

        # The single desire-row action this tick: a create, or a transition
        # target for the live desire. At most one of the two ever fires.
        create_desire = False
        transition_to: str | None = None

        # effective pressure at verdict time (from persisted inhibition) — staleness input
        effective_now = effective_pressure(
            u_now,
            inhibition_at(
                state.action_pending_since,
                now,
                i0=self._i0,
                grace_min=self._grace_min,
                halflife_min=self._halflife_min,
            ),
        )

        # 1) real contact resets clocks and terminalizes a live desire (before outcome/wake).
        #    A real reply this frame dominates any same-frame proactive_outcome: it resolves
        #    the pull, so the outcome loop below is skipped (the desire is already gone).
        had_exchange = any(
            sig.kind == KIND_CONTACT_OBSERVED
            and read_contact_observed(sig)[0] != "proactive_internal"
            for sig in signals
        )
        if had_exchange:
            last_exchange_at = to_iso(now)
            silence_anchor_at = None  # a real exchange re-anchors the gate on itself
            declined_at = None
            decline_count = 0
            action_pending_since = None
            unanswered_outbound_count = 0  # a genuine reply resets the longing bid
            # Contact DOMINATES a same-frame/in-flight proactive attempt (spec §7.3):
            # clear the pending-proactive anchor in LOCKSTEP with terminalizing the
            # desire. Leaving it set would strand pending_proactive_id — the desire is
            # gone, so the eventual LLM-turn completion's ASYNC_COMPLETION frame no
            # longer runs (hooks returns early on the missing desire) and the launcher
            # HOLDs every future launch forever (cognition deadlocked). Clearing it
            # also makes that stale completion a clean no-op (the crossed outreach is
            # ignored — the accepted stale-outreach semantics).
            pending_id = None
            pending_since = None
            pending_origin = None
            if desire_state in (DesireState.ACTIVE, DesireState.DEFERRED):
                transition_to = DesireState.SATISFIED  # exchange terminalizes the live desire
            desire_state = _NONE

        # 2) a proactive outcome resolves the woken desire — dropped if stale (async
        #    invalidation §7.3). Only reached when no contact dominated this frame
        #    (contact_observed-dominates-proactive_outcome).
        if not had_exchange:
            for sig in signals:
                if sig.kind != KIND_PROACTIVE_OUTCOME:
                    continue
                stale, _reason = is_proactive_outcome_stale(
                    desire_state=desire_state,
                    pending_id=pending_id,
                    outcome_correlation_id=read_proactive_outcome_correlation(sig),
                    last_exchange_at=last_exchange_at,
                    pending_since=pending_since,
                    effective=effective_now,
                    threshold=self._theta,
                    now=now,
                    deadline_min=self._verdict_deadline_min,
                )
                if stale:
                    continue
                po = read_proactive_outcome(sig)
                if po is ProactiveOutcome.SENT:
                    transition_to = DesireState.SATISFIED
                    action_pending_since = to_iso(now)  # send -> inhibition starts
                    last_contact_at = to_iso(now)
                    send_log = record_send(send_log, now)  # backstop counter (spec §14)
                    # Pure-longing outreach counter: a SENT drive-only outreach (the
                    # only kind now, T3) is a repeat longing bid -> bump. A legacy
                    # THOUGHT/MIXED-sprung desire (persisted from before the cut) is a
                    # materially-new reason -> not a repeat, does not bump.
                    if live is not None and live.spring == DesireSpring.DRIVE:
                        unanswered_outbound_count += 1
                elif po is ProactiveOutcome.SILENT:
                    transition_to = DesireState.DROPPED
                    declined_at = to_iso(now)
                    decline_count += 1
                else:  # FAILED / STALE — the attempt ended with nothing to reinforce:
                    # clear the pending + drop the desire, but no decline backoff and no
                    # ActionPending window (no send happened).
                    transition_to = DesireState.DROPPED
                resolved_correlation = read_proactive_outcome_correlation(sig)
                resolved_origin = pending_origin
                resolved_outcome = po
                pending_id = None
                pending_since = None
                pending_origin = None
                desire_state = _NONE
                break  # a resolved desire is no longer active — later outcomes are stale

        # duration on latent u (never shrinks; latent, not effective — accrues under inhibition)
        dt = max(0.0, minutes_between(state.last_tick_at, now))
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0

        # effective pressure for the wake gate (post-verdict inhibition)
        effective = effective_pressure(
            u_now,
            inhibition_at(
                action_pending_since,
                now,
                i0=self._i0,
                grace_min=self._grace_min,
                halflife_min=self._halflife_min,
            ),
        )

        # The silence-window gate measures from the anchor: an admin override if set,
        # else the real last exchange (lm-md6.1). ``last_exchange_at`` still feeds the
        # verdict-staleness check above and the wake-packet fact — untouched here.
        silence_anchor = silence_anchor_at if silence_anchor_at is not None else last_exchange_at
        exch_min = -minutes_between(silence_anchor, now) if silence_anchor is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=self._params)
        drive_urge = outcome.is_urge

        # Anti-repeat HOLD (T3, simplified — concern kept, machinery shed): a SECOND
        # pure-longing bid while one is unanswered (unanswered_outbound_count >= 1)
        # must HOLD — don't drum a second reach into the void. Aggregation is
        # drive-only now, so a held urge is always a pure-longing repeat (there is no
        # top-down override). A genuine exchange THIS tick reset the counter above
        # (same-tick visible), so a same-tick exchange+re-urge does not self-deadlock.
        pure_longing_repeat_hold = unanswered_outbound_count >= 1 and drive_urge

        # A wake-eligible urge births a desire only when none is live and nothing
        # resolved one this tick (dedup / anti-drum).
        if (
            drive_urge
            and desire_state == _NONE
            and transition_to is None
            and not pure_longing_repeat_hold
        ):
            create_desire = True

        changes: dict[str, object] = {
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "silence_anchor_at": silence_anchor_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
            "action_pending_since": action_pending_since,
            "pending_proactive_id": pending_id,
            "pending_proactive_since": pending_since,
            "pending_proactive_origin_traceparent": pending_origin,
            "proactive_send_log": send_log,
            "unanswered_outbound_count": unanswered_outbound_count,
        }
        intents: list[Intent] = [UpdateState(changes)]

        if create_desire:
            # Drive-only (T3): spring is always DRIVE, salience is the effective
            # pressure that cleared the wake bar. A birth only happens when no live
            # desire exists, so it is always a NEW episode → stamp a fresh trace.
            provenance = creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="aggregation",
                reason="contact desire (drive)",
            )
            desire = build_contact_desire(
                state=DesireState.ACTIVE,
                salience=effective,
                source_drive=u_now,
                spring=DesireSpring.DRIVE,
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_contact_desire(desire))))
        elif transition_to is not None and live is not None:
            intents.append(
                TransitionRecord(
                    op=TransitionOp(
                        kind="desire",
                        id=live.id,
                        from_state=live.state,
                        to_state=transition_to,
                    )
                )
            )
            # Atomic interlock: transition the live intention in the SAME commit as
            # its desire — never one without the other (split-brain guard). Only
            # when an intention exists AND the edge is a real change (skip a
            # deferred→deferred no-op, which the machine would reject and roll the
            # whole tick — and the desire resolution — back).
            intention_target = _INTENTION_TARGET.get(str(transition_to))
            if (
                live_intention is not None
                and intention_target is not None
                and live_intention.state != intention_target
            ):
                intents.append(
                    TransitionRecord(
                        op=TransitionOp(
                            kind="intention",
                            id=live_intention.id,
                            from_state=live_intention.state,
                            to_state=intention_target,
                        )
                    )
                )

        # Observability (spec §5): drop this tick's decision values onto the span so
        # it is SELF-EXPLAINING (u / effective pressure / wake outcome / what the
        # desire row did), and on a QUIET tick — no desire born, nothing resolved —
        # emit a suppression span naming the gate that held fire, so silence is a
        # logged decision, not the absence of a record. (A creation or resolution is
        # NOT silent; a dedup'd URGE with a live desire is "already reaching", not a
        # suppression.) Only when a span-bound logger is wired (the live tick); a
        # bare unit-test context skips it.
        if ctx.logger is not None:
            ctx.logger.span.set(
                u=u_now,
                effective_pressure=effective,
                wake_outcome=outcome.name,
                created=create_desire,
                transition_to=str(transition_to) if transition_to is not None else None,
            )
            if not create_desire and transition_to is None:
                reason = (
                    SuppressionReason.REPEAT_PURE_LONGING
                    if pure_longing_repeat_hold
                    else _WAKE_SUPPRESSION.get(outcome)
                )
                if reason is not None:
                    emit_suppression_span(
                        ctx.logger, reason=reason, component=self.id, metrics=ctx.metrics
                    )

        # Async-bridge resolution span (spec §4.4 / §5 step 6): when this frame
        # consumes the outcome that resolves an in-flight proactive attempt, weave the
        # resolution UNDER THE ORIGIN TRACE (raised from the state anchor, NOT this
        # frame's trace) and stamp ``resolved_at`` on the disposable correlation index
        # so retention can eventually reclaim the origin trace. The anchor itself is
        # already cleared in ``changes`` above (durable half); this is the observable
        # + index half. Best-effort, fail-open (§4.2): a bare context skips it.
        if (
            resolved_correlation
            and resolved_origin is not None
            and ctx.tracer is not None
            and ctx.event_ring is not None
        ):
            resolved_at = to_iso(now)
            outcome_value = resolved_outcome.value if resolved_outcome is not None else None
            bridge = open_correlated_span(
                tracer=ctx.tracer,
                writer=ctx.trace_writer,
                ring=ctx.event_ring,
                origin_traceparent=resolved_origin,
                component=self.id,
                tick=state.tick_count + 1,
                started_at=to_iso(now),
            )
            bridge.span.set(
                correlation_id=resolved_correlation,
                outcome=outcome_value,
                resolved_at=resolved_at,
            )
            bridge.logger.info(
                "proactive_resolution", correlation_id=resolved_correlation, outcome=outcome_value
            )
            bridge.span.end(status="ok", ended_at=resolved_at)
            bridge.persist(ended_at=resolved_at)
            ctx.trace_writer.submit_correlation(
                correlation_id=resolved_correlation,
                origin_trace_id=bridge.span.context.trace_id,
                created_at=resolved_at,
                resolved_at=resolved_at,
            )

        return intents

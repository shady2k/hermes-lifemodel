"""ContactAggregation — the AGGREGATION layer for the contact desire (spec §7, §12).

Stateless: every tick it reads the live contact desire from the start-of-tick
records snapshot (``ctx.objects``, via :func:`live_contact_desire`) plus the
neuron's transient ``contact`` value and the durable ``exchange``/``verdict``/
``in_flight`` inputs, applies them in the order exchange → verdict → wake
(threaded through locals, like ``core/decision.py``'s functions), and emits at
most ONE desire-row mutation (a :class:`PutRecord` to birth an ``active`` desire,
or a :class:`TransitionRecord` to advance a live one) plus the residual-field
:class:`UpdateState`.

The desire *lifecycle* is a first-class typed row now, not a ``State`` flag
(lm-27n.3): ``urge → PutRecord(active)``; ``FULFILL → active→satisfied``;
``REJECT → active→dropped``; ``DEFER → active→deferred``; a real ``exchange``
terminalizes the live desire (``→satisfied``) and **dominates a same-tick
verdict**. ``satisfied``/``dropped``/``expired`` are terminal — aggregation never
transitions out of them (it only ever births a fresh ``active`` desire, an upsert
on the singleton id). This layer keeps ALL its gates (effective pressure, silence
window, decline backoff, ActionPending grace/decay, in-flight, the ``evaluate_wake``
gate); the residual policy scalars (``decline_count``, ``action_pending_since``,
``pending_proactive_id``, ``proactive_send_log`` …) stay on ``State``.

The neuron owns ``u`` on rise and exchange-satiation; this layer never writes
``u`` (send ≠ contact: FULFILL starts an ActionPending inhibition window but does
not satiate the drive). Only a real exchange clears ActionPending (the neuron
satiates ``u`` separately). Aggregation is the SOLE contact-desire writer — it folds
BOTH springs into the singleton (lm-27n.9): bottom-up drive AND the top-down
``thought_contact_proposal`` that ``ThoughtCrystallization`` emits (in-tick, just
upstream). ``spring`` = drive / thought / mixed accordingly. So the start-of-tick
snapshot plus its single decision remain a sufficient in-tick dedup guard even with
two springs.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import PutOp, TransitionOp
from ..domain.objects import DesireSpring, DesireState, IntentionState
from ..sim.aggregation import Verdict
from ..sim.wake import GateParams, LaneState, evaluate_wake
from .backstop import record_send
from .component import TickContext
from .desire_view import build_contact_desire, encode_contact_desire, live_contact_desire
from .intention_view import live_contact_intention
from .intents import EmitSignal, Intent, PutRecord, TransitionRecord, UpdateState
from .invalidation import is_verdict_stale
from .pressure import effective_pressure, inhibition_at
from .receptivity import appraise_receptivity
from .relationship_view import DEFAULT_RELATIONSHIP, live_owner_relationship
from .taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    contact_value,
    is_in_flight,
    read_exchange,
    read_thought_contact_proposal,
    read_verdict,
    read_verdict_correlation,
    thought_contact_created_signal,
)
from .timeutil import minutes_between
from .trace import creation_provenance

#: The logical "no live desire" sentinel — the old ``desire_status == "none"``.
_NONE = "none"

#: The atomic lifecycle interlock (lm-27n.4): when a desire resolution transitions
#: the desire, the live intention (the decision record) is transitioned in lockstep
#: — in the SAME tick commit — so the pair can never split-brain. Maps the desire's
#: resolution target to the intention's. FULFILL/exchange → ``completed``; REJECT →
#: ``dropped``; DEFER → ``deferred`` (each legal from both ``active`` and, for the
#: terminal targets, ``deferred``).
_INTENTION_TARGET: dict[str, str] = {
    DesireState.SATISFIED.value: IntentionState.COMPLETED.value,
    DesireState.DROPPED.value: IntentionState.DROPPED.value,
    DesireState.DEFERRED.value: IntentionState.DEFERRED.value,
}


class ContactAggregation:
    """Owns the contact-desire lifecycle (one desire per lane)."""

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
        u_now = contact_value(ctx.signals, default=state.u)

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
        declined_at = state.declined_at
        decline_count = state.decline_count
        last_contact_at = state.last_contact_at
        action_pending_since = state.action_pending_since
        pending_id = state.pending_proactive_id
        pending_since = state.pending_proactive_since
        send_log = state.proactive_send_log
        unanswered_outbound_count = state.unanswered_outbound_count

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

        # 1) real exchanges reset clocks and terminalize a live desire (before verdict/wake).
        #    A real reply this tick dominates any same-tick verdict: it resolves the
        #    pull, so the verdict loop below is skipped (the desire is already gone).
        had_exchange = any(
            sig.kind == KIND_EXCHANGE and read_exchange(sig)[0] != "proactive_internal"
            for sig in ctx.signals
        )
        if had_exchange:
            last_exchange_at = now.isoformat()
            declined_at = None
            decline_count = 0
            action_pending_since = None
            unanswered_outbound_count = 0  # a genuine reply resets the longing bid (Task 7)
            if desire_state in (DesireState.ACTIVE, DesireState.DEFERRED):
                transition_to = DesireState.SATISFIED  # exchange terminalizes the live desire
            desire_state = _NONE

        # 2) a verdict resolves the woken desire — dropped if stale (async invalidation §7.3).
        #    Only reached when no exchange dominated this tick (exchange-dominates-verdict).
        if not had_exchange:
            for sig in ctx.signals:
                if sig.kind != KIND_VERDICT:
                    continue
                stale, _reason = is_verdict_stale(
                    desire_state=desire_state,
                    pending_id=pending_id,
                    verdict_correlation_id=read_verdict_correlation(sig),
                    last_exchange_at=last_exchange_at,
                    pending_since=pending_since,
                    effective=effective_now,
                    threshold=self._theta,
                    now=now,
                    deadline_min=self._verdict_deadline_min,
                )
                if stale:
                    continue
                verdict = read_verdict(sig)
                if verdict is Verdict.FULFILL:
                    transition_to = DesireState.SATISFIED
                    action_pending_since = now.isoformat()  # send -> inhibition starts
                    last_contact_at = now.isoformat()
                    send_log = record_send(send_log, now)  # backstop counter (spec §14)
                    # Pure-longing outreach counter (Task 7, lm-8o3.1): a FULFILLED
                    # drive-only send (no crystallized-thought backing) is a repeat
                    # longing bid -> bump. THOUGHT/MIXED carries a genuine new reason
                    # (a source thought) -> not a repeat, does not bump.
                    if live is not None and live.spring == DesireSpring.DRIVE:
                        unanswered_outbound_count += 1
                    pending_id = None
                    pending_since = None
                    desire_state = _NONE
                elif verdict is Verdict.REJECT:
                    transition_to = DesireState.DROPPED
                    declined_at = now.isoformat()
                    decline_count += 1
                    pending_id = None
                    pending_since = None
                    desire_state = _NONE
                else:  # Verdict.DEFER — hold the intention (never reached in live Model A)
                    transition_to = DesireState.DEFERRED
                    desire_state = DesireState.DEFERRED
                break  # a resolved desire is no longer active — later verdicts are stale

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

        # Receptivity appraisal (lm-27n.5): the NEW owner-appropriateness gate,
        # DISJOINT from the wake gates below (it never re-derives silence / backoff /
        # inhibition / in-flight — those stay in ``evaluate_wake`` + the effective
        # pressure math above). A hard veto (explicit boundary — quiet hours,
        # cadence min, no-contact) SUPPRESSES the birth; soft norms scale the
        # effective pressure so a borderline urge that would have woken is held.
        # With no relationship row (or the permissive DEFAULT) this returns
        # allowed=True / multiplier=1.0 → behaviour-identical to .4.
        relationship = live_owner_relationship(ctx.objects) or DEFAULT_RELATIONSHIP
        appraisal = appraise_receptivity(relationship, state, now)
        gated_effective = effective * appraisal.pressure_multiplier

        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=gated_effective, now=0.0, state=lane, params=self._params)
        drive_urge = outcome.is_urge

        # Top-down spring (lm-27n.9): a crystallized-thought proposal from the
        # in-tick signal (ThoughtCrystallization ran just before this component).
        # It BYPASSES the drive threshold — contact from a genuine reason, not
        # accumulated pressure — but STILL respects every appropriateness gate:
        # in-flight / silence window / decline backoff (re-evaluated by forcing the
        # threshold pass while keeping the real lane state), the receptivity hard
        # veto (``appraisal.allowed`` below — AUTHORITATIVE here), and the singleton
        # dedup (``desire_state == _NONE``). Energy is inherited downstream (cognition
        # will not launch a turn it cannot afford), exactly as the bottom-up path.
        proposal = read_thought_contact_proposal(ctx.signals)
        top_down_admissible = (
            proposal is not None
            and evaluate_wake(
                u=max(gated_effective, self._theta), now=0.0, state=lane, params=self._params
            ).is_urge
        )

        # A wake-eligible urge (bottom-up drive OR top-down proposal) births a desire
        # only when none is live, nothing resolved one this tick (dedup / anti-drum),
        # AND the appraisal admits it (no explicit-boundary hard veto). With no
        # proposal and the permissive default this is behaviour-identical to .5.
        # Task 8 HOLD gate (lm-8o3.1): with an unanswered pure-longing send still
        # out (``unanswered_outbound_count >= 1``, Task 7), a SECOND pure-longing
        # bid (``drive_urge and not top_down_admissible`` — no materially-new
        # reason backing it) must HOLD rather than birth another desire. A
        # top-down-admissible proposal still overrides — it IS a materially-new
        # reason. Uses the tick-local ``unanswered_outbound_count`` (not
        # ``state.unanswered_outbound_count`` directly) so a genuine exchange
        # THIS SAME tick (which resets it to 0 above, §1) already lifts the hold
        # — the reset is the escape hatch and must be visible same-tick, or a
        # same-tick exchange+re-urge could spuriously self-deadlock.
        pure_longing_repeat_hold = (
            unanswered_outbound_count >= 1 and drive_urge and not top_down_admissible
        )
        if (
            (drive_urge or top_down_admissible)
            and desire_state == _NONE
            and transition_to is None
            and appraisal.allowed
            and not pure_longing_repeat_hold
        ):
            create_desire = True

        changes: dict[str, object] = {
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
            "action_pending_since": action_pending_since,
            "pending_proactive_id": pending_id,
            "pending_proactive_since": pending_since,
            "proactive_send_log": send_log,
            "unanswered_outbound_count": unanswered_outbound_count,
        }
        intents: list[Intent] = [UpdateState(changes)]

        if create_desire:
            # Fold the two springs into the singleton (lm-27n.9). Bottom-up only →
            # ``spring=DRIVE``, salience = the gated effective pressure that cleared
            # the wake bar (behaviour-identical to .5). Top-down proposal only →
            # ``spring=THOUGHT``, salience from the proposal score, carrying
            # ``source_thought_ids`` (the concrete reason — the [SILENT] fix). Both
            # in one tick → ``spring=MIXED`` (still carrying the source thought).
            source_thought_ids: tuple[str, ...]
            if top_down_admissible and proposal is not None:
                spring = DesireSpring.MIXED if drive_urge else DesireSpring.THOUGHT
                salience = max(gated_effective, proposal.score) if drive_urge else proposal.score
                source_thought_ids = (proposal.thought_id,)
                risk_if_ignored = proposal.other_regarding
            else:
                spring = DesireSpring.DRIVE
                salience = gated_effective
                source_thought_ids = ()
                risk_if_ignored = 0.0
            # Creation provenance (lm-27n.11): a contact-desire birth ONLY happens when
            # no live desire exists (create-if-none, guarded by ``desire_state == _NONE``
            # → ``live is None`` here), so a birth is ALWAYS a NEW episode → stamp a
            # fresh trace. The ``live is not None`` branch is the uniform preserve-if-live
            # guard (defensive/documentary — unreachable in the create path today).
            provenance = (
                live.provenance
                if live is not None
                else creation_provenance(
                    ctx.trace,
                    created_by=self.id,
                    component="aggregation",
                    reason=f"contact desire ({spring})",
                )
            )
            desire = build_contact_desire(
                state=DesireState.ACTIVE,
                salience=salience,
                source_drive=u_now,
                spring=spring,
                source_thought_ids=source_thought_ids,
                risk_if_ignored=risk_if_ignored,
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_contact_desire(desire))))
            # Tell ThoughtAttention (same tick, downstream) the source thought's reason
            # ACTUALLY became a desire, so it resolves that thought. Only on genuine
            # creation — a proposal aggregation suppressed leaves its thought live
            # (lm-27n.9; codex): the reason is not silently spent by timing.
            for thought_id in source_thought_ids:
                intents.append(
                    EmitSignal(
                        signal=thought_contact_created_signal(
                            origin_id=self.id, thought_id=thought_id, timestamp=now.isoformat()
                        )
                    )
                )
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
        return intents

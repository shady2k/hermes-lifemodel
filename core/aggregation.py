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
satiates ``u`` separately). Aggregation is the SOLE contact-desire writer in this
task (top-down/thought desire is a later task), so the start-of-tick snapshot plus
its single decision are a sufficient in-tick dedup guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import PutOp, TransitionOp
from ..domain.objects import DesireState, IntentionState
from ..sim.aggregation import Verdict
from ..sim.wake import GateParams, LaneState, evaluate_wake
from .backstop import record_send
from .component import TickContext
from .desire_view import build_contact_desire, encode_contact_desire, live_contact_desire
from .intention_view import live_contact_intention
from .intents import Intent, PutRecord, TransitionRecord, UpdateState
from .invalidation import is_verdict_stale
from .pressure import effective_pressure, inhibition_at
from .taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    contact_value,
    is_in_flight,
    read_exchange,
    read_verdict,
    read_verdict_correlation,
)
from .timeutil import minutes_between

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

        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=self._params)
        # A wake-eligible urge births a desire only when none is live and nothing
        # resolved one this tick (dedup / anti-drum). After any resolution the
        # residual gates already veto a same-tick re-wake, so this is behaviour-
        # identical to the old ``on_urge`` on a ``NONE`` status.
        if outcome.is_urge and desire_state == _NONE and transition_to is None:
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
        }
        intents: list[Intent] = [UpdateState(changes)]

        if create_desire:
            desire = build_contact_desire(
                state=DesireState.ACTIVE, salience=effective, source_drive=u_now
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
        return intents

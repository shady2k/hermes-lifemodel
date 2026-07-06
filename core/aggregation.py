"""ContactAggregation — the AGGREGATION layer for the contact desire (spec §7, §12).

Stateless: every tick it reconstructs the certified ``sim`` primitives from the
persisted state and drives the desire lifecycle. It reads the neuron's transient
``contact`` value plus durable ``exchange``/``verdict``/``in_flight`` inputs,
applies them in the order exchange → verdict → wake (threaded through locals,
like ``core/decision.py``'s functions), and emits one ``UpdateState``.

The neuron owns ``u`` on rise and exchange-satiation; this layer never writes
``u`` (send ≠ contact: FULFILL starts an ActionPending inhibition window but does
not satiate the drive). Only a real exchange clears ActionPending (the neuron
satiates ``u`` separately). This is the port of ``core/decision.py`` onto the
layer boundary — the wake/lifecycle math is the reused ``sim`` code, never
reimplemented here.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.aggregation import Aggregator, DesireStatus, Verdict
from ..sim.wake import GateParams, LaneState, evaluate_wake
from .component import TickContext
from .intents import Intent, UpdateState
from .pressure import effective_pressure, inhibition_at
from .taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    contact_value,
    is_in_flight,
    read_exchange,
    read_verdict,
)
from .timeutil import minutes_between


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

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        now = ctx.now
        u_now = contact_value(ctx.signals, default=state.u)

        # working copies of the policy fields (threaded like decision.py)
        last_exchange_at = state.last_exchange_at
        declined_at = state.declined_at
        decline_count = state.decline_count
        agg = Aggregator(status=DesireStatus(state.desire_status))
        last_contact_at = state.last_contact_at
        action_pending_since = state.action_pending_since

        # 1) real exchanges reset the policy clocks and clear the desire (before wake)
        for sig in ctx.signals:
            if sig.kind == KIND_EXCHANGE:
                actor, _label = read_exchange(sig)
                if actor != "proactive_internal":
                    last_exchange_at = now.isoformat()
                    declined_at = None
                    decline_count = 0
                    action_pending_since = None  # real contact resolves the pull
                    agg.on_exchange()

        # 2) a verdict resolves the woken desire (after exchange, before wake)
        for sig in ctx.signals:
            if sig.kind == KIND_VERDICT:
                verdict = read_verdict(sig)
                agg.apply_verdict(verdict)
                if verdict is Verdict.FULFILL:
                    action_pending_since = now.isoformat()  # send happened -> inhibition starts
                    last_contact_at = now.isoformat()  # record our outreach (observability only)
                elif verdict is Verdict.REJECT:
                    declined_at = now.isoformat()
                    decline_count += 1

        # duration-over-threshold accumulates on latent u (not effective)
        dt = minutes_between(state.last_tick_at, now)
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0

        # compute effective pressure: latent u gated by ActionPending inhibition
        inhibition = inhibition_at(
            action_pending_since,
            now,
            i0=self._i0,
            grace_min=self._grace_min,
            halflife_min=self._halflife_min,
        )
        effective = effective_pressure(u_now, inhibition)

        # wake gates — every quantity as minutes relative to now (now = 0.0)
        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=self._params)
        if outcome.is_urge:
            agg.on_urge()

        changes: dict[str, object] = {
            "desire_status": agg.status.value,
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
            "action_pending_since": action_pending_since,
        }
        return [UpdateState(changes)]

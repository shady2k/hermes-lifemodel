"""ContactAggregation — the AGGREGATION layer for the contact desire (spec §7, §12).

Stateless: every tick it reconstructs the certified ``sim`` primitives from the
persisted state and drives the desire lifecycle. It reads the neuron's transient
``contact`` value plus durable ``exchange``/``verdict``/``in_flight`` inputs,
applies them in the order exchange → verdict → wake (threaded through locals,
like ``core/decision.py``'s functions), and emits one ``UpdateState``.

The neuron owns ``u`` on rise/exchange-satiation; this layer writes ``u`` only on
a ``FULFILL`` verdict (the certified model's delivery satiation). This is the port
of ``core/decision.py`` onto the layer boundary — the wake/lifecycle math is the
reused ``sim`` code, never reimplemented here.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.aggregation import Aggregator, DesireStatus
from ..sim.wake import GateParams, LaneState, evaluate_wake
from .component import TickContext
from .intents import Intent, UpdateState
from .taxonomy import KIND_EXCHANGE, contact_value, is_in_flight, read_exchange
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
        id: str = "contact-aggregation",
    ) -> None:
        self.id = id
        self._params = params
        self._theta = theta
        self._beta = beta
        self._u_max = u_max

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        now = ctx.now
        u_now = contact_value(ctx.signals, default=state.u)

        # working copies of the policy fields (threaded like decision.py)
        last_exchange_at = state.last_exchange_at
        declined_at = state.declined_at
        decline_count = state.decline_count
        agg = Aggregator(status=DesireStatus(state.desire_status))

        # 1) real exchanges reset the policy clocks and clear the desire (before wake)
        for sig in ctx.signals:
            if sig.kind == KIND_EXCHANGE:
                actor, _label = read_exchange(sig)
                if actor != "proactive_internal":
                    last_exchange_at = now.isoformat()
                    declined_at = None
                    decline_count = 0
                    agg.on_exchange()

        # duration-over-threshold accumulates on the current (risen) u
        dt = minutes_between(state.last_tick_at, now)
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0

        # wake gates — every quantity as minutes relative to now (now = 0.0)
        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=u_now, now=0.0, state=lane, params=self._params)
        if outcome.is_urge:
            agg.on_urge()

        return [
            UpdateState(
                {
                    "desire_status": agg.status.value,
                    "duration_over_theta": duration,
                    "last_exchange_at": last_exchange_at,
                    "declined_at": declined_at,
                    "decline_count": decline_count,
                }
            )
        ]

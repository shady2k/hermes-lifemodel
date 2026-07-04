"""In-process proactive tick (spec §3.2/§6).

:func:`run_proactive_tick` is the delivery-aware analog of :func:`lifemodel.tick.run_tick`:
it accumulates pressure and decides via the aggregator exactly as the cron tick does
(both brains must agree), but drains pressure / stamps contact ONLY after a native
reach-out is DELIVERED — so a failed or busy delivery retries next tick instead of
silently consuming the wake. The supervised ``proactive_service_loop`` that drives
this on a timer lands in task 7.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta

from .composition import LifeModel
from .domain.egress import ReachOutcome
from .impulse import compose_impulse
from .logging import EventLogger
from .ports.proactive import ProactiveEgressPort
from .tick import DEFAULT_WAKE_COOLDOWN


def run_proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
    cooldown: timedelta = DEFAULT_WAKE_COOLDOWN,
    busy: bool = False,
) -> ReachOutcome:
    """One in-process proactive tick. Delivery-aware; fail-closed on the caller side.

    Accumulates pressure exactly like :func:`lifemodel.tick.run_tick` (same neuron
    loop, same ``State.pressure`` sum, same aggregator decision) so both brains see
    the same drive. The only divergence is delivery-awareness: pressure is drained
    and contact/cooldown stamped ONLY when ``egress.reach_out`` returns DELIVERED —
    on any other outcome pressure is left intact so the wake retries next tick.
    Always bumps bookkeeping and the liveness stamp, commits once.
    """
    state = lm.state.load()
    now = lm.clock.now()

    # Accumulate pressure from this tick's neurons — identical to run_tick so both
    # brains agree on the drive (the in-proc service and the cron fallback must not
    # diverge on what "above threshold" means).
    for neuron in lm.neurons:
        for signal in neuron.tick(state):
            lm.bus.publish(signal)
    signals = lm.bus.consume_unprocessed()
    state.pressure += sum((signal.salience for signal in signals), 0.0)

    decision = lm.aggregator.decide(signals, pressure=state.pressure)
    in_cooldown = (
        state.cooldown_until is not None and now < datetime.fromisoformat(state.cooldown_until)
    )

    outcome = ReachOutcome.SKIPPED_BUSY  # default "did not reach out this tick"
    if decision.wake and decision.packet is not None and not in_cooldown and not busy:
        last_contact = (
            datetime.fromisoformat(state.last_contact_at)
            if state.last_contact_at is not None
            else None
        )
        impulse = compose_impulse(decision.packet, now=now, last_contact_at=last_contact)
        outcome = egress.reach_out(target, impulse)
        if outcome is ReachOutcome.DELIVERED:
            # Drain + stamp + cooldown ONLY on a confirmed delivery — a failed/busy
            # delivery leaves pressure intact so the wake retries next tick.
            state.pressure = 0.0
            state.last_contact_at = now.isoformat()
            state.cooldown_until = (now + cooldown).isoformat()
        else:
            logger.info("proactive_not_delivered", outcome=outcome.value)
    elif busy:
        outcome = ReachOutcome.SKIPPED_BUSY

    state.tick_count += 1
    state.last_tick_at = now.isoformat()
    state.egress_service_alive_at = now.isoformat()  # liveness stamp (Task 6)
    lm.state.commit(state)
    logger.info("proactive_tick", pressure=state.pressure, outcome=outcome.value)
    return outcome

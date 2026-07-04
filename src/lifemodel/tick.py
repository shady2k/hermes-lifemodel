"""The heartbeat tick — the cron ``--script`` entrypoint (roadmap 1.1, HLA D1).

A Hermes cron job fires ~every minute and runs this module as its pre-check
``--script`` (via ``sys.executable`` inside Hermes' interpreter). The tick is the
**autonomic pulse** of the being: it loads state, lets the neurons read it and
emit signals, asks the aggregator whether to wake cognition, advances the
heartbeat bookkeeping, and commits — all at **zero LLM cost** (HLA §1).

**The wake-gate contract (verified against ``cron/scheduler.py``).** The
scheduler runs this script *first* and parses its **last non-empty stdout line**
as a wake gate (``_parse_wake_gate``): a JSON object ``{"wakeAgent": false}``
skips the agent entirely — no LLM, silent, zero cost; anything else wakes it and
injects the script's full stdout as context (HLA D4). So this module prints
**exactly one** JSON line to stdout, derived from the aggregator's decision.

As of Phase 1.3 the graph carries one autonomic neuron
(:class:`~lifemodel.core.neuron.StubTimerNeuron`) accumulating pressure into
state each tick, and the real
:class:`~lifemodel.core.aggregator.ThresholdAggregator`: while the accumulated
pressure stays below its threshold the decision is "stay asleep" → the gate is
``{"wakeAgent": false}`` (silent, zero LLM), and the tick the pressure crosses
the threshold the aggregator returns a waking decision → this same code emits the
wake-packet with ``wakeAgent: true`` — no rewrite of the tick (HLA §11). Phase 1.4
closes the loop: on a wake the tick **drains** the pressure, stamps the contact,
and opens a **cooldown** that vetoes the next would-be wakes (single-fire, ≤ 1
message per threshold cycle); the message itself is sent by Hermes' cron when it
wakes the agent (delivery = gateway, HLA §7 / D4).

**stdout discipline.** Only the single wake-gate line is written to stdout;
structured logs go to stderr (see :func:`~lifemodel.logging.configure`) and to
the queryable :class:`~lifemodel.events.EventSink`, so nothing corrupts the gate
the scheduler parses.

This module is an *entrypoint/adapter*, not core: :func:`main` resolves the
Hermes profile home through a lazy, override-able seam (mirroring
:func:`lifemodel._hermes_home`), while :func:`run_tick` is Hermes-free and driven
in tests with injected fakes.
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from .composition import LifeModel, build_lifemodel
from .domain.wake import WakeDecision
from .events import EVENT_TICK, EVENT_TICK_FAILED, EVENTS_FILENAME, EventSink
from .logging import EventLogger, EventTee, configure, get_logger
from .paths import state_dir
from .state.model import State

#: How long after a wake the being stays quiet before it may wake again (roadmap
#: 1.4). A wall-clock duration compared against ``State.cooldown_until``: while it
#: is active the tick stays asleep even above threshold, so at most one message
#: fires per threshold cycle. Cooldown/quiet-hours are *ours*, not Hermes' (HLA
#: §7). A sane default; a disk-backed, hot-reloadable value plugs in here later
#: (same seam as the aggregator's threshold) without reshaping ``run_tick``.
DEFAULT_WAKE_COOLDOWN = timedelta(minutes=30)

#: How long after a liveness stamp the cron heartbeat still considers the
#: in-process egress service "alive" and defers to it (lm-64s, spec §6). ~3× the
#: service's 60s tick interval: as long as the in-proc brain keeps stamping, the
#: cron stays out of the way; once the stamp goes stale the cron takes over as the
#: fallback brain. ``None`` / unparseable stamps are treated as "dead" (fail-over).
SERVICE_LIVENESS_MAX_AGE = timedelta(minutes=3)


def service_is_alive(
    state: State, *, now: datetime, max_age: timedelta = SERVICE_LIVENESS_MAX_AGE
) -> bool:
    """True if the in-process egress service stamped liveness within *max_age*.

    The cron heartbeat calls this before ticking: a fresh stamp means the
    in-process service owns the brain (it ticks pressure + decides reach-out
    itself), so the cron defers to avoid two brains writing state at once. Stale
    or absent stamps fall back to the cron path. Unparseable stamps are treated as
    dead rather than raising — fail-over to cron, never a mid-cron crash.
    """
    stamp = state.egress_service_alive_at
    if stamp is None:
        return False
    try:
        return now - datetime.fromisoformat(stamp) <= max_age
    except ValueError:
        return False


def _hermes_home() -> Path:
    """Resolve the active Hermes profile home via the host API (lazy seam).

    The only Hermes touchpoint in the tick path. Imported lazily so
    :mod:`lifemodel.tick` stays importable — and :func:`run_tick` stays
    unit-testable — without Hermes on ``sys.path``; tests override this to inject
    a throwaway home. Mirrors :func:`lifemodel._hermes_home` (HLA §3/§4).
    """
    from hermes_constants import get_hermes_home

    # get_hermes_home() already returns a Path; re-wrap so the host module's
    # untyped (Any) return narrows to Path for the strict type checker.
    return Path(get_hermes_home())


def wake_gate_line(decision: WakeDecision) -> str:
    """Render the single stdout line the scheduler parses as the wake gate.

    Derived from the aggregator's :class:`WakeDecision` (never hardcoded), so the
    same tick that stays silent in Phase 1.1 emits a real wake-packet once 1.3's
    aggregator starts waking:

    * stay-asleep → ``{"wakeAgent": false}`` — the scheduler skips the agent.
    * wake → the wake-packet fields plus ``"wakeAgent": true`` — the scheduler
      wakes cognition and injects this line as context (HLA §11 / D4).
    """
    if decision.wake and decision.packet is not None:
        gate = {**decision.packet.to_dict(), "wakeAgent": True}
    else:
        gate = {"wakeAgent": False}
    return json.dumps(gate, ensure_ascii=False)


def run_tick(
    lm: LifeModel, *, logger: EventLogger, cooldown: timedelta = DEFAULT_WAKE_COOLDOWN
) -> WakeDecision:
    """Run one heartbeat tick over the assembled graph and return the decision.

    Hermes-free and side-effecting only through the injected collaborators, so a
    test drives it with fakes (``FakeClock`` / ``FakeStateStore`` / ...). The
    orchestration keeps every layer's seam live while hardcoding none of their
    logic (roadmap "interfaces from day one"):

    1. **Autonomic layer** — each neuron reads state and emits signals onto the
       bus. The first neuron (``StubTimerNeuron``) lands in 1.2.
    2. **Accumulate** — sum the consumed signals' pressure *deltas* into
       ``State.pressure`` (each neuron owns the delta it emitted as its
       ``salience``; the tick only *sums* them — no threshold/wake logic lives
       here). This happens *before* the decision so the aggregator weighs the
       accumulated pressure, not just this tick's transient signals.
    3. **Aggregation layer** — ask the aggregator for a wake decision against the
       accumulated pressure. ``ThresholdAggregator`` wakes once it crosses the
       threshold (1.3); the threshold call is entirely the aggregator's.
    4. **Drain + cooldown + single-fire (roadmap 1.4)** — an active cooldown
       *vetoes* a would-be wake (stay asleep even above threshold), so at most one
       message fires per threshold cycle. When a wake is honoured we **drain**
       ``State.pressure`` to zero, stamp ``last_contact_at``, and open a fresh
       ``cooldown_until = now + cooldown``. The threshold call stays the
       aggregator's; the cooldown is *ours* (HLA §7 "quiet-hours/cooldown —
       наши"), enforced here as an orchestration guard rather than by widening
       the pure-threshold aggregator. The emitted wake-packet is enriched with
       the *prior* contact time (HLA §11) before it is overwritten.
    5. **Heartbeat bookkeeping** — advance ``tick_count`` and stamp
       ``last_tick_at`` from the injected clock, then commit atomically. This is
       the single state writer of the tick (HLA §9); neurons never persist.
    6. **Observability** — emit the structured ``tick`` event so ``/lifemodel
       debug`` can answer "last tick" from the event sink (HLA §12).

    Delivery is *not* here: on a wake this tick prints ``wakeAgent: true`` and
    Hermes' cron wakes the agent and delivers via the gateway (HLA §7 "decision —
    ours, delivery — gateway"; D1/D4). Draining on the *wake decision* (not on a
    delivery ack) is deliberate: the tick is the single state writer and cannot
    observe the async gateway send, and draining here is exactly what guarantees
    the next tick sees no pressure + an open cooldown → no second message.
    """
    state = lm.state.load()
    now = lm.clock.now()

    # Liveness watchdog (lm-64s, spec §6): while the in-process egress service is
    # alive it owns the brain (it ticks pressure + decides reach-out itself), so
    # the cron heartbeat defers here — before any accumulation — and stays a silent
    # fallback. Pressure is left untouched; the in-proc service is the sole writer
    # while alive. When the stamp goes stale/absent the cron takes over as the
    # fallback brain and falls through to the normal tick below.
    if service_is_alive(state, now=now):
        logger.info(EVENT_TICK, deferred="service_alive", pressure=state.pressure)
        return WakeDecision.stay_asleep()

    for neuron in lm.neurons:
        for signal in neuron.tick(state):
            lm.bus.publish(signal)

    signals = lm.bus.consume_unprocessed()

    # Orchestration only: sum the deltas the neurons emitted (each carried as a
    # signal's salience) into the persistent accumulator, *before* deciding — the
    # aggregator weighs the accumulated ``State.pressure``, not just this tick's
    # transient signals. Below-threshold pressure keeps growing tick over tick;
    # whether it is enough to *wake* is the aggregator's call (1.3), never the
    # tick's — no threshold literal lives here.
    state.pressure += sum((signal.salience for signal in signals), 0.0)
    proposed = lm.aggregator.decide(signals, pressure=state.pressure)

    # Cooldown guard (roadmap 1.4): honour a wake only when no cooldown is active.
    # While ``now < cooldown_until`` we stay asleep even above threshold — this is
    # the "≤ 1 message per threshold cycle" + cooldown rail. Threading ``now`` +
    # the stored cooldown here keeps the aggregator a pure thalamus (pressure vs
    # threshold) and the time/cooldown policy in the orchestrator (HLA §7).
    in_cooldown = state.cooldown_until is not None and now < datetime.fromisoformat(
        state.cooldown_until
    )

    if proposed.wake and proposed.packet is not None and not in_cooldown:
        # Enrich the packet with the PRIOR contact time (HLA §11: the wake-packet
        # carries "last contact") before we overwrite ``last_contact_at`` below.
        decision = WakeDecision.wake_with(
            replace(proposed.packet, last_contact_at=state.last_contact_at)
        )
        # Drain on wake: reset the drive, stamp the contact, open the cooldown.
        state.pressure = 0.0
        state.last_contact_at = now.isoformat()
        state.cooldown_until = (now + cooldown).isoformat()
    else:
        # Below threshold, or vetoed by an active cooldown — stay silent (zero
        # LLM). Pressure is committed as-is so it keeps accumulating toward the
        # next cycle; the cooldown (if any) stays untouched.
        decision = WakeDecision.stay_asleep()

    state.tick_count += 1
    state.last_tick_at = now.isoformat()
    lm.state.commit(state)

    logger.info(
        EVENT_TICK,
        tick_count=state.tick_count,
        last_tick_at=state.last_tick_at,
        pressure=state.pressure,
        signals=len(signals),
        wake=decision.wake,
        in_cooldown=in_cooldown,
        cooldown_until=state.cooldown_until,
    )
    return decision


def main() -> int:
    """Run one tick against the real profile home and print the wake gate.

    The cron ``--script`` target (takes no arguments). Wires the real graph from
    the profile state dir, teeing the ``tick`` event into the on-disk
    :class:`EventSink` so the debug command can read it. Routes structlog to
    **stderr** first (:func:`configure`) so the only thing on **stdout** is the
    single wake-gate line. Exits 0; the scheduler reads the gate, not the exit
    code, to decide whether to wake.

    **Fail closed (safety).** Hermes does *not* stay silent when a cron
    ``--script`` exits non-zero: it injects a "Script Error" prompt and **wakes
    the agent**, delivering a crash message to the user (``cron/scheduler.py``
    ``_run_job_script`` → ``_build_job_prompt``). So a crash on a would-be-wake
    tick would otherwise fire an unintended, undrained, repeating delivery. We
    therefore catch **any** unhandled exception, emit the stay-asleep gate
    (``{"wakeAgent": false}``) and exit 0 — Hermes stays silent, nothing is
    delivered, and because the failing tick never committed, a clean wake retries
    on a later healthy tick (a delayed real message beats a delivered crash). The
    error is recorded to stderr and, best-effort, to the on-disk event sink.
    """
    # Resolve the safe gate line and a base logger up front, with operations that
    # cannot raise, so failure reporting and the stay-asleep fallback are always
    # available. Rendering the stay-asleep gate is pure JSON and cannot fail.
    # ``logger`` starts stderr-only and is upgraded to the sink-teed one once the
    # sink exists, so a ``tick_failed`` record lands wherever we got to.
    safe_gate = wake_gate_line(WakeDecision.stay_asleep())  # {"wakeAgent": false}
    logger: EventLogger = get_logger("lifemodel.tick")
    line = safe_gate
    try:
        configure()
        sdir = state_dir(_hermes_home())
        sink = EventSink(sdir / EVENTS_FILENAME)
        logger = EventTee(logger, sink)
        lm = build_lifemodel(base_dir=sdir, logger=logger)
        line = wake_gate_line(run_tick(lm, logger=logger))
    except Exception as exc:  # noqa: BLE001 - fail closed: a crash must NOT wake
        line = safe_gate  # keep the stay-asleep gate — Hermes stays silent (see above)
        with contextlib.suppress(Exception):  # observability must never re-raise
            logger.info(EVENT_TICK_FAILED, error=f"{type(exc).__name__}: {exc}")

    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())

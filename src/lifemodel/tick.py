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
wake-packet with ``wakeAgent: true`` — no rewrite of the tick (HLA §11). Draining
the pressure, the cooldown, and single-fire on delivery are task 1.4.

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

import json
import sys
from pathlib import Path

from .composition import LifeModel, build_lifemodel
from .domain.wake import WakeDecision
from .events import EVENT_TICK, EVENTS_FILENAME, EventSink
from .logging import EventLogger, EventTee, configure, get_logger
from .paths import state_dir


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


def run_tick(lm: LifeModel, *, logger: EventLogger) -> WakeDecision:
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
    4. **Heartbeat bookkeeping** — advance ``tick_count`` and stamp
       ``last_tick_at`` from the injected clock, then commit atomically. This is
       the single state writer of the tick (HLA §9); neurons never persist. The
       pressure is committed undrained — draining on wake is task 1.4.
    5. **Observability** — emit the structured ``tick`` event so ``/lifemodel
       debug`` can answer "last tick" from the event sink (HLA §12).
    """
    state = lm.state.load()

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
    decision = lm.aggregator.decide(signals, pressure=state.pressure)

    # NB (roadmap 1.4): the accumulated pressure is committed *as is* — the wake
    # decision does not drain it here. Draining on delivery + cooldown is 1.4;
    # this is the obvious seam for it (after decide, before commit).
    now = lm.clock.now()
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
    """
    configure()
    sdir = state_dir(_hermes_home())
    sink = EventSink(sdir / EVENTS_FILENAME)
    logger = EventTee(get_logger("lifemodel.tick"), sink)
    lm = build_lifemodel(base_dir=sdir, logger=logger)

    decision = run_tick(lm, logger=logger)

    sys.stdout.write(wake_gate_line(decision) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())

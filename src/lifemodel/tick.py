"""The heartbeat tick — the cron ``--script`` entrypoint (roadmap 1.1, HLA D1).

A Hermes cron job fires ~every minute and runs this module as its pre-check
``--script`` (via ``sys.executable`` inside Hermes' interpreter).

**Phase-1 drum-killer (wire-desire-model plan, Task 4): the cron tick is a
silent watchdog, never a second brain.** The in-process egress service
(:func:`~lifemodel.egress_service.run_proactive_tick`, reusing
:mod:`lifemodel.core.decision`) is the **sole** decision brain — it is the only
code path that rises the drive, evaluates the wake gates, and launches a
proactive turn. This cron tick **never** decides to wake: :func:`run_tick`
always returns :meth:`~lifemodel.domain.wake.WakeDecision.stay_asleep`,
whatever the persisted state. Its only remaining jobs are (1) a liveness
watchdog — while the in-process service's stamp is fresh it owns state
exclusively, so this tick writes nothing and gets out of the way — and (2),
once that stamp goes stale/absent, plain heartbeat bookkeeping
(``tick_count``/``last_tick_at``) so ``/lifemodel debug`` still has a "last
tick" to show, at zero LLM cost and with no proactive fallback (Global
Constraint: "cron never wakes proactively... both must never disagree, so cron
simply never decides").

**The wake-gate contract (verified against ``cron/scheduler.py``).** The
scheduler runs this script *first* and parses its **last non-empty stdout line**
as a wake gate (``_parse_wake_gate``): a JSON object ``{"wakeAgent": false}``
skips the agent entirely — no LLM, silent, zero cost; anything else wakes it and
injects the script's full stdout as context (HLA D4). Because :func:`run_tick`
always stays asleep now, this module always prints ``{"wakeAgent": false}`` —
:func:`wake_gate_line` keeps the general wake-packet rendering path (still
exercised by its own unit tests / other wake producers), it is simply never fed
a waking decision from this entrypoint.

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
from datetime import datetime, timedelta
from pathlib import Path

from .composition import LifeModel, build_lifemodel
from .domain.wake import WakeDecision
from .events import EVENT_TICK, EVENT_TICK_FAILED, EVENTS_FILENAME, EventSink
from .logging import EventLogger, EventTee, configure, get_logger
from .paths import state_dir
from .state.model import State

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
    in-process service owns state exclusively (it is the sole decision brain —
    see :mod:`lifemodel.core.decision` / :mod:`lifemodel.egress_service`), so the
    cron defers and writes nothing to avoid two brains racing the same commit.
    Stale or absent stamps fall back to the cron's own (purely bookkeeping) path.
    Unparseable stamps are treated as dead rather than raising — fail-over to
    cron, never a mid-cron crash.
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


def run_tick(lm: LifeModel, *, logger: EventLogger) -> WakeDecision:
    """Run one cron heartbeat tick as a **silent watchdog** and never wake.

    Hermes-free and side-effecting only through the injected collaborators, so a
    test drives it with fakes (``FakeClock`` / ``FakeStateStore`` / ...).

    Per the wire-desire-model plan (Task 4), the in-process egress service is
    the sole decision brain (:mod:`lifemodel.core.decision` /
    :mod:`lifemodel.egress_service`) — this tick decides nothing and **always**
    returns :meth:`WakeDecision.stay_asleep`, whatever state it finds:

    1. **Liveness watchdog** — while the in-process service's liveness stamp is
       fresh (:func:`service_is_alive`) it owns state exclusively, so this tick
       writes nothing and returns immediately (avoids two brains racing the same
       commit).
    2. **Heartbeat bookkeeping (fallback only)** — once that stamp goes
       stale/absent, advance ``tick_count`` and stamp ``last_tick_at`` from the
       injected clock, then commit — the only reason this cron path still runs
       at all is so ``/lifemodel debug`` keeps a "last tick" even if the
       in-process service is down. No proactive fallback: the cron never
       launches a reach-out on its own (Global Constraint — "both must never
       disagree, so cron simply never decides").
    3. **Observability** — emit the structured ``tick`` event either way.
    """
    state = lm.state.load()
    now = lm.clock.now()

    # Liveness watchdog (lm-64s, spec §6): while the in-process egress service is
    # alive it owns state exclusively, so the cron heartbeat defers here — before
    # writing anything — and stays a silent fallback. When the stamp goes
    # stale/absent the cron takes over the (bookkeeping-only) path below.
    if service_is_alive(state, now=now):
        logger.info(EVENT_TICK, deferred="service_alive")
        return WakeDecision.stay_asleep()

    state.tick_count += 1
    state.last_tick_at = now.isoformat()
    lm.state.commit(state)

    logger.info(
        EVENT_TICK,
        tick_count=state.tick_count,
        last_tick_at=state.last_tick_at,
        wake=False,
    )
    return WakeDecision.stay_asleep()


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

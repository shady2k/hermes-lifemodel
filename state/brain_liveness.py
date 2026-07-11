"""The ``/lifemodel status`` brain-liveness block (spec §4.4, bead lm-fib.9.3).

This is the surface that makes the being's liveness READABLE without grepping
logs — the display half of the fail-loud invariant. It renders a compact,
owner-facing section from BOTH sources the spec §4.4 names:

* the process-local :class:`~lifemodel.state.brain_health.BrainHealth` (the same
  gateway process hosts the being AND the ``/lifemodel`` command, so this is one
  shared singleton): ``state`` / ``loop_alive`` / ``death_count`` / the last loop
  death / the last observer error / ``boot_error``;
* the **durable, PRIMARY** liveness — ``last_tick_at`` / ``ticks_total`` read
  straight from :class:`~lifemodel.state.model.State` (advanced every tick by the
  CoreLoop) — never a parallel counter, and never the (supporting-only) heartbeat
  metric (codex MAJOR-8);
* the durable **BOOT record** (:func:`~lifemodel.state.brain_health.read_boot_record`)
  so that after a re-raise+restart the block still shows ``boot_failed: <reason>``
  in a FRESH process where the in-memory :class:`BrainHealth` is ``never_started``.

Rendering is **fail-soft by construction** (spec §5 item, acceptance): a flaky /
absent health read degrades to a clear ``unknown`` line (logged), and the durable
read degrades to ``ticks_total: ?`` — the status command must never raise.

Split into a PURE :func:`render_brain_liveness` (fed values → deterministic) and a
thin :func:`brain_liveness_lines` reader that gathers the three sources fail-soft.
All stdlib.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .brain_health import (
    STALE_AFTER_SECONDS,
    BrainHealthSnapshot,
    get_brain_health,
    read_boot_record,
    tick_staleness,
)

_LOG = logging.getLogger("lifemodel.brain_health")


def _system_now() -> datetime:
    """The default render "now" — the ONE sanctioned system-time read (spec §5).

    Sources through :class:`~lifemodel.adapters.clock.SystemClock` (imported lazily
    so this leaf view module never eagerly pulls the adapters package) rather than
    ``datetime.now`` directly; tests inject ``now`` for determinism.
    """
    from ..adapters.clock import SystemClock

    return SystemClock().now()


_BOOT_FAILED = "boot_failed"


def render_brain_liveness(
    snapshot: BrainHealthSnapshot | None,
    *,
    last_tick_at: str | None,
    ticks_total: int | None,
    boot_record: dict[str, str] | None,
    now: datetime,
    stale_after_seconds: float = STALE_AFTER_SECONDS,
) -> list[str]:
    """Render the brain-liveness block as ``**label:** value`` lines (spec §4.4).

    Pure over its inputs so tests feed values directly. *snapshot* is the in-memory
    :class:`BrainHealth` read (``None`` when it could not be read — the caller logs
    that and passes ``None``); *last_tick_at* / *ticks_total* are the DURABLE primary
    liveness (``ticks_total=None`` when the state read degraded); *boot_record* is the
    durable boot-health record, which UPGRADES a ``never_started`` / unknown in-memory
    read to ``boot_failed`` (the fresh-process case).
    """
    # Displayed state + boot_error, with the durable boot record's precedence rules:
    # an in-memory boot_failed wins; else a durable boot_failed record upgrades ONLY a
    # fresh-process read (snapshot None, or the singleton still ``never_started`` after a
    # re-raise+restart) — a LIVE snapshot (connected / connecting / loop_dead) MUST win,
    # so a STALE brain_boot.json cannot mislabel a running brain as boot_failed (codex
    # MAJOR); else the in-memory state; else unknown (the health read itself failed).
    record_is_boot_failed = (
        boot_record is not None and str(boot_record.get("state")) == _BOOT_FAILED
    )
    snapshot_is_fresh = snapshot is None or snapshot.state == "never_started"

    boot_error: str | None = None
    displayed: str  # a BrainState OR "unknown" (a failed health read) — widened to str
    if snapshot is not None and snapshot.state == _BOOT_FAILED:
        displayed = _BOOT_FAILED
        boot_error = snapshot.boot_error
    elif record_is_boot_failed and snapshot_is_fresh:
        displayed = _BOOT_FAILED
        boot_error = boot_record.get("boot_error") if boot_record is not None else None
    elif snapshot is not None:
        displayed = snapshot.state
    else:
        displayed = "unknown"

    if displayed == "connected":
        loop_alive = "yes"
    elif displayed == "unknown":
        loop_alive = "unknown"
    else:
        loop_alive = "no"

    connected_at = snapshot.connected_at if snapshot is not None else None
    age, stale = tick_staleness(
        connected_at, last_tick_at, now=now, stale_after_seconds=stale_after_seconds
    )

    lines = ["**brain liveness**"]
    state_line = f"**state:** {displayed}"
    if displayed == "unknown":
        state_line += " (health read failed — see log)"
    lines.append(state_line)
    lines.append(f"**loop_alive:** {loop_alive}")
    lines.append(f"**last_tick_at:** {last_tick_at or 'never'}")
    lines.append(f"**ticks_total:** {'?' if ticks_total is None else ticks_total}")

    # A wedged-but-connected loop (ticks went quiet) shows a visible staleness warning;
    # a dead / failed brain needs no redundant flag — the state itself is the alarm.
    if displayed == "connected" and stale and age is not None:
        lines.append(f"**⚠ stale:** no tick for {age:.0f}s (> {stale_after_seconds:.0f}s)")

    death_count = snapshot.death_count if snapshot is not None else 0
    lines.append(f"**death_count:** {death_count}")

    last_loop_death = snapshot.last_loop_death if snapshot is not None else None
    if last_loop_death:
        lines.append(f"**last_loop_death:** {last_loop_death.splitlines()[0]}")

    if boot_error:
        lines.append(f"**boot_error:** {boot_error}")

    observer_errors = snapshot.last_observer_error if snapshot is not None else {}
    if observer_errors:
        rendered = "; ".join(f"{name}: {err}" for name, err in sorted(observer_errors.items()))
        lines.append(f"**observer_errors:** {rendered}")

    return lines


def brain_liveness_lines(base_dir: Path, *, now: datetime | None = None) -> list[str]:
    """Gather the three liveness sources for *base_dir* and render the block (fail-soft).

    Reads the in-memory :class:`BrainHealth` snapshot, the DURABLE
    ``last_tick_at`` / ``ticks_total``, and the durable boot record — each guarded so a
    flaky read degrades (a clear ``unknown`` state / ``ticks_total: ?``) and is logged,
    never raised. ``now`` is injectable for deterministic tests; it defaults to real UTC.
    """
    now = now if now is not None else _system_now()

    snapshot: BrainHealthSnapshot | None
    try:
        snapshot = get_brain_health(base_dir).snapshot()
    except Exception:  # noqa: BLE001 - a status render must never crash on a flaky read
        _LOG.warning("brain_liveness_health_read_failed base_dir=%s", base_dir, exc_info=True)
        snapshot = None

    last_tick_at, ticks_total = _read_durable_liveness(base_dir)
    boot_record = read_boot_record(base_dir)  # itself defensive → None on any trouble

    return render_brain_liveness(
        snapshot,
        last_tick_at=last_tick_at,
        ticks_total=ticks_total,
        boot_record=boot_record,
        now=now,
        stale_after_seconds=STALE_AFTER_SECONDS,
    )


def _read_durable_liveness(base_dir: Path) -> tuple[str | None, int | None]:
    """Read the DURABLE ``last_tick_at`` / ``tick_count`` from ``State`` (spec §4.2).

    This is the PRIMARY liveness — advanced every tick by the CoreLoop into
    ``AgentState`` — read through the composition root's ``StatePort`` exactly as the
    other ``/lifemodel`` read paths do. Fail-soft: a locked / corrupt / unbuildable read
    logs a WARNING (observable) and degrades to ``(None, None)`` → ``ticks_total: ?``,
    never a crash. The import is lazy so this leaf module never pulls the graph at load.
    """
    try:
        from ..composition import build_lifemodel

        state = build_lifemodel(base_dir=base_dir).state.load()
        return state.last_tick_at, state.tick_count
    except Exception:  # noqa: BLE001 - the status render must never crash on this read
        _LOG.warning("brain_liveness_state_read_failed base_dir=%s", base_dir, exc_info=True)
        return None, None

"""FR20 — a durable daily ceiling on expensive internal-cognition calls (design §3.4).

The aux model *slot* (``ctx.register_auxiliary_task``) is routing, not a cost
ceiling — Hermes never enforces a spend cap on our behalf. This module is the
ceiling WE build: a plain, pure function over :class:`~lifemodel.state.model.State`
that atomically reserves one call (incrementing the day's counter, rolling the
day over when it has turned) or refuses when the ceiling is already reached.

Pure and stdlib-only — no clock port, no I/O. The caller (an adapter-owned
runner, :mod:`lifemodel.adapters.internal_runner`) reserves BEFORE creating the
async task, and commits the returned ``State`` via an ``UpdateState`` inside a
frame (the reservation is one of the two frame-serialized halves of the seam;
the aux LLM call itself runs off the state-actor lock).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

from ..state.model import State
from .timeutil import from_iso


def _day(now: datetime) -> str:
    """The plain ISO date (``YYYY-MM-DD``) *now* falls on — the day-rollover key."""
    return now.date().isoformat()


#: FR20 v1 default daily ceiling (spec §4.5) — the ONE source of truth, imported by
#: both the runner (the atomic reserve) and the selector (the pre-check), so the two
#: can never drift. A generous-but-real cap: idle ticks stay 0-LLM, this only bounds
#: the spike once a live emitter (lm-705.2) exists. Tune against live traces.
DEFAULT_DAILY_INTERNAL_CALL_CEILING = 50

#: Min wall-clock gap between two internal-cognition launches (spec §4.5) — paces
#: rumination so a live backlog is chewed a little at a time, not all at once.
DEFAULT_MIN_INTERPROCESSING_INTERVAL = timedelta(minutes=30)


def internal_budget_available(state: State, *, now: datetime, daily_ceiling: int) -> bool:
    """True iff another internal-cognition call fits under today's FR20 ceiling.

    The read-only pre-check the selector runs so it never emits a launch the runner's
    atomic :func:`reserve_internal_call` would just deny (and to log the honest
    ``skipped_no_budget`` reason). Shares the day-rollover convention with
    ``reserve_internal_call``: a call on a new day counts against 0."""
    used = state.internal_calls_today if state.internal_calls_day == _day(now) else 0
    return used < daily_ceiling


def internal_interval_elapsed(state: State, *, now: datetime, min_interval: timedelta) -> bool:
    """True iff at least *min_interval* has passed since the last launch (spec §4.5).

    ``True`` when no pass has ever run (``last_internal_call_at is None``). Fail-open:
    an unparseable stored timestamp reads as elapsed rather than wedging processing."""
    if state.last_internal_call_at is None:
        return True
    try:
        last = from_iso(state.last_internal_call_at)
    except (ValueError, TypeError):
        return True
    return now - last >= min_interval


def reserve_internal_call(state: State, *, now: datetime, daily_ceiling: int) -> State | None:
    """Reserve one internal-cognition call against *state*'s FR20 quota.

    Returns a NEW ``State`` with :attr:`~State.internal_calls_today` incremented
    and :attr:`~State.internal_calls_day` set to *now*'s date — rolling the
    counter over to 1 when the stored day does not match *now*'s day (a fresh
    day starts a fresh budget). Returns ``None`` when the ceiling for the
    CURRENT day is already reached — the caller must not create the async task.

    Does not touch :attr:`~State.pending_internal_id` — that is the runner's
    concern (a second field in the same commit), kept orthogonal here so the
    quota half stays a pure, single-purpose function.
    """
    today = _day(now)
    used = state.internal_calls_today if state.internal_calls_day == today else 0
    if used >= daily_ceiling:
        return None
    return dataclasses.replace(state, internal_calls_today=used + 1, internal_calls_day=today)

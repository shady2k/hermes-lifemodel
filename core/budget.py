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
from datetime import datetime

from ..state.model import State


def _day(now: datetime) -> str:
    """The plain ISO date (``YYYY-MM-DD``) *now* falls on — the day-rollover key."""
    return now.date().isoformat()


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

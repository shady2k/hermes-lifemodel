"""The wake-decision: hard policy gates over the drive (spec §7).

An urge (``u ≥ θ_u``) can only *wake cognition*; it never sends a message. Four
gates bound when an accumulated urge is allowed to wake, independent of the
drive's dynamics.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class WakeOutcome(enum.Enum):
    """The wake-decision's verdict for one evaluation."""

    BELOW_THRESHOLD = "no_wake_below_threshold"
    IN_FLIGHT = "no_wake_in_flight"
    SILENCE_WINDOW = "no_wake_silence_window"
    DECLINE_BACKOFF = "no_wake_decline_backoff"
    URGE = "URGE"

    @property
    def is_urge(self) -> bool:
        return self is WakeOutcome.URGE


@dataclass
class LaneState:
    """Per-lane policy memory (spec §8) — *not* a second drive variable.

    Bounded, discrete bookkeeping the wake-decision reads: the conversation clock
    and the decline record. It does not accumulate a deficit and does not feed
    the threshold rule; the drive's ``u`` is the only continuous state.
    """

    last_exchange_at: float | None = None
    in_flight: bool = False
    declined_at: float | None = None
    decline_count: int = 0


@dataclass(frozen=True)
class GateParams:
    """The wake-decision's constants (calibrated; §9). Time in consistent units."""

    theta_u: float = 1.0
    w: float = 15.0
    r0: float = 30.0
    k: float = 2.0
    r_max: float = 1440.0


def evaluate_wake(
    *,
    u: float,
    now: float,
    state: LaneState,
    params: GateParams,
    waive_threshold: bool = False,
) -> WakeOutcome:
    """Decide whether the urge ``u`` is allowed to wake cognition at ``now``.

    Fixed precedence: no urge at all (below threshold) → a turn already running
    (in flight) → inside the active-silence window ``W`` → inside the growing
    decline backoff ``R``. Only a clean pass yields an ``URGE``. Crossing the
    threshold never sends a message — the URGE merely wakes cognition.

    *waive_threshold* skips the FIRST gate and NOTHING else (Phase 4 genesis, spec
    §6.2): a being that is nobody yet wakes to be born, and waiting for ``u ≥ θ``
    would be a category error — ``u`` models a contact deficit inside an EXISTING
    relationship, and a newborn has none. There is nobody to miss, so ``u`` stays 0
    and no caller may inflate it to buy this wake. The other three gates are about
    the CONVERSATION, not the drive, and they all still bind: a turn already in
    flight is still in flight, a live conversation still suppresses, and a newborn
    that woke and chose ``[SILENT]`` is held by the decline backoff exactly like any
    other decline — that backoff IS how it is re-woken later.
    """
    if not waive_threshold and u < params.theta_u:
        return WakeOutcome.BELOW_THRESHOLD
    if state.in_flight:
        return WakeOutcome.IN_FLIGHT
    if state.last_exchange_at is not None and now - state.last_exchange_at < params.w:
        return WakeOutcome.SILENCE_WINDOW
    if state.declined_at is not None:
        r = backoff_interval(
            decline_count=state.decline_count, r0=params.r0, k=params.k, r_max=params.r_max
        )
        if now - state.declined_at < r:
            return WakeOutcome.DECLINE_BACKOFF
    return WakeOutcome.URGE


def backoff_interval(*, decline_count: int, r0: float, k: float, r_max: float) -> float:
    """The *growing* decline backoff ``R_n`` (spec §7 gate 3).

    After ``n = decline_count`` consecutive declines, the next urge is suppressed
    for ``min(r_max, r0·k^(n−1))`` — the first decline suppresses for the base
    ``r0``, and each further consecutive decline grows the interval geometrically
    (a *fixed* backoff would merely relabel the drum period; growth is the v1
    requirement). ``decline_count`` is ``≥ 1`` whenever a backoff is active.
    """
    raw = r0 * (k ** (decline_count - 1))
    return min(r_max, raw)

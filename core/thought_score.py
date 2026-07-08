"""The 0-LLM attention score + decay/park arithmetic (lm-27n.7).

Pure, deterministic, stdlib-only functions the ``ThoughtAttention`` component
(:mod:`lifemodel.core.thought_attention`) uses to score, decay, and park the
being's live thoughts — the anti-rumination brake (Nolen-Hoeksema: healthy
reflection *converges or parks*; sick reflection *loops*). No LLM, no clock
beyond the ``now`` passed in, no state:
every function is a total function of its arguments, unit-tested in isolation.

The score is a **bounded product** of five factors, each in ``(floor .. 1]`` so
one weak factor can down-weight but never annihilate the rest — except the single
real veto (:func:`unresolvedness` returns ``0.0`` for a thought still inside its
park window, which *should* be zero: it is suspended, not competing for
attention)::

    score = salience_c * unresolvedness * novelty * safety * relevance

Decay is a **multiplicative half-life on elapsed time** (robust to irregular
ticks — a 60-minute gap decays sixty times as much as a 1-minute gap), floored so
a thought parks *before* it expires (recoverability). Parking backs off
exponentially by cycle (6h → 24h → 72h); past the cap the thought expires.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime

from ..domain.objects import Sensitivity, Thought, ThoughtState
from .timeutil import minutes_between

# --- Scan / attend width caps (codex's two bounds). --------------------------
#: How many live thoughts a single tick scans (top by salience, id tiebreak). The
#: whole table is never processed — attention is bounded work per tick.
SCAN_WIDTH = 32
#: How many thoughts a single tick *attends* (marks as the turned-over one).
ATTEND_K = 1

# --- Decay. ------------------------------------------------------------------
#: Half-life (minutes) of an unresolved thought's salience — one day. After this
#: much elapsed time an untended thought has lost half its pull.
THOUGHT_SALIENCE_HALFLIFE_MIN = 1440.0
#: At/below this salience an active thought **parks** (not expires) — a floor that
#: keeps a faded thought recoverable rather than destroyed.
SALIENCE_FLOOR = 0.03

# --- Parking / loop detection. -----------------------------------------------
#: No-progress count at which a repeatedly-attended, never-resolved thought parks.
PARK_AFTER = 3
#: Park cycles a thought may survive; once it has been parked this many times an
#: elapsed park window **expires** it instead of re-arming (bounded rumination).
MAX_PARK_CYCLES = 3
#: Exponential park backoff (hours) by 1-indexed park cycle; capped at the last.
_PARK_BACKOFF_HOURS: dict[int, float] = {1: 6.0, 2: 24.0, 3: 72.0}
_PARK_BACKOFF_CAP_HOURS = 72.0

# --- Score factor calibration. -----------------------------------------------
#: Multiplicative privacy penalty base by sensitivity (lower = more guarded).
_SENSITIVITY_BASE: dict[Sensitivity, float] = {
    Sensitivity.NORMAL: 1.0,
    Sensitivity.SENSITIVE: 0.6,
    Sensitivity.PRIVATE: 0.3,
}
#: The floor of the ``safety`` factor — even the most guarded thought keeps a
#: little pull (a veto is the receptivity gate's job, not the score's).
SAFETY_FLOOR = 0.2
#: How far a high other-regarding value may raise ``safety`` toward 1.0 (only for
#: low/moderate sensitivity — a PRIVATE thought is not un-guarded by altruism).
_SAFETY_LIFT = 0.4
#: ``unresolvedness`` for a parked thought whose window has elapsed (a re-entry
#: candidate — it competes again, but at a discount to a fresh active thought).
_PARKED_REENTRY = 0.35
#: Deterministic ``trigger_relevance`` by trigger family: a drive/event beats idle
#: mind-wandering, which beats a ``thought:<parent>`` chain (rich context is .8).
_TRIGGER_DRIVE_EVENT = 0.9
_TRIGGER_IDLE = 0.5
_TRIGGER_CHAIN = 0.3
_TRIGGER_DEFAULT = 0.4


def _clamp01(x: float) -> float:
    """Clamp *x* into ``[0.0, 1.0]``."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _trigger_relevance(trigger: str) -> float:
    """Deterministic relevance from the trigger family (never zero)."""
    if trigger.startswith(("drive", "event")):
        return _TRIGGER_DRIVE_EVENT
    if trigger == "idle":
        return _TRIGGER_IDLE
    if trigger.startswith("thought:"):
        return _TRIGGER_CHAIN
    return _TRIGGER_DEFAULT


def unresolvedness(thought: Thought, now: datetime) -> float:
    """How much the thought is still *open* — the one factor that may veto (0.0).

    ``1.0`` for an active thought; ``0.0`` for a parked thought still inside its
    window (suspended — not competing); ``_PARKED_REENTRY`` for a parked thought
    whose window has elapsed (a re-entry candidate)."""
    if thought.state == ThoughtState.ACTIVE.value:
        return 1.0
    # parked: elapsed iff parked_until is absent/unparseable (>=0) or already past.
    return _PARKED_REENTRY if park_window_elapsed(thought.parked_until, now) else 0.0


def novelty(no_progress_count: int) -> float:
    """Progress penalty: ``1 / (1 + no_progress_count)`` — not mere recency."""
    return 1.0 / (1.0 + max(0, no_progress_count))


def safety(thought: Thought) -> float:
    """Privacy penalty in ``(SAFETY_FLOOR .. 1]``.

    Lower for more sensitive content (PRIVATE < SENSITIVE < NORMAL); a high
    ``other_regarding_value`` lifts it back toward 1.0 **only** when sensitivity is
    low/moderate (a PRIVATE thought stays guarded no matter how altruistic)."""
    base = _SENSITIVITY_BASE.get(thought.sensitivity, 1.0)
    if thought.sensitivity in (Sensitivity.NORMAL, Sensitivity.SENSITIVE):
        base += _SAFETY_LIFT * _clamp01(thought.other_regarding_value) * (1.0 - base)
    return max(SAFETY_FLOOR, min(1.0, base))


def relevance(thought: Thought) -> float:
    """How much attending would *lead somewhere*: ``max`` of actionability,
    other-regarding value, and the deterministic trigger relevance (floored by the
    trigger family, so never zero)."""
    return max(
        _clamp01(thought.actionability),
        _clamp01(thought.other_regarding_value),
        _trigger_relevance(thought.trigger),
    )


def attention_score(thought: Thought, now: datetime) -> float:
    """The bounded-product attention score in ``[0.0, 1.0]`` (0-LLM, deterministic).

    ``salience_c * unresolvedness * novelty * safety * relevance`` — the leading
    magnitude is the clamped salience; the four modifiers each live in
    ``(floor .. 1]`` (bar the parked-window veto), so a single weak factor
    down-weights without erasing the thought."""
    salience_c = _clamp01(thought.salience)
    return (
        salience_c
        * unresolvedness(thought, now)
        * novelty(thought.no_progress_count)
        * safety(thought)
        * relevance(thought)
    )


def decay_salience(
    salience: float,
    elapsed_min: float,
    *,
    halflife_min: float = THOUGHT_SALIENCE_HALFLIFE_MIN,
) -> float:
    """Multiplicative half-life decay over *elapsed_min* (monotone non-increasing).

    ``salience * 0.5 ** (elapsed_min / halflife_min)`` — robust to irregular ticks
    because it is a function of real elapsed time, not tick count. Non-positive
    elapsed (a first touch, or clock skew) leaves salience unchanged; the result
    never goes below 0."""
    if elapsed_min <= 0.0 or halflife_min <= 0.0:
        return max(0.0, salience)
    return max(0.0, salience * math.pow(0.5, elapsed_min / halflife_min))


def park_backoff_hours(park_count: int) -> float:
    """Park window (hours) for the 1-indexed *park_count* — 6h, 24h, 72h, cap 72h."""
    return _PARK_BACKOFF_HOURS.get(park_count, _PARK_BACKOFF_CAP_HOURS)


def park_window_elapsed(parked_until: str | None, now: datetime) -> bool:
    """Has a parked thought's window run out? (absent/unparseable ⇒ elapsed).

    ``minutes_between`` returns ``0.0`` for ``None``/naive/unparseable inputs, so a
    parked row without a valid ``parked_until`` reads as immediately re-entrant
    rather than stuck forever; a genuine future instant returns negative."""
    return minutes_between(parked_until, now) >= 0.0


_WHITESPACE = re.compile(r"\s+")


def loop_signature(thought: Thought) -> str:
    """A deterministic loop signature = normalized-content hash + trigger family.

    Used to detect a thought that keeps returning in the same shape. Content is
    lowercased and whitespace-collapsed before hashing so trivial re-phrasings map
    to one signature; the trigger family disambiguates a drive loop from an idle
    one. Only computed when a thought has none yet (an empty ``loop_signature``)."""
    normalized = _WHITESPACE.sub(" ", thought.content.strip().lower())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    family = thought.trigger.split(":", 1)[0]
    return f"{family}:{digest}"

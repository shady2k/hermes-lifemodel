"""Receptivity appraisal — the pure, 0-LLM "is NOW appropriate?" computation
(lm-27n.5; HLA §4.1, "not-needy is architecture, not prompt vibe").

:func:`appraise_receptivity` reads the owner :class:`~lifemodel.domain.objects.UserModel`
(our DERIVED norms) plus the being's live :class:`~lifemodel.state.model.State`
counters and ``now``, and answers whether a proactive contact is appropriate
*right now* — as a :class:`ReceptivityResult`. It **persists nothing** and calls
no LLM.

It computes ONLY the NEW user-model gates (hours / cadence / privacy / style /
confidence). It deliberately does **not** re-derive the wake gates that already
live in aggregation — silence window, decline backoff, ActionPending inhibition,
in-flight, energy, the send backstop — so there is no duplicate gate; those stay
where they are and this layer is disjoint from them (it never reads
``last_exchange_at``/``declined_at``/``action_pending_since`` for a gate).

**Sovereignty (the split that keeps the being from disappearing):**

* HARD veto (``allowed=False``, a ``hard_reason``) only for an **explicit** owner
  boundary — quiet hours, an explicit cadence minimum, a blanket no-contact
  privacy boundary. "Explicit" = the user-model's ``confidence`` is at/above
  :data:`~lifemodel.core.user_model_view.EXPLICIT_CONFIDENCE`. A seeded/default
  or low-confidence row NEVER hard-vetoes.
* SOFT down-weight (``pressure_multiplier`` < 1, a ``soft_reason``) for weak
  norms — weak negative valence, known load, slow reply latency, or a
  low-confidence would-be boundary. Soft scales the effective pressure (raising
  the effective wake bar); it never zeroes it.
* CONSTRAINT (recorded, not a veto) for an acceptable-style preference and for
  topic-scoped sensitivities — a topic-less proactive *wake* cannot evaluate a
  topic, so those ride along as ``constraints`` the being honors when it composes
  its own turn.

**Behavior-neutral by default:** with the permissive
:data:`~lifemodel.core.user_model_view.DEFAULT_USER_MODEL` (no row) the result is
``allowed=True, pressure_multiplier=1.0, hard_reasons=(), soft_reasons=()`` — so
aggregation and cognition behave EXACTLY as before this task.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain.objects import UserModel
from ..state.model import State
from .user_model_view import EXPLICIT_CONFIDENCE


@dataclass(frozen=True)
class ReceptivityResult:
    """The appraisal verdict for one moment (persists nothing)."""

    #: ``False`` iff an EXPLICIT owner boundary hard-vetoes proactive contact now.
    allowed: bool
    #: Explicit-boundary veto reasons (empty ⇒ nothing hard-vetoed).
    hard_reasons: tuple[str, ...]
    #: Weak-norm down-weight reasons (present even when ``allowed`` is ``True``).
    soft_reasons: tuple[str, ...]
    #: ~0.5..1.0 — scales effective pressure for soft norms (1.0 = no effect).
    pressure_multiplier: float
    #: Composing constraints (styles, topic sensitivities) for the wake packet /
    #: intention audit — NOT vetoes.
    constraints: tuple[str, ...]
    #: The user_model's confidence (how much to trust these norms).
    confidence: float


#: Down-weight applied per soft (weak-norm) consideration; multiplied, then floored.
SOFT_FACTOR = 0.75

#: The floor for ``pressure_multiplier`` — a stack of soft norms scales, never
#: zeroes, the effective pressure (soft is a down-weight, not a veto).
MIN_MULTIPLIER = 0.5

#: Privacy/topic tokens that mean "do not proactively reach out at all" → a
#: blanket contact veto (distinct from a topic-scoped sensitivity, which is a
#: composing constraint, not a wake veto).
BLANKET_NO_CONTACT: frozenset[str] = frozenset(
    {"no_proactive_contact", "no_unprompted_contact", "do_not_initiate", "*"}
)

#: Small 0-LLM lexicons for the soft considerations (free-text norm fields). A
#: match down-weights; the empty/neutral defaults match nothing, so the default
#: user_model never down-weights.
_NEGATIVE_VALENCE: tuple[str, ...] = (
    "slow",
    "cool",
    "cold",
    "negative",
    "reluctant",
    "curt",
    "annoyed",
    "distant",
)
_BUSY_LOAD: tuple[str, ...] = (
    "busy",
    "overloaded",
    "swamped",
    "stressed",
    "slammed",
    "crunch",
    "heavy",
)
_SLOW_LATENCY: tuple[str, ...] = ("slow", "days", "delayed", "sporadic")

#: Cadence words → an explicit minimum spacing (minutes) between proactive contacts.
_CADENCE_WORDS: dict[str, float] = {
    "hourly": 60.0,
    "daily": 1440.0,
    "weekly": 10080.0,
    "monthly": 43200.0,
}
#: Unit suffixes for a numeric cadence ("2h", "90m", "1d") → minutes-per-unit.
_UNIT_MINUTES: dict[str, float] = {"min": 1.0, "hr": 60.0, "m": 1.0, "h": 60.0, "d": 1440.0}


def appraise_receptivity(user_model: UserModel, state: State, now: datetime) -> ReceptivityResult:
    """Appraise whether a proactive contact is appropriate *now* (pure, 0-LLM).

    Reads *user_model*'s learned norms + *state*'s live counters
    (``proactive_send_log``, for the cadence gate) + *now*. Computes only the new
    user_model gates; never re-derives the aggregation wake gates.
    """
    confidence = user_model.confidence if user_model.confidence is not None else 0.0
    explicit = confidence >= EXPLICIT_CONFIDENCE

    # Resolve each DERIVED field at `now`: a field whose inference has gone stale
    # (spec §8, ``inferred_at + ttl`` past) reads as its permissive default here —
    # "no information" — which is exactly the behavior-neutral fallback, so an
    # expired inference silently stops gating rather than gating on a stale value.
    bad_hours = user_model.bad_hours.resolve_or(now, ())
    good_hours = user_model.good_hours.resolve_or(now, ())
    cadence = user_model.cadence.resolve_or(now, "")
    topic_sensitivity = user_model.topic_sensitivity.resolve_or(now, ())
    privacy_boundaries = user_model.privacy_boundaries.resolve_or(now, ())
    acceptable_styles = user_model.acceptable_styles.resolve_or(now, ())
    response_valence_pattern = user_model.response_valence_pattern.resolve_or(now, "")
    known_load = user_model.known_load.resolve_or(now, "")
    reply_latency_norm = user_model.reply_latency_norm.resolve_or(now, "")

    hard: list[str] = []
    soft: list[str] = []
    constraints: list[str] = []
    multiplier = 1.0

    # --- hours_fit: owner good/bad HOURS (distinct from the being's circadian
    #     energy). Explicit bad hour now → hard veto; a low-confidence bad hour →
    #     soft. Hours are UTC hours-of-day, matching the engine's circadian clock
    #     (owner-local TZ conversion is a later concern).
    if now.hour in bad_hours:
        if explicit:
            hard.append(f"owner quiet hours (hour {now.hour:02d}:00 UTC is off-limits)")
        else:
            soft.append("inferred bad hour")
            multiplier *= SOFT_FACTOR

    # --- good_hours: the owner's PREFERRED hours (softer than bad_hours). Being
    #     OUTSIDE an explicitly-set preferred window is a SOFT down-weight (a
    #     preference, not a forbidden hour — bad_hours is the hard boundary). Empty
    #     (the default) means "no preference" → inert, so the default never gates.
    if good_hours and now.hour not in good_hours:
        soft.append("outside owner preferred hours")
        multiplier *= SOFT_FACTOR

    # --- cadence_fit: an explicit preferred MIN spacing between proactive
    #     contacts (beyond the send-rate backstop), computed from the live
    #     proactive_send_log. Only an EXPLICIT cadence hard-vetoes.
    min_gap = cadence_min_minutes(cadence)
    if min_gap is not None and explicit:
        since = _minutes_since_last_send(state.proactive_send_log, now)
        if since is not None and since < min_gap:
            hard.append(
                f"owner cadence: min {min_gap:.0f} min between proactive contacts "
                f"(last was {since:.0f} min ago)"
            )

    # --- privacy_fit: a blanket "no proactive contact" boundary hard-vetoes when
    #     explicit; topic-scoped sensitivities/boundaries ride along as composing
    #     constraints (a topic-less wake cannot evaluate them → never a veto).
    if _blanket_no_contact(privacy_boundaries, topic_sensitivity):
        if explicit:
            hard.append("owner privacy boundary: no proactive contact")
        else:
            soft.append("inferred no-contact preference")
            multiplier *= SOFT_FACTOR
    for topic in topic_sensitivity:
        if topic not in BLANKET_NO_CONTACT:
            constraints.append(f"avoid topic: {topic}")
    for boundary in privacy_boundaries:
        if boundary not in BLANKET_NO_CONTACT:
            constraints.append(f"respect boundary: {boundary}")

    # --- style_fit: allowed proactive styles are a CONSTRAINT on the packet, not
    #     a veto.
    if acceptable_styles:
        constraints.append("style: " + "|".join(acceptable_styles))

    # --- soft considerations: weak negative valence / known load / slow latency.
    if _matches(response_valence_pattern, _NEGATIVE_VALENCE):
        soft.append("weak negative valence")
        multiplier *= SOFT_FACTOR
    if _matches(known_load, _BUSY_LOAD):
        soft.append("known load")
        multiplier *= SOFT_FACTOR
    if _matches(reply_latency_norm, _SLOW_LATENCY):
        soft.append("inferred latency")
        multiplier *= SOFT_FACTOR

    multiplier = max(MIN_MULTIPLIER, multiplier)
    return ReceptivityResult(
        allowed=not hard,
        hard_reasons=tuple(hard),
        soft_reasons=tuple(soft),
        pressure_multiplier=multiplier,
        constraints=tuple(constraints),
        confidence=confidence,
    )


def cadence_min_minutes(cadence: str) -> float | None:
    """Parse an explicit minimum proactive-contact spacing (minutes) from a
    user_model's ``cadence`` string, or ``None`` when it sets no minimum.

    Accepts a bare number of minutes (``"120"``), a number+unit (``"2h"``,
    ``"90m"``, ``"1d"``), or a cadence word (``"hourly"``/``"daily"``/``"weekly"``/
    ``"monthly"``). Anything else — empty, ``"flexible"``, free prose — sets no
    minimum, so the cadence gate stays inert (the default ``cadence=""`` never
    gates).
    """
    text = cadence.strip().lower()
    if not text:
        return None
    if text in _CADENCE_WORDS:
        return _CADENCE_WORDS[text]
    for suffix in sorted(_UNIT_MINUTES, key=len, reverse=True):
        if text.endswith(suffix):
            value = _try_float(text[: -len(suffix)].strip())
            if value is not None and value >= 0.0:
                return value * _UNIT_MINUTES[suffix]
    bare = _try_float(text)
    if bare is not None and bare >= 0.0:
        return bare
    return None


def _matches(text: str, needles: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(needle in low for needle in needles)


def _blanket_no_contact(
    privacy_boundaries: tuple[str, ...], topic_sensitivity: tuple[str, ...]
) -> bool:
    tokens = set(privacy_boundaries) | set(topic_sensitivity)
    return bool(tokens & BLANKET_NO_CONTACT)


def _minutes_since_last_send(send_log: list[str], now: datetime) -> float | None:
    """Minutes since the most recent parseable proactive send in *send_log*.

    ``None`` when the log is empty or holds no usable (parseable, tz-aware, past)
    timestamp. This is the cadence gate's OWN reading of the send log — the
    owner's min-spacing preference — NOT the send backstop's ≤3/day-≥60m rule
    (that stays in aggregation/egress); the two never share a computation.
    """
    latest_gap: float | None = None
    for ts in send_log:
        try:
            sent = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if sent.tzinfo is None or sent.utcoffset() is None:
            continue
        gap = (now - sent).total_seconds() / 60.0
        if gap < 0.0:
            continue
        if latest_gap is None or gap < latest_gap:
            latest_gap = gap
    return latest_gap


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None

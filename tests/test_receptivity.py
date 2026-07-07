# tests/test_receptivity.py
#
# The pure 0-LLM receptivity appraisal (lm-27n.5). Behavior-neutral by default:
# the permissive DEFAULT_RELATIONSHIP returns allowed/1.0 with no reasons, so
# aggregation/cognition behave exactly as .4. Only EXPLICIT boundaries hard-veto;
# weak norms only down-weight; styles/topics are constraints, not vetoes.
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.receptivity import (
    MIN_MULTIPLIER,
    ReceptivityResult,
    appraise_receptivity,
    cadence_min_minutes,
)
from lifemodel.core.relationship_view import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RELATIONSHIP,
    EXPLICIT_CONFIDENCE,
    build_owner_relationship,
)
from lifemodel.state.model import State

NOW = datetime(2026, 7, 6, 3, 0, tzinfo=UTC)  # hour 03 UTC


def _explicit(**prefs):
    return build_owner_relationship(confidence=EXPLICIT_CONFIDENCE, **prefs)


# --- behavior-neutral default ------------------------------------------------


def test_default_relationship_is_permissive() -> None:
    r = appraise_receptivity(DEFAULT_RELATIONSHIP, State(), NOW)
    assert r.allowed is True
    assert r.pressure_multiplier == 1.0
    assert r.hard_reasons == ()
    assert r.soft_reasons == ()
    assert r.constraints == ()
    assert r.confidence == DEFAULT_CONFIDENCE  # low: a seed, not an explicit norm


def test_default_never_hard_vetoes_even_at_a_populated_but_low_confidence_hour() -> None:
    # A LOW-confidence (seeded/would-be-inferred) bad hour NEVER hard-vetoes; it
    # only soft down-weights — the "being doesn't disappear" guard.
    rel = build_owner_relationship(bad_hours=(3,), confidence=DEFAULT_CONFIDENCE)
    r = appraise_receptivity(rel, State(), NOW)
    assert r.allowed is True
    assert r.pressure_multiplier < 1.0
    assert "inferred bad hour" in r.soft_reasons


# --- hours_fit (explicit quiet hours -> hard veto) ---------------------------


def test_explicit_bad_hour_hard_vetoes() -> None:
    r = appraise_receptivity(_explicit(bad_hours=(3, 4)), State(), NOW)
    assert r.allowed is False
    assert any("quiet hours" in reason for reason in r.hard_reasons)


def test_explicit_bad_hour_outside_the_hour_allows() -> None:
    # bad hours 4,5 but now is 03 -> not in a bad hour -> allowed.
    r = appraise_receptivity(_explicit(bad_hours=(4, 5)), State(), NOW)
    assert r.allowed is True
    assert r.hard_reasons == ()


# --- good_hours (preferred window -> SOFT down-weight, never a veto) ---------


def test_outside_preferred_hours_is_soft_downweight_not_veto() -> None:
    # now is 03; preferred hours are 9,10 -> OUTSIDE -> soft, still allowed.
    r = appraise_receptivity(_explicit(good_hours=(9, 10)), State(), NOW)
    assert r.allowed is True  # a preference, not a boundary
    assert r.pressure_multiplier < 1.0
    assert any("preferred hours" in s for s in r.soft_reasons)


def test_inside_preferred_hours_no_downweight() -> None:
    r = appraise_receptivity(_explicit(good_hours=(NOW.hour,)), State(), NOW)
    assert r.allowed is True
    assert r.pressure_multiplier == 1.0


def test_empty_good_hours_is_inert() -> None:
    # the default (no preference) never down-weights.
    r = appraise_receptivity(_explicit(good_hours=()), State(), NOW)
    assert r.pressure_multiplier == 1.0
    assert r.soft_reasons == ()


# --- cadence_fit (explicit min spacing -> hard veto) -------------------------


def test_explicit_cadence_min_vetoes_when_inside_the_gap() -> None:
    # min 2h spacing; last proactive send 30 min ago -> inside gap -> veto.
    state = State(proactive_send_log=["2026-07-06T02:30:00+00:00"])
    r = appraise_receptivity(_explicit(cadence="2h"), state, NOW)
    assert r.allowed is False
    assert any("cadence" in reason for reason in r.hard_reasons)


def test_explicit_cadence_allows_when_gap_elapsed() -> None:
    # min 2h; last send 3h ago -> gap elapsed -> allowed.
    state = State(proactive_send_log=["2026-07-06T00:00:00+00:00"])
    r = appraise_receptivity(_explicit(cadence="2h"), state, NOW)
    assert r.allowed is True


def test_cadence_with_empty_send_log_allows() -> None:
    r = appraise_receptivity(_explicit(cadence="2h"), State(), NOW)
    assert r.allowed is True  # never sent -> no spacing to violate


def test_cadence_min_minutes_parsing() -> None:
    assert cadence_min_minutes("") is None
    assert cadence_min_minutes("flexible") is None
    assert cadence_min_minutes("120") == 120.0
    assert cadence_min_minutes("2h") == 120.0
    assert cadence_min_minutes("90m") == 90.0
    assert cadence_min_minutes("1d") == 1440.0
    assert cadence_min_minutes("daily") == 1440.0
    assert cadence_min_minutes("weekly") == 10080.0


# --- privacy_fit (explicit blanket no-contact -> hard veto) ------------------


def test_blanket_no_contact_privacy_boundary_vetoes() -> None:
    r = appraise_receptivity(_explicit(privacy_boundaries=("no_proactive_contact",)), State(), NOW)
    assert r.allowed is False
    assert any("privacy" in reason for reason in r.hard_reasons)


def test_topic_sensitivity_is_a_constraint_not_a_veto() -> None:
    # A topic-scoped sensitivity cannot be evaluated at a topic-less wake -> it is
    # recorded as a composing constraint, never a veto.
    r = appraise_receptivity(_explicit(topic_sensitivity=("work",)), State(), NOW)
    assert r.allowed is True
    assert any("work" in c for c in r.constraints)


def test_non_blanket_privacy_boundary_is_a_constraint() -> None:
    r = appraise_receptivity(_explicit(privacy_boundaries=("no health details",)), State(), NOW)
    assert r.allowed is True
    assert any("no health details" in c for c in r.constraints)


# --- style_fit (constraint, never a veto) ------------------------------------


def test_acceptable_styles_are_a_constraint_not_a_veto() -> None:
    r = appraise_receptivity(_explicit(acceptable_styles=("playful", "concise")), State(), NOW)
    assert r.allowed is True
    assert any("playful" in c and "concise" in c for c in r.constraints)


# --- soft down-weight (weak norms) -------------------------------------------


def test_weak_negative_valence_down_weights_but_allows() -> None:
    r = appraise_receptivity(_explicit(response_valence_pattern="warm-but-slow"), State(), NOW)
    assert r.allowed is True
    assert r.pressure_multiplier < 1.0
    assert "weak negative valence" in r.soft_reasons


def test_known_load_down_weights_but_allows() -> None:
    r = appraise_receptivity(_explicit(known_load="busy at work"), State(), NOW)
    assert r.allowed is True
    assert r.pressure_multiplier < 1.0
    assert "known load" in r.soft_reasons


def test_stacked_soft_norms_floor_the_multiplier() -> None:
    rel = _explicit(
        response_valence_pattern="cold and distant",
        known_load="swamped",
        reply_latency_norm="days",
    )
    r = appraise_receptivity(rel, State(), NOW)
    assert r.allowed is True  # soft never vetoes
    assert r.pressure_multiplier == MIN_MULTIPLIER  # floored, never zero


# --- no re-derivation of the existing aggregation gates ----------------------


def test_appraisal_ignores_silence_window_and_decline_backoff() -> None:
    # A state that would be inside the silence window AND inside decline backoff
    # still returns allowed from appraise: those gates live in aggregation, not
    # here — appraise must not re-derive them.
    state = State(
        last_exchange_at="2026-07-06T02:59:00+00:00",  # 1 min ago (silence window)
        declined_at="2026-07-06T02:59:00+00:00",  # just declined (backoff)
        decline_count=5,
        action_pending_since="2026-07-06T02:59:00+00:00",  # ActionPending grace
    )
    r = appraise_receptivity(DEFAULT_RELATIONSHIP, state, NOW)
    assert r.allowed is True
    assert r.pressure_multiplier == 1.0
    assert isinstance(r, ReceptivityResult)

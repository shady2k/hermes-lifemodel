"""The 0-LLM attention score + decay/park arithmetic (lm-27n.7), unit-tested pure.

Every function here is a total function of its args — no store, no LLM, no clock
beyond the ``now`` passed in — so it is pinned in isolation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from lifemodel.core.thought_score import (
    MAX_PARK_CYCLES,
    PARK_AFTER,
    SAFETY_FLOOR,
    SALIENCE_FLOOR,
    THOUGHT_SALIENCE_HALFLIFE_MIN,
    attention_score,
    decay_salience,
    loop_signature,
    novelty,
    park_backoff_hours,
    park_window_elapsed,
    relevance,
    safety,
    unresolvedness,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.objects import Sensitivity, Thought, ThoughtState

_NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
_BASE = build_thought(id="t1", content="turning this over")


def _t(**kw: object) -> Thought:
    # ``replace`` reaches every field — including the envelope ``sensitivity`` and
    # ``state`` that the ``build_thought`` constructor does not surface — so the
    # pure-score factors can be exercised across the whole thought shape.
    return replace(_BASE, **kw)  # type: ignore[arg-type]


# --- unresolvedness: the one factor that may veto (0.0) ----------------------


def test_unresolvedness_active_is_one() -> None:
    assert unresolvedness(_t(state=ThoughtState.ACTIVE), _NOW) == 1.0


def test_unresolvedness_parked_within_window_is_zero_veto() -> None:
    future = (_NOW + timedelta(hours=1)).isoformat()
    t = _t(state=ThoughtState.PARKED, parked_until=future)
    assert unresolvedness(t, _NOW) == 0.0  # suspended: a real veto


def test_unresolvedness_parked_elapsed_is_reentry_discount() -> None:
    past = (_NOW - timedelta(hours=1)).isoformat()
    t = _t(state=ThoughtState.PARKED, parked_until=past)
    score = unresolvedness(t, _NOW)
    assert 0.0 < score < 1.0  # re-entry candidate, discounted below a fresh active


# --- novelty: progress penalty, not recency ---------------------------------


def test_novelty_falls_with_no_progress_count() -> None:
    assert novelty(0) == 1.0
    assert novelty(1) == 0.5
    assert novelty(3) == 0.25
    assert novelty(0) > novelty(1) > novelty(3)  # monotone decreasing


# --- safety: privacy penalty, floored, altruism-liftable ---------------------


def test_safety_decreases_with_sensitivity() -> None:
    normal = _t(sensitivity=Sensitivity.NORMAL)
    sensitive = _t(sensitivity=Sensitivity.SENSITIVE)
    private = _t(sensitivity=Sensitivity.PRIVATE)
    assert safety(normal) > safety(sensitive) > safety(private)
    assert safety(normal) == 1.0
    assert safety(private) >= SAFETY_FLOOR  # never a hard zero


def test_other_regarding_value_lifts_safety_only_when_not_private() -> None:
    sensitive_altruistic = _t(sensitivity=Sensitivity.SENSITIVE, other_regarding_value=1.0)
    sensitive_selfish = _t(sensitivity=Sensitivity.SENSITIVE, other_regarding_value=0.0)
    assert safety(sensitive_altruistic) > safety(sensitive_selfish)
    private_altruistic = _t(sensitivity=Sensitivity.PRIVATE, other_regarding_value=1.0)
    private_selfish = _t(sensitivity=Sensitivity.PRIVATE, other_regarding_value=0.0)
    assert safety(private_altruistic) == safety(private_selfish)  # PRIVATE stays guarded


# --- relevance: max(actionability, other-regarding, trigger family) ----------


def test_relevance_prefers_drive_event_over_idle_over_chain() -> None:
    drive = _t(trigger="drive:contact")
    event = _t(trigger="event:owner_reply")
    idle = _t(trigger="idle")
    chain = _t(trigger="thought:parent")
    assert relevance(drive) == relevance(event)
    assert relevance(drive) > relevance(idle) > relevance(chain)


def test_relevance_takes_the_max_of_appraisal_and_trigger() -> None:
    # A high actionability beats even an idle trigger's relevance.
    t = _t(trigger="idle", actionability=0.95)
    assert relevance(t) == 0.95


def test_relevance_is_never_zero() -> None:
    t = _t(trigger="whatever", actionability=0.0, other_regarding_value=0.0)
    assert relevance(t) > 0.0  # a trigger-family floor keeps it positive


# --- attention_score: the bounded product -----------------------------------


def test_attention_score_is_bounded_in_unit_interval() -> None:
    t = _t(salience=5.0, actionability=9.0, other_regarding_value=9.0)  # over-1 inputs
    assert 0.0 <= attention_score(t, _NOW) <= 1.0


def test_attention_score_is_the_factor_product() -> None:
    t = _t(salience=0.8, trigger="idle", no_progress_count=1)
    expected = (
        0.8 * unresolvedness(t, _NOW) * novelty(t.no_progress_count) * safety(t) * relevance(t)
    )
    assert abs(attention_score(t, _NOW) - expected) < 1e-12


def test_attention_score_zero_for_parked_within_window() -> None:
    future = (_NOW + timedelta(hours=5)).isoformat()
    t = _t(state=ThoughtState.PARKED, salience=1.0, parked_until=future)
    assert attention_score(t, _NOW) == 0.0  # the veto annihilates the product


def test_no_progress_lowers_the_score() -> None:
    fresh = _t(salience=0.7)
    looping = _t(salience=0.7, no_progress_count=4)
    assert attention_score(fresh, _NOW) > attention_score(looping, _NOW)


# --- decay: multiplicative half-life, monotone, floored ----------------------


def test_decay_halves_over_one_halflife() -> None:
    assert abs(decay_salience(1.0, THOUGHT_SALIENCE_HALFLIFE_MIN) - 0.5) < 1e-9


def test_decay_is_robust_to_irregular_ticks() -> None:
    # A 60-min gap decays sixty times as much as a 1-min gap (function of elapsed
    # real time, not tick count).
    one_min = decay_salience(1.0, 1.0)
    sixty_min = decay_salience(1.0, 60.0)
    assert sixty_min < one_min < 1.0


def test_decay_is_monotone_non_increasing_and_nonnegative() -> None:
    prev = 1.0
    for elapsed in (0.0, 10.0, 100.0, 1000.0, 100000.0):
        cur = decay_salience(1.0, elapsed)
        assert 0.0 <= cur <= prev
        prev = cur


def test_decay_no_elapsed_or_skew_leaves_salience_unchanged() -> None:
    assert decay_salience(0.6, 0.0) == 0.6
    assert decay_salience(0.6, -5.0) == 0.6  # clock skew never grows salience


# --- park backoff + window + loop signature ----------------------------------


def test_park_backoff_is_exponential_and_capped() -> None:
    assert park_backoff_hours(1) == 6.0
    assert park_backoff_hours(2) == 24.0
    assert park_backoff_hours(3) == 72.0
    assert park_backoff_hours(4) == 72.0  # capped, never grows past the last band
    assert park_backoff_hours(1) < park_backoff_hours(2) < park_backoff_hours(3)


def test_park_window_elapsed() -> None:
    assert park_window_elapsed((_NOW - timedelta(minutes=1)).isoformat(), _NOW) is True
    assert park_window_elapsed((_NOW + timedelta(minutes=1)).isoformat(), _NOW) is False
    assert park_window_elapsed(None, _NOW) is True  # absent → immediately re-entrant


def test_loop_signature_normalizes_content_and_is_deterministic() -> None:
    a = loop_signature(_t(content="The  Owner   seems Sad", trigger="idle"))
    b = loop_signature(_t(content="the owner seems sad", trigger="idle"))
    assert a == b  # whitespace/case-insensitive
    assert a.startswith("idle:")  # trigger family prefix
    diff = loop_signature(_t(content="a wholly different thought", trigger="idle"))
    assert a != diff


def test_thresholds_are_sane() -> None:
    assert PARK_AFTER == 3
    assert MAX_PARK_CYCLES == 3
    assert 0.0 < SALIENCE_FLOOR < 0.1

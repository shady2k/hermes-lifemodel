"""The appraisal seam — the deterministic slice-1 ``HeuristicAppraiser`` (lm-705.1 Task 2)."""

from __future__ import annotations

from lifemodel.core.appraisal import HeuristicAppraiser


def test_heuristic_seeds_on_a_forward_reference():
    seed = HeuristicAppraiser().appraise(
        user_message="I've got a dentist appointment on Friday, dreading it",
        assistant_response="Ah, hope it goes smoothly — tell me how it went?",
    )
    assert seed is not None
    assert "friday" in seed.content.lower() or "dentist" in seed.content.lower()
    assert 0.0 < seed.salience <= 1.0


def test_heuristic_declines_on_small_talk():
    seed = HeuristicAppraiser().appraise(
        user_message="ok thanks",
        assistant_response="anytime!",
    )
    assert seed is None


def test_heuristic_declines_on_empty():
    assert HeuristicAppraiser().appraise(user_message="", assistant_response="") is None

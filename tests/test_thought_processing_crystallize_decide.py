from datetime import UTC, datetime

from lifemodel.core.thought_processing import (
    ProcessingReason,
    decide_processing_transition,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.objects import ThoughtState

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


def _t(no_progress=0):
    return build_thought(
        id="thought:seed:x", content="interview Friday", no_progress_count=no_progress
    )


_GOOD = {
    "content": "ask how their interview went",
    "basis": "follow_up",
    "trigger_kind": "event",
    "trigger_value": "next time we talk",
}


def test_crystallize_carries_the_fields_and_leaves_the_transition_to_apply():
    d = decide_processing_transition(
        _t(),
        parsed={"outcome": "crystallize_commitment", "commitment": _GOOD},
        raw="{...}",
        now=NOW,
    )
    assert d.reason == ProcessingReason.CRYSTALLIZED_COMMITMENT
    assert d.crystallize == _GOOD  # the validated sub-object rides the decision
    assert d.transition is None  # apply computes active→resolved after a successful build


def test_crystallize_without_a_commitment_object_is_no_progress():
    d = decide_processing_transition(
        _t(), parsed={"outcome": "crystallize_commitment"}, raw="{...}", now=NOW
    )
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS
    assert d.crystallize is None
    assert d.transition.to_state == ThoughtState.PARKED.value


def test_crystallize_with_a_non_object_commitment_is_no_progress():
    d = decide_processing_transition(
        _t(),
        parsed={"outcome": "crystallize_commitment", "commitment": "oops"},
        raw="{...}",
        now=NOW,
    )
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS


def test_existing_outcomes_unchanged():
    assert (
        decide_processing_transition(_t(), parsed={"outcome": "resolve"}, raw="x", now=NOW).reason
        == ProcessingReason.RESOLVED
    )
    assert (
        decide_processing_transition(_t(), parsed=None, raw="   ", now=NOW).reason
        == ProcessingReason.TRANSIENT_FAILURE
    )

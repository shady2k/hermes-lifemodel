from datetime import UTC, datetime

from lifemodel.core.thought_processing import (
    MAX_NO_PROGRESS_COUNT,
    MAX_PARK_CYCLES,
    ProcessingReason,
    decide_processing_transition,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.objects import ThoughtState

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)


def _t(*, no_progress=0, park=0):
    return build_thought(
        id="thought:seed:abc",
        content="dentist on Friday",
        no_progress_count=no_progress,
        park_count=park,
    )


def test_resolve_is_terminal():
    d = decide_processing_transition(_t(), parsed={"outcome": "resolve"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.RESOLVED
    assert d.transition.from_state == ThoughtState.ACTIVE.value
    assert d.transition.to_state == ThoughtState.RESOLVED.value


def test_drop_is_terminal():
    d = decide_processing_transition(_t(), parsed={"outcome": "drop"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.DROPPED
    assert d.transition.to_state == ThoughtState.DROPPED.value


def test_park_sets_backoff_and_bumps_park_count():
    d = decide_processing_transition(_t(park=0), parsed={"outcome": "park"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.PARKED
    assert d.transition.to_state == ThoughtState.PARKED.value
    assert d.transition.patch.payload_merge["park_count"] == 1
    assert (
        d.transition.patch.payload_merge["parked_until"] == "2026-07-16T18:00:00.000000+00:00"
    )  # +6h


def test_park_at_cap_expires_instead():
    d = decide_processing_transition(
        _t(park=MAX_PARK_CYCLES), parsed={"outcome": "park"}, raw="{...}", now=NOW
    )
    assert d.reason == ProcessingReason.EXPIRED_PARK_CAP
    assert d.transition.to_state == ThoughtState.EXPIRED.value


def test_malformed_parks_and_bumps_no_progress():
    d = decide_processing_transition(_t(no_progress=0), parsed=None, raw="not json at all", now=NOW)
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS
    assert d.transition.to_state == ThoughtState.PARKED.value
    assert d.transition.patch.payload_merge["no_progress_count"] == 1


def test_malformed_at_no_progress_cap_drops():
    d = decide_processing_transition(
        _t(no_progress=MAX_NO_PROGRESS_COUNT - 1), parsed=None, raw="junk", now=NOW
    )
    assert d.reason == ProcessingReason.DROPPED_NO_PROGRESS
    assert d.transition.to_state == ThoughtState.DROPPED.value
    assert d.transition.patch.payload_merge["no_progress_count"] == MAX_NO_PROGRESS_COUNT


def test_transient_failure_does_not_touch_the_thought():
    d = decide_processing_transition(_t(), parsed=None, raw="   ", now=NOW)
    assert d.reason == ProcessingReason.TRANSIENT_FAILURE
    assert d.transition is None  # thought stays active, retried next interval


def test_unknown_outcome_string_is_malformed_not_a_crash():
    d = decide_processing_transition(_t(), parsed={"outcome": "banana"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS

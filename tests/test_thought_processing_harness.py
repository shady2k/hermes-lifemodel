"""Real-code sim: backlog health, bounds terminate, idle 0-LLM, cost <= ceiling
(lm-705.2 Task 8, spec §6).

Drives the REAL frame (:func:`run_frame` -> the registered
:class:`~lifemodel.core.thought_processing.ThoughtProcessingSelector`) and the REAL
completion (:func:`run_internal_completion` -> the injected
:class:`~lifemodel.core.thought_processing.ThoughtProcessingApply`) over the actual
on-disk store (:func:`build_processing_lifemodel`) -- not mocks, so a green
scenario honestly predicts live behaviour. The harness has no gateway asyncio loop,
so the runner's own reserve (setting ``pending_internal_id``/
``pending_internal_subject_id``) is simulated with a direct ``UpdateState`` under
the same state-actor lock the real runner would take.
"""

from __future__ import annotations

from pathlib import Path

from lifemodel.composition import LifeModel
from lifemodel.core.budget import DEFAULT_DAILY_INTERNAL_CALL_CEILING
from lifemodel.core.frame import FrameTrigger, run_frame, state_actor_lock
from lifemodel.core.intents import LaunchInternalCognition, UpdateState
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.thought_processing import (
    MAX_NO_PROGRESS_COUNT,
    MAX_PARK_CYCLES,
    ThoughtProcessingApply,
)
from lifemodel.core.thought_view import build_thought, encode_thought, read_thought
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import ThoughtState
from lifemodel.testing.harness import build_processing_lifemodel

TARGET: dict[str, str | None] = {"platform": "test", "chat_id": "1", "thread_id": None}


class _FakeEgress:
    """A no-op :class:`~lifemodel.ports.proactive.ProactiveEgressPort` -- these
    scenarios never expect a delivery (non-delivery is structural, spec §4.1),
    but ``run_internal_completion`` still needs a real port to hand to
    ``dispatch_launches``."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        self.calls.append((target, impulse))
        return ReachOutcome.DELIVERED


def _seed_active_thought(
    lm: LifeModel,
    *,
    id: str,
    content: str,
    salience: float,
    no_progress_count: int = 0,
    park_count: int = 0,
) -> None:
    thought = build_thought(
        id=id,
        content=content,
        state=ThoughtState.ACTIVE,
        salience=salience,
        no_progress_count=no_progress_count,
        park_count=park_count,
    )
    lm.state.put(encode_thought(thought))


def _set_pending(lm: LifeModel, launch: LaunchInternalCognition) -> None:
    """Simulate the runner's reserve (``InternalCognitionRunner.launch``, minus the
    async task): stamp the single-flight markers the selector's next look would see
    as "a pass is in flight", under the same lock a real frame takes."""
    assert lm.state_actor is not None
    with state_actor_lock():
        lm.state_actor.apply(
            [
                UpdateState(
                    {
                        "pending_internal_id": launch.correlation_id,
                        "pending_internal_subject_id": launch.subject_id,
                    }
                )
            ]
        )


def test_seeded_thought_is_processed_to_resolved(tmp_path: Path) -> None:
    """Backlog health: a seeded active thought is picked up and, on a ``resolve``
    completion, is no longer live -- the backlog neither starves nor spirals."""
    lm = build_processing_lifemodel(base_dir=tmp_path)
    _seed_active_thought(lm, id="thought:seed:a", content="dentist Friday", salience=0.8)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]
    assert launch.subject_id == "thought:seed:a"

    _set_pending(lm, launch)
    run_internal_completion(
        lm,
        _FakeEgress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw='{"outcome":"resolve"}', parsed={"outcome": "resolve"}),
        apply=ThoughtProcessingApply(),
    )

    assert read_thought(lm.state, "thought:seed:a") is None  # resolved -> no longer live


def test_idle_empty_backlog_is_zero_launches(tmp_path: Path) -> None:
    """Idle 0-LLM: an empty backlog emits no launch on a heartbeat."""
    lm = build_processing_lifemodel(base_dir=tmp_path)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    assert report.internal_launches == ()


def test_repeated_malformed_drops_at_no_progress_cap(tmp_path: Path) -> None:
    """Bounds terminate (drop): a thought already one attempt short of the
    no-progress cap is DROPPED by a single further malformed completion -- bounded,
    never spiralling."""
    lm = build_processing_lifemodel(base_dir=tmp_path)
    _seed_active_thought(
        lm,
        id="thought:seed:a",
        content="c",
        salience=0.8,
        no_progress_count=MAX_NO_PROGRESS_COUNT - 1,
    )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]

    _set_pending(lm, launch)
    run_internal_completion(
        lm,
        _FakeEgress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw="junk", parsed=None),
        apply=ThoughtProcessingApply(),
    )

    assert read_thought(lm.state, "thought:seed:a") is None  # dropped -- bounded, no spiral


def test_park_cap_expires_thought(tmp_path: Path) -> None:
    """Bounds terminate (expire): a thought already at the park cap EXPIRES on a
    further ``park`` completion instead of parking forever."""
    lm = build_processing_lifemodel(base_dir=tmp_path)
    _seed_active_thought(
        lm, id="thought:seed:b", content="c", salience=0.8, park_count=MAX_PARK_CYCLES
    )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]

    _set_pending(lm, launch)
    run_internal_completion(
        lm,
        _FakeEgress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw='{"outcome":"park"}', parsed={"outcome": "park"}),
        apply=ThoughtProcessingApply(),
    )

    assert read_thought(lm.state, "thought:seed:b") is None  # expired at the park cap


def test_daily_ceiling_caps_launches(tmp_path: Path) -> None:
    """Cost <= FR20 ceiling: with today's counter already at the daily ceiling, the
    selector's budget gate blocks the launch on a live backlog."""
    lm = build_processing_lifemodel(base_dir=tmp_path)
    _seed_active_thought(lm, id="thought:seed:a", content="c", salience=0.8)
    assert lm.state_actor is not None
    with state_actor_lock():
        lm.state_actor.apply(
            [
                UpdateState(
                    {
                        "internal_calls_today": DEFAULT_DAILY_INTERNAL_CALL_CEILING,
                        # matches build_processing_lifemodel's default clock day
                        "internal_calls_day": "2026-01-01",
                    }
                )
            ]
        )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    assert report.internal_launches == ()

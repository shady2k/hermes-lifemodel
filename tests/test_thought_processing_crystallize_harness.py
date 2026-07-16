"""Real-code sim: crystallization end-to-end, zero-launch, bounded atomic failure
(lm-705.3 Task 5, spec §6 crystallization-slice).

Drives the REAL frame (:func:`run_frame` -> the registered
:class:`~lifemodel.core.thought_processing.ThoughtProcessingSelector`) and the REAL
completion (:func:`run_internal_completion` -> the injected
:class:`~lifemodel.core.thought_processing.ThoughtProcessingApply`) over the actual
on-disk store (:func:`build_processing_lifemodel`) -- not mocks, so a green scenario
honestly predicts live behaviour. Mirrors ``tests/test_thought_processing_harness.py``
(slice 2) exactly, adding the crystallization outcome:

* a seeded thought crystallizes into a real ``kind=commitment`` row and the source
  thought resolves (no longer live);
* crystallization is non-delivering -- with NO pre-existing contact desire, the
  completion frame reaches the fake egress ZERO times (codex I4 no-send acceptance);
* a domain-invalid commitment payload (``basis="BAD"``) is an ATOMIC failure: no
  stray commitment row, and the source thought falls back to the bounded
  no-progress path (still live, ``no_progress_count == 1``) rather than vanishing.
"""

from __future__ import annotations

from lifemodel.composition import LifeModel
from lifemodel.core.commitment_view import read_live_commitments
from lifemodel.core.frame import FrameTrigger, run_frame, state_actor_lock
from lifemodel.core.intents import LaunchInternalCognition, UpdateState
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.thought_processing import ThoughtProcessingApply
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


def _fake_egress() -> _FakeEgress:
    return _FakeEgress()


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


def test_seeded_thought_crystallizes_into_a_real_commitment() -> None:
    """Crystallization: a seeded thought's ``crystallize_commitment`` completion
    produces a real ``Commitment`` row (content + ``source_thought_ids``) and the
    source thought is resolved (no longer live)."""
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="interview Friday", salience=0.8)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]
    assert launch.subject_id == "thought:seed:x"
    _set_pending(lm, launch)

    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw='{"outcome":"crystallize_commitment","commitment":{'
            '"content":"ask how their interview went","basis":"follow_up",'
            '"trigger_kind":"event","trigger_value":"next time we talk"}}',
            parsed={
                "outcome": "crystallize_commitment",
                "commitment": {
                    "content": "ask how their interview went",
                    "basis": "follow_up",
                    "trigger_kind": "event",
                    "trigger_value": "next time we talk",
                },
            },
        ),
        apply=ThoughtProcessingApply(),
    )

    commitments = read_live_commitments(lm.state)
    assert len(commitments) == 1
    assert commitments[0].content == "ask how their interview went"
    assert commitments[0].source_thought_ids == ("thought:seed:x",)
    assert read_thought(lm.state, "thought:seed:x") is None  # source thought resolved


def test_crystallization_emits_no_proactive_launch() -> None:
    """No-send (codex I4 acceptance): with NO pre-existing contact desire, a
    crystallization completion frame reaches the fake egress ZERO times --
    non-delivery is structural, not incidental."""
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="interview Friday", salience=0.8)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = report.internal_launches[0]
    _set_pending(lm, launch)

    egress = _fake_egress()
    run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw='{"outcome":"crystallize_commitment","commitment":{"content":"c",'
            '"basis":"follow_up","trigger_kind":"event","trigger_value":"later"}}',
            parsed={
                "outcome": "crystallize_commitment",
                "commitment": {
                    "content": "c",
                    "basis": "follow_up",
                    "trigger_kind": "event",
                    "trigger_value": "later",
                },
            },
        ),
        apply=ThoughtProcessingApply(),
    )

    assert egress.calls == []  # non-delivery, structural


def test_atomicity_bad_commitment_leaves_no_row_and_no_resolve() -> None:
    """Atomic failure: a domain-invalid ``basis`` leaves NO stray commitment row --
    the source thought falls back to the bounded no-progress path (still live,
    ``no_progress_count == 1``), never silently vanishing with nothing to show."""
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="c", salience=0.8)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = report.internal_launches[0]
    _set_pending(lm, launch)

    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw='{"outcome":"crystallize_commitment","commitment":{"content":"c",'
            '"basis":"BAD","trigger_kind":"event","trigger_value":"later"}}',
            parsed={
                "outcome": "crystallize_commitment",
                "commitment": {
                    "content": "c",
                    "basis": "BAD",
                    "trigger_kind": "event",
                    "trigger_value": "later",
                },
            },
        ),
        apply=ThoughtProcessingApply(),
    )

    assert read_live_commitments(lm.state) == ()  # no stray commitment
    t = read_thought(lm.state, "thought:seed:x")
    assert t is not None and t.no_progress_count == 1  # bounded no-progress, still live

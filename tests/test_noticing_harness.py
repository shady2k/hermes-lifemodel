"""Real-code sim: buffer -> thoughts, source ids, dedup, 0-LLM idle, strand-dispatch
(lm-705.5 Task 7, spec §6 noticing-slice).

Drives the REAL frame (:func:`run_frame` -> the registered
:class:`~lifemodel.core.noticing.NoticingTrigger`) and the REAL completion
(:func:`run_internal_completion` -> the injected
:class:`~lifemodel.core.noticing.NoticingApply`) over the actual on-disk store
(:func:`build_noticing_lifemodel`) and a real
:class:`~lifemodel.core.noticing_buffer.NoticingBuffer` -- not mocks, so a green
scenario honestly predicts live behaviour. Mirrors
``tests/test_thought_processing_crystallize_harness.py``/``test_thought_processing_harness.py``
(slices 2/3) exactly: the harness has no gateway asyncio loop, so the runner's own
reserve (setting ``pending_internal_id``/``pending_internal_subject_id``) is simulated
with a direct ``UpdateState`` under the same state-actor lock the real runner would take.

``build_noticing_lifemodel`` builds the ORDINARY real graph, so
:class:`~lifemodel.core.thought_processing.ThoughtProcessingSelector` is ALSO live
(``build_lifemodel`` always registers it) -- a scenario that seeds an ACTIVE thought
(continuity/top-K/dedup) can make BOTH selectors due on the SAME heartbeat. That is
real coexistence (spec §6 Task 6 reconciliation: two selectors sharing one seam,
disambiguated by ``pending_internal_subject_id``), never a bug, so
:func:`_only_noticing_launch` filters to the subjectless (noticing) one -- exactly
what a real dispatch loop would still find and route correctly regardless of what
order the frame happened to emit them in.

Each test's ``NoticingApply(buffer)`` is a FRESH instance sharing the SAME *buffer*
already wired as the STANDING completion consumer by ``build_noticing_lifemodel``
(``composition.py``'s ``noticing_buffer=`` branch) -- passing it again to
``run_internal_completion`` is idempotent (the frame actually runs whichever instance
the registry already holds, keyed by ``NoticingApply.id``; both wrap the identical
buffer, so which one runs is behaviour-irrelevant), mirroring how the real runner's
injected ``apply`` interacts with this seam (see ``core/internal_cognition.py``'s
module docstring).
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.composition import LifeModel
from lifemodel.core.coreloop import TickReport
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.frame import FrameTrigger, run_frame, state_actor_lock
from lifemodel.core.intents import LaunchInternalCognition, UpdateState
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.noticing import (
    BACKLOG_TOP_M,
    DEFAULT_NOTICING_IDLE,
    DEFAULT_NOTICING_SIZE_CAP,
    NoticingApply,
)
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.thought_view import (
    build_thought,
    encode_thought,
    read_thought,
    seed_thought_id,
)
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import DesireState
from lifemodel.state.sqlite_store import SqliteBufferStore
from lifemodel.testing import FakeClock
from lifemodel.testing.harness import build_noticing_lifemodel

TARGET: dict[str, str | None] = {"platform": "test", "chat_id": "1", "thread_id": None}


def _noticing_lm() -> tuple[LifeModel, NoticingBuffer]:
    """A real-graph noticing ``LifeModel`` whose ``NoticingBuffer`` is backed by a
    durable :class:`SqliteBufferStore` over the SAME ``base_dir`` (and clock) as the
    runtime store — so the two share ONE ``lifemodel.sqlite``. This is load-bearing
    for the claim/finalize round-trip (lm-705.13): the trigger's ``claim`` and the
    apply's ``FinalizeBuffer`` DELETE both land in that one file, so a completed pass
    actually clears the surveyed prefix (an in-memory buffer would be invisible to
    the real committer's raw-SQL finalize, and the cursor would never advance)."""
    base_dir = Path(tempfile.mkdtemp(prefix="lifemodel-noticing-harness-"))
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    buffer = NoticingBuffer(store=SqliteBufferStore(base_dir, clock=clock))
    lm = build_noticing_lifemodel(buffer=buffer, base_dir=base_dir, clock=clock)
    return lm, buffer


class _FakeEgress:
    """A no-op :class:`~lifemodel.ports.proactive.ProactiveEgressPort` -- these
    scenarios never expect a delivery FROM THE NOTICING PASS ITSELF (non-delivery
    is structural, spec §4.1), but ``run_internal_completion`` still needs a real
    port to hand to ``dispatch_launches``."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        self.calls.append((target, impulse))
        return ReachOutcome.DELIVERED


def _fake_egress() -> _FakeEgress:
    return _FakeEgress()


def _complete_turn(
    buffer: NoticingBuffer,
    session_id: str,
    turn_id: str,
    *,
    user_text: str = "hi",
    assistant_text: str = "hello",
    ts: datetime,
) -> None:
    """Drive the buffer's real public API to land one ``complete`` entry."""
    buffer.open_pending(session_id, user_text=user_text, now=ts)
    buffer.complete(session_id, turn_id, assistant_text=assistant_text, now=ts)


def _set_pending(lm: LifeModel, launch: LaunchInternalCognition) -> None:
    """Simulate the runner's reserve (``InternalCognitionRunner.launch``, minus the
    async task): stamp the single-flight markers the trigger's next look would see
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


def _only_noticing_launch(report: TickReport) -> LaunchInternalCognition:
    """The one subjectless (noticing) launch this frame produced.

    A scenario that seeds an ACTIVE thought can make ``ThoughtProcessingSelector``
    due on the SAME heartbeat as the noticing trigger (real coexistence, spec §6
    Task 6 -- both selectors share one seam, disambiguated by
    ``pending_internal_subject_id``, not a same-tick single-flight race; the
    runner's own sequential ``launch()`` calls are what actually serialize two
    same-tick launches live). Filtering on ``subject_id is None`` finds the
    noticing one regardless of how many launches this frame produced or in what
    order.
    """
    noticing = [launch for launch in report.internal_launches if launch.subject_id is None]
    assert len(noticing) == 1, report.internal_launches
    return noticing[0]


def test_buffered_sitting_becomes_a_real_thought_with_source_ids() -> None:
    """A buffered sitting -> a real ``kind=thought`` row whose provenance carries
    the exact source ids/turn_id the model was shown (anti-hallucination + lineage)."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(
        buffer,
        "s1",
        "t1",
        user_text="I have an interview Friday",
        assistant_text="Good luck!",
        ts=old_ts,
    )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = _only_noticing_launch(report)
    assert launch.subject_id is None  # subjectless (noticing, not processing)
    _set_pending(lm, launch)

    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw='{"seeds":[{"gist":"they have an interview Friday",'
            '"source_message_ids":["t1"],"turn_id":"t1","salience":0.7}]}',
            parsed={
                "seeds": [
                    {
                        "gist": "they have an interview Friday",
                        "source_message_ids": ["t1"],
                        "turn_id": "t1",
                        "salience": 0.7,
                    }
                ]
            },
        ),
        apply=NoticingApply(buffer),
    )

    thought_id = seed_thought_id("they have an interview Friday")
    thought = read_thought(lm.state, thought_id)
    assert thought is not None
    assert thought.content == "they have an interview Friday"
    assert thought.source == "noticing"
    assert thought.provenance is not None
    assert thought.provenance.source_object_ids == ("t1",)
    assert thought.provenance.turn_id == "t1"


def test_continuity_uses_the_backlog_not_raw_old_text() -> None:
    """Continuity: a PRIOR thought is folded into the prompt (spec §4.2's "what am I
    already turning over"); a new pass's real thought THEMATICALLY builds on it, but
    is formally GROUNDED only in the new segment's own turn id -- continuity never
    resurrects raw old text as a source."""
    lm, buffer = _noticing_lm()
    prior = build_thought(
        id="thought:seed:prior", content="worried about the Friday interview", salience=0.9
    )
    lm.state.put(encode_thought(prior))

    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(
        buffer,
        "s1",
        "t-new",
        user_text="I'm nervous about it",
        assistant_text="that's understandable",
        ts=old_ts,
    )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = _only_noticing_launch(report)
    # the trigger's own 0-LLM prompt build folded the backlog in for real
    assert "worried about the Friday interview" in launch.prompt
    assert "t-new" in launch.prompt
    _set_pending(lm, launch)

    gist = "still turning over the interview nerves, and now noticing the anxiety too"
    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw="...",
            parsed={"seeds": [{"gist": gist, "source_message_ids": ["t-new"], "turn_id": "t-new"}]},
        ),
        apply=NoticingApply(buffer),
    )

    new_thought = read_thought(lm.state, seed_thought_id(gist))
    assert new_thought is not None
    # grounded ONLY in the segment actually surveyed this pass, never the old thought's id
    assert new_thought.provenance is not None
    assert new_thought.provenance.source_object_ids == ("t-new",)
    # continuity READ the prior thought; it did not consume/alter it
    assert read_thought(lm.state, "thought:seed:prior") is not None


def test_idle_elapsed_alone_fires_the_launch() -> None:
    """Idle ∨ size-cap (idle branch): a single old closed turn is due once it has
    sat past ``idle``, even nowhere near the size cap."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    _only_noticing_launch(report)


def test_size_cap_alone_fires_the_launch_even_though_idle_has_not_elapsed() -> None:
    """Idle ∨ size-cap (size-cap branch): a fresh, size-cap-deep segment is due
    even though none of it is anywhere near ``idle``."""
    lm, buffer = _noticing_lm()
    now = lm.clock.now()
    for i in range(DEFAULT_NOTICING_SIZE_CAP):
        _complete_turn(
            buffer, "s1", f"t{i}", ts=now - timedelta(seconds=DEFAULT_NOTICING_SIZE_CAP - i)
        )

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    _only_noticing_launch(report)


def test_pending_turn_blocks_the_launch() -> None:
    """The closed-prefix rule: a lane with a live (fresh, within-TTL) pending turn
    yields NO segment, even though its ALREADY-closed entries are well past idle."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)
    # a NEW turn opens (mid-flight) on the SAME lane, never completed
    buffer.open_pending("s1", user_text="still talking", now=lm.clock.now())

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    assert [launch for launch in report.internal_launches if launch.subject_id is None] == []
    assert buffer.closed_segment("s1", now=lm.clock.now()) == []


def test_cursor_clears_after_a_pass() -> None:
    """The cursor: a genuinely-surveyed segment is cleared through its anchor once
    the pass completes, fruitless or not -- never re-shown forever."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = _only_noticing_launch(report)
    _set_pending(lm, launch)

    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw="{}", parsed={"seeds": []}),
        apply=NoticingApply(buffer),
    )

    assert buffer.closed_segment("s1", now=lm.clock.now()) == []


def test_top_k_backlog_holds() -> None:
    """The bounded backlog (``BACKLOG_TOP_M``): with more live thoughts than the
    cap, only the top-K by salience are folded into the prompt."""
    lm, buffer = _noticing_lm()
    total = BACKLOG_TOP_M + 2
    for i in range(total):
        salience = (i + 1) / 10.0  # strictly ascending, so rank == i
        content = f"backlog thought number {i}"
        lm.state.put(
            encode_thought(
                build_thought(id=f"thought:seed:b{i}", content=content, salience=salience)
            )
        )
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    launch = _only_noticing_launch(report)
    prompt = launch.prompt
    included = {i for i in range(total) if f"backlog thought number {i}" in prompt}
    assert len(included) == BACKLOG_TOP_M
    # top-K BY SALIENCE -- the highest-salience (highest-i) ones, never the bottom
    assert included == set(range(total - BACKLOG_TOP_M, total))


def test_idle_with_empty_buffer_stays_zero_llm() -> None:
    """0-LLM idle: a heartbeat over a buffer that has never seen a single turn
    emits no launch at all."""
    lm, buffer = _noticing_lm()

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    assert report.internal_launches == ()


def test_internal_correlation_never_collides_with_pending_proactive_id() -> None:
    """Separate correlation spaces: an in-flight PROACTIVE turn does not block a
    noticing launch, and the noticing pass's own completion never touches
    ``pending_proactive_id``."""
    lm, buffer = _noticing_lm()
    assert lm.state_actor is not None
    with state_actor_lock():
        lm.state_actor.apply([UpdateState({"pending_proactive_id": "proactive-xyz"})])
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", "t1", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)

    launch = _only_noticing_launch(report)
    assert launch.correlation_id != "proactive-xyz"
    _set_pending(lm, launch)
    assert lm.state.load().pending_proactive_id == "proactive-xyz"  # untouched by the reserve

    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw="{}", parsed={"seeds": []}),
        apply=NoticingApply(buffer),
    )

    final = lm.state.load()
    assert final.pending_internal_id is None  # the internal correlation cleared
    assert final.pending_proactive_id == "proactive-xyz"  # separate space, never touched


def test_completion_frame_with_incidental_proactive_launch_dispatches_it() -> None:
    """The §3.2 strand regression, over a REAL noticing completion: an unrelated,
    already-registered ``CognitionLauncher`` incidentally waking on the SAME
    ASYNC_COMPLETION frame this noticing pass's own apply also runs on must still
    have its ``LaunchProactive`` dispatched -- and the noticing pass's own
    (non-delivered) work still lands in the same frame."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", "t1", user_text="a big life update", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = _only_noticing_launch(report)
    _set_pending(lm, launch)

    # ONLY after the noticing launch is captured does an unrelated proactive desire
    # become live -- mirrors test_internal_cognition.py's own codex-#2 regression
    # recipe exactly (u=3.0/energy=1.0/salience=3.0), so CognitionLauncher (which
    # runs on EVERY frame regardless of trigger) incidentally wakes on the SAME
    # completion frame this noticing pass's own apply runs on.
    assert lm.state_actor is not None
    with state_actor_lock():
        lm.state_actor.apply([UpdateState({"u": 3.0, "energy": 1.0})])
    lm.state.put(
        encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=3.0))
    )

    egress = _fake_egress()
    gist = "noticed the big life update"
    outcome = run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw="...",
            parsed={"seeds": [{"gist": gist, "source_message_ids": ["t1"], "turn_id": "t1"}]},
        ),
        apply=NoticingApply(buffer),
    )

    assert outcome is ReachOutcome.DELIVERED  # the incidental proactive launch WAS dispatched
    assert len(egress.calls) == 1
    # the noticing pass itself STILL did its own (non-delivered) work this same frame
    assert read_thought(lm.state, seed_thought_id(gist)) is not None
    final = lm.state.load()
    assert final.pending_internal_id is None  # the internal correlation still clears
    assert final.pending_proactive_id is not None  # a real proactive turn now in flight


def test_dedup_ring_prevents_a_duplicate_thought_on_a_resurvey_of_the_same_id() -> None:
    """The consumed-id ring dedups across a re-survey: a SECOND real pass whose
    (scripted) response cites a source id already consumed by an earlier pass is
    dropped -- no duplicate thought for that citation, one real thought total."""
    lm, buffer = _noticing_lm()
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=10)
    _complete_turn(buffer, "s1", "t1", user_text="first mention", ts=old_ts)

    report1 = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch1 = _only_noticing_launch(report1)
    _set_pending(lm, launch1)
    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch1.correlation_id,
        result=InternalCognitionResult(
            raw="...",
            parsed={
                "seeds": [
                    {
                        "gist": "first noticed thing",
                        "source_message_ids": ["t1"],
                        "turn_id": "t1",
                    }
                ]
            },
        ),
        apply=NoticingApply(buffer),
    )
    first_thought_id = seed_thought_id("first noticed thing")
    assert read_thought(lm.state, first_thought_id) is not None
    assert "t1" in lm.state.load().noticed_source_ids

    # a SECOND, later real pass resurfaces the SAME id "t1" (e.g. a redelivered/
    # replayed turn reusing the id) -- the trigger fires again for real; the
    # (scripted) response again cites "t1" as a source. (The FIRST pass's own real
    # thought is now itself live, so ThoughtProcessingSelector is ALSO due this
    # heartbeat -- real coexistence, filtered out by ``_only_noticing_launch``.)
    later_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=1)
    _complete_turn(buffer, "s1", "t1", user_text="first mention, again", ts=later_ts)

    report2 = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch2 = _only_noticing_launch(report2)
    _set_pending(lm, launch2)
    duplicate_gist = "noticed it again"
    run_internal_completion(
        lm,
        _fake_egress(),
        TARGET,
        correlation_id=launch2.correlation_id,
        result=InternalCognitionResult(
            raw="...",
            parsed={
                "seeds": [{"gist": duplicate_gist, "source_message_ids": ["t1"], "turn_id": "t1"}]
            },
        ),
        apply=NoticingApply(buffer),
    )

    # dropped -- "t1" is already in the consumed ring, so no thought for THIS gist
    assert read_thought(lm.state, seed_thought_id(duplicate_gist)) is None
    # the first pass's real thought is untouched -- one thought, no duplicate
    assert read_thought(lm.state, first_thought_id) is not None

"""``build_lifemodel``'s noticing wiring (lm-705.5, plan Task 6 / E5).

Wires the two noticing components (E4, ``core/noticing.py``) into the live
graph: ``NoticingTrigger`` (heartbeat emitter) and ``NoticingApply`` (the
STANDING completion-frame consumer) register only when a ``NoticingBuffer`` is
injected — ``None`` (every existing ``build_lifemodel`` caller) registers
neither, back-compat.

The end-to-end tests drive the REAL graph (``build_lifemodel`` +
``lm.coreloop.tick()`` + ``run_internal_completion``) — mirrors
``tests/test_internal_cognition.py``'s own style — so the coexistence
guarantee (a subject-set completion is ``ThoughtProcessingApply``'s; a
subjectless one is ``NoticingApply``'s; never both) is exercised through the
SAME wiring the live runner uses, not by calling the two components by hand
(already covered by ``tests/test_noticing_trigger.py``/``test_noticing_apply.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.composition import build_lifemodel
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.noticing import NOTICING_APPLY_ID, NOTICING_TRIGGER_ID
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought, read_live_thoughts
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State

BORN_AT = "2026-07-01T10:00:00+00:00"
TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}
NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class FakeEgress:
    """Records every ``reach_out`` call — an empty ``calls`` list proves the
    non-delivery invariant (nothing in this seam ever reaches the egress)."""

    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, object]] = []

    def reach_out(self, target: object, impulse: object) -> ReachOutcome:
        self.calls.append((target, impulse))
        return self.outcome


def _seeded_buffer() -> NoticingBuffer:
    """One closed turn on lane ``s1``, aged well past the trigger's default idle
    (30 min) as of :data:`NOW` — a due segment."""
    buffer = NoticingBuffer()
    old = NOW - timedelta(hours=1)
    buffer.open_pending("s1", user_text="hi", now=old)
    buffer.stamp_source("s1", "m1")
    buffer.complete("s1", "t1", assistant_text="hello", now=old + timedelta(seconds=1))
    return buffer


# --- registration -------------------------------------------------------------


def test_build_lifemodel_registers_noticing_pair_when_buffer_injected(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path, noticing_buffer=NoticingBuffer())
    ids = {m.id for m in lm.registry.manifests()}
    assert NOTICING_TRIGGER_ID in ids
    assert NOTICING_APPLY_ID in ids


def test_build_lifemodel_registers_neither_noticing_component_without_a_buffer(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)  # noticing_buffer=None, the default
    ids = {m.id for m in lm.registry.manifests()}
    assert NOTICING_TRIGGER_ID not in ids
    assert NOTICING_APPLY_ID not in ids


def test_build_lifemodel_is_idempotent_on_repeated_registration(tmp_path) -> None:
    # A shared registry (a caller that builds twice over the same ComponentRegistry,
    # mirroring the per-tick "resolved_registry.manifest(...) except UnknownComponent"
    # guard already used by every other component here) must not raise a
    # DuplicateComponent on the second pass.
    buffer = NoticingBuffer()
    registry = build_lifemodel(base_dir=tmp_path, noticing_buffer=buffer).registry
    lm2 = build_lifemodel(base_dir=tmp_path, registry=registry, noticing_buffer=buffer)
    ids = {m.id for m in lm2.registry.manifests()}
    assert NOTICING_TRIGGER_ID in ids
    assert NOTICING_APPLY_ID in ids


# --- the trigger emits a subjectless launch on a full-graph tick --------------


def test_full_graph_tick_emits_a_subjectless_launch_for_a_due_segment(tmp_path) -> None:
    buffer = _seeded_buffer()
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(NOW), noticing_buffer=buffer)
    lm.state.commit(State(genesis_completed_at=BORN_AT, last_tick_at="2026-07-17T11:59:00+00:00"))

    report = lm.coreloop.tick()

    launches = [
        launch for launch in report.internal_launches if launch.correlation_id.startswith("notice-")
    ]
    assert len(launches) == 1
    assert launches[0].subject_id is None


# --- coexistence: a subjectless completion is NoticingApply's -----------------


def test_subjectless_completion_seeds_a_thought_never_delivers(tmp_path) -> None:
    buffer = _seeded_buffer()
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(NOW), noticing_buffer=buffer)
    # The launched pass claimed its surveyed prefix (t1) under a survey_id; the
    # correlation carries it as notice-<session>#<survey_id>.
    survey_id = f"t1@{NOW.isoformat()}"
    correlation_id = f"notice-s1#{survey_id}"
    buffer.claim("s1", ("t1",), survey_id)
    lm.state.commit(
        State(
            genesis_completed_at=BORN_AT,
            pending_internal_id=correlation_id,
            pending_internal_subject_id=None,  # a noticing pass, not processing
            last_tick_at="2026-07-17T11:59:00+00:00",
        )
    )
    egress = FakeEgress()
    parsed = {"seeds": [{"gist": "carry this", "source_message_ids": ["m1"]}]}

    # The runner's injected apply STAYS ThoughtProcessingApply() (never changes) —
    # it guards on subject_id, so on a subjectless completion it contributes nothing
    # and the STANDING NoticingApply (registered by build_lifemodel above) is the
    # one that actually turns the result into a thought.
    outcome = run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id=correlation_id,
        result=InternalCognitionResult(raw="{...}", parsed=parsed),
        apply=ThoughtProcessingApply(),
    )

    assert outcome is None  # nothing proactive this frame — non-delivery is structural
    assert egress.calls == []  # never reached the egress
    assert lm.state.load().pending_internal_id is None  # cleared like every completion

    live = read_live_thoughts(lm.state)
    assert len(live) == 1
    assert live[0].content == "carry this"
    # the surveyed prefix was claimed away — nothing left in the closed segment
    assert buffer.closed_segment("s1", now=NOW) == []


# --- coexistence: a subject-set completion is ThoughtProcessingApply's --------


def test_subject_set_completion_resolves_the_thought_noticing_stays_silent(tmp_path) -> None:
    buffer = _seeded_buffer()
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(NOW), noticing_buffer=buffer)
    lm.state.commit(
        State(
            genesis_completed_at=BORN_AT,
            pending_internal_id="process-x",
            pending_internal_subject_id="thought:seed:a",  # a processing pass
            last_tick_at="2026-07-17T11:59:00+00:00",
        )
    )
    thought = build_thought(id="thought:seed:a", content="ca", state=ThoughtState.ACTIVE)
    lm.state.put(encode_thought(thought))
    egress = FakeEgress()

    outcome = run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id="process-x",
        result=InternalCognitionResult(raw="{...}", parsed={"outcome": "resolve"}),
        apply=ThoughtProcessingApply(),
    )

    assert outcome is None
    assert egress.calls == []
    assert lm.state.load().pending_internal_id is None

    resolved = lm.state.get("thought", "thought:seed:a")
    assert resolved is not None and resolved.state == ThoughtState.RESOLVED.value
    # NoticingApply (standing, registered above) guarded off on the subject-set
    # completion: it never touched the buffer, and minted no thought of its own —
    # the only thought in the store is the (now-resolved, terminal) processed one.
    assert [e.turn_id for e in buffer.closed_segment("s1", now=NOW)] == ["t1"]
    assert read_live_thoughts(lm.state) == ()

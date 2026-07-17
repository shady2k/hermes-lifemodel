"""Real-code sim: noticing forms a defeasible ``belief``, a turn surfaces it once
(lm-705.19 Task 5, belief-track v1 — the plan's final task).

Mirrors ``tests/test_noticing_harness.py`` exactly (see its module docstring for
the full rationale): drives the REAL frame (:func:`run_frame` ->
:class:`~lifemodel.core.noticing.NoticingTrigger`) and the REAL completion
(:func:`run_internal_completion` -> :class:`~lifemodel.core.noticing.NoticingApply`)
over the actual on-disk store (:func:`build_noticing_lifemodel`) and a real
:class:`~lifemodel.core.noticing_buffer.NoticingBuffer` — not mocks, so a green
scenario honestly predicts live behaviour. Then it drives the REAL third
``pre_llm_call`` hook (:func:`~lifemodel.hooks.make_belief_injector`) over the
SAME on-disk store to prove the whole belief-track end to end: born in noticing,
gated + cooldown-bounded at the door into a live turn.

Two more things this file checks that ``test_noticing_apply.py``/
``test_belief_injector.py`` (Tasks 3/4) already prove in isolation, re-verified
here as the plan's Self-Review demands: the D10 observability redaction (a
created belief's ``id``/``subject``/``confidence``/``sensitivity`` plus the
pass's ``reflection`` ride the noticing/apply span — never the belief's full
``content``; the injector's log line carries count/ids/latency — never
content). Those two checks reuse the SAME capture mechanisms the existing
suites do (:class:`~lifemodel.testing.FakeSpanLogger` + a hand-built
``TickContext`` for the span; ``caplog`` for the injector's stdlib log line) —
a full ``run_frame`` never surfaces a component's raw ``span.set(...)`` fields
(those ride the durable trace sink, not the in-memory event ring), so the
span-redaction check is deliberately the same lightweight, direct-component
style ``test_noticing_apply.py`` already uses, not a second real-store harness.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lifemodel.composition import LifeModel, build_lifemodel
from lifemodel.core.belief_view import belief_id, read_active_beliefs
from lifemodel.core.component import TickContext
from lifemodel.core.coreloop import TickReport
from lifemodel.core.frame import FrameTrigger, run_frame, state_actor_lock
from lifemodel.core.intents import LaunchInternalCognition, UpdateState
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.noticing import (
    DEFAULT_NOTICING_IDLE,
    NOTICING_APPLY_ID,
    NoticingApply,
)
from lifemodel.core.noticing_buffer import NoticingBuffer
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.timeutil import to_iso
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.hooks import make_belief_injector
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SqliteBufferStore
from lifemodel.testing import FakeActiveSpan, FakeClock, FakeSpanLogger
from lifemodel.testing.harness import build_noticing_lifemodel

TARGET: dict[str, str | None] = {"platform": "test", "chat_id": "1", "thread_id": None}

_CONFIDENT = "They get anxious before a loss of status."
_TENTATIVE = "They might prefer tea over coffee."
_PRIVATE = "They are secretly stressed about a possible layoff."


def _belief_lm() -> tuple[LifeModel, NoticingBuffer, Path, FakeClock]:
    """A real-graph noticing ``LifeModel`` over a durable on-disk store (D7: one
    physical store), mirroring ``test_noticing_harness.py``'s ``_noticing_lm``.
    Returns ``base_dir``/``clock`` too, so a caller can rebuild a FRESH
    ``LifeModel`` over the SAME store later (what the injector's ``build_lm``
    does every call, live) without re-running ``build_noticing_lifemodel``'s own
    born-state commit, which would clobber the state a noticing pass already
    wrote (the cooldown ring included)."""
    base_dir = Path(tempfile.mkdtemp(prefix="lifemodel-belief-harness-"))
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    buffer = NoticingBuffer(store=SqliteBufferStore(base_dir, clock=clock))
    lm = build_noticing_lifemodel(buffer=buffer, base_dir=base_dir, clock=clock)
    return lm, buffer, base_dir, clock


class _FakeEgress:
    """A no-op ``ProactiveEgressPort`` — these scenarios never expect a delivery
    FROM the noticing pass itself (non-delivery is structural), but
    ``run_internal_completion`` still needs a real port to hand to
    ``dispatch_launches``."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def reach_out(self, target: object, impulse: str) -> ReachOutcome:
        self.calls.append((target, impulse))
        return ReachOutcome.DELIVERED


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
    """Simulate the runner's reserve (minus the async task): stamp the
    single-flight markers under the same lock a real frame would take."""
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
    """The one subjectless (noticing) launch this frame produced (see
    ``test_noticing_harness.py``'s identically-named helper for why this
    filter, not a bare index, is the honest way to pick it out)."""
    noticing = [launch for launch in report.internal_launches if launch.subject_id is None]
    assert len(noticing) == 1, report.internal_launches
    return noticing[0]


def _seed_belief_via_noticing(
    lm: LifeModel,
    buffer: NoticingBuffer,
    *,
    content: str,
    confidence: float,
    sensitivity: str = "sensitive",
    turn_id: str = "t1",
) -> None:
    """Drive ONE real noticing pass — heartbeat launch, then a completion whose
    sole seed is a grounded ``belief`` — over *lm*/*buffer*. The shared setup
    every belief-creation scenario below starts from."""
    old_ts = lm.clock.now() - DEFAULT_NOTICING_IDLE - timedelta(minutes=5)
    _complete_turn(buffer, "s1", turn_id, user_text=f"about {content}", ts=old_ts)

    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = _only_noticing_launch(report)
    _set_pending(lm, launch)

    run_internal_completion(
        lm,
        _FakeEgress(),
        TARGET,
        correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw="...",
            parsed={
                "seeds": [
                    {
                        "kind": "belief",
                        "gist": f"noticed: {content}",
                        "content": content,
                        "source_message_ids": [turn_id],
                        "turn_id": turn_id,
                        "confidence": confidence,
                        "sensitivity": sensitivity,
                    }
                ],
                "reflection": "a quiet read on what they said, held loosely",
            },
        ),
        apply=NoticingApply(buffer),
    )


# --- Step 1: a grounded belief seed becomes a stored Belief row -------------


def test_belief_seed_becomes_a_stored_belief_with_evidence_and_confidence() -> None:
    """A ``kind:"belief"`` seed grounded in the surveyed segment becomes a real
    ``Belief`` row: the exact evidence/confidence/sensitivity the model gave,
    never re-derived or defaulted away."""
    lm, buffer, _base_dir, _clock = _belief_lm()

    _seed_belief_via_noticing(lm, buffer, content=_CONFIDENT, confidence=0.8)

    beliefs = read_active_beliefs(lm.state, min_confidence=0.0, exclude_private=False, limit=10)
    assert len(beliefs) == 1
    belief = beliefs[0]
    assert belief.content == _CONFIDENT
    assert belief.confidence == 0.8
    assert belief.sensitivity == Sensitivity.SENSITIVE
    assert belief.source_message_ids == ("t1",)


# --- Step 1: the injector surfaces it once, fallible-framed, then cools down --


def test_injector_surfaces_the_grounded_belief_once_then_cooldown_blocks_a_second_turn() -> None:
    """The belief-track payoff, end to end: a belief a noticing pass genuinely
    formed colours the being's NEXT live turn — once. A second immediate turn
    does not re-surface it (the ``surfaced_belief_ids`` cooldown ring)."""
    lm, buffer, base_dir, clock = _belief_lm()
    _seed_belief_via_noticing(lm, buffer, content=_CONFIDENT, confidence=0.8)

    # The injector rebuilds its OWN LifeModel every call (mirrors the live
    # pre_llm_call hook, and ``test_belief_injector.py``'s own ``_lm`` helper) —
    # a bare ``build_lifemodel`` never re-commits a fresh ``State`` the way
    # ``build_noticing_lifemodel`` does, so the noticing pass's cooldown ring
    # (and every other persisted field) survives across these fresh instances,
    # exactly as it would across live turns against the SAME on-disk store.
    injector = make_belief_injector(lambda: build_lifemodel(base_dir=base_dir, clock=clock))

    first = injector(session_id="s1", user_message="hi")
    assert first is not None
    context = first["context"]
    assert "I could be wrong" in context  # the fallible framing (D framing)
    assert _CONFIDENT in context

    second = injector(session_id="s1", user_message="hi again")
    assert second is None  # cooled down; nothing else qualifies


def test_below_threshold_belief_is_stored_but_never_surfaced() -> None:
    """A genuinely-held but tentative belief (confidence 0.4, below the
    injector's default θ=0.6) is still durably STORED — noticing never
    discards a validly-grounded seed for being unconfident — but the injector
    never surfaces it into a live turn."""
    lm, buffer, base_dir, clock = _belief_lm()
    _seed_belief_via_noticing(lm, buffer, content=_TENTATIVE, confidence=0.4)

    beliefs = read_active_beliefs(lm.state, min_confidence=0.0, exclude_private=False, limit=10)
    assert len(beliefs) == 1
    assert beliefs[0].confidence == 0.4

    injector = make_belief_injector(lambda: build_lifemodel(base_dir=base_dir, clock=clock))
    assert injector(session_id="s1", user_message="hi") is None


def test_private_belief_is_stored_but_never_surfaced() -> None:
    """A belief the model marks ``sensitivity:"private"`` is still durably
    stored (readable with ``exclude_private=False``) but is NEVER a candidate
    for the live-turn injector, and NEVER visible to the ordinary
    ``exclude_private=True`` read either — both this plan's floor (Task 1) and
    its gate (Task 4)."""
    lm, buffer, base_dir, clock = _belief_lm()
    _seed_belief_via_noticing(lm, buffer, content=_PRIVATE, confidence=0.9, sensitivity="private")

    everything = read_active_beliefs(lm.state, min_confidence=0.0, exclude_private=False, limit=10)
    assert len(everything) == 1
    assert everything[0].sensitivity == Sensitivity.PRIVATE

    visible = read_active_beliefs(lm.state, min_confidence=0.0, exclude_private=True, limit=10)
    assert visible == []

    injector = make_belief_injector(lambda: build_lifemodel(base_dir=base_dir, clock=clock))
    assert injector(session_id="s1", user_message="hi") is None


# --- Step 2: observability (D10) --------------------------------------------


def test_belief_creation_span_carries_metadata_and_reflection_never_content() -> None:
    """The noticing/apply span carries the created belief's ``id``/``subject``/
    ``confidence``/``sensitivity`` plus the pass's ``reflection`` — but NEVER
    the belief's full ``content`` in any span field (D10, tightened for this
    kind). Exercises the REAL :class:`NoticingApply` directly over a hand-built
    :class:`~lifemodel.testing.FakeSpanLogger`/``TickContext`` — the same
    capture style ``test_noticing_apply.py``'s own
    ``test_belief_creation_logs_redacted_metadata_not_content`` uses (a real
    ``run_frame`` never routes a component's ``span.set(...)`` fields back
    through anything a test can read without standing up a durable trace
    store, since those ride the async trace sink, not the in-memory event
    ring) — re-verified here alongside a ``reflection``, which that Task 3 test
    did not also carry."""
    now = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)
    buffer = NoticingBuffer()
    buffer.open_pending("s1", user_text="first", now=now - timedelta(hours=1))
    buffer.complete("s1", "t1", assistant_text="reply", now=now - timedelta(minutes=59))
    survey_id = f"t1@{to_iso(now)}"
    buffer.claim("s1", ("t1",), survey_id)
    correlation_id = f"notice-s1#{survey_id}"

    secret = "A secret worry about their job security, never to be repeated verbatim."
    parsed = {
        "seeds": [
            {
                "kind": "belief",
                "gist": "worry",
                "content": secret,
                "source_message_ids": ["t1"],
                "confidence": 0.8,
                "sensitivity": "private",
            }
        ],
        "reflection": "I noticed something delicate and want to hold it carefully.",
    }
    signal = internal_result_signal(
        origin_id="r1",
        correlation_id=correlation_id,
        raw="...",  # the placeholder ``aux_raw`` every noticing-apply test uses (never a re-dump)
        parsed=parsed,
        timestamp=to_iso(now),
    )
    trace = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    logger = FakeSpanLogger(FakeActiveSpan(trace, component=NOTICING_APPLY_ID, tick=1))
    ctx = TickContext(
        state=State(pending_internal_id=correlation_id, pending_internal_subject_id=None),
        now=now,
        trace=trace,
        objects=(),
        signals=[signal],
        logger=logger,
    )

    list(NoticingApply(buffer).step(ctx))

    beliefs = logger.span.attrs["beliefs"]
    assert len(beliefs) == 1
    assert "id" in beliefs[0]
    assert beliefs[0]["subject"] == "owner"
    assert beliefs[0]["confidence"] == 0.8
    assert beliefs[0]["sensitivity"] == Sensitivity.PRIVATE.value
    assert (
        logger.span.attrs["reflection"]
        == "I noticed something delicate and want to hold it carefully."
    )
    # the secret content must never ride ANY span field (D10 redaction)
    assert all(secret not in str(value) for value in logger.span.attrs.values())


def test_injector_log_line_carries_count_ids_latency_never_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The injector's own surfacing log line carries count/ids/latency ONLY —
    never the belief's ``content`` (D10). ``caplog`` is the same capture
    mechanism ``test_belief_injector.py``'s own fail-soft test already uses for
    this module's logger (``lifemodel.hooks``)."""
    lm, buffer, base_dir, clock = _belief_lm()
    _seed_belief_via_noticing(lm, buffer, content=_CONFIDENT, confidence=0.8)
    injector = make_belief_injector(lambda: build_lifemodel(base_dir=base_dir, clock=clock))

    with caplog.at_level(logging.INFO, logger="lifemodel.hooks"):
        result = injector(session_id="s1", user_message="hi")

    assert result is not None
    surfaced = [r for r in caplog.records if "belief_injector surfaced" in r.getMessage()]
    assert surfaced, "expected a belief_injector surfaced log line"
    message = surfaced[0].getMessage()
    assert "count=1" in message
    assert "latency_ms=" in message
    assert belief_id("noticing", _CONFIDENT) in message  # opaque id — fine to log
    assert _CONFIDENT not in message  # the belief's own words — never logged

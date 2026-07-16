"""Tests for :func:`lifemodel.core.internal_cognition.run_internal_completion` (lm-705.6).

The completion-frame body: apply the typed ``internal_result`` re-entry through an
injected ``apply`` component, dispatch any launches the frame returns (the strand
fix, codex #2 — Task 4), then clear ``pending_internal_id``. Driven over the REAL
graph (``build_lifemodel``) so the codex-#2 regression is genuine: an unrelated,
already-registered ``CognitionLauncher`` incidentally waking on this same
ASYNC_COMPLETION frame must still have its ``LaunchProactive`` dispatched, not
dropped just because this frame's PURPOSE was internal cognition.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.composition import build_lifemodel
from lifemodel.core.component import TickContext
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.intents import Intent, PutRecord
from lifemodel.core.internal_cognition import run_internal_completion
from lifemodel.core.llm_port import InternalCognitionResult
from lifemodel.core.taxonomy import KIND_INTERNAL_RESULT, read_internal_result
from lifemodel.domain.egress import ReachOutcome
from lifemodel.domain.memory import MemoryDraft, PutOp
from lifemodel.domain.objects import DesireState
from lifemodel.state.model import State

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}
BORN_AT = "2026-07-01T10:00:00+00:00"
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class FakeEgress:
    def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


class RecordingApply:
    """Applies the internal_result by writing it as a memory record — proves the
    typed re-entry actually reaches an injected component."""

    id = "recording-apply"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        results = [s for s in ctx.signals if s.kind == KIND_INTERNAL_RESULT]
        intents: list[Intent] = []
        for sig in results:
            read = read_internal_result(sig)
            self.seen.append(read.raw)
            intents.append(
                PutRecord(
                    op=PutOp(
                        draft=MemoryDraft(
                            kind="note",
                            id="n1",
                            state="active",
                            payload={"raw": read.raw, "correlation_id": read.correlation_id},
                            source="test",
                        )
                    )
                )
            )
        return intents


class NoopApply:
    id = "noop-apply"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return []


def _lm(tmp_path, state: State):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(NOW))
    lm.state.commit(state)
    return lm


def test_apply_component_commits_its_intent_and_pending_clears(tmp_path) -> None:
    lm = _lm(
        tmp_path,
        State(
            genesis_completed_at=BORN_AT,
            pending_internal_id="internal-1",
            last_tick_at="2026-07-16T11:59:00+00:00",
        ),
    )
    egress = FakeEgress()
    apply = RecordingApply()

    outcome = run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id="internal-1",
        result=InternalCognitionResult(raw="hello from the aux model", parsed=None),
        apply=apply,
    )

    assert outcome is None  # no proactive launch this frame — nothing to deliver
    assert apply.seen == ["hello from the aux model"]
    note = lm.state.get("note", "n1")
    assert note is not None and note.payload["raw"] == "hello from the aux model"
    assert lm.state.load().pending_internal_id is None  # cleared
    assert egress.calls == []  # non-delivery is structural — nothing reached the egress


def test_a_completion_frame_whose_unrelated_cognition_launcher_wakes_still_dispatches_it(
    tmp_path,
) -> None:
    # THE regression (codex #2), end-to-end over the REAL graph: CognitionLauncher is
    # registered by build_lifemodel like any live component, and it does not know or
    # care that this frame's TRIGGER is an internal-cognition completion — its gate is
    # only "a live ACTIVE desire + no proactive turn in flight + affordable energy". A
    # naive completion executor that only applied ITS OWN result-component's intents
    # (ignoring report.launches) would leave pending_proactive_id set with nothing
    # actually injected — real outreach blocked forever.
    lm = _lm(
        tmp_path,
        State(
            genesis_completed_at=BORN_AT,
            u=3.0,
            energy=1.0,
            pending_internal_id="internal-2",
            last_tick_at="2026-07-16T11:59:00+00:00",
        ),
    )
    lm.state.put(
        encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=3.0))
    )
    egress = FakeEgress()

    outcome = run_internal_completion(
        lm,
        egress,
        TARGET,
        correlation_id="internal-2",
        result=InternalCognitionResult(raw="", parsed=None),
        apply=NoopApply(),
    )

    assert outcome is ReachOutcome.DELIVERED  # the incidental proactive launch WAS dispatched
    assert len(egress.calls) == 1
    final = lm.state.load()
    assert final.pending_internal_id is None  # the internal correlation still clears
    assert final.pending_proactive_id is not None  # a real proactive turn is now in flight
    # separate correlation spaces — never collide
    assert final.pending_proactive_id != "internal-2"


def test_a_failed_call_result_still_clears_pending(tmp_path) -> None:
    # The runner (Task 6) maps a failed/timed-out aux call to an empty result; this
    # completion path must still clear pending_internal_id — a strand here blocks
    # every future internal launch, mirroring the proactive gate.
    lm = _lm(
        tmp_path,
        State(
            genesis_completed_at=BORN_AT,
            pending_internal_id="internal-3",
            last_tick_at="2026-07-16T11:59:00+00:00",
        ),
    )
    apply = RecordingApply()

    run_internal_completion(
        lm,
        FakeEgress(),
        TARGET,
        correlation_id="internal-3",
        result=InternalCognitionResult(raw="", parsed=None),
        apply=apply,
    )

    assert lm.state.load().pending_internal_id is None
    # apply still ran (over an empty result) — it just had nothing to say.
    assert apply.seen == [""]

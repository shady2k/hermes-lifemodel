"""ThoughtCrystallization — the TOP-DOWN Rubicon gate + the pure proposer (lm-27n.9).

Contract:
* the deterministic Rubicon gate (:func:`should_crystallize`) — anti-frivolity is
  the headline: a fresh idle thought NEVER crosses (fails salience AND persistence,
  no event bypass); a genuinely-salient other-serving thought crosses only after
  PERSIST_MIN ticks of sustained viable attention; a strong external event crosses
  immediately; receptivity veto / a pending turn / a non-active thought all block;
* the component is a PURE PROPOSER — it emits exactly one ``EmitSignal`` (the
  ``thought_contact_proposal``) and NO record mutation, and is silent when the
  singleton contact desire already exists.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.component import TickContext
from lifemodel.core.intents import EmitSignal, Intent, PutRecord, TransitionRecord
from lifemodel.core.taxonomy import KIND_THOUGHT_CONTACT_PROPOSAL, read_thought_contact_proposal
from lifemodel.core.thought_crystallization import (
    ACTIONABILITY_MIN,
    CRYSTALLIZE_SALIENCE,
    OTHER_REGARDING_MIN,
    PERSIST_MIN,
    STRONG_LONGING_PERSIST,
    STRONG_LONGING_SALIENCE,
    ThoughtCrystallization,
    should_crystallize,
    strong_event_trigger,
    strong_own_longing,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import Thought, ThoughtState
from lifemodel.state.model import State
from lifemodel.testing import FakeSignalBus, contact_desire_record, thought_record

_NOW = datetime(2026, 7, 6, 0, 1, tzinfo=UTC)


def _thought(
    *,
    salience: float = 0.8,
    other_regarding: float = 0.6,
    actionability: float = 0.3,
    sustained: int = PERSIST_MIN,
    trigger: str = "idle",
    state: str = "active",
) -> Thought:
    return build_thought(
        id="t-x",
        content="I should check the owner is ok after that hard week",
        trigger=trigger,
        state=ThoughtState(state),
        salience=salience,
        other_regarding_value=other_regarding,
        actionability=actionability,
        sustained_attention_count=sustained,
    )


# --- the pure Rubicon gate (anti-frivolity is the headline) -----------------


def test_fresh_idle_thought_never_crystallizes() -> None:
    # low salience, no persistence, idle trigger → fails BOTH bars, no event bypass.
    idle = _thought(salience=0.15, other_regarding=0.10, actionability=0.05, sustained=0)
    assert (
        should_crystallize(idle, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


def test_salient_other_serving_but_not_persistent_does_not_crystallize() -> None:
    # clears salience + reason, but sustained=0 and no strong event → persistence bar blocks.
    t = _thought(salience=0.8, other_regarding=0.6, sustained=0, trigger="idle")
    assert not strong_event_trigger(t)
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


def test_persistent_salient_other_serving_thought_crystallizes() -> None:
    t = _thought(salience=0.8, other_regarding=0.6, sustained=PERSIST_MIN, trigger="idle")
    assert should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)


def test_actionability_alone_is_a_valid_reason() -> None:
    t = _thought(salience=0.7, other_regarding=0.0, actionability=0.7, sustained=PERSIST_MIN)
    assert should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)


def test_no_reason_never_crystallizes_however_persistent() -> None:
    # salient + very persistent, but neither serves the owner nor is actionable, and
    # not a drive/event own-longing → no legitimate reason → blocked.
    t = _thought(salience=0.9, other_regarding=0.3, actionability=0.3, sustained=99, trigger="idle")
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


def test_strong_event_crystallizes_without_persistence() -> None:
    # an urgent external event crosses immediately (deliberation happened out there).
    t = _thought(
        salience=0.8, other_regarding=0.75, actionability=0.3, sustained=0, trigger="event"
    )
    assert strong_event_trigger(t)
    assert should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)


def test_idle_never_bypasses_persistence_even_at_event_salience() -> None:
    # same scores as the strong event, but the trigger is idle → NO bypass.
    t = _thought(salience=0.9, other_regarding=0.9, actionability=0.9, sustained=0, trigger="idle")
    assert not strong_event_trigger(t)
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


def test_receptivity_hard_veto_blocks_crystallization() -> None:
    t = _thought(salience=0.8, other_regarding=0.6, sustained=PERSIST_MIN)
    assert (
        should_crystallize(t, receptivity_allowed=False, pending=False, action_pending=False)
        is False
    )


def test_pending_turn_blocks_crystallization() -> None:
    t = _thought(salience=0.8, other_regarding=0.6, sustained=PERSIST_MIN)
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=True, action_pending=False) is False
    )


def test_below_salience_bar_blocks() -> None:
    t = _thought(salience=CRYSTALLIZE_SALIENCE - 0.01, other_regarding=0.9, sustained=PERSIST_MIN)
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


def test_strong_own_longing_is_a_gated_reason() -> None:
    # a high-salience drive thought with STRONGER persistence and no send pressure is
    # a legitimate own-longing reason — even with weak other-regarding/actionability.
    t = _thought(
        salience=0.8,
        other_regarding=0.2,
        actionability=0.2,
        sustained=PERSIST_MIN + 1,
        trigger="drive",
    )
    assert strong_own_longing(t, action_pending=False)
    assert should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=False)
    # recent send pressure (ActionPending) closes the own-longing door → blocked.
    assert not strong_own_longing(t, action_pending=True)
    assert (
        should_crystallize(t, receptivity_allowed=True, pending=False, action_pending=True) is False
    )


def test_strong_longing_crystallizes_below_the_actionability_and_other_regarding_bar() -> None:
    # Characterizes design point (2) of lm-8o3 (the "sliding reason-threshold keyed
    # to accumulated longing"): a strong-longing drive/event thought — high salience,
    # STRONGER persistence, no send pressure — crystallizes even though BOTH the
    # actionability AND other-regarding scores sit BELOW their ordinary floors. The
    # own-longing path is a substitute reason, not an addition on top of those
    # floors, so longing genuinely lowers the actionability/other-regarding bar to
    # zero (not just "a little"), in exchange for a stricter salience + persistence
    # requirement — the respect gates (receptivity/pending/action_pending) are
    # untouched and still fully enforced by ``should_crystallize`` itself.
    longing = _thought(
        salience=STRONG_LONGING_SALIENCE,
        other_regarding=OTHER_REGARDING_MIN - 0.1,
        actionability=ACTIONABILITY_MIN - 0.1,
        sustained=STRONG_LONGING_PERSIST,
        trigger="drive",
    )
    assert longing.other_regarding_value < OTHER_REGARDING_MIN
    assert longing.actionability < ACTIONABILITY_MIN
    assert strong_own_longing(longing, action_pending=False)
    assert should_crystallize(
        longing, receptivity_allowed=True, pending=False, action_pending=False
    )

    # The SAME low actionability/other-regarding and the SAME persistence, but an
    # idle trigger (no own-longing path available) — the ordinary reason bar still
    # applies and blocks it, proving the lower bar is genuinely longing-gated, not a
    # side effect of salience or persistence alone.
    neutral = _thought(
        salience=STRONG_LONGING_SALIENCE,
        other_regarding=OTHER_REGARDING_MIN - 0.1,
        actionability=ACTIONABILITY_MIN - 0.1,
        sustained=STRONG_LONGING_PERSIST,
        trigger="idle",
    )
    assert not strong_own_longing(neutral, action_pending=False)
    assert (
        should_crystallize(neutral, receptivity_allowed=True, pending=False, action_pending=False)
        is False
    )


# --- the component: a PURE PROPOSER (writes nothing) ------------------------


def _ctx(objects: tuple[MemoryRecord, ...], *, state: State | None = None) -> TickContext:
    return TickContext(state=state or State(), now=_NOW, bus=FakeSignalBus(), objects=objects)


def _crystallizer() -> ThoughtCrystallization:
    return ThoughtCrystallization()


def test_emits_one_proposal_and_writes_nothing_when_gate_passes() -> None:
    objects = (
        thought_record(
            "check in on the owner",
            "active",
            id="t-serve",
            salience=0.8,
            other_regarding_value=0.6,
            sustained_attention_count=PERSIST_MIN,
        ),
    )
    intents: list[Intent] = list(_crystallizer().step(_ctx(objects)))
    assert len(intents) == 1
    signal_intent = intents[0]
    assert isinstance(signal_intent, EmitSignal)  # a proposal, not a command
    # PURE PROPOSER: NO record mutation of its own.
    assert not [i for i in intents if isinstance(i, PutRecord | TransitionRecord)]
    proposal = read_thought_contact_proposal([signal_intent.signal])
    assert proposal is not None
    assert proposal.thought_id == "t-serve"
    assert signal_intent.signal.kind == KIND_THOUGHT_CONTACT_PROPOSAL


def test_no_proposal_when_no_live_thought() -> None:
    assert list(_crystallizer().step(_ctx(()))) == []


def test_no_proposal_when_gate_fails() -> None:
    # a fresh idle thought in the snapshot → the gate blocks, no proposal.
    objects = (
        thought_record(
            "just wandering",
            "active",
            id="t-idle",
            salience=0.15,
            trigger="idle",
            other_regarding_value=0.10,
            sustained_attention_count=0,
        ),
    )
    assert list(_crystallizer().step(_ctx(objects))) == []


def test_no_proposal_when_a_live_contact_desire_already_exists() -> None:
    # the singleton is taken — proposing would resolve a thought whose reason never
    # became a (new) desire. Silence until the desire clears.
    objects = (
        contact_desire_record("active"),
        thought_record(
            "check in on the owner",
            "active",
            id="t-serve",
            salience=0.9,
            other_regarding_value=0.8,
            sustained_attention_count=PERSIST_MIN,
        ),
    )
    assert list(_crystallizer().step(_ctx(objects))) == []


def test_pending_turn_state_suppresses_the_proposal() -> None:
    objects = (
        thought_record(
            "check in on the owner",
            "active",
            id="t-serve",
            salience=0.9,
            other_regarding_value=0.8,
            sustained_attention_count=PERSIST_MIN,
        ),
    )
    state = State(pending_proactive_id="proactive-inflight")
    assert list(_crystallizer().step(_ctx(objects, state=state))) == []

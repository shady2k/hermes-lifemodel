import pytest

from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import Belief, BeliefState, InvalidTransition, default_registry
from lifemodel.domain.objects.provenance import Sensitivity


def _belief(**over):
    base = dict(
        id="belief:seed:abcd",
        state=BeliefState.ACTIVE.value,
        source="noticing",
        content="They tend to get anxious before a loss of status.",
        subject="owner",
        source_message_ids=("t1",),
        source_thought_ids=("thought:seed:x",),
        confidence=0.7,
        sensitivity=Sensitivity.SENSITIVE,
    )
    base.update(over)
    return Belief(**base)


def _as_record(draft) -> MemoryRecord:
    return MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at="2026-07-17T00:00:00+00:00",
        updated_at="2026-07-17T00:00:00+00:00",
        revision=0,
        schema_version=draft.schema_version,
    )


def test_belief_round_trips_through_the_registry():
    reg = default_registry()
    b = _belief()
    draft = reg.encode(b)
    record = _as_record(draft)
    assert reg.decode(record) == b


def test_belief_is_a_known_kind():
    assert "belief" in default_registry().kinds()


def test_active_reaches_every_terminal_but_terminals_are_sealed():
    reg = default_registry()
    for term in (BeliefState.SUPERSEDED, BeliefState.DROPPED, BeliefState.EXPIRED):
        reg.validate_transition("belief", BeliefState.ACTIVE.value, term.value)  # no raise
        with pytest.raises(InvalidTransition):
            reg.validate_transition("belief", term.value, BeliefState.ACTIVE.value)


def test_catalog_is_terminal_consistent_including_belief():
    reg = default_registry()
    live = reg.live_states()
    terminal = reg.terminal_states_of("belief")
    assert not (terminal & live), f"belief: {terminal & live} both terminal and live"

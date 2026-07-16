import pytest

from lifemodel.domain.objects import (
    Commitment,
    CommitmentBasis,
    CommitmentState,
    CommitmentTriggerKind,
    InvalidTransition,
    default_registry,
)


def _commitment(**over):
    base = dict(
        id="commitment:seed:abc",
        state=CommitmentState.ACTIVE.value,
        source="thought-processing-apply",
        content="ask how their interview on Friday went",
        basis=CommitmentBasis.FOLLOW_UP,
        trigger_kind=CommitmentTriggerKind.EVENT,
        trigger_value="next time we talk",
        due_at=None,
        source_thought_ids=("thought:seed:xyz",),
        other_regarding_value=0.8,
        salience=0.6,
    )
    base.update(over)
    return Commitment(**base)


def test_commitment_roundtrips_through_the_registry():
    reg = default_registry()
    c = _commitment()
    record = reg.encode(c)
    assert record.kind == "commitment"
    assert record.salience == 0.6  # base-envelope field, not the payload
    back = reg.decode(_as_record(record))
    assert isinstance(back, Commitment)
    assert back.content == c.content
    assert back.basis == CommitmentBasis.FOLLOW_UP
    assert back.trigger_kind == CommitmentTriggerKind.EVENT
    assert back.source_thought_ids == ("thought:seed:xyz",)


def test_commitment_is_a_known_kind():
    assert "commitment" in default_registry().kinds()


def test_commitment_transition_table_is_complete():
    reg = default_registry()
    reg.validate_transition("commitment", "active", "honoured")
    reg.validate_transition("commitment", "active", "deferred")
    reg.validate_transition("commitment", "deferred", "active")
    with pytest.raises(InvalidTransition):
        reg.validate_transition("commitment", "honoured", "active")  # terminal


def test_catalog_is_terminal_consistent_including_commitment():
    # no state string may be terminal for one kind and live (non-terminal) for another
    reg = default_registry()
    live = reg.live_states()
    for kind in reg.kinds():
        terminal = reg.terminal_states_of(kind)
        assert not (terminal & live), f"{kind}: {terminal & live} both terminal and live"


def _as_record(draft):
    from lifemodel.domain.memory import MemoryRecord

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

from lifemodel.core.commitment_view import (
    build_commitment,
    crystallized_commitment_id,
    encode_commitment,
)
from lifemodel.domain.objects import CommitmentBasis, CommitmentState, CommitmentTriggerKind


def test_id_is_deterministic_and_scoped_to_the_source_thought():
    a = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    b = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    c = crystallized_commitment_id("thought:seed:y", "ask about the interview")  # other source
    d = crystallized_commitment_id("thought:seed:x", "different content")
    assert a == b  # reproducible → idempotent
    assert a != c and a != d  # distinct episode / distinct content ≠ conflated
    assert a.startswith("commitment:")


def test_build_and_encode_roundtrip():
    c = build_commitment(
        id=crystallized_commitment_id("thought:seed:x", "ask about the interview"),
        content="ask how their interview went",
        basis=CommitmentBasis.FOLLOW_UP,
        trigger_kind=CommitmentTriggerKind.EVENT,
        trigger_value="next time we talk",
        source_thought_ids=("thought:seed:x",),
        other_regarding_value=0.8,
        salience=0.6,
    )
    assert c.state == CommitmentState.ACTIVE.value
    draft = encode_commitment(c)  # goes through registry.encode → validates
    assert draft.kind == "commitment"
    assert draft.salience == 0.6

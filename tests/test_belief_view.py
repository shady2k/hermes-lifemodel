from datetime import UTC, datetime

import pytest

from lifemodel.core.belief_view import (
    belief_from_seed_fields,
    belief_id,
    build_belief,
    encode_belief,
    live_beliefs,
    read_active_beliefs,
)
from lifemodel.domain.memory import MemoryRecord
from lifemodel.domain.objects import BeliefState, InvalidPayload
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.testing import FakeClock, FakeMemoryStore

_CLOCK = FakeClock(datetime(2026, 7, 17, 0, 0, tzinfo=UTC))


def test_id_is_deterministic_content_digest_scoped_to_thought():
    a = belief_id("thought:seed:x", "They get anxious before status loss.")
    assert a == belief_id("thought:seed:x", "  They get anxious before status loss. ")
    assert a != belief_id("thought:seed:y", "They get anxious before status loss.")
    assert a.startswith("belief:seed:")


def test_id_rejects_non_utf8_encodable_content_as_invalid_payload():
    with pytest.raises(InvalidPayload):
        belief_id("thought:seed:x", "\ud800")


def test_from_seed_rejects_out_of_range_confidence():
    with pytest.raises(InvalidPayload):
        belief_from_seed_fields(
            source_thought_id="thought:seed:x",
            fields={"content": "c", "confidence": 1.5},
            source_message_ids=("t1",),
            provenance=None,
        )


def test_from_seed_rejects_missing_confidence():
    with pytest.raises(InvalidPayload):
        belief_from_seed_fields(
            source_thought_id="thought:seed:x",
            fields={"content": "c"},
            source_message_ids=("t1",),
            provenance=None,
        )


def test_from_seed_rejects_empty_content():
    with pytest.raises(InvalidPayload):
        belief_from_seed_fields(
            source_thought_id="thought:seed:x",
            fields={"content": "   ", "confidence": 0.5},
            source_message_ids=("t1",),
            provenance=None,
        )


def test_from_seed_floors_sensitivity_to_sensitive_by_default():
    b = belief_from_seed_fields(
        source_thought_id="thought:seed:x",
        fields={"content": "c", "confidence": 0.7},
        source_message_ids=("t1",),
        provenance=None,
    )
    assert b.sensitivity == Sensitivity.SENSITIVE
    assert b.state == BeliefState.ACTIVE.value
    assert b.source_message_ids == ("t1",)


def test_from_seed_allows_model_escalation_to_private():
    b = belief_from_seed_fields(
        source_thought_id="thought:seed:x",
        fields={"content": "c", "confidence": 0.7, "sensitivity": "private"},
        source_message_ids=("t1",),
        provenance=None,
    )
    assert b.sensitivity == Sensitivity.PRIVATE


def test_build_and_encode_roundtrip():
    b = build_belief(
        id=belief_id("thought:seed:x", "they get anxious before status loss"),
        content="they get anxious before status loss",
        source_message_ids=("t1",),
        source_thought_ids=("thought:seed:x",),
        confidence=0.7,
        salience=0.6,
    )
    assert b.state == BeliefState.ACTIVE.value
    draft = encode_belief(b)  # goes through registry.encode -> validates
    assert draft.kind == "belief"
    assert draft.salience == 0.6


def test_build_belief_rejects_out_of_range_confidence():
    with pytest.raises(InvalidPayload):
        build_belief(id="belief:seed:x", content="c", confidence=-0.1)


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


def test_live_beliefs_filters_and_sorts_by_salience_then_id():
    high = build_belief(
        id="belief:seed:high",
        content="high",
        confidence=0.7,
        salience=0.9,
        source_thought_ids=("thought:t1",),
    )
    low = build_belief(
        id="belief:seed:low",
        content="low",
        confidence=0.7,
        salience=0.1,
        source_thought_ids=("thought:t1",),
    )
    records = [_as_record(encode_belief(high)), _as_record(encode_belief(low))]
    live = live_beliefs(records)
    assert [b.id for b in live] == ["belief:seed:high", "belief:seed:low"]


def test_live_beliefs_excludes_non_active_records():
    b = build_belief(
        id="belief:seed:x", content="x", confidence=0.5, source_thought_ids=("thought:t1",)
    )
    record = _as_record(encode_belief(b))
    dropped_record = MemoryRecord(
        kind=record.kind,
        id=record.id,
        state=BeliefState.DROPPED.value,
        payload=record.payload,
        source=record.source,
        recipient_id=record.recipient_id,
        salience=record.salience,
        confidence=record.confidence,
        expires_at=record.expires_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
        revision=record.revision,
        schema_version=record.schema_version,
    )
    assert live_beliefs([dropped_record]) == ()


def test_read_active_beliefs_applies_confidence_and_privacy_filters():
    store = FakeMemoryStore(clock=_CLOCK)
    high_conf = build_belief(
        id="belief:seed:a", content="a", confidence=0.9, source_thought_ids=("thought:t1",)
    )
    low_conf = build_belief(
        id="belief:seed:b", content="b", confidence=0.2, source_thought_ids=("thought:t1",)
    )
    private = build_belief(
        id="belief:seed:c",
        content="c",
        confidence=0.9,
        sensitivity=Sensitivity.PRIVATE,
        source_thought_ids=("thought:t1",),
    )
    for b in (high_conf, low_conf, private):
        store.put(encode_belief(b))

    out = read_active_beliefs(store, min_confidence=0.5, exclude_private=True, limit=10)
    assert [b.id for b in out] == ["belief:seed:a"]


def test_read_active_beliefs_can_include_private():
    store = FakeMemoryStore(clock=_CLOCK)
    private = build_belief(
        id="belief:seed:c",
        content="c",
        confidence=0.9,
        sensitivity=Sensitivity.PRIVATE,
        source_thought_ids=("thought:t1",),
    )
    store.put(encode_belief(private))

    out = read_active_beliefs(store, min_confidence=0.0, exclude_private=False, limit=10)
    assert [b.id for b in out] == ["belief:seed:c"]


def test_read_active_beliefs_respects_limit():
    store = FakeMemoryStore(clock=_CLOCK)
    for i in range(5):
        b = build_belief(
            id=f"belief:seed:{i}",
            content=f"content {i}",
            confidence=0.9,
            source_thought_ids=("thought:t1",),
        )
        store.put(encode_belief(b))

    out = read_active_beliefs(store, limit=2)
    assert len(out) == 2

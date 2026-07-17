import dataclasses
from datetime import UTC, datetime

import pytest

from lifemodel.core.commitment_view import (
    build_commitment,
    commitment_from_live_fields,
    crystallized_commitment_id,
    encode_commitment,
    live_commitment_id,
    read_active_commitments,
)
from lifemodel.domain.objects import (
    CommitmentBasis,
    CommitmentState,
    CommitmentTriggerKind,
    InvalidPayload,
)
from lifemodel.testing import FakeClock
from lifemodel.testing.fakes import FakeMemoryStore

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_id_is_deterministic_and_scoped_to_the_source_thought():
    a = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    b = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    c = crystallized_commitment_id("thought:seed:y", "ask about the interview")  # other source
    d = crystallized_commitment_id("thought:seed:x", "different content")
    assert a == b  # reproducible → idempotent
    assert a != c and a != d  # distinct episode / distinct content ≠ conflated
    assert a.startswith("commitment:")


def test_id_rejects_non_utf8_encodable_content_as_invalid_payload():
    # a lone Unicode surrogate is not UTF-8-encodable → the fingerprint .encode() raises
    # UnicodeEncodeError, which must be translated to InvalidPayload so a crystallize
    # caller's narrow `except InvalidPayload` bounds it (lm-705.3 review I1c), never a strand.
    with pytest.raises(InvalidPayload):
        crystallized_commitment_id("thought:seed:x", "\ud800")


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


# ---- lm-705.21: live id + in-the-moment create-parse + bounded active reader ----


def test_live_commitment_id_deterministic_whitespace_normalized_and_namespaced():
    a = live_commitment_id("reflect the question back")
    assert a == live_commitment_id("  reflect the question back ")  # strip-normalized
    assert a != live_commitment_id("something else")
    assert a.startswith("commitment:live:")  # distinct from crystallization's :seed:
    assert len(a.rsplit(":", 1)[1]) == 16  # same 16-hex digest length


def test_live_commitment_id_rejects_lone_surrogate():
    with pytest.raises(InvalidPayload):
        live_commitment_id("\ud800")


def test_commitment_from_live_fields_builds_active_with_tool_source_and_mid_salience():
    c = commitment_from_live_fields(
        fields={
            "content": " reflect it back ",
            "basis": "self_assumed",
            "trigger_kind": "condition",
            "trigger_value": "he asks permission to spend on himself",
        }
    )
    assert c.state == CommitmentState.ACTIVE.value
    assert c.content == "reflect it back"  # stripped
    assert c.source == "commitment-tool"
    assert c.salience == 0.5
    assert c.source_thought_ids == ()
    assert c.id == live_commitment_id("reflect it back")


def test_commitment_from_live_fields_rejects_empty_content_and_bad_enum():
    with pytest.raises(InvalidPayload):
        commitment_from_live_fields(
            fields={
                "content": "   ",
                "basis": "self_assumed",
                "trigger_kind": "condition",
                "trigger_value": "x",
            }
        )
    with pytest.raises(InvalidPayload):
        commitment_from_live_fields(
            fields={
                "content": "c",
                "basis": "nope",
                "trigger_kind": "condition",
                "trigger_value": "x",
            }
        )


def _put_active(store: FakeMemoryStore, content: str, *, salience: float) -> str:
    c = commitment_from_live_fields(
        fields={
            "content": content,
            "basis": "follow_up",
            "trigger_kind": "event",
            "trigger_value": "when we next talk",
        }
    )
    c = dataclasses.replace(c, salience=salience)
    store.put(encode_commitment(c))
    return c.id


def test_read_active_commitments_active_only_salience_ordered_and_overflow_probe():
    store = FakeMemoryStore(clock=FakeClock(_NOW))
    low = _put_active(store, "low salience", salience=0.1)
    high = _put_active(store, "high salience", salience=0.9)
    deferred = _put_active(store, "deferred one", salience=0.8)
    store.transition("commitment", deferred, "active", "deferred")  # not active → excluded

    got = read_active_commitments(store, limit=8)
    ids = [c.id for c in got]
    assert deferred not in ids  # active-only
    assert ids == [high, low]  # salience_desc

    # overflow probe: with limit=1 and 2 active rows, the reader returns limit+1
    assert len(read_active_commitments(store, limit=1)) == 2

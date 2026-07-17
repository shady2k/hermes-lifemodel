"""``thought_seed`` source lineage (lm-705.5 Task 1, waking-mind Slice 5).

Extends the appraisal-result signal with WHICH conversation messages/turn a
noticed thought came from — foundational for the later noticing pass's dedup
(the consumed-source-id ring lives on ``State``, see ``test_state_model.py``).
Additive: an older seed with no lineage keys must still read cleanly.
"""

from __future__ import annotations

from lifemodel.core.taxonomy import (
    KIND_THOUGHT_SEED,
    read_thought_seed,
    thought_seed_signal,
)
from lifemodel.domain.signal import Signal


def test_thought_seed_roundtrips_source_lineage() -> None:
    sig = thought_seed_signal(
        origin_id="seed-2",
        content="the owner asked about the trip itinerary",
        salience=0.5,
        actionability=0.2,
        other_regarding_value=0.1,
        source_message_ids=("m1", "m2"),
        turn_id="t1",
        timestamp="2026-07-16T00:00:00+00:00",
    )
    assert sig.kind == KIND_THOUGHT_SEED
    read = read_thought_seed(sig)
    assert read.content == "the owner asked about the trip itinerary"
    assert read.source_message_ids == ("m1", "m2")
    assert read.turn_id == "t1"


def test_thought_seed_source_lineage_defaults_when_not_passed() -> None:
    # A caller (e.g. the existing post_llm appraisal seam, hooks.py) that does not
    # pass the new keyword args still builds a valid seed that reads with defaults.
    sig = thought_seed_signal(
        origin_id="seed-3",
        content="a plain seed with no lineage",
        salience=0.4,
        timestamp="2026-07-16T00:00:00+00:00",
    )
    read = read_thought_seed(sig)
    assert read.source_message_ids == ()
    assert read.turn_id is None


def test_read_thought_seed_defaults_on_hand_built_legacy_signal() -> None:
    # A hand-built Signal whose payload entirely lacks the new keys (as if written
    # by a pre-lm-705.5 build) must still decode cleanly — back-compat, not a
    # migration: the reader defaults, it never raises for an absent key.
    sig = Signal(
        origin_id="seed-4",
        kind=KIND_THOUGHT_SEED,
        payload={"content": "legacy seed", "salience": 0.2},
        timestamp=None,
    )
    read = read_thought_seed(sig)
    assert read.content == "legacy seed"
    assert read.source_message_ids == ()
    assert read.turn_id is None


def test_read_thought_seed_ignores_malformed_lineage_fields() -> None:
    # A corrupt/foreign payload (wrong types for the new keys) degrades to the
    # back-compat default rather than raising — mirrors the module's existing
    # defensive readers (e.g. read_contact_presence) for non-required fields.
    sig = Signal(
        origin_id="seed-5",
        kind=KIND_THOUGHT_SEED,
        payload={
            "content": "malformed lineage",
            "salience": 0.3,
            "source_message_ids": "not-a-list",
            "turn_id": 123,
        },
        timestamp=None,
    )
    read = read_thought_seed(sig)
    assert read.source_message_ids == ()
    assert read.turn_id is None

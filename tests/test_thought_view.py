"""The thought view (lm-27n.6): the registry door onto the NON-singleton
``kind='thought'`` rows — build/encode/decode, live-set reads, deterministic ids.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.thought_view import (
    LIVE_THOUGHT_STATES,
    THOUGHT_KIND,
    build_thought,
    encode_thought,
    live_thoughts,
    read_live_thoughts,
    read_thought,
    seed_thought_id,
    thought_id,
)
from lifemodel.domain.objects import Thought, ThoughtState, derive_id
from lifemodel.testing import (
    FakeClock,
    FakeMemoryStore,
    contact_desire_record,
    thought_objects,
    thought_record,
)

_CLOCK = FakeClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC))


# --- deterministic ids ------------------------------------------------------


def test_thought_id_is_derive_id_and_reproducible() -> None:
    assert thought_id("seed", "abc") == derive_id(THOUGHT_KIND, "seed", "abc")
    assert thought_id("a", "b") == thought_id("a", "b")  # never random


def test_seed_thought_id_is_stable_per_content() -> None:
    a = seed_thought_id("I wonder how the owner is")
    b = seed_thought_id("  I wonder how the owner is  ")  # whitespace-insensitive
    assert a == b  # idempotent: same content -> one row
    assert a != seed_thought_id("a different thought")
    assert a.startswith(f"{THOUGHT_KIND}:seed:")


# --- build / encode / decode round-trip -------------------------------------


def test_build_defaults_to_active_neutral_thought() -> None:
    t = build_thought(id="t1", content="hello")
    assert isinstance(t, Thought)
    assert t.KIND == THOUGHT_KIND
    assert t.state == ThoughtState.ACTIVE.value
    assert t.no_progress_count == 0
    assert t.parent_id is None


def test_encode_round_trips_through_the_store() -> None:
    store = FakeMemoryStore(clock=_CLOCK)
    store.put(encode_thought(build_thought(id="t1", content="turn this over", salience=0.4)))
    got = read_thought(store, "t1")
    assert got is not None
    assert got.content == "turn this over"
    assert got.salience == 0.4


# --- live_thoughts (from a snapshot) ----------------------------------------


def test_live_thoughts_includes_active_and_parked() -> None:
    objects = (
        thought_record("active one", "active", id="t-a"),
        thought_record("parked one", "parked", id="t-p"),
    )
    ids = {t.id for t in live_thoughts(objects)}
    assert ids == {"t-a", "t-p"}
    assert set(LIVE_THOUGHT_STATES) == {"active", "parked"}


def test_live_thoughts_excludes_terminal_states() -> None:
    for terminal in ("resolved", "dropped", "expired", "merged"):
        assert live_thoughts(thought_objects("gone", terminal)) == ()


def test_live_thoughts_ordered_by_salience_then_id() -> None:
    objects = (
        thought_record("low", "active", id="t-low", salience=0.1),
        thought_record("high", "active", id="t-high", salience=0.9),
        thought_record("mid-b", "active", id="t-b", salience=0.5),
        thought_record("mid-a", "active", id="t-a", salience=0.5),  # tie -> id asc
    )
    ordered = live_thoughts(objects)
    assert [t.id for t in ordered] == ["t-high", "t-a", "t-b", "t-low"]


def test_live_thoughts_ignores_non_thought_records() -> None:
    assert live_thoughts((contact_desire_record("active"),)) == ()
    assert live_thoughts(()) == ()


# --- read_live_thoughts (point-in-time from the store) ----------------------


def test_read_live_thoughts_orders_and_filters_and_bounds() -> None:
    store = FakeMemoryStore(clock=_CLOCK)
    store.put(encode_thought(build_thought(id="t-hi", content="hi", salience=0.9)))
    store.put(encode_thought(build_thought(id="t-lo", content="lo", salience=0.1)))
    store.put(
        encode_thought(
            build_thought(id="t-dead", content="dead", state=ThoughtState.DROPPED, salience=1.0)
        )
    )
    live = read_live_thoughts(store)
    assert [t.id for t in live] == ["t-hi", "t-lo"]  # terminal excluded, salience order
    assert [t.id for t in read_live_thoughts(store, limit=1)] == ["t-hi"]  # bounded


def test_read_thought_absent_is_none() -> None:
    assert read_thought(FakeMemoryStore(clock=_CLOCK), "nope") is None

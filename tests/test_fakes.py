"""Tests for the shipped test fakes (HLA §13, "imitations before code").

Later tasks (1.1–1.4) build on these, so their behaviour is pinned here: the
clock is controllable, delivery records, the state store isolates and persists,
and the in-memory signal bus honours the same dedup contract as the file bus.
Imports no Hermes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.domain.signal import Signal
from lifemodel.state.model import State
from lifemodel.testing import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore


def test_fake_clock_is_controllable() -> None:
    start = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
    clock = FakeClock(start)
    assert clock.now() == start
    clock.advance(timedelta(minutes=5))
    assert clock.now() == start + timedelta(minutes=5)
    clock.set(start)
    assert clock.now() == start


def test_fake_delivery_records_sends() -> None:
    delivery = FakeDelivery()
    delivery.send("author", "hello")
    delivery.send("author", "again")
    assert delivery.sent == [("author", "hello"), ("author", "again")]


def test_fake_state_store_defaults_and_round_trips() -> None:
    store = FakeStateStore()
    assert store.load() == State()
    store.commit(State(pressure=2.0))
    assert store.load().pressure == 2.0


def test_fake_state_store_isolates_from_caller_mutation() -> None:
    store = FakeStateStore()
    state = State(pressure=1.0)
    store.commit(state)
    # Mutating the caller's object must not reach into the store (copy-in).
    state.pressure = 99.0
    assert store.load().pressure == 1.0
    # ...and a loaded copy is likewise detached (copy-out).
    loaded = store.load()
    loaded.pressure = 42.0
    assert store.load().pressure == 1.0


def test_fake_signal_bus_matches_dedup_contract() -> None:
    bus = FakeSignalBus()
    bus.publish(Signal(origin_id="m1", kind="x"))
    bus.publish(Signal(origin_id="m1", kind="y"))  # dup id
    bus.publish(Signal(origin_id="m2", kind="x"))
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m1", "m2"]
    # Idempotent across calls; re-published id does not re-fire.
    bus.publish(Signal(origin_id="m1", kind="z"))
    assert bus.consume_unprocessed() == []

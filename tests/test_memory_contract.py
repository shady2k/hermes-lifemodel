"""Shared contract suite for ``MemoryPort``/``PressureSensorPort`` (lm-fib.6.1).

Parametrized over a fake (:class:`~lifemodel.testing.fakes.FakeMemoryStore`)
and the real :class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`, so
every test here proves the two implementations agree bit-for-bit on the
contract (HLA §4.1/D7). SQLite-only concerns (recovery, migrations,
STRICT/epoch storage details, fail-soft) live in ``test_sqlite_store.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pytest

from lifemodel.core.timeutil import to_iso
from lifemodel.domain.memory import (
    MemoryDraft,
    MemoryPatch,
    MemorySerializationError,
    PressureIndex,
    StaleTransition,
)
from lifemodel.ports.memory import MemoryPort, OrderBy
from lifemodel.ports.pressure import PressureSensorPort
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock, FakeMemoryStore

BASE_TIME = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


class _MemoryStore(MemoryPort, PressureSensorPort, Protocol):
    """The combined shape both backends under test satisfy."""


@pytest.fixture(params=["fake", "sqlite"])
def backend(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[_MemoryStore, FakeClock]]:
    clock = FakeClock(BASE_TIME)
    store: _MemoryStore
    if request.param == "fake":
        store = FakeMemoryStore(clock=clock)
    else:
        store = SQLiteRuntimeStore(tmp_path, clock=clock)
    yield store, clock


def _draft(**overrides: object) -> MemoryDraft:
    base = dict(
        kind="desire",
        id="d1",
        state="active",
        payload={"note": "hello"},
        source="test",
    )
    base.update(overrides)
    return MemoryDraft(**base)  # type: ignore[arg-type]


# ---- isinstance / port conformance -----------------------------------------


def test_backend_satisfies_both_ports(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, _clock = backend
    assert isinstance(store, MemoryPort)
    assert isinstance(store, PressureSensorPort)


# ---- put / get --------------------------------------------------------------


def test_put_then_get_round_trip(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, clock = backend
    returned_id = store.put(_draft())
    assert returned_id == "d1"

    record = store.get("desire", "d1")
    assert record is not None
    assert record.kind == "desire"
    assert record.id == "d1"
    assert record.state == "active"
    assert record.payload == {"note": "hello"}
    assert record.source == "test"
    assert record.recipient_id == "owner"
    assert record.salience == 0.0
    assert record.confidence is None
    assert record.expires_at is None
    assert record.created_at == to_iso(clock.now())
    assert record.updated_at == to_iso(clock.now())
    assert record.revision == 0
    assert record.schema_version == 1


def test_get_missing_record_returns_none(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, _clock = backend
    assert store.get("desire", "nope") is None


def test_put_upsert_keeps_created_at_and_bumps_revision(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    store.put(_draft(payload={"note": "v1"}))
    first = store.get("desire", "d1")
    assert first is not None

    clock.advance(timedelta(minutes=5))
    store.put(_draft(payload={"note": "v2"}, salience=0.7))

    second = store.get("desire", "d1")
    assert second is not None
    assert second.created_at == first.created_at  # kept
    assert second.updated_at != first.updated_at  # bumped
    assert second.revision == first.revision + 1
    assert second.payload == {"note": "v2"}  # wholesale replace, not merge
    assert second.salience == 0.7


def test_put_returned_payload_is_a_copy_not_a_live_reference(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    payload = {"note": "hello"}
    store.put(_draft(payload=payload))
    record = store.get("desire", "d1")
    assert record is not None
    record.payload["mutated"] = True  # mutate the returned dict
    assert store.get("desire", "d1").payload == {"note": "hello"}  # store unaffected


def test_put_rejects_non_json_serializable_payload(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    with pytest.raises(MemorySerializationError):
        store.put(_draft(payload={"bad": object()}))  # type: ignore[dict-item]
    assert store.get("desire", "d1") is None  # nothing written


def test_put_rejects_naive_expires_at(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, _clock = backend
    with pytest.raises(MemorySerializationError):
        store.put(_draft(expires_at="2026-07-06T12:00:00"))  # naive, no tz
    assert store.get("desire", "d1") is None


# ---- find ---------------------------------------------------------------


def test_find_filters_by_kind(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, _clock = backend
    store.put(_draft(kind="desire", id="d1"))
    store.put(_draft(kind="fact", id="f1"))
    results = store.find(kind="fact")
    assert [r.id for r in results] == ["f1"]


def test_find_filters_by_state(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, _clock = backend
    store.put(_draft(id="d1", state="active"))
    store.put(_draft(id="d2", state="archived"))
    results = store.find(state="active")
    assert [r.id for r in results] == ["d1"]


def test_find_filters_by_kind_and_state_combined(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(kind="desire", id="d1", state="active"))
    store.put(_draft(kind="desire", id="d2", state="archived"))
    store.put(_draft(kind="fact", id="f1", state="active"))
    results = store.find(kind="desire", state="active")
    assert [r.id for r in results] == ["d1"]


def test_find_respects_limit(backend: tuple[_MemoryStore, FakeClock]) -> None:
    store, clock = backend
    for i in range(5):
        store.put(_draft(id=f"d{i}"))
        clock.advance(timedelta(minutes=1))
    results = store.find(limit=2)
    assert len(results) == 2


def test_find_rejects_negative_limit(backend: tuple[_MemoryStore, FakeClock]) -> None:
    # SQLite `LIMIT -1` means "no limit"; the fake slice would drop one. Both must
    # reject identically rather than silently diverge.
    store, _clock = backend
    store.put(_draft())
    with pytest.raises(ValueError):
        store.find(limit=-1)


def test_find_order_by_updated_desc_with_id_tiebreak(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    # b and c share an updated_at timestamp; a is older, d is newest.
    store.put(_draft(id="a"))
    clock.advance(timedelta(minutes=1))
    store.put(_draft(id="c"))
    store.put(_draft(id="b"))
    clock.advance(timedelta(minutes=1))
    store.put(_draft(id="d"))

    results = store.find(order_by="updated_desc")
    assert [r.id for r in results] == ["d", "b", "c", "a"]


def test_find_order_by_created_desc_with_id_tiebreak(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    store.put(_draft(id="a"))
    clock.advance(timedelta(minutes=1))
    store.put(_draft(id="c"))
    store.put(_draft(id="b"))

    results = store.find(order_by="created_desc")
    assert [r.id for r in results] == ["b", "c", "a"]


def test_find_order_by_salience_desc_with_id_tiebreak(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(id="a", salience=0.5))
    store.put(_draft(id="c", salience=0.9))
    store.put(_draft(id="b", salience=0.9))

    results = store.find(order_by="salience_desc")
    assert [r.id for r in results] == ["b", "c", "a"]


# ---- transition -----------------------------------------------------------


def test_transition_valid_from_state_bumps_revision_and_state(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    store.put(_draft())
    clock.advance(timedelta(minutes=1))
    record = store.transition("desire", "d1", "active", "fulfilled")
    assert record.state == "fulfilled"
    assert record.revision == 1
    assert record.updated_at == to_iso(clock.now())


def test_transition_applies_payload_merge_shallow(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(payload={"a": 1, "b": 2}))
    record = store.transition(
        "desire", "d1", "active", "active", patch=MemoryPatch(payload_merge={"b": 3, "c": 4})
    )
    assert record.payload == {"a": 1, "b": 3, "c": 4}


def test_transition_applies_top_level_field_replace(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(salience=0.1, confidence=0.2, source="orig"))
    record = store.transition(
        "desire",
        "d1",
        "active",
        "active",
        patch=MemoryPatch(salience=0.9, confidence=0.8, source="revised"),
    )
    assert record.salience == 0.9
    assert record.confidence == 0.8
    assert record.source == "revised"


def test_transition_with_no_patch_only_changes_state(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(payload={"a": 1}, salience=0.3))
    record = store.transition("desire", "d1", "active", "archived")
    assert record.state == "archived"
    assert record.payload == {"a": 1}
    assert record.salience == 0.3


def test_transition_invalid_from_state_raises_stale_transition(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft(state="archived"))
    with pytest.raises(StaleTransition):
        store.transition("desire", "d1", "active", "fulfilled")
    # nothing changed
    assert store.get("desire", "d1").state == "archived"  # type: ignore[union-attr]


def test_transition_missing_record_raises_stale_transition(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    with pytest.raises(StaleTransition):
        store.transition("desire", "nope", "active", "archived")


def test_transition_soft_delete_then_excluded_from_active_find(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, _clock = backend
    store.put(_draft())
    store.transition("desire", "d1", "active", "archived")
    assert store.find(state="active") == []
    archived = store.find(state="archived")
    assert [r.id for r in archived] == ["d1"]


# ---- read_pressure_index ----------------------------------------------------


def test_pressure_index_counts_active_desires_and_max_salience(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    store.put(_draft(id="d1", salience=0.4))
    store.put(_draft(id="d2", salience=0.8))
    store.put(_draft(kind="fact", id="f1", salience=0.99))  # wrong kind
    store.put(_draft(id="d3", state="archived", salience=0.99))  # wrong state

    index = store.read_pressure_index(clock.now())
    assert index == PressureIndex(
        active_desire_count=2, max_desire_salience=0.8, contact_frame_available=True
    )


def test_pressure_index_excludes_expired_desire(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    expired_at = (clock.now() - timedelta(minutes=1)).isoformat()
    alive_at = (clock.now() + timedelta(minutes=1)).isoformat()
    store.put(_draft(id="expired", salience=0.9, expires_at=expired_at))
    store.put(_draft(id="alive", salience=0.5, expires_at=alive_at))

    index = store.read_pressure_index(clock.now())
    assert index.active_desire_count == 1
    assert index.max_desire_salience == 0.5


def test_pressure_index_empty_store_returns_default(
    backend: tuple[_MemoryStore, FakeClock],
) -> None:
    store, clock = backend
    assert store.read_pressure_index(clock.now()) == PressureIndex()


# ---- clock canonicalization / cross-backend parity --------------------------


def test_put_rejects_a_timezone_naive_clock(tmp_path: Path) -> None:
    # ClockPort promises tz-aware UTC; a naive value must be rejected (before any
    # write) by both backends, not silently misinterpreted as local time.
    naive_clock = FakeClock(datetime(2026, 7, 6, 12, 0))  # no tzinfo
    fake = FakeMemoryStore(clock=naive_clock)
    real = SQLiteRuntimeStore(tmp_path, clock=naive_clock)  # construction stays tolerant
    for store in (fake, real):
        with pytest.raises(MemorySerializationError):
            store.put(_draft())


def test_timestamps_are_canonicalized_to_utc_under_a_non_utc_clock(tmp_path: Path) -> None:
    tz = timezone(timedelta(hours=5))
    clock = FakeClock(datetime(2026, 7, 6, 17, 0, tzinfo=tz))  # == 12:00 UTC
    fake = FakeMemoryStore(clock=clock)
    real = SQLiteRuntimeStore(tmp_path, clock=clock)
    for store in (fake, real):
        store.put(_draft(id="z"))
        record = store.get("desire", "z")
        assert record is not None
        assert record.created_at == "2026-07-06T12:00:00.000000+00:00"
        assert record.updated_at == "2026-07-06T12:00:00.000000+00:00"


def test_fake_and_real_find_ordering_match_under_non_utc_clock(tmp_path: Path) -> None:
    # The fake and store both sort by the stored, normalized fixed-width ISO TEXT
    # (a non-UTC clock is canonicalized to UTC on write), so a lexical sort is
    # byte-identical across the two backends.
    tz = timezone(timedelta(hours=5))

    def populate(store: _MemoryStore, clock: FakeClock) -> None:
        clock.set(datetime(2026, 7, 6, 17, 0, tzinfo=tz))
        store.put(_draft(id="a", salience=0.2))
        clock.advance(timedelta(minutes=1))
        store.put(_draft(id="c", salience=0.9))
        store.put(_draft(id="b", salience=0.9))  # shares updated_at with c
        clock.advance(timedelta(minutes=1))
        store.put(_draft(id="d", salience=0.1))

    fake_clock = FakeClock(datetime(2026, 7, 6, 17, 0, tzinfo=tz))
    real_clock = FakeClock(datetime(2026, 7, 6, 17, 0, tzinfo=tz))
    fake = FakeMemoryStore(clock=fake_clock)
    real = SQLiteRuntimeStore(tmp_path, clock=real_clock)
    populate(fake, fake_clock)
    populate(real, real_clock)

    orders: list[OrderBy] = ["updated_desc", "created_desc", "salience_desc"]
    for order in orders:
        assert [r.id for r in fake.find(order_by=order)] == [
            r.id for r in real.find(order_by=order)
        ]

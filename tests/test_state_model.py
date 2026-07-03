"""Unit tests for the pure :class:`State` model and its (de)serialization.

The model is stdlib-only and Hermes-free: it owns its own wire format
(``to_dict``/``from_dict``) and validates types on the way in, raising a typed
:class:`StateCorruptError` for malformed data.
"""

from __future__ import annotations

import pytest

from lifemodel.state import SCHEMA_VERSION, State, StateCorruptError


def test_defaults_are_documented_and_current_schema() -> None:
    state = State()
    assert state.schema_version == SCHEMA_VERSION == 1
    assert state.pressure == 0.0
    assert state.energy == 1.0
    assert state.last_tick_at is None
    assert state.last_contact_at is None
    assert state.processed_signal_ids == []


def test_default_processed_ids_are_not_shared_between_instances() -> None:
    # Guard against a mutable-default footgun: each State gets its own list.
    a = State()
    b = State()
    a.processed_signal_ids.append("m1")
    assert b.processed_signal_ids == []


def test_to_dict_puts_schema_version_first_as_a_header() -> None:
    keys = list(State().to_dict().keys())
    assert keys[0] == "schema_version"


def test_round_trip_through_dict_is_identity() -> None:
    state = State(
        pressure=3.5,
        energy=0.25,
        last_tick_at="2026-07-03T12:00:00Z",
        last_contact_at="2026-07-03T11:00:00Z",
        processed_signal_ids=["m1", "m2"],
    )
    assert State.from_dict(state.to_dict()) == state


def test_from_dict_tolerates_missing_optional_fields() -> None:
    # A minimal (e.g. hand-written) file with only the header still loads,
    # filling documented defaults — "graceful defaults" per the task.
    state = State.from_dict({"schema_version": SCHEMA_VERSION})
    assert state == State()


def test_from_dict_rejects_non_integer_schema_version() -> None:
    # from_dict validates the header type too (the store gates the *value*).
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": "one"})


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": SCHEMA_VERSION, "pressure": "high"},
        {"schema_version": SCHEMA_VERSION, "energy": None},
        {"schema_version": SCHEMA_VERSION, "pressure": True},  # bool is not a number
        {"schema_version": SCHEMA_VERSION, "last_tick_at": 123},
        {"schema_version": SCHEMA_VERSION, "processed_signal_ids": "m1"},
        {"schema_version": SCHEMA_VERSION, "processed_signal_ids": [1, 2]},
    ],
)
def test_from_dict_rejects_wrong_field_types(payload: dict[str, object]) -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict(payload)

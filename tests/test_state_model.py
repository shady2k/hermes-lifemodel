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
    assert state.tick_count == 0
    assert state.pressure == 0.0
    assert state.energy == 1.0
    assert state.last_tick_at is None
    assert state.last_contact_at is None
    assert state.cooldown_until is None


def test_no_processed_signal_ids_field() -> None:
    # Finding 4: dedup ownership lives in the SignalBus consumed-ledger, not in
    # State. The dead field is gone from the model and its serialized shape, so
    # nothing surfaces it as an always-zero (misleading) dedup metric.
    assert not hasattr(State(), "processed_signal_ids")
    assert "processed_signal_ids" not in State().to_dict()


def test_to_dict_puts_schema_version_first_as_a_header() -> None:
    keys = list(State().to_dict().keys())
    assert keys[0] == "schema_version"


def test_round_trip_through_dict_is_identity() -> None:
    state = State(
        tick_count=42,
        pressure=3.5,
        energy=0.25,
        last_tick_at="2026-07-03T12:00:00Z",
        last_contact_at="2026-07-03T11:00:00Z",
        cooldown_until="2026-07-03T11:30:00Z",
    )
    assert State.from_dict(state.to_dict()) == state


def test_cooldown_until_is_additive_and_schema_stays_v1() -> None:
    # cooldown_until (roadmap 1.4) is a new optional field; a file written before
    # it existed (only the header + pre-1.4 fields) still loads under schema v1,
    # defaulting cooldown_until to None — additive, no version bump.
    legacy = {
        "schema_version": SCHEMA_VERSION,
        "tick_count": 7,
        "pressure": 2.0,
        "energy": 1.0,
        "last_tick_at": "2026-07-03T12:00:00Z",
        "last_contact_at": None,
    }
    state = State.from_dict(legacy)
    assert state.cooldown_until is None
    assert state.tick_count == 7


def test_cooldown_until_rejects_wrong_type() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "cooldown_until": 123})


def test_cooldown_until_rejects_unparseable_iso() -> None:
    # cooldown_until is the one timestamp the engine parses/branches on, so a
    # malformed string is corruption caught loud at load — never a mid-tick crash.
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "cooldown_until": "not-a-timestamp"})


def test_cooldown_until_accepts_valid_iso_forms() -> None:
    # Both an explicit +00:00 offset and the 'Z' suffix parse (Python 3.11+).
    for ts in ("2026-07-04T12:00:00+00:00", "2026-07-04T12:00:00Z"):
        assert (
            State.from_dict({"schema_version": SCHEMA_VERSION, "cooldown_until": ts}).cooldown_until
            == ts
        )


def test_tick_count_rejects_non_integer() -> None:
    # tick_count is a strict integer counter; a bool (int subclass) or a float
    # in the file signals corruption, not a valid count.
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "tick_count": True})
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "tick_count": 1.5})


def test_from_dict_tolerates_missing_optional_fields() -> None:
    # A minimal (e.g. hand-written) file with only the header still loads,
    # filling documented defaults — "graceful defaults" per the task.
    state = State.from_dict({"schema_version": SCHEMA_VERSION})
    assert state == State()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_from_dict_rejects_non_finite_floats(bad: float) -> None:
    # Non-finite floats are not valid JSON and poison downstream comparisons;
    # from_dict must reject them as corruption.
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "pressure": bad})


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
    ],
)
def test_from_dict_rejects_wrong_field_types(payload: dict[str, object]) -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict(payload)

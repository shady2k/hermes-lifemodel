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
    assert state.schema_version == SCHEMA_VERSION == 2
    assert state.tick_count == 0
    assert state.energy == 1.0
    assert state.last_tick_at is None
    assert state.last_contact_at is None


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
        energy=0.25,
        last_tick_at="2026-07-03T12:00:00Z",
        last_contact_at="2026-07-03T11:00:00Z",
    )
    assert State.from_dict(state.to_dict()) == state


def test_declined_at_is_additive_and_schema_stays_v1() -> None:
    # declined_at (the desire-lifecycle model's reject bookkeeping) is a new
    # optional field; a file written before it existed (only the header + prior
    # fields) still loads under schema v1, defaulting declined_at to None —
    # additive, no version bump.
    legacy = {
        "schema_version": SCHEMA_VERSION,
        "tick_count": 7,
        "energy": 1.0,
        "last_tick_at": "2026-07-03T12:00:00Z",
        "last_contact_at": None,
    }
    state = State.from_dict(legacy)
    assert state.declined_at is None
    assert state.tick_count == 7


def test_declined_at_rejects_wrong_type() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "declined_at": 123})


def test_declined_at_rejects_unparseable_iso() -> None:
    # declined_at is one of the timestamps the engine parses/branches on, so a
    # malformed string is corruption caught loud at load, never a mid-tick crash.
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, "declined_at": "not-a-timestamp"})


def test_declined_at_accepts_valid_iso_forms() -> None:
    # Both an explicit +00:00 offset and the 'Z' suffix parse (Python 3.11+).
    for ts in ("2026-07-04T12:00:00+00:00", "2026-07-04T12:00:00Z"):
        assert (
            State.from_dict({"schema_version": SCHEMA_VERSION, "declined_at": ts}).declined_at == ts
        )


@pytest.mark.parametrize(
    "field",
    [
        "last_contact_at",
        "last_exchange_at",
        "silence_anchor_at",
        "declined_at",
        "pending_proactive_since",
    ],
)
def test_iso_fields_reject_timezone_naive_values(field: str) -> None:
    # FINDING 2: a tz-naive value parses fine via fromisoformat but the tick
    # compares it against the clock's aware UTC ``now`` → TypeError mid-tick. The
    # engine's instant fields must be tz-AWARE, so a naive value is rejected as
    # corruption at load, never left to crash (or, under fail-closed main, wedge)
    # the tick.
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": SCHEMA_VERSION, field: "2026-07-04T12:00:00"})


def test_last_contact_at_accepts_aware_iso() -> None:
    aware = "2026-07-04T12:00:00+00:00"
    assert (
        State.from_dict(
            {"schema_version": SCHEMA_VERSION, "last_contact_at": aware}
        ).last_contact_at
        == aware
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
        State.from_dict({"schema_version": SCHEMA_VERSION, "u": bad})


def test_from_dict_rejects_non_integer_schema_version() -> None:
    # from_dict validates the header type too (the store gates the *value*).
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": "one"})


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": SCHEMA_VERSION, "u": "high"},
        {"schema_version": SCHEMA_VERSION, "energy": None},
        {"schema_version": SCHEMA_VERSION, "u": True},  # bool is not a number
        {"schema_version": SCHEMA_VERSION, "last_tick_at": 123},
    ],
)
def test_from_dict_rejects_wrong_field_types(payload: dict[str, object]) -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict(payload)


def test_state_has_lifecycle_fields_with_defaults() -> None:
    s = State()
    assert s.u == 0.0
    assert s.duration_over_theta == 0.0
    assert s.last_exchange_at is None
    assert s.declined_at is None
    assert s.decline_count == 0
    assert s.pending_proactive_id is None
    assert s.pending_proactive_since is None
    # The async-correlation anchor (§4.4) defaults absent, in lockstep with pending_id.
    assert s.pending_proactive_origin_traceparent is None


def test_state_roundtrips_lifecycle_fields() -> None:
    s = State(
        u=42.0,
        duration_over_theta=7.0,
        last_exchange_at="2026-07-05T10:00:00+00:00",
        declined_at="2026-07-05T09:00:00+00:00",
        decline_count=3,
        pending_proactive_id="p-1",
        pending_proactive_since="2026-07-05T10:01:00+00:00",
        pending_proactive_origin_traceparent=(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ),
    )
    assert State.from_dict(s.to_dict()) == s


def test_state_anchor_survives_missing_key_from_older_files() -> None:
    # Additive/back-compatible (spec §9): an older runtime_state JSON without the
    # anchor key loads cleanly with the field defaulting to ``None``.
    data = State(pending_proactive_id="p-1").to_dict()
    del data["pending_proactive_origin_traceparent"]
    loaded = State.from_dict(data)
    assert loaded.pending_proactive_origin_traceparent is None
    assert loaded.pending_proactive_id == "p-1"


def test_from_dict_ignores_unknown_legacy_keys() -> None:
    # Old state.json carried pressure/cooldown_until; they must be dropped, not crash.
    data = {
        "schema_version": 1,
        "pressure": 5.0,
        "cooldown_until": "2026-01-01T00:00:00+00:00",
        "u": 3.0,
    }
    s = State.from_dict(data)
    assert s.u == 3.0
    assert not hasattr(s, "pressure")


def test_naive_lifecycle_timestamp_is_corruption() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": 1, "last_exchange_at": "2026-07-05T10:00:00"})  # no tz


def test_action_pending_since_roundtrips() -> None:
    s = State(action_pending_since="2026-07-06T12:00:00+00:00")
    assert State.from_dict(s.to_dict()).action_pending_since == "2026-07-06T12:00:00+00:00"


def test_action_pending_since_defaults_none() -> None:
    assert State().action_pending_since is None
    assert State.from_dict({}).action_pending_since is None  # additive: missing key is fine


def test_fatigue_defaults_zero_and_roundtrips() -> None:
    assert State().fatigue == 0.0
    assert State.from_dict({}).fatigue == 0.0  # additive
    assert State.from_dict(State(fatigue=0.4).to_dict()).fatigue == 0.4


def test_proactive_send_log_defaults_empty_and_roundtrips() -> None:
    assert State().proactive_send_log == []
    assert State.from_dict({}).proactive_send_log == []  # additive
    s = State(proactive_send_log=["2026-07-06T20:00:00+00:00"])
    assert State.from_dict(s.to_dict()).proactive_send_log == ["2026-07-06T20:00:00+00:00"]


def test_proactive_send_log_rejects_non_list() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"proactive_send_log": "nope"})


def test_unanswered_outbound_count_defaults_zero_and_roundtrips() -> None:
    assert State().unanswered_outbound_count == 0
    assert State.from_dict({}).unanswered_outbound_count == 0  # additive: old file loads clean
    s = State(unanswered_outbound_count=3)
    assert State.from_dict(s.to_dict()).unanswered_outbound_count == 3


def test_unanswered_outbound_count_rejects_non_int() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"unanswered_outbound_count": "x"})


def test_silence_anchor_at_defaults_none_and_roundtrips() -> None:
    # lm-md6.1: the decoupled silence-window anchor. Additive (missing key → None, so
    # older runtime_state files load clean, no schema bump) and round-trips as an aware
    # ISO instant.
    assert State().silence_anchor_at is None
    assert State.from_dict({}).silence_anchor_at is None
    s = State(silence_anchor_at="2026-07-06T11:40:00+00:00")
    assert State.from_dict(s.to_dict()).silence_anchor_at == "2026-07-06T11:40:00+00:00"

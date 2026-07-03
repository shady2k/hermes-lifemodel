"""Tests for the ``Signal`` domain model — the one nervous-impulse type.

A signal must round-trip through JSON (it is a durable log record) and reject
corrupt records with a typed error. Imports no Hermes.
"""

from __future__ import annotations

import json

import pytest

from lifemodel.domain.signal import Signal, SignalDecodeError


def test_defaults_are_minimal_and_json_native() -> None:
    sig = Signal(origin_id="turn-1", kind="incoming")
    assert sig.payload == {}
    assert sig.timestamp is None
    assert sig.salience == 1.0


def test_is_immutable() -> None:
    sig = Signal(origin_id="turn-1", kind="incoming")
    with pytest.raises((AttributeError, TypeError)):
        sig.origin_id = "other"  # type: ignore[misc]


def test_round_trips_through_dict() -> None:
    sig = Signal(
        origin_id="msg-42",
        kind="connection",
        payload={"who": "author", "n": 3},
        timestamp="2026-07-03T12:00:00Z",
        salience=2.5,
    )
    assert Signal.from_dict(sig.to_dict()) == sig


def test_round_trips_through_json_serialization() -> None:
    sig = Signal(origin_id="m1", kind="thought", payload={"text": "hmm"}, salience=0.25)
    restored = Signal.from_dict(json.loads(json.dumps(sig.to_dict())))
    assert restored == sig


def test_to_dict_copies_payload_so_mutation_does_not_leak() -> None:
    sig = Signal(origin_id="m1", kind="incoming", payload={"a": 1})
    dumped = sig.to_dict()
    dumped["payload"]["a"] = 999
    assert sig.payload == {"a": 1}


def test_from_dict_tolerates_missing_optional_keys() -> None:
    sig = Signal.from_dict({"origin_id": "m1", "kind": "incoming"})
    assert sig == Signal(origin_id="m1", kind="incoming")


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "incoming"},  # missing origin_id
        {"origin_id": 5, "kind": "incoming"},  # origin_id wrong type
        {"origin_id": "m1"},  # missing kind
        {"origin_id": "m1", "kind": 7},  # kind wrong type
        {"origin_id": "m1", "kind": "x", "payload": [1, 2]},  # payload not object
        {"origin_id": "m1", "kind": "x", "timestamp": 123},  # timestamp wrong type
        {"origin_id": "m1", "kind": "x", "salience": "hi"},  # salience not number
        {"origin_id": "m1", "kind": "x", "salience": True},  # bool is not a number
    ],
)
def test_from_dict_rejects_corrupt_records(bad: dict[str, object]) -> None:
    with pytest.raises(SignalDecodeError):
        Signal.from_dict(bad)

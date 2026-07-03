"""Tests for the wake path domain values — ``WakePacket`` and ``WakeDecision``.

The wake-packet is the neuron script's stdout schema (HLA §11), so its JSON
round-trip and its rejection of malformed input matter. The wake-decision guards
the "waking implies a packet" invariant. Imports no Hermes.
"""

from __future__ import annotations

import json

import pytest

from lifemodel.domain.wake import (
    WAKE_PACKET_VERSION,
    WakeDecision,
    WakePacket,
    WakePacketError,
)


def test_packet_round_trips_through_json() -> None:
    packet = WakePacket(
        reason="haven't spoken in a while",
        pressure_kind="connection",
        pressure=1.4,
        energy=0.8,
        budget=None,
        last_contact_at="2026-07-02T09:00:00Z",
    )
    assert WakePacket.from_json(packet.to_json()) == packet


def test_packet_to_json_is_compact_and_version_headed() -> None:
    packet = WakePacket(reason="r", pressure_kind="connection", pressure=1.0)
    data = json.loads(packet.to_json())
    assert next(iter(data)) == "version"
    assert data["version"] == WAKE_PACKET_VERSION


def test_packet_from_json_rejects_non_object() -> None:
    with pytest.raises(WakePacketError):
        WakePacket.from_json("[1, 2, 3]")


def test_packet_from_dict_rejects_missing_required_field() -> None:
    with pytest.raises(WakePacketError):
        WakePacket.from_dict({"reason": "r"})  # no pressure_kind / pressure


@pytest.mark.parametrize(
    "bad",
    [
        {"reason": "r", "pressure_kind": "k", "pressure": "hi"},  # pressure not number
        {"reason": "r", "pressure_kind": "k", "pressure": 1.0, "energy": None},  # energy null
        {"reason": "r", "pressure_kind": "k", "pressure": 1.0, "budget": "x"},  # budget bad
        {"reason": "r", "pressure_kind": "k", "pressure": 1.0, "last_contact_at": 5},  # ts bad
        {"reason": "r", "pressure_kind": "k", "pressure": 1.0, "version": "1"},  # version bad
        {"reason": "r", "pressure_kind": 7, "pressure": 1.0},  # pressure_kind bad
    ],
)
def test_packet_from_dict_rejects_wrong_types(bad: dict[str, object]) -> None:
    with pytest.raises(WakePacketError):
        WakePacket.from_dict(bad)


def test_packet_accepts_explicit_null_budget_and_last_contact() -> None:
    packet = WakePacket.from_dict(
        {"reason": "r", "pressure_kind": "k", "pressure": 1.0, "budget": None}
    )
    assert packet.budget is None
    assert packet.last_contact_at is None


def test_stay_asleep_is_the_quiet_default() -> None:
    decision = WakeDecision.stay_asleep()
    assert decision.wake is False
    assert decision.packet is None


def test_wake_with_carries_the_packet() -> None:
    packet = WakePacket(reason="r", pressure_kind="connection", pressure=2.0)
    decision = WakeDecision.wake_with(packet)
    assert decision.wake is True
    assert decision.packet is packet


def test_waking_without_a_packet_is_rejected() -> None:
    with pytest.raises(ValueError, match="must carry a WakePacket"):
        WakeDecision(wake=True, packet=None)

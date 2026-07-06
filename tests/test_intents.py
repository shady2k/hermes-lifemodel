from __future__ import annotations

import dataclasses

import pytest

from lifemodel.core.intents import CheckpointState, EmitSignal, Intent, UpdateState
from lifemodel.domain.signal import Signal


def test_update_state_carries_changes() -> None:
    intent = UpdateState({"u": 0.5, "tick_count": 3})
    assert isinstance(intent, Intent)
    assert intent.changes == {"u": 0.5, "tick_count": 3}


def test_update_state_is_frozen() -> None:
    intent = UpdateState({"u": 0.5})
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.changes = {"u": 0.9}  # type: ignore[misc]


def test_emit_signal_wraps_a_signal() -> None:
    sig = Signal(origin_id="n1", kind="contact")
    intent = EmitSignal(sig)
    assert isinstance(intent, Intent)
    assert intent.signal is sig


def test_checkpoint_state_is_a_marker_intent() -> None:
    intent = CheckpointState()
    assert isinstance(intent, Intent)
    assert CheckpointState() == CheckpointState()


def test_intents_are_equal_by_value() -> None:
    assert UpdateState({"u": 1.0}) == UpdateState({"u": 1.0})
    assert UpdateState({"u": 1.0}) != UpdateState({"u": 2.0})

"""The ``thought_seed`` signal — the appraisal result on the bus (lm-705.1 Task 1)."""

from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_THOUGHT_SEED,
    read_thought_seed,
    thought_seed_signal,
)


def test_thought_seed_roundtrip():
    sig = thought_seed_signal(
        origin_id="seed-1",
        content="the owner mentioned a dentist appointment on Friday",
        salience=0.6,
        actionability=0.3,
        other_regarding_value=0.5,
        timestamp="2026-07-16T00:00:00+00:00",
    )
    assert sig.kind == KIND_THOUGHT_SEED
    read = read_thought_seed(sig)
    assert read.content == "the owner mentioned a dentist appointment on Friday"
    assert read.salience == 0.6
    assert read.actionability == 0.3
    assert read.other_regarding_value == 0.5


def test_read_thought_seed_rejects_wrong_kind():
    from lifemodel.domain.signal import Signal

    with pytest.raises(ValueError):
        read_thought_seed(Signal(origin_id="x", kind="not_a_seed", payload={}, timestamp=None))


def test_thought_seed_carries_producer() -> None:
    sig = thought_seed_signal(
        origin_id="o",
        content="c",
        salience=0.5,
        producer="create-thought-tool",
        timestamp=None,
    )
    assert read_thought_seed(sig).producer == "create-thought-tool"


def test_thought_seed_producer_defaults_unknown() -> None:
    sig = thought_seed_signal(origin_id="o", content="c", salience=0.5, timestamp=None)
    assert read_thought_seed(sig).producer == "unknown"

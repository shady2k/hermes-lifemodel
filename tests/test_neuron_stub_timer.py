"""Unit tests for the stub timer neuron (roadmap 1.2, HLA §1/§2).

The first autonomic neuron: it carries no behaviour (the real time-since-contact
neuron is Phase 2.1), only the engine-validation contract — one fixed-delta
pressure signal per tick, with a stable-per-tick, distinct-across-ticks origin
id so the bus dedup counts each tick exactly once (HLA §10). The neuron owns the
delta and never mutates state (the tick is the single writer, HLA §9).

Imports no Hermes.
"""

from __future__ import annotations

from typing import Any

from lifemodel.core.neuron import (
    EVENT_NEURON_FIRED,
    TIMER_PRESSURE_KIND,
    Neuron,
    StubTimerNeuron,
)
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


class _RecordingLogger:
    """Minimal ``EventLogger`` that records the events it is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


def test_is_a_neuron() -> None:
    assert isinstance(StubTimerNeuron(), Neuron)


def test_tick_returns_exactly_one_pressure_signal_with_default_delta() -> None:
    signals = StubTimerNeuron().tick(State())

    assert len(signals) == 1
    signal = signals[0]
    assert isinstance(signal, Signal)
    assert signal.kind == TIMER_PRESSURE_KIND
    assert signal.salience == 1.0  # the fixed pressure delta (weight)


def test_delta_is_configurable_and_carried_as_salience() -> None:
    signals = StubTimerNeuron(delta=2.5).tick(State())

    assert [s.salience for s in signals] == [2.5]


def test_origin_id_is_tied_to_tick_identity_and_stable_within_a_tick() -> None:
    neuron = StubTimerNeuron()
    state = State(tick_count=5)

    first = neuron.tick(state)[0].origin_id
    second = neuron.tick(state)[0].origin_id

    assert first == second  # stable for a given tick identity → dedup counts once
    assert "5" in first  # derived from tick_count (the tick's identity)


def test_origin_id_is_distinct_across_ticks() -> None:
    neuron = StubTimerNeuron()

    id_at_0 = neuron.tick(State(tick_count=0))[0].origin_id
    id_at_1 = neuron.tick(State(tick_count=1))[0].origin_id

    assert id_at_0 != id_at_1  # a fresh id each tick → not collapsed by dedup


def test_tick_does_not_mutate_state() -> None:
    neuron = StubTimerNeuron()
    state = State(tick_count=3, pressure=7.0)

    neuron.tick(state)

    # The neuron reads state and returns signals; the tick is the single writer.
    assert state.tick_count == 3
    assert state.pressure == 7.0


def test_emits_neuron_fired_event_when_a_logger_is_injected() -> None:
    logger = _RecordingLogger()
    neuron = StubTimerNeuron(delta=1.0, logger=logger)

    neuron.tick(State(tick_count=2))

    fired = [fields for event, fields in logger.calls if event == EVENT_NEURON_FIRED]
    assert len(fired) == 1
    assert fired[0]["delta"] == 1.0
    assert "2" in str(fired[0]["origin_id"])


def test_no_logger_is_a_silent_no_op() -> None:
    # Without an injected logger the neuron still fires its signal (DI: emission
    # is optional observability, not required for correctness).
    signals = StubTimerNeuron(logger=None).tick(State())

    assert len(signals) == 1

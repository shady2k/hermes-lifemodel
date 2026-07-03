"""Unit tests for the heartbeat tick entrypoint (roadmap 1.1).

These drive the tick with injected fakes (``FakeClock`` / ``FakeStateStore`` /
``FakeSignalBus``) — no Hermes, no LLM. They assert the walking-skeleton
contract: the tick loads state, advances it, commits, emits a ``tick`` event,
stays asleep (``{"wakeAgent": false}``), and persists across runs. A seam test
proves the neuron→bus→aggregator wiring is live so 1.2/1.3 only fill it in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import lifemodel.tick as tick_mod
from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import Aggregator
from lifemodel.core.neuron import Neuron
from lifemodel.domain.signal import Signal
from lifemodel.domain.wake import WakeDecision, WakePacket
from lifemodel.events import EVENT_TICK, EVENTS_FILENAME, EventSink
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore
from lifemodel.tick import main, run_tick, wake_gate_line

_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


class _RecordingLogger:
    """Minimal :class:`EventLogger` that records the events it is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


class _EmitOnceNeuron(Neuron):
    """A stub neuron that emits exactly one signal per tick (seam probe only)."""

    def __init__(self, origin_id: str) -> None:
        self._origin_id = origin_id

    def tick(self, state: State) -> list[Signal]:
        return [Signal(origin_id=self._origin_id, kind="probe")]


class _RecordingAggregator(Aggregator):
    """Captures the signals it was handed; always stays asleep (like Silent)."""

    def __init__(self) -> None:
        self.seen: list[Signal] = []

    def decide(self, signals: Any) -> WakeDecision:
        self.seen.extend(signals)
        return WakeDecision.stay_asleep()


def _build(**overrides: Any) -> Any:
    """Assemble a graph over fakes; callers override individual collaborators."""
    params: dict[str, Any] = {
        "base_dir": Path("/unused-for-fakes"),
        "state": FakeStateStore(),
        "bus": FakeSignalBus(),
        "clock": FakeClock(_T0),
        "delivery": FakeDelivery(),
        "neurons": (),
    }
    params.update(overrides)
    return build_lifemodel(**params)


def test_run_tick_advances_state_and_commits() -> None:
    store = FakeStateStore()
    lm = _build(state=store, clock=FakeClock(_T0))

    decision = run_tick(lm, logger=_RecordingLogger())

    persisted = store.load()
    assert persisted.tick_count == 1
    assert persisted.last_tick_at == _T0.isoformat()
    assert decision.wake is False


def test_run_tick_persists_between_ticks() -> None:
    store = FakeStateStore()
    clock = FakeClock(_T0)
    lm = _build(state=store, clock=clock)

    run_tick(lm, logger=_RecordingLogger())
    first = store.load()
    clock.advance(timedelta(minutes=1))
    run_tick(lm, logger=_RecordingLogger())
    second = store.load()

    assert (first.tick_count, second.tick_count) == (1, 2)
    assert second.last_tick_at == (_T0 + timedelta(minutes=1)).isoformat()
    assert second.last_tick_at != first.last_tick_at


def test_run_tick_emits_tick_event_with_bookkeeping() -> None:
    logger = _RecordingLogger()
    lm = _build(clock=FakeClock(_T0))

    run_tick(lm, logger=logger)

    ticks = [fields for event, fields in logger.calls if event == EVENT_TICK]
    assert len(ticks) == 1
    assert ticks[0]["tick_count"] == 1
    assert ticks[0]["last_tick_at"] == _T0.isoformat()
    assert ticks[0]["wake"] is False


def test_run_tick_never_wakes_and_never_delivers() -> None:
    # Below-threshold contract (HLA §1): no wake, so nothing is delivered and no
    # LLM path is entered — the walking skeleton is zero-cost every tick.
    delivery = FakeDelivery()
    lm = _build(delivery=delivery)

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert delivery.sent == []


def test_run_tick_wires_neurons_through_bus_to_aggregator() -> None:
    # Seam probe: a neuron's signal flows onto the bus and reaches the
    # aggregator's decide(). 1.1 ships no neurons; this proves 1.2/1.3 only fill
    # the seam rather than reshape the tick.
    aggregator = _RecordingAggregator()
    lm = _build(neurons=(_EmitOnceNeuron("sig-1"),), aggregator=aggregator)

    run_tick(lm, logger=_RecordingLogger())

    assert [s.origin_id for s in aggregator.seen] == ["sig-1"]


def test_wake_gate_line_stays_asleep_is_wakeagent_false() -> None:
    line = wake_gate_line(WakeDecision.stay_asleep())
    assert json.loads(line) == {"wakeAgent": False}


def test_wake_gate_line_wake_carries_packet_and_true_flag() -> None:
    # Forward seam (1.3): a waking decision emits the wake-packet fields plus the
    # wakeAgent:true flag on the single stdout line.
    packet = WakePacket(reason="overdue", pressure_kind="connection", pressure=2.0)
    gate = json.loads(wake_gate_line(WakeDecision.wake_with(packet)))
    assert gate["wakeAgent"] is True
    assert gate["reason"] == "overdue"
    assert gate["pressure_kind"] == "connection"


def test_main_prints_wake_gate_and_persists_across_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Drive the real entrypoint against a throwaway home: the wake gate the
    # scheduler parses (the last non-empty stdout line) is {"wakeAgent": false},
    # and state persists across two invocations.
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    sdir = tmp_path / "lifemodel"
    state = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    assert state["tick_count"] == 2

    records = [
        json.loads(line)
        for line in (sdir / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    tick_events = [r for r in records if r.get("event") == EVENT_TICK]
    assert [r["tick_count"] for r in tick_events] == [1, 2]
    assert all(r["wake"] is False for r in tick_events)


def test_main_stdout_last_line_is_readable_by_the_scheduler_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Belt-and-suspenders: reproduce the scheduler's own _parse_wake_gate logic
    # over main()'s stdout and confirm it decides "do not wake".
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    main()
    out = capsys.readouterr().out
    gate = _last_gate_line(out)
    assert gate.get("wakeAgent", True) is False  # scheduler: False => skip agent


def _last_gate_line(stdout: str) -> dict[str, Any]:
    """Mirror cron/scheduler.py:_parse_wake_gate — parse the last non-empty line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, "tick produced no stdout"
    parsed = json.loads(lines[-1])
    assert isinstance(parsed, dict)
    return parsed


def test_sink_reads_back_the_last_tick(tmp_path: Path) -> None:
    # The tick event lands in the queryable EventSink so /lifemodel debug can
    # answer "last tick" (HLA §12) — exercised via the real EventTee wiring.
    from lifemodel.logging import EventTee, get_logger

    sink = EventSink(tmp_path / EVENTS_FILENAME)
    logger = EventTee(get_logger("lifemodel.tick.test"), sink)
    lm = _build(clock=FakeClock(_T0))

    run_tick(lm, logger=logger)

    last = sink.read()[-1]
    assert last["event"] == EVENT_TICK
    assert last["tick_count"] == 1

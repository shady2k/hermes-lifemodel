"""Unit tests for the heartbeat tick entrypoint (roadmap 1.1).

These drive the tick with injected fakes (``FakeClock`` / ``FakeStateStore`` /
``FakeSignalBus``) — no Hermes, no LLM. They assert the walking-skeleton
contract: the tick loads state, advances it, commits, emits a ``tick`` event,
stays asleep (``{"wakeAgent": false}``), and persists across runs. A seam test
proves the neuron→bus→aggregator wiring is live so 1.2/1.3 only fill it in.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import lifemodel.tick as tick_mod
from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import DEFAULT_WAKE_THRESHOLD, Aggregator, ThresholdAggregator
from lifemodel.core.neuron import EVENT_NEURON_FIRED, Neuron, StubTimerNeuron
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
    """Captures the signals and accumulated pressure it was handed; stays asleep."""

    def __init__(self) -> None:
        self.seen: list[Signal] = []
        self.pressures: list[float] = []

    def decide(self, signals: Any, *, pressure: float) -> WakeDecision:
        self.seen.extend(signals)
        self.pressures.append(pressure)
        return WakeDecision.stay_asleep()


class _ConstantAggregator(Aggregator):
    """Returns a fixed decision regardless of pressure; records what it received.

    Proves the tick *delegates* the wake call entirely — the orchestrator applies
    no threshold of its own, so whatever this returns is what run_tick returns.
    """

    def __init__(self, decision: WakeDecision) -> None:
        self._decision = decision
        self.pressures: list[float] = []

    def decide(self, signals: Any, *, pressure: float) -> WakeDecision:
        self.pressures.append(pressure)
        return self._decision


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


def test_run_tick_accumulates_one_delta_into_pressure_each_tick() -> None:
    # 1.2 engine validation: the stub neuron owns a fixed delta; run_tick sums
    # the consumed signals' deltas into State.pressure. K ticks → K*delta, the
    # number PERSISTS between ticks (loaded from the committed state, not
    # recomputed from zero), bookkeeping still advances, a neuron_fired event is
    # recorded each tick, and the wake gate stays asleep the whole time (0 LLM).
    store = FakeStateStore()
    clock = FakeClock(_T0)
    logger = _RecordingLogger()
    delta = 2.0
    neuron = StubTimerNeuron(delta=delta, logger=logger)
    lm = _build(state=store, clock=clock, neurons=(neuron,))

    k = 3
    for i in range(k):
        decision = run_tick(lm, logger=logger)
        assert decision.wake is False  # wake gate stays {"wakeAgent": false}
        assert store.load().pressure == (i + 1) * delta  # grows + persists
        clock.advance(timedelta(minutes=1))

    final = store.load()
    assert final.pressure == k * delta
    assert final.tick_count == k
    assert final.last_tick_at == (_T0 + timedelta(minutes=k - 1)).isoformat()

    fired = [fields for event, fields in logger.calls if event == EVENT_NEURON_FIRED]
    assert len(fired) == k  # the neuron fired once per tick
    ticks = [fields for event, fields in logger.calls if event == EVENT_TICK]
    assert all(t["wake"] is False for t in ticks)


def test_run_tick_wakes_when_accumulated_pressure_crosses_threshold() -> None:
    # 1.3 acceptance: below-threshold ticks accumulate silently until the crossing
    # tick flips the wake gate. Drive the real run_tick harness with the real
    # ThresholdAggregator + StubTimerNeuron; assert the gate progression and that
    # the crossing line is a single JSON object that parses back into a WakePacket
    # carrying reason + pressure + threshold.
    store = FakeStateStore()
    clock = FakeClock(_T0)
    threshold = 3.0
    lm = _build(
        state=store,
        clock=clock,
        neurons=(StubTimerNeuron(delta=1.0),),
        aggregator=ThresholdAggregator(threshold=threshold),
    )

    # Ticks 1 and 2: pressure 1.0, 2.0 — below threshold, gate stays asleep.
    for _ in range(2):
        decision = run_tick(lm, logger=_RecordingLogger())
        assert decision.wake is False
        assert json.loads(wake_gate_line(decision)) == {"wakeAgent": False}
        clock.advance(timedelta(minutes=1))

    # Tick 3: pressure reaches 3.0 == threshold → wake.
    decision = run_tick(lm, logger=_RecordingLogger())
    assert decision.wake is True
    assert decision.packet is not None

    line = wake_gate_line(decision)
    gate = json.loads(line)  # a single, parseable JSON object
    assert gate["wakeAgent"] is True

    packet = WakePacket.from_dict(gate)  # parses back via the hardened schema
    assert packet.reason
    assert packet.pressure == 3.0
    assert packet.threshold == threshold
    assert store.load().pressure == 3.0  # committed, undrained (drain is 1.4)


def test_run_tick_below_threshold_is_zero_llm_and_delivers_nothing() -> None:
    # Below threshold → no wake, so nothing is delivered and no LLM path opens.
    delivery = FakeDelivery()
    lm = _build(
        delivery=delivery,
        neurons=(StubTimerNeuron(delta=1.0),),
        aggregator=ThresholdAggregator(threshold=100.0),
    )

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert json.loads(wake_gate_line(decision)) == {"wakeAgent": False}
    assert delivery.sent == []


def test_run_tick_decides_against_accumulated_pressure_not_transient_signal() -> None:
    # The aggregator is handed the *accumulated* State.pressure (post-sum), not the
    # bare per-tick delta — proving accumulate-happens-before-decide.
    store = FakeStateStore(State(pressure=5.0))
    agg = _RecordingAggregator()
    lm = _build(state=store, neurons=(StubTimerNeuron(delta=2.0),), aggregator=agg)

    run_tick(lm, logger=_RecordingLogger())

    # 5.0 carried over + 2.0 this tick = 7.0 seen by decide().
    assert agg.pressures == [7.0]


def test_run_tick_delegates_the_wake_call_entirely_to_the_aggregator() -> None:
    # The orchestrator holds no threshold: whatever the aggregator returns is
    # what run_tick returns. A stub that wakes at pressure 0 proves the tick
    # applies no suppression; a stub that sleeps at huge pressure proves it adds
    # no wake logic of its own.
    waker = _ConstantAggregator(
        WakeDecision.wake_with(WakePacket(reason="x", pressure_kind="k", pressure=0.0))
    )
    lm_wake = _build(neurons=(), aggregator=waker)
    assert run_tick(lm_wake, logger=_RecordingLogger()).wake is True
    assert waker.pressures == [0.0]  # woke at zero pressure — no tick-side gate

    sleeper = _ConstantAggregator(WakeDecision.stay_asleep())
    lm_sleep = _build(
        state=FakeStateStore(State(pressure=1_000.0)),
        neurons=(),
        aggregator=sleeper,
    )
    assert run_tick(lm_sleep, logger=_RecordingLogger()).wake is False  # slept at huge pressure


def test_run_tick_source_holds_no_threshold_literal() -> None:
    # The threshold decision lives in the aggregator, never in the orchestrator:
    # run_tick's source carries no copy of the threshold value.
    src = inspect.getsource(run_tick)
    assert str(DEFAULT_WAKE_THRESHOLD) not in src


def test_pressure_persists_across_a_fresh_graph_over_the_same_store() -> None:
    # Persistence proof: a *new* LifeModel (fresh bus/clock) over the same
    # committed store keeps accumulating from the persisted pressure, not from
    # zero — the number lives in State, not in the process.
    store = FakeStateStore()
    delta = 1.0
    lm1 = _build(state=store, clock=FakeClock(_T0), neurons=(StubTimerNeuron(delta=delta),))
    run_tick(lm1, logger=_RecordingLogger())
    assert store.load().pressure == delta

    lm2 = _build(
        state=store,
        clock=FakeClock(_T0 + timedelta(minutes=1)),
        neurons=(StubTimerNeuron(delta=delta),),
    )
    run_tick(lm2, logger=_RecordingLogger())
    assert store.load().pressure == 2 * delta  # continued, not reset


def test_default_neuron_signal_is_distinct_each_tick_so_dedup_never_collapses() -> None:
    # The stub neuron's origin_id is tied to tick_count, so a persistent dedup
    # ledger never swallows a later tick's impulse — every tick contributes.
    store = FakeStateStore()
    clock = FakeClock(_T0)
    bus = FakeSignalBus()
    lm = _build(state=store, bus=bus, clock=clock, neurons=(StubTimerNeuron(delta=1.0),))

    run_tick(lm, logger=_RecordingLogger())
    clock.advance(timedelta(minutes=1))
    run_tick(lm, logger=_RecordingLogger())

    # Two ticks, two distinct origin ids logged on the shared bus → pressure 2.0.
    assert store.load().pressure == 2.0


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


def test_main_default_neuron_accumulates_pressure_and_stays_asleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # End-to-end via the real entrypoint: the DEFAULT graph now wires the stub
    # timer neuron (both call sites get it), so State.pressure grows one delta
    # per tick, a neuron_fired event lands in the sink, and the wake gate stays
    # {"wakeAgent": false} — zero LLM.
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}
    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    sdir = tmp_path / "lifemodel"
    state = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    assert state["tick_count"] == 2
    assert state["pressure"] == 2.0  # default delta 1.0, accumulated over 2 ticks

    records = [
        json.loads(line)
        for line in (sdir / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    fired = [r for r in records if r.get("event") == EVENT_NEURON_FIRED]
    assert len(fired) == 2  # the wired-in neuron fired each tick


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

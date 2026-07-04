"""Unit tests for the wake → drain → cooldown → single-fire loop (roadmap 1.4).

These drive the real :func:`lifemodel.tick.run_tick` orchestrator with fakes
(``FakeClock`` / ``FakeStateStore`` / ``FakeSignalBus`` / ``FakeDelivery``), the
real :class:`ThresholdAggregator`, and the real ``StubTimerNeuron`` — no Hermes,
no LLM. They pin the Phase-1.4 safety rails that live in the tick:

* on a wake the pressure **drains** to zero, ``last_contact_at`` is stamped, and a
  ``cooldown_until`` is opened;
* an active cooldown **vetoes** a would-be wake even when pressure is above
  threshold → the gate stays ``{"wakeAgent": false}`` (zero LLM);
* therefore **exactly one** wake fires per threshold cycle, and the cooldown is
  time-bounded (it expires, allowing the next cycle to wake);
* the emitted wake-packet carries the *prior* contact time (HLA §11).

``run_tick`` never delivers — the message is Hermes' cron job's to send when the
gate flips ``true`` (HLA §7) — so ``FakeDelivery`` stays empty throughout; these
tests assert that invariant too.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import ThresholdAggregator
from lifemodel.core.neuron import StubTimerNeuron
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore
from lifemodel.tick import run_tick, wake_gate_line

_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
_MINUTE = timedelta(minutes=1)


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


def _build(
    *,
    store: FakeStateStore,
    clock: FakeClock,
    threshold: float = 3.0,
    delta: float = 1.0,
    delivery: FakeDelivery | None = None,
) -> Any:
    return build_lifemodel(
        base_dir=Path("/unused-for-fakes"),
        state=store,
        bus=FakeSignalBus(),
        clock=clock,
        delivery=delivery or FakeDelivery(),
        neurons=(StubTimerNeuron(delta=delta),),
        aggregator=ThresholdAggregator(threshold=threshold),
    )


def test_wake_drains_pressure_stamps_contact_and_opens_cooldown() -> None:
    store = FakeStateStore()
    clock = FakeClock(_T0)
    cooldown = timedelta(minutes=5)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    # Ticks 1, 2: below threshold, no drain, no cooldown yet.
    for _ in range(2):
        decision = run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown)
        assert decision.wake is False
        assert store.load().cooldown_until is None
        clock.advance(_MINUTE)

    # Tick 3: pressure reaches 3.0 == threshold → wake.
    wake_now = clock.now()  # _T0 + 2 minutes
    decision = run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown)

    assert decision.wake is True
    persisted = store.load()
    assert persisted.pressure == 0.0  # drained
    assert persisted.last_contact_at == wake_now.isoformat()  # contact stamped
    assert persisted.cooldown_until == (wake_now + cooldown).isoformat()  # cooldown open


def test_active_cooldown_vetoes_wake_even_above_threshold() -> None:
    # The heart of the safety rail: pressure is far above threshold, but an
    # unexpired cooldown keeps the gate asleep (zero LLM) and does NOT drain.
    cooldown_until = (_T0 + timedelta(minutes=5)).isoformat()
    store = FakeStateStore(State(pressure=100.0, cooldown_until=cooldown_until))
    lm = build_lifemodel(
        base_dir=Path("/unused"),
        state=store,
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),  # now < cooldown_until
        delivery=FakeDelivery(),
        neurons=(),  # keep pressure fixed at 100
        aggregator=ThresholdAggregator(threshold=3.0),
    )

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert json.loads(wake_gate_line(decision)) == {"wakeAgent": False}
    persisted = store.load()
    assert persisted.pressure == 100.0  # NOT drained while vetoed
    assert persisted.cooldown_until == cooldown_until  # cooldown untouched


def test_exactly_one_wake_per_threshold_cycle() -> None:
    # With a cooldown longer than the refill time, a long run wakes exactly once:
    # the crossing tick fires, then every subsequent above-threshold tick is
    # vetoed by the still-active cooldown.
    store = FakeStateStore()
    clock = FakeClock(_T0)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    wakes = 0
    for _ in range(12):
        if run_tick(lm, logger=_RecordingLogger(), cooldown=timedelta(minutes=30)).wake:
            wakes += 1
        clock.advance(_MINUTE)

    assert wakes == 1


def test_cooldown_expiry_allows_the_next_cycle_to_wake() -> None:
    # The cooldown is time-bounded: once it expires and pressure has rebuilt past
    # the threshold, the being wakes again — a second, distinct cycle.
    store = FakeStateStore()
    clock = FakeClock(_T0)
    cooldown = timedelta(minutes=5)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    wake_ticks: list[int] = []
    for i in range(1, 13):
        if run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown).wake:
            wake_ticks.append(i)
            assert store.load().pressure == 0.0  # each wake drains
        clock.advance(_MINUTE)

    # First wake at tick 3 (pressure 3). After draining, the cooldown expires and
    # pressure rebuilds to threshold again → a second wake in a later tick.
    assert len(wake_ticks) == 2
    assert wake_ticks[0] == 3
    assert wake_ticks[1] > wake_ticks[0]


def test_wake_packet_carries_prior_last_contact_at() -> None:
    # HLA §11: the wake-packet carries the last-contact time. The second wake's
    # packet must report the FIRST wake's timestamp (the prior contact), not the
    # instant of the second wake.
    store = FakeStateStore()
    clock = FakeClock(_T0)
    cooldown = timedelta(minutes=5)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    first_contact: str | None = None
    second_packet_contact: str | None = "unset"
    for _ in range(12):
        decision = run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown)
        if decision.wake:
            assert decision.packet is not None
            if first_contact is None:
                first_contact = store.load().last_contact_at
            else:
                second_packet_contact = decision.packet.last_contact_at
                break
        clock.advance(_MINUTE)

    assert first_contact is not None
    assert second_packet_contact == first_contact


def test_first_wake_packet_last_contact_is_none() -> None:
    # On the very first wake there is no prior contact, so the packet's
    # last_contact_at is None (the slot exists but is unpopulated).
    store = FakeStateStore()
    clock = FakeClock(_T0)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    decision = None
    for _ in range(3):
        decision = run_tick(lm, logger=_RecordingLogger())
        clock.advance(_MINUTE)

    assert decision is not None and decision.wake is True
    assert decision.packet is not None
    assert decision.packet.last_contact_at is None


def test_below_threshold_never_drains_or_sets_cooldown_and_never_delivers() -> None:
    store = FakeStateStore()
    clock = FakeClock(_T0)
    delivery = FakeDelivery()
    lm = _build(store=store, clock=clock, threshold=100.0, delta=1.0, delivery=delivery)

    for _ in range(5):
        decision = run_tick(lm, logger=_RecordingLogger())
        assert decision.wake is False
        clock.advance(_MINUTE)

    persisted = store.load()
    assert persisted.pressure == 5.0  # kept accumulating, never drained
    assert persisted.cooldown_until is None
    assert persisted.last_contact_at is None
    assert delivery.sent == []  # run_tick never delivers (that is Hermes' cron)


def test_cooldown_ticks_stay_asleep_while_pressure_rebuilds() -> None:
    # Immediately after a wake, the next ticks accumulate fresh pressure but the
    # gate stays {"wakeAgent": false} until the cooldown lifts — the drain + the
    # cooldown together enforce "no immediate second fire".
    store = FakeStateStore()
    clock = FakeClock(_T0)
    cooldown = timedelta(minutes=30)
    lm = _build(store=store, clock=clock, threshold=3.0, delta=1.0)

    # Reach and take the first wake (tick 3).
    for _ in range(3):
        run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown)
        clock.advance(_MINUTE)
    assert store.load().pressure == 0.0

    # The following ticks rebuild pressure but stay asleep under the cooldown.
    for _ in range(5):
        decision = run_tick(lm, logger=_RecordingLogger(), cooldown=cooldown)
        assert decision.wake is False
        assert json.loads(wake_gate_line(decision)) == {"wakeAgent": False}
        clock.advance(_MINUTE)

    assert store.load().pressure > 0.0  # pressure did rebuild, yet no second wake

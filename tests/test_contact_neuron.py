from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.contact_neuron import PresenceNeuron
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.taxonomy import KIND_CONTACT_PRESENCE, exchange_signal, read_contact_presence
from lifemodel.state.model import State


def _sensor() -> PresenceNeuron:
    return PresenceNeuron()


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals))


def test_emits_contact_presence_reading_with_dt(tmp_path) -> None:
    state = State(u=5.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min of silence
    intents = _sensor().step(_ctx(state, now, tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert emit.signal.kind == KIND_CONTACT_PRESENCE
    reading = read_contact_presence([emit.signal])
    assert reading is not None
    assert abs(reading.dt - 240.0) < 1e-9
    assert reading.qualities == ()


def test_sensor_writes_no_state_and_never_touches_u(tmp_path) -> None:
    # spec §3: PresenceNeuron is strictly instantaneous — it owns no durable state,
    # does NOT integrate, does NOT write u. Its sole output is the raw reading signal.
    state = State(u=5.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _sensor().step(_ctx(state, now, tmp_path=tmp_path))
    assert not any(isinstance(i, UpdateState) for i in intents)
    emits = [i for i in intents if isinstance(i, EmitSignal)]
    assert len(emits) == 1


def test_senses_exchange_quality_into_the_reading(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)  # dt=0
    ex = exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)  # q=1.0
    intents = _sensor().step(_ctx(state, now, [ex], tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    reading = read_contact_presence([emit.signal])
    assert reading is not None
    assert reading.qualities == (1.0,)


def test_own_impulse_reads_as_zero_quality(tmp_path) -> None:
    # The being's own proactive impulse is never user contact (q=0): the sensor
    # reports it as zero quality, so the drive never self-satiates on its own nudge.
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
    own = exchange_signal(
        origin_id="e-2", actor="proactive_internal", label="two_way", timestamp=None
    )
    intents = _sensor().step(_ctx(state, now, [own], tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    reading = read_contact_presence([emit.signal])
    assert reading is not None
    assert reading.qualities == (0.0,)


def test_no_last_tick_reads_zero_dt(tmp_path) -> None:
    # First tick (last_tick_at=None) → dt=0 ("no elapsed rise"); never crashes.
    state = State(u=0.0)
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _sensor().step(_ctx(state, now, tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    reading = read_contact_presence([emit.signal])
    assert reading is not None
    assert reading.dt == 0.0

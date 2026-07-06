from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.contact_neuron import ContactNeuron
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.taxonomy import KIND_CONTACT, exchange_signal
from lifemodel.state.model import State

ALPHA = 1.0 / 240.0


def _neuron() -> ContactNeuron:
    return ContactNeuron(alpha=ALPHA, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals))


def test_rises_by_elapsed_silence(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min → +1.0
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert abs(update.changes["u"] - 1.0) < 1e-9


def test_emits_contact_signal_with_value_and_delta(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert emit.signal.kind == KIND_CONTACT
    assert abs(emit.signal.payload["value"] - 1.0) < 1e-9
    assert abs(emit.signal.payload["delta"] - 1.0) < 1e-9


def test_exchange_satiates_the_drive(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)  # dt=0 → no rise
    ex = exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)  # q=1.0
    intents = _neuron().step(_ctx(state, now, [ex], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.0  # 1.0 - beta*1.0


def test_own_impulse_does_not_satiate(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
    own = exchange_signal(
        origin_id="e-2", actor="proactive_internal", label="two_way", timestamp=None
    )
    intents = _neuron().step(_ctx(state, now, [own], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 1.0  # proactive_internal → q=0 → unchanged


def test_neuron_writes_only_u(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert set(update.changes) == {"u"}

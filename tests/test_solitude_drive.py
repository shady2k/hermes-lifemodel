from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.solitude_drive import SolitudeDrive
from lifemodel.core.taxonomy import (
    KIND_CONTACT_PRESSURE,
    contact_presence_signal,
    contact_pressure_value,
)
from lifemodel.domain.signal import Signal
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State

ALPHA = 1.0 / 240.0

# ctx.trace is non-optional (spec §4.1); this drive writes no objects, so a literal
# span's ids suffice for the unit fixture.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


def _drive() -> SolitudeDrive:
    return SolitudeDrive(alpha=ALPHA, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(
        state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals), trace=_TRACE
    )


def _presence(dt: float, qualities: tuple[float, ...], *, origin_id: str = "p") -> Signal:
    return contact_presence_signal(origin_id=origin_id, dt=dt, qualities=qualities, timestamp=None)


def test_rises_by_elapsed_silence_from_presence_reading(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min → +1.0 at alpha=1/240
    intents = _drive().step(_ctx(state, now, [_presence(240.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert abs(update.changes["u"] - 1.0) < 1e-9


def test_emits_contact_signal_with_fresh_u_and_delta(tmp_path) -> None:
    # The snapshot-per-tick seam (T2 critical note): aggregation reads the fresh u
    # from this transient contact signal, NOT from ctx.state.u (which only updates
    # after commit). The signal carries value=fresh-u + the per-tick delta.
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [_presence(240.0, ())], tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert emit.signal.kind == KIND_CONTACT_PRESSURE
    assert abs(emit.signal.payload["value"] - 1.0) < 1e-9
    assert abs(emit.signal.payload["delta"] - 1.0) < 1e-9
    # ...readable exactly the way aggregation reads it (contact_pressure_value).
    assert abs(contact_pressure_value([emit.signal], default=0.0) - 1.0) < 1e-9


def test_satiate_quality_drains_u(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)  # dt=0 in the reading
    intents = _drive().step(_ctx(state, now, [_presence(0.0, (1.0,))], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.0  # 1.0 - beta*1.0


def test_zero_quality_does_not_satiate(tmp_path) -> None:
    # An own-impulse quality (q=0) never self-satiates: u is held (only the rise,
    # which is zero here since dt=0).
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [_presence(0.0, (0.0,))], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 1.0


def test_no_presence_reading_holds_u(tmp_path) -> None:
    # No contact_presence signal this tick (sensor absent / corrupt) → no rise, no
    # satiate: the drive HOLDS its value rather than guessing from stale state.
    state = State(u=0.7, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.7


def test_drive_writes_only_u(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
    intents = _drive().step(_ctx(state, now, [_presence(60.0, ())], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert set(update.changes) == {"u"}

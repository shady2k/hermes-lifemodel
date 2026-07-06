from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent, UpdateState
from lifemodel.core.personality import Personality
from lifemodel.state.model import State

PEAK = 13.0


def _p() -> Personality:
    return Personality(
        e_max=1.0,
        recovery_per_min=0.01,
        night_boost=0.5,
        fatigue_decay_per_min=0.002,
        peak_hour_utc=PEAK,
    )


def _ctx(state: State, now: datetime, *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=())


def _changes(intents: Sequence[Intent]) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def test_energy_recovers_during_idle(tmp_path) -> None:
    state = State(energy=0.5, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 12, 10, tzinfo=UTC)  # dt=10 min, near peak -> ~1x recovery
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["energy"] > 0.5  # recovered
    assert changes["energy"] <= 1.0


def test_energy_clamped_at_max(tmp_path) -> None:
    state = State(energy=0.99, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 18, 0, tzinfo=UTC)  # long dt
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["energy"] == 1.0


def test_fatigue_decays_during_rest(tmp_path) -> None:
    state = State(fatigue=0.5, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 13, 0, tzinfo=UTC)  # dt=60 min -> -0.12
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert abs(changes["fatigue"] - 0.38) < 1e-9


def test_fatigue_never_negative(tmp_path) -> None:
    state = State(fatigue=0.01, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 18, 0, tzinfo=UTC)  # long dt
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["fatigue"] == 0.0


def test_night_recovers_faster_than_day(tmp_path) -> None:
    # same dt, but at circadian trough (01:00 UTC) recovery is boosted vs peak (13:00 UTC)
    day = State(energy=0.5, last_tick_at="2026-07-06T13:00:00+00:00")
    day_changes = _changes(
        _p().step(_ctx(day, datetime(2026, 7, 6, 13, 10, tzinfo=UTC), tmp_path=tmp_path))
    )
    night = State(energy=0.5, last_tick_at="2026-07-06T01:00:00+00:00")
    night_changes = _changes(
        _p().step(_ctx(night, datetime(2026, 7, 6, 1, 10, tzinfo=UTC), tmp_path=tmp_path))
    )
    assert night_changes["energy"] > day_changes["energy"]  # night rest recovers faster

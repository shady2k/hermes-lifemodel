from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.cognition import Cognition
from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchProactive, UpdateState
from lifemodel.state.model import State

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _cog() -> Cognition:
    return Cognition(fast_cost=0.02, send_cost=0.03, alpha=2.0)


def _ctx(state: State, *, tmp_path) -> TickContext:
    return TickContext(state=state, now=NOW, bus=FileSignalBus(tmp_path), signals=())


def _launch(intents):
    return next((i for i in intents if isinstance(i, LaunchProactive)), None)


def _update(intents):
    return next((i for i in intents if isinstance(i, UpdateState)), None)


def test_no_active_desire_does_nothing(tmp_path) -> None:
    intents = _cog().step(_ctx(State(desire_status="none", u=2.0), tmp_path=tmp_path))
    assert list(intents) == []


def test_active_desire_launches_proactive_turn(tmp_path) -> None:
    state = State(desire_status="active", u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    launch = _launch(intents)
    assert launch is not None
    assert launch.correlation_id == f"proactive-{NOW.isoformat()}"
    assert launch.prompt  # carries the wake-packet prompt
    upd = _update(intents)
    assert upd.changes["pending_proactive_id"] == launch.correlation_id
    assert upd.changes["pending_proactive_since"] == NOW.isoformat()
    assert upd.changes["energy"] < 1.0  # reserved


def test_pending_turn_is_not_relaunched(tmp_path) -> None:
    state = State(desire_status="active", u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    assert _launch(intents) is None  # idempotent — a turn is already in flight


def test_insufficient_energy_holds_no_launch(tmp_path) -> None:
    # estimate = (0.02+0.03)*(1+2*1.0)=0.15 at max fatigue; energy 0.05 can't afford
    state = State(desire_status="active", u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    assert _launch(intents) is None  # emergent shutoff — hold
    assert _update(intents) is None  # energy untouched, desire stays active


def test_prompt_has_no_raw_numbers(tmp_path) -> None:
    import re

    state = State(desire_status="active", u=3.2, energy=1.0)
    launch = _launch(_cog().step(_ctx(state, tmp_path=tmp_path)))
    assert not re.search(r"\d", launch.prompt)

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import Component, TickContext
from lifemodel.core.intents import Intent, UpdateState
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State

# ctx.trace is non-optional now (spec §4.1) — tracing is mandatory, so every
# construction site supplies a span's ids (here a literal for a bare unit test).
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)


class Ticker:
    id = "ticker"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [UpdateState({"tick_count": ctx.state.tick_count + 1})]


def test_tick_context_exposes_state_now_bus(tmp_path) -> None:
    bus = FileSignalBus(tmp_path)
    now = datetime(2026, 7, 6, tzinfo=UTC)
    ctx = TickContext(state=State(tick_count=4), now=now, bus=bus, trace=_TRACE)
    assert ctx.state.tick_count == 4
    assert ctx.now is now
    assert ctx.bus is bus


def test_component_protocol_is_satisfied_structurally(tmp_path) -> None:
    ticker = Ticker()
    assert isinstance(ticker, Component)
    ctx = TickContext(
        state=State(tick_count=4),
        now=datetime(2026, 7, 6, tzinfo=UTC),
        bus=FileSignalBus(tmp_path),
        trace=_TRACE,
    )
    (intent,) = ticker.step(ctx)
    assert isinstance(intent, UpdateState)
    assert intent.changes == {"tick_count": 5}

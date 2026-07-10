from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

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


def test_tick_context_exposes_state_and_now() -> None:
    now = datetime(2026, 7, 6, tzinfo=UTC)
    ctx = TickContext(state=State(tick_count=4), now=now, trace=_TRACE)
    assert ctx.state.tick_count == 4
    assert ctx.now is now
    assert ctx.signals == ()  # the frame's signals default empty (spec §3)


def test_component_protocol_is_satisfied_structurally() -> None:
    ticker = Ticker()
    assert isinstance(ticker, Component)
    ctx = TickContext(
        state=State(tick_count=4),
        now=datetime(2026, 7, 6, tzinfo=UTC),
        trace=_TRACE,
    )
    (intent,) = ticker.step(ctx)
    assert isinstance(intent, UpdateState)
    assert intent.changes == {"tick_count": 5}

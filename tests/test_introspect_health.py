"""Brain-liveness readings: derived from ``last_tick_at`` freshness (spec §16).

The debug command reads only from disk, so it cannot see the live adapter loop.
Liveness is instead inferred from how recently the loop stamped ``last_tick_at``
(via ``coreloop.tick()``). Fresh ⇒ the brain is ticking; stale ⇒ the loop is
probably down and the gateway should have reconnected the adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.introspect import BRAIN_STALE_MIN, compute_readings
from lifemodel.debug import _cfg
from lifemodel.state.model import State

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _readings(last_tick_iso: str | None):
    return compute_readings(State(last_tick_at=last_tick_iso), now=NOW, cfg=_cfg())


def test_brain_alive_when_last_tick_fresh() -> None:
    assert BRAIN_STALE_MIN == 2.0
    assert _readings("2026-07-06T11:59:00+00:00").brain_alive is True  # 1 min ago


def test_brain_stale_when_last_tick_old() -> None:
    assert _readings("2026-07-06T11:40:00+00:00").brain_alive is False  # 20 min ago


def test_brain_stale_when_never_ticked() -> None:
    assert _readings(None).brain_alive is False

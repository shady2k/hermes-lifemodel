from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.pressure import effective_pressure, inhibition_at

BASE = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def test_no_action_pending_is_zero_inhibition() -> None:
    assert inhibition_at(None, BASE, i0=1.0, grace_min=45.0, halflife_min=60.0) == 0.0


def test_grace_plateau_holds_full_inhibition() -> None:
    since = BASE.isoformat()
    assert inhibition_at(since, _at(0), i0=1.0, grace_min=45.0, halflife_min=60.0) == 1.0
    assert inhibition_at(since, _at(44), i0=1.0, grace_min=45.0, halflife_min=60.0) == 1.0


def test_decays_by_halflife_after_grace() -> None:
    since = BASE.isoformat()
    # one half-life (60 min) after the grace end (45 min) -> ~0.5
    val = inhibition_at(since, _at(45 + 60), i0=1.0, grace_min=45.0, halflife_min=60.0)
    assert abs(val - 0.5) < 1e-9


def test_inhibition_clamped_to_unit_interval() -> None:
    since = BASE.isoformat()
    v = inhibition_at(since, _at(10_000), i0=1.0, grace_min=45.0, halflife_min=60.0)
    assert 0.0 <= v <= 1.0
    assert v < 1e-3  # long after: essentially gone


def test_effective_pressure_suppressed_by_inhibition() -> None:
    assert effective_pressure(2.0, 1.0) == 0.0  # fully inhibited
    assert effective_pressure(2.0, 0.0) == 2.0  # no inhibition
    assert abs(effective_pressure(2.0, 0.5) - 1.0) < 1e-9
    assert effective_pressure(0.0, 0.0) == 0.0

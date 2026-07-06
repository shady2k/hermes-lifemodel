from __future__ import annotations

from lifemodel.core.intake import IntakeLimits, IntakeResult, apply_intake
from lifemodel.core.taxonomy import lane_of
from lifemodel.domain.signal import Signal


def _c(i: int) -> Signal:  # a contact (sensor) signal
    return Signal(origin_id=f"c{i}", kind="contact", payload={"value": float(i)})


def _ex(i: int) -> Signal:  # an exchange (control) signal
    return Signal(origin_id=f"e{i}", kind="exchange", payload={"actor": "user", "label": "two_way"})


def _limits(**kw) -> IntakeLimits:
    return IntakeLimits(**kw)


def test_control_signals_are_kept_lossless_and_first() -> None:
    signals = [_c(1), _ex(1), _c(2), _ex(2)]
    res = apply_intake(signals, limits=_limits(), lane_of=lane_of)
    kept_kinds = [s.kind for s in res.kept]
    assert kept_kinds[:2] == ["exchange", "exchange"]  # control first
    assert res.shed_control == 0


def test_sensor_kind_coalesces_to_latest_wins() -> None:
    # three contact signals in one batch collapse to the last one
    res = apply_intake([_c(1), _c(2), _c(3)], limits=_limits(), lane_of=lane_of)
    contacts = [s for s in res.kept if s.kind == "contact"]
    assert len(contacts) == 1
    assert contacts[0].origin_id == "c3"  # latest wins
    assert res.coalesced_sensor == 2


def test_control_overflow_is_counted_not_reordered() -> None:
    signals = [_ex(i) for i in range(10)]
    res = apply_intake(signals, limits=_limits(max_control=4), lane_of=lane_of)
    assert len([s for s in res.kept if s.kind == "exchange"]) == 4
    assert res.shed_control == 6
    assert [s.origin_id for s in res.kept[:4]] == ["e0", "e1", "e2", "e3"]  # prefix, in order


def test_flood_never_raises_and_is_bounded() -> None:
    # 10_000 mixed signals in one batch -> bounded output, control preserved, no exception
    flood = []
    for i in range(5000):
        flood.append(_c(i))
        flood.append(_ex(i))
    res = apply_intake(flood, limits=_limits(max_control=256, max_sensor=64), lane_of=lane_of)
    assert isinstance(res, IntakeResult)
    exchanges = [s for s in res.kept if s.kind == "exchange"]
    contacts = [s for s in res.kept if s.kind == "contact"]
    assert len(exchanges) == 256  # control kept up to the cap
    assert len(contacts) == 1  # 5000 contacts coalesced to latest
    assert res.shed_control == 5000 - 256
    assert len(res.kept) <= 256 + 64  # O(bounded)


def test_unknown_kinds_are_sensors_and_droppable() -> None:
    weird = [Signal(origin_id=f"w{i}", kind="mystery", payload={}) for i in range(100)]
    res = apply_intake(weird, limits=_limits(max_sensor=64), lane_of=lane_of)
    # 'mystery' is a sensor kind -> coalesced to latest-wins (1 survives)
    assert len([s for s in res.kept if s.kind == "mystery"]) == 1

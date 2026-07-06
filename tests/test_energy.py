from __future__ import annotations

from lifemodel.core.energy import Reservation, can_afford, cost_real, reserve, settle

COST_FAST = 0.02
ALPHA = 2.0
E_MIN_AFFORDABLE = 0.1


def test_cost_inflates_with_fatigue() -> None:
    assert cost_real(0.02, 0.0, alpha=ALPHA) == 0.02
    assert abs(cost_real(0.02, 1.0, alpha=ALPHA) - 0.06) < 1e-9  # 0.02*(1+2*1)


def test_dont_petrify_cheapest_act_affordable_at_max_fatigue() -> None:
    # the calibrated invariant: cheapest thought at S=1 stays under the affordability floor
    assert cost_real(COST_FAST, 1.0, alpha=ALPHA) < E_MIN_AFFORDABLE


def test_reserve_gates_on_affordability() -> None:
    assert reserve(0.05, 0.10) is None  # can't afford
    result = reserve(0.30, 0.10)
    assert result is not None
    energy_after, res = result
    assert abs(energy_after - 0.20) < 1e-9  # estimate held
    assert res == Reservation(reserved=0.10)


def test_settle_refunds_unused_estimate() -> None:
    energy_after, res = reserve(0.30, 0.10)  # energy 0.30 -> 0.20, reserved 0.10
    final = settle(energy_after, res, actual=0.04)  # only spent 0.04
    assert abs(final - 0.26) < 1e-9  # 0.20 + (0.10 - 0.04) => net -0.04 from 0.30


def test_overspend_pushes_energy_negative() -> None:
    energy_after, res = reserve(0.05, 0.05)  # energy -> 0.0, reserved 0.05
    final = settle(energy_after, res, actual=0.12)  # blew the estimate
    assert final < 0.0  # 0.0 + (0.05 - 0.12) = -0.07 — allowed, forces recovery


def test_can_afford() -> None:
    assert can_afford(0.10, 0.06) is True
    assert can_afford(0.05, 0.06) is False

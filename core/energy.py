"""Energy budget: cost model + reserve/settle lifecycle (spec §8).

Only cognition pays. Before an expensive act the CoreLoop (Phase D) reserves the
*expected* cost (gate: is it affordable?); afterwards it settles the *actual*
cost and refunds the unused estimate. Fatigue ``S`` inflates cost —
``cost_real = cost_base·(1+α·S)`` — so a tired being naturally drops to reflexes
without any ``if energy < X`` (progressive shutoff is emergent). Overspend is
allowed: energy may go negative and recover. Ego-depletion is NOT modelled.
"""

from __future__ import annotations

from dataclasses import dataclass


def cost_real(cost_base: float, s: float, *, alpha: float) -> float:
    """The fatigue-inflated cost of an act: ``cost_base·(1+α·s)``."""
    return cost_base * (1.0 + alpha * s)


def can_afford(energy: float, cost: float) -> bool:
    return energy >= cost


@dataclass(frozen=True)
class Reservation:
    """A held energy estimate, settled once the act's actual cost is known."""

    reserved: float


def reserve(energy: float, estimate: float) -> tuple[float, Reservation] | None:
    """Gate on affordability and hold ``estimate``. Returns ``(energy_after,
    reservation)`` or ``None`` if unaffordable."""
    if not can_afford(energy, estimate):
        return None
    return energy - estimate, Reservation(reserved=estimate)


def settle(energy: float, reservation: Reservation, actual: float) -> float:
    """Refund the unused estimate; net deduction is the actual cost. Overspend
    (``actual > reserved``) drives energy negative — allowed (forced recovery)."""
    return energy + (reservation.reserved - actual)

"""The drive-component — the contact urge ``u`` and its dynamics (spec §5).

The drive is the one *continuous* state variable. It rises linearly in genuine
silence, is satiated by positive exchanges, and is drained (fully or partially)
when the wake-decision consumes an URGE. Project convention: the dynamics live
in this dedicated drive-component, not smeared into a neuron.

Constants are normalised so ``θ_u = 1`` and ``β = 1`` (spec §6): a genuine
two-way exchange (``q = 1``) fully drains a threshold-height urge.
"""

from __future__ import annotations

import math

from lifemodel.sim.drive import Drive


def test_rises_by_alpha_times_dt_in_silence() -> None:
    drive = Drive(alpha=0.25, u=0.0)

    drive.rise(dt=2.0)

    assert drive.u == 0.5


def test_rise_is_capped_at_u_max() -> None:
    drive = Drive(alpha=1.0, u_max=1.5, u=1.0)

    drive.rise(dt=10.0)

    assert drive.u == 1.5


def test_positive_exchange_satiates_by_beta_times_q() -> None:
    drive = Drive(alpha=0.1, beta=1.0, u=0.9)

    drive.satiate(q=0.5)  # an ack drains half a threshold

    assert drive.u == 0.4


def test_satiation_floors_at_zero() -> None:
    drive = Drive(alpha=0.1, beta=1.0, u=0.3)

    drive.satiate(q=1.0)  # would push below zero

    assert drive.u == 0.0


def test_nonpositive_quality_does_not_satiate() -> None:
    # q = 0 (monologue / internal impulse) and q < 0 (rejection) never reduce u.
    drive = Drive(alpha=0.1, beta=1.0, u=0.7)

    drive.satiate(q=0.0)
    assert drive.u == 0.7

    drive.satiate(q=-0.5)
    assert drive.u == 0.7


def test_full_drain_zeroes_the_urge() -> None:
    drive = Drive(alpha=0.1, u=0.95)

    drive.drain()  # default: full drain (u ← 0)

    assert drive.u == 0.0


def test_partial_drain_scales_the_urge() -> None:
    # no_drain_on_decline uses a partial drain u ← (1 − fraction)·u.
    drive = Drive(alpha=0.1, u=1.0)

    drive.drain(fraction=0.3)

    assert math.isclose(drive.u, 0.7)


def test_u_max_defaults_to_unbounded() -> None:
    # With linear rise and θ_u = 1 < u_max, the ceiling is purely defensive;
    # by default it never binds (spec §11 "confirm it never binds").
    drive = Drive(alpha=1.0, u=0.0)

    drive.rise(dt=1000.0)

    assert drive.u == 1000.0

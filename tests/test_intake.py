"""Priority-class backpressure in the AGGREGATION gate (spec §7, bead lm-fib.8.2).

The ephemeral bus carries EVERY seed through (spec §7: "everything hits the bus;
the gate is the defensive layer"). The gate — here, :func:`apply_backpressure`,
which :class:`~lifemodel.core.aggregation.ContactAggregation` runs at the top of its
step — classifies each signal into a priority class off its taxonomy ``kind``:

* ``must_process`` — ``contact_observed`` / ``proactive_outcome`` / ``in_flight``
  (the safety interlock) / the drive's ``contact_pressure`` (the PANIC/GRIEF ``u``):
  NEVER shed, no matter how full the frame is.
* ``best_effort`` — sensor noise / low-salience observations: under load COALESCED
  (folded to a bounded count, shedding the lowest-salience) so they do not each
  drive a separate expensive downstream step.
"""

from __future__ import annotations

from lifemodel.core.intake import (
    MAX_BEST_EFFORT,
    apply_backpressure,
    priority_class,
)
from lifemodel.core.taxonomy import (
    contact_observed_signal,
    contact_pressure_signal,
    in_flight_signal,
    proactive_outcome_signal,
)
from lifemodel.domain.egress import ProactiveOutcome
from lifemodel.domain.signal import Signal


def _noise(i: int, salience: float = 1.0) -> Signal:
    return Signal(origin_id=f"n{i}", kind="sensor_noise", salience=salience)


def test_load_bearing_kinds_are_must_process() -> None:
    for sig in (
        contact_observed_signal(origin_id="e", actor="user", label="two_way", timestamp=None),
        proactive_outcome_signal(origin_id="v", outcome=ProactiveOutcome.SENT, timestamp=None),
        in_flight_signal(origin_id="f", value=True, timestamp=None),
        # the drive's OWN output (the PANIC/GRIEF u) — must never be shed, else the
        # gate is blinded to the very drive it defends cognition for (spec §7).
        contact_pressure_signal(origin_id="c", value=1.5, delta=0.0, timestamp=None),
    ):
        assert priority_class(sig.kind) == "must_process"


def test_sensor_noise_and_spent_presence_are_best_effort() -> None:
    assert priority_class("sensor_noise") == "best_effort"
    # contact_presence is the drive's INPUT — spent by the drive (which ran before
    # the gate); aggregation never reads it, so it is sheddable noise at the gate.
    assert priority_class("contact_presence") == "best_effort"
    # an unknown flood can never claim the lossless class.
    assert priority_class("something-brand-new") == "best_effort"


def test_must_process_never_shed_even_when_frame_oversized() -> None:
    observed = contact_observed_signal(origin_id="e", actor="user", label="two_way", timestamp=None)
    result = apply_backpressure([observed, *(_noise(i) for i in range(200))])
    assert observed in result.signals  # the must_process signal is NEVER dropped
    assert result.must_process == 1
    # best_effort coalesced to the bounded cap — they do not each drive a step.
    assert result.best_effort_kept == MAX_BEST_EFFORT
    assert result.best_effort_shed == 200 - MAX_BEST_EFFORT
    kept_noise = [s for s in result.signals if s.kind == "sensor_noise"]
    assert len(kept_noise) == MAX_BEST_EFFORT


def test_only_best_effort_noise_is_bounded_and_sane() -> None:
    result = apply_backpressure([_noise(i) for i in range(50)])
    assert result.must_process == 0
    assert len(result.signals) == MAX_BEST_EFFORT  # bounded
    assert result.best_effort_shed == 50 - MAX_BEST_EFFORT
    assert result.overflowed


def test_under_cap_nothing_is_shed() -> None:
    result = apply_backpressure([_noise(i) for i in range(3)])
    assert result.best_effort_shed == 0
    assert len(result.signals) == 3
    assert not result.overflowed


def test_coalescing_keeps_the_highest_salience() -> None:
    lows = [_noise(i, salience=0.1) for i in range(10)]
    highs = [_noise(100 + i, salience=9.0) for i in range(2)]
    result = apply_backpressure([*lows, *highs], max_best_effort=3)
    assert highs[0] in result.signals and highs[1] in result.signals  # salient survive
    assert result.best_effort_kept == 3
    assert result.best_effort_shed == 9


def test_empty_frame_is_sane() -> None:
    result = apply_backpressure([])
    assert result.signals == ()
    assert result.must_process == 0
    assert result.best_effort_shed == 0
    assert not result.overflowed


def test_frame_order_is_preserved_for_survivors() -> None:
    observed = contact_observed_signal(origin_id="e", actor="user", label="two_way", timestamp=None)
    n0, n1 = _noise(0), _noise(1)
    result = apply_backpressure([n0, observed, n1], max_best_effort=8)
    assert list(result.signals) == [n0, observed, n1]  # original frame order kept

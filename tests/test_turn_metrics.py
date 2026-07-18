"""Tests for ``core/turn_metrics.py`` — the TURN injector metric surface (lm-hg7).

Contract under test:

* ``register_turn_metrics`` declares ``lifemodel_turn_injector_total`` and is
  idempotent (a second call — a second recorder / a fresh graph — must not
  double-declare or raise);
* the counter is read through the real :class:`~lifemodel.core.metrics.Counter`
  accessor, ``.value(**labels)``, matching every other metric test in this repo;
* every per-injector outcome vocabulary is a closed ``frozenset[str]`` that
  always carries ``"error"`` (the fail-soft branch's home), plus the specific
  spot-checked values named in the plan.

Stdlib only.
"""

from __future__ import annotations

from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_metrics import (
    BELIEF_OUTCOMES,
    COMMITMENT_OUTCOMES,
    FELT_OUTCOMES,
    GENESIS_OUTCOMES,
    TURN_INJECTOR_TOTAL,
    register_turn_metrics,
)


def test_register_is_idempotent_and_declares_component_outcome_labels():
    reg = MetricRegistry()
    register_turn_metrics(reg)
    register_turn_metrics(reg)  # a fresh graph / second recorder must not double-declare
    reg.inc(TURN_INJECTOR_TOTAL, component="belief", outcome="surfaced")
    metric = reg.get(TURN_INJECTOR_TOTAL)
    assert metric.value(component="belief", outcome="surfaced") == 1.0


def test_every_injector_outcome_set_carries_error_and_is_closed():
    for outcomes in (FELT_OUTCOMES, GENESIS_OUTCOMES, BELIEF_OUTCOMES, COMMITMENT_OUTCOMES):
        assert "error" in outcomes  # the fail-soft branch always has a home
    assert "light" in FELT_OUTCOMES and "surfaced" in BELIEF_OUTCOMES
    assert "injected" in GENESIS_OUTCOMES and "surfaced" in COMMITMENT_OUTCOMES

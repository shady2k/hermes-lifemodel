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

Codex review M3: the above only proved the frozensets were self-consistent —
nothing tied them to what the injectors actually emit, so a stale/dead value
(``FELT_OUTCOMES`` carried a ``"cooldown_unchanged"`` from the RETIRED
``lifemodel_felt_display_total`` counter that ``Decision`` never had a member
for) could sit there as false assurance indefinitely. The lightest honest fix
that does not require faking a live call per injector:

* ``FELT_OUTCOMES`` is asserted EQUAL to
  ``{d.value for d in core.felt_display.Decision} | {"error"}`` — felt-state's
  outcome IS that enum's ``.value`` (``hooks.py``'s
  ``span.set(outcome=decision.value, ...)``), so this ties the frozenset
  directly to its one real source rather than a hand-copied literal;
* genesis/belief/commitment have no such enum (each ``outcome=`` is a literal
  string dropped straight into ``hooks.py``), so those are checked by
  INTROSPECTING each injector factory's own source for every literal
  ``outcome="..."`` call site and asserting that set, plus ``"error"`` (the
  fail-soft branch's home, never a literal call site in the factory itself),
  is EQUAL to the declared frozenset — not merely a subset. A subset check
  only catches a typo'd/renamed outcome the injector emits that the frozenset
  doesn't cover; it says nothing about the OTHER direction — a dead value
  sitting in the frozenset that no code path emits (exactly the class of bug
  ``FELT_OUTCOMES``' retired ``"cooldown_unchanged"`` was) survives a subset
  check forever. Equality catches both directions.

Stdlib only.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from lifemodel import hooks as hooks_module
from lifemodel.core.felt_display import Decision
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


def test_felt_outcomes_matches_the_decide_gates_own_vocabulary():
    # M3: this is the drift FELT_OUTCOMES actually had — a dead
    # "cooldown_unchanged" from the retired lifemodel_felt_display_total counter
    # that Decision has never had a member for. Tying the frozenset directly to
    # the enum (felt-state's real outcome source, `span.set(outcome=decision.value,
    # ...)`) makes this kind of drift structurally impossible to miss again.
    assert {d.value for d in Decision} | {"error"} == FELT_OUTCOMES


def _literal_outcomes(factory: Any) -> set[str]:
    """Every literal ``outcome="..."`` string in *factory*'s own source — a
    lightweight static scan of its ``span.set(outcome=...)`` call sites, not a
    live call (genesis/belief/commitment have no enum to tie the frozenset to
    the way felt-state does)."""
    source = inspect.getsource(factory)
    return set(re.findall(r'outcome="([a-z_]+)"', source))


def test_genesis_outcomes_cover_every_literal_outcome_the_injector_emits():
    literal = _literal_outcomes(hooks_module.make_genesis_injector)
    assert literal  # sanity: the scan actually found real span.set(outcome=...) calls
    # C-M3: EQUAL, not a subset — a subset check would let a dead value (one the
    # frozenset declares but no code path ever emits) survive undetected forever.
    assert literal | {"error"} == GENESIS_OUTCOMES


def test_belief_outcomes_cover_every_literal_outcome_the_injector_emits():
    literal = _literal_outcomes(hooks_module.make_belief_injector)
    assert literal
    assert literal | {"error"} == BELIEF_OUTCOMES


def test_commitment_outcomes_cover_every_literal_outcome_the_injector_emits():
    literal = _literal_outcomes(hooks_module.make_commitment_injector)
    assert literal
    assert literal | {"error"} == COMMITMENT_OUTCOMES

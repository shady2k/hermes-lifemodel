"""Turn-hook metric surface (lm-hg7) — the per-turn injector outcome counter.

Symmetric to :mod:`core.tick_metrics` (the tick surface): the TURN path (the
``pre_llm_call`` injectors) gets ONE shared counter,
``lifemodel_turn_injector_total``, carrying the injector name on ``component`` and
its per-call verdict on ``outcome`` (both allowed keys in ``MetricSpec``'s closed
label set). It REPLACES felt-state's retired ``lifemodel_felt_display_total`` —
that was exactly this counter for one injector. Emitted once per injector
invocation by the :class:`~lifemodel.core.turn_recorder.TurnRecorder`'s
``injector_span`` close, fail-open.

The outcome strings are the SINGLE SOURCE (the registry validates label *keys*,
not *values*, so a typo silently forks a series): keep every emission's outcome in
the matching frozenset here.
"""

from __future__ import annotations

from .metrics import MetricRegistry, MetricSpec

TURN_INJECTOR_TOTAL = "lifemodel_turn_injector_total"

INJECTOR_FELT = "felt_state"
INJECTOR_GENESIS = "genesis"
INJECTOR_BELIEF = "belief"
INJECTOR_COMMITMENT = "commitment"

#: felt-state's per-call gate verdict (the retired FELT_DISPLAY_TOTAL vocabulary) + error.
FELT_OUTCOMES = frozenset(
    {"light", "not_warmed", "not_salient", "task", "cooldown_unchanged", "error"}
)
#: genesis's disjoint no-inject branches + the one inject + error.
GENESIS_OUTCOMES = frozenset(
    {"injected", "born", "carried_by_impulse", "own_impulse", "not_due", "stale_identity", "error"}
)
BELIEF_OUTCOMES = frozenset({"surfaced", "empty", "unavailable", "error"})
COMMITMENT_OUTCOMES = frozenset({"surfaced", "empty", "unavailable", "error"})

TURN_INJECTOR_SPEC = MetricSpec(
    name=TURN_INJECTOR_TOTAL,
    kind="counter",
    help="pre_llm_call injector invocations by injector (component) and per-call verdict "
    "(outcome).",
    label_keys=("component", "outcome"),
)


def register_turn_metrics(registry: MetricRegistry) -> None:
    """Declare the turn metric into *registry* (fail-fast on a bad spec, idempotent
    for the identical one — safe for a second recorder / a fresh graph to re-run)."""
    registry.register(TURN_INJECTOR_SPEC)

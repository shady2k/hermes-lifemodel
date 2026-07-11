"""Universal tick metrics — the harness-emitted operational surface (telemetry-core §4.2).

The CoreLoop wraps every ``step()`` and auto-emits these into the shared
:class:`~lifemodel.core.metrics.MetricRegistry` WITHOUT the component cooperating
(§3 invariant, lock #1: the only prod execution path is the wrapper, so a
component cannot run unmeasured). This module owns their canonical NAMES and their
declarative :class:`~lifemodel.core.metrics.MetricSpec`s; :func:`register_universal_metrics`
declares them fail-fast into a registry (idempotent — a plugin ``register()`` and
every fresh per-tick graph re-run it, so declaration must be safe to repeat).

**Label mapping to the closed set (bead lm-fib.7.2).** ``MetricSpec`` enforces the
closed low-cardinality label set ``{component, layer, phase, reason, outcome,
model}`` — a label outside it fails fast. Design §4.2 sketches
``component_runs_total{…,status}`` and ``signals_intake_total{lane,result}``, but
``status``/``lane``/``result`` are not closed-set keys, so we carry the run
outcome and the intake bucket on the allowed ``outcome`` label instead:

* ``lifemodel_component_runs_total`` → ``outcome ∈ {ok, suppressed, failed}``
  (the derived status of §4.2);
* ``lifemodel_signals_intake_total`` → ``outcome ∈ {kept, shed_control,
  shed_sensor, coalesced}`` (the lane is folded into the value, preserving every
  :class:`~lifemodel.core.intake.IntakeResult` bucket without a ``lane`` key).

``lifemodel_suppressions_total{component,reason}`` is declared here too but emitted
at the choke-point :func:`~lifemodel.core.suppression.emit_suppression_span`, so
EVERY suppression (in-tick + out-of-tick) is counted through one door.
"""

from __future__ import annotations

from .metrics import DEFAULT_BUCKETS, MetricRegistry, MetricSpec

# --- Canonical metric names (design §4.2 — exact) --------------------------- #

TICK_DURATION = "lifemodel_tick_duration_seconds"
TICK_LAG = "lifemodel_tick_lag_seconds"
COMPONENT_DURATION = "lifemodel_component_duration_seconds"
COMPONENT_RUNS = "lifemodel_component_runs_total"
SIGNALS_INTAKE = "lifemodel_signals_intake_total"
LAYER_ACCEPTS_SIGNALS = "lifemodel_layer_accepts_signals"
TRACE_WRITER_DROPPED = "lifemodel_trace_writer_dropped_records"
TRACE_WRITER_WRITE_ERRORS = "lifemodel_trace_writer_write_errors"
SUPPRESSIONS_TOTAL = "lifemodel_suppressions_total"
#: Afferent observer bodies that raised (fail-loud, spec §4.3/MAJOR-4). The observer
#: NAME rides the ``component`` label (closed-set), e.g. ``post_llm_call`` /
#: ``pre_gateway_dispatch`` — a runtime observer failure is plugin-owned + counted,
#: never left to Hermes' hook wrapper.
OBSERVER_ERRORS = "lifemodel_observer_errors_total"

#: The closed run-status vocabulary carried on the ``outcome`` label (§4.2 derivation:
#: failed on exception, else suppressed on a suppressed span, else ok).
RUN_OK = "ok"
RUN_SUPPRESSED = "suppressed"
RUN_FAILED = "failed"

#: The closed intake-bucket vocabulary carried on the ``outcome`` label — one value
#: per :class:`~lifemodel.core.intake.IntakeResult` field (lane folded into the value).
INTAKE_KEPT = "kept"
INTAKE_SHED_CONTROL = "shed_control"
INTAKE_SHED_SENSOR = "shed_sensor"
INTAKE_COALESCED = "coalesced"


# --- Declarative specs ------------------------------------------------------ #

UNIVERSAL_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name=TICK_DURATION,
        kind="histogram",
        unit="seconds",
        help="Wall-clock duration of one CoreLoop tick.",
        buckets=DEFAULT_BUCKETS,
    ),
    MetricSpec(
        name=TICK_LAG,
        kind="gauge",
        unit="seconds",
        help="Seconds between the previous tick and this tick (now - last_tick_at).",
    ),
    MetricSpec(
        name=COMPONENT_DURATION,
        kind="histogram",
        unit="seconds",
        help="Wall-clock duration of one component's step().",
        label_keys=("component", "layer"),
        buckets=DEFAULT_BUCKETS,
    ),
    MetricSpec(
        name=COMPONENT_RUNS,
        kind="counter",
        help="Component step() runs by derived status (ok/suppressed/failed).",
        label_keys=("component", "layer", "outcome"),
    ),
    MetricSpec(
        name=SIGNALS_INTAKE,
        kind="counter",
        help="Signals through intake by bucket (kept/shed_control/shed_sensor/coalesced).",
        label_keys=("outcome",),
    ),
    MetricSpec(
        name=LAYER_ACCEPTS_SIGNALS,
        kind="gauge",
        help="1 if any component in the layer consumes signals, else 0 (from the manifest).",
        label_keys=("layer",),
    ),
    MetricSpec(
        name=TRACE_WRITER_DROPPED,
        kind="gauge",
        help="Absolute trace-writer records dropped since boot (queue-full snapshot).",
    ),
    MetricSpec(
        name=TRACE_WRITER_WRITE_ERRORS,
        kind="gauge",
        help="Absolute trace-writer write errors since boot (swallowed, snapshot).",
    ),
    MetricSpec(
        name=SUPPRESSIONS_TOTAL,
        kind="counter",
        help="Suppression spans by component + closed reason (choke-point counted).",
        label_keys=("component", "reason"),
    ),
    MetricSpec(
        name=OBSERVER_ERRORS,
        kind="counter",
        help="Afferent observer bodies that raised, by observer name (component label).",
        label_keys=("component",),
    ),
)


def register_universal_metrics(registry: MetricRegistry) -> None:
    """Declare every universal spec into *registry* (fail-fast, idempotent).

    Safe to call repeatedly: :meth:`MetricRegistry.register` returns the existing
    metric for an identical spec, so the per-tick graph rebuild (a fresh
    :class:`~lifemodel.core.coreloop.CoreLoop` every tick) never double-declares.
    """
    for spec in UNIVERSAL_SPECS:
        registry.register(spec)

"""Tests for ``core/metrics.py`` — the ``MetricRegistry`` + metric types (spec §4.1).

Contract under test (telemetry-core design §4.1/§4.3/§7):

* the three stdlib metric types — ``Counter`` (monotonic), ``Gauge`` (set), and
  ``Histogram`` (fixed buckets + ``count`` + ``sum``);
* declaration/registration + spec validation is **fail-fast** (a malformed
  ``MetricSpec`` or a label key outside the closed low-cardinality set raises at
  build/test time);
* runtime emission (``inc``/``set``/``observe`` on the registry) is **fail-open**:
  an unknown metric, wrong kind, or undeclared label is a no-op that bumps the
  self-registered ``lifemodel_metrics_emit_errors_total{reason}`` counter and
  NEVER raises (a tick must not die on a bad metric);
* ``get_metric_registry(base_dir)`` is a singleton-per-base_dir, idempotent;
* ``label_key`` is a deterministic, order-independent canonical key.

Stdlib only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifemodel.core.metrics import (
    ALLOWED_LABEL_KEYS,
    DEFAULT_BUCKETS,
    EMIT_ERRORS_METRIC,
    Counter,
    Gauge,
    Histogram,
    MetricEmitError,
    MetricRegistry,
    MetricSpec,
    MetricSpecError,
    get_metric_registry,
    label_key,
)

# --------------------------------------------------------------------------- #
# Metric types — Counter
# --------------------------------------------------------------------------- #


def test_counter_inc_accumulates_and_defaults_to_one() -> None:
    counter = Counter(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    counter.inc(component="neuron")
    counter.inc(component="neuron")
    counter.inc(2.0, component="neuron")
    assert counter.value(component="neuron") == 4.0


def test_counter_series_are_isolated_per_label_set() -> None:
    counter = Counter(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    counter.inc(component="neuron")
    counter.inc(3.0, component="drive")
    assert counter.value(component="neuron") == 1.0
    assert counter.value(component="drive") == 3.0
    assert counter.value(component="unseen") == 0.0


def test_counter_with_no_labels_uses_the_empty_series() -> None:
    counter = Counter(MetricSpec(name="c_total", kind="counter"))
    counter.inc()
    counter.inc()
    assert counter.value() == 2.0


def test_counter_rejects_negative_increment_to_stay_monotonic() -> None:
    counter = Counter(MetricSpec(name="c_total", kind="counter"))
    with pytest.raises(MetricEmitError):
        counter.inc(-1.0)


# --------------------------------------------------------------------------- #
# Metric types — Gauge
# --------------------------------------------------------------------------- #


def test_gauge_set_replaces_the_current_value() -> None:
    gauge = Gauge(MetricSpec(name="g", kind="gauge", label_keys=("layer",)))
    gauge.set(1.0, layer="AUTONOMIC")
    gauge.set(0.0, layer="AUTONOMIC")
    gauge.set(5.0, layer="COGNITION")
    assert gauge.value(layer="AUTONOMIC") == 0.0
    assert gauge.value(layer="COGNITION") == 5.0


# --------------------------------------------------------------------------- #
# Metric types — Histogram
# --------------------------------------------------------------------------- #


def test_histogram_default_buckets_match_the_spec() -> None:
    assert DEFAULT_BUCKETS == (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    )


def test_histogram_observe_tracks_cumulative_buckets_count_and_sum() -> None:
    hist = Histogram(MetricSpec(name="h_seconds", kind="histogram", label_keys=("component",)))
    for value in (0.03, 0.3, 7.0, 100.0):
        hist.observe(value, component="tick")

    snap = hist.snapshot(component="tick")
    assert snap.count == 4
    assert snap.sum == pytest.approx(107.33)

    # Cumulative "≤ le" counts (Prometheus semantics).
    cumulative = dict(snap.buckets)
    assert cumulative[0.025] == 0
    assert cumulative[0.05] == 1  # 0.03
    assert cumulative[0.5] == 2  # 0.03, 0.3
    assert cumulative[10.0] == 3  # + 7.0
    assert cumulative[30.0] == 3  # 100.0 overflows the finite buckets
    # The +Inf bucket equals the total count.
    assert snap.count == 4


def test_histogram_snapshot_of_unseen_label_set_is_empty() -> None:
    hist = Histogram(MetricSpec(name="h_seconds", kind="histogram"))
    snap = hist.snapshot()
    assert snap.count == 0
    assert snap.sum == 0.0
    assert all(count == 0 for _, count in snap.buckets)


# --------------------------------------------------------------------------- #
# MetricSpec — fail-fast declaration validation
# --------------------------------------------------------------------------- #


def test_spec_rejects_label_key_outside_the_closed_set() -> None:
    with pytest.raises(MetricSpecError):
        MetricSpec(name="bad", kind="counter", label_keys=("trace_id",))


def test_spec_accepts_only_the_closed_label_key_set() -> None:
    spec = MetricSpec(name="ok", kind="counter", label_keys=tuple(ALLOWED_LABEL_KEYS))
    assert frozenset(spec.label_keys) == ALLOWED_LABEL_KEYS


def test_spec_rejects_unknown_kind() -> None:
    with pytest.raises(MetricSpecError):
        MetricSpec(name="bad", kind="summary")  # type: ignore[arg-type]


def test_spec_rejects_empty_name() -> None:
    with pytest.raises(MetricSpecError):
        MetricSpec(name="", kind="counter")


def test_spec_normalises_label_keys_to_a_sorted_tuple() -> None:
    a = MetricSpec(name="x", kind="counter", label_keys=("layer", "component"))
    b = MetricSpec(name="x", kind="counter", label_keys=("component", "layer"))
    assert a.label_keys == ("component", "layer")
    assert a == b  # order-independent equality → idempotent re-registration


def test_spec_rejects_non_increasing_histogram_buckets() -> None:
    with pytest.raises(MetricSpecError):
        MetricSpec(name="h", kind="histogram", buckets=(1.0, 0.5))


# --------------------------------------------------------------------------- #
# label_key helper — deterministic, order-independent
# --------------------------------------------------------------------------- #


def test_label_key_is_order_independent() -> None:
    assert label_key({"component": "n", "layer": "A"}) == label_key(
        {"layer": "A", "component": "n"}
    )


def test_label_key_distinguishes_different_label_sets() -> None:
    assert label_key({"component": "n"}) != label_key({"component": "m"})
    assert label_key({}) != label_key({"component": "n"})


def test_label_key_of_empty_labels_is_stable() -> None:
    assert label_key({}) == label_key({})


# --------------------------------------------------------------------------- #
# get_metric_registry — singleton per base_dir, idempotent
# --------------------------------------------------------------------------- #


def test_get_metric_registry_returns_same_instance_for_a_base_dir(tmp_path: Path) -> None:
    first = get_metric_registry(tmp_path)
    second = get_metric_registry(tmp_path)
    assert first is second


def test_get_metric_registry_is_distinct_per_base_dir(tmp_path: Path) -> None:
    a = get_metric_registry(tmp_path / "a")
    b = get_metric_registry(tmp_path / "b")
    assert a is not b


# --------------------------------------------------------------------------- #
# MetricRegistry.register — fail-fast, idempotent for equal specs
# --------------------------------------------------------------------------- #


def test_register_returns_the_metric_object() -> None:
    reg = MetricRegistry()
    metric = reg.register(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    assert isinstance(metric, Counter)
    assert reg.get("c_total") is metric


def test_register_is_idempotent_for_an_identical_spec() -> None:
    reg = MetricRegistry()
    spec = MetricSpec(name="c_total", kind="counter", label_keys=("component",))
    first = reg.register(spec)
    second = reg.register(spec)
    assert first is second


def test_register_conflicting_spec_for_same_name_fails_fast() -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="c_total", kind="counter"))
    with pytest.raises(MetricSpecError):
        reg.register(MetricSpec(name="c_total", kind="gauge"))


def test_registry_convenience_registrars() -> None:
    reg = MetricRegistry()
    assert isinstance(reg.counter("c_total", label_keys=("component",)), Counter)
    assert isinstance(reg.gauge("g", label_keys=("layer",)), Gauge)
    assert isinstance(reg.histogram("h_seconds", label_keys=("component",)), Histogram)


# --------------------------------------------------------------------------- #
# Runtime emission — fail-open (no-op + emit-error counter, never raises)
# --------------------------------------------------------------------------- #


def _emit_errors(reg: MetricRegistry, reason: str) -> float:
    errors = reg.get(EMIT_ERRORS_METRIC)
    assert isinstance(errors, Counter)
    return errors.value(reason=reason)


def test_emit_errors_counter_is_self_registered() -> None:
    reg = MetricRegistry()
    assert isinstance(reg.get(EMIT_ERRORS_METRIC), Counter)


def test_runtime_inc_of_unknown_metric_is_noop_and_counts_error() -> None:
    reg = MetricRegistry()
    reg.inc("never_declared", component="x")  # must not raise
    assert _emit_errors(reg, "unknown_metric") == 1.0


def test_runtime_emit_with_undeclared_label_is_noop_and_counts_error() -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    reg.inc("c_total", nonsense="x")  # undeclared + outside closed set
    counter = reg.get("c_total")
    assert isinstance(counter, Counter)
    assert counter.value(nonsense="x") == 0.0  # value not recorded
    assert _emit_errors(reg, "unknown_label") == 1.0


def test_runtime_emit_with_valid_but_undeclared_closed_label_is_rejected() -> None:
    # ``layer`` is in the closed set, but this metric only declared ``component``.
    reg = MetricRegistry()
    reg.register(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    reg.inc("c_total", layer="AUTONOMIC")
    assert _emit_errors(reg, "unknown_label") == 1.0


def test_runtime_wrong_kind_is_noop_and_counts_error() -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="g", kind="gauge", label_keys=("layer",)))
    reg.inc("g", layer="AUTONOMIC")  # inc on a gauge
    assert _emit_errors(reg, "wrong_kind") == 1.0


def test_runtime_emission_records_when_declared() -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="c_total", kind="counter", label_keys=("component",)))
    reg.register(MetricSpec(name="g", kind="gauge", label_keys=("layer",)))
    reg.register(MetricSpec(name="h_seconds", kind="histogram", label_keys=("component",)))

    reg.inc("c_total", component="neuron")
    reg.set("g", 0.75, layer="AUTONOMIC")
    reg.observe("h_seconds", 0.2, component="tick")

    counter = reg.get("c_total")
    gauge = reg.get("g")
    hist = reg.get("h_seconds")
    assert isinstance(counter, Counter) and counter.value(component="neuron") == 1.0
    assert isinstance(gauge, Gauge) and gauge.value(layer="AUTONOMIC") == 0.75
    assert isinstance(hist, Histogram) and hist.snapshot(component="tick").count == 1


def test_runtime_emission_never_raises_on_bad_value() -> None:
    reg = MetricRegistry()
    reg.register(MetricSpec(name="g", kind="gauge", label_keys=("layer",)))
    reg.set("g", float("nan"), layer="AUTONOMIC")  # must not raise
    assert _emit_errors(reg, "bad_value") == 1.0


def test_counting_an_emit_error_does_not_recurse() -> None:
    # Repeated bad emits just accumulate on the self-registered counter.
    reg = MetricRegistry()
    for _ in range(3):
        reg.inc("missing")
    assert _emit_errors(reg, "unknown_metric") == 3.0

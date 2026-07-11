"""Tests for ``/lifemodel stats`` — the read-only telemetry view (design §4.5).

Covers the pure seams (layer rollup, histogram-bucket p95, window rate) and the
fail-soft command edges (missing/corrupt ``metrics.sqlite``, an un-instrumented
registry → ``n/a`` rather than a crash), mirroring ``tests/test_trace_view.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.core.metrics import MetricRegistry, get_metric_registry
from lifemodel.core.timeutil import to_iso
from lifemodel.state.metrics_store import MetricSample, MetricsSampler, metrics_db_path
from lifemodel.stats_view import (
    approx_quantile,
    layer_for_component,
    render_now,
    render_window,
    stats_for_dir,
)

#: Time is ISO-8601 UTC TEXT (spec §4). A fixed aware-UTC anchor lets a test name an
#: instant by "seconds since anchor" and keep the old 60s window spacing intact.
_ANCHOR = datetime(2026, 7, 11, 0, 0, 0, tzinfo=UTC)


def _dt(offset_seconds: float) -> datetime:
    return _ANCHOR + timedelta(seconds=offset_seconds)


def _ts(offset_seconds: float) -> str:
    return to_iso(_dt(offset_seconds))


def _sample(ts: float, name: str, value: float, run_id: str = "R", **labels: str) -> MetricSample:
    label_key = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
    return MetricSample(
        ts=_ts(ts), run_id=run_id, name=name, label_key=label_key, value=value, labels=dict(labels)
    )


# --------------------------------------------------------------------------- #
# component -> layer rollup (design §4.5; static map from core.component)
# --------------------------------------------------------------------------- #


def test_layer_for_component_maps_known_types() -> None:
    assert layer_for_component("neuron") == "autonomic"
    assert layer_for_component("drive") == "autonomic"
    assert layer_for_component("aggregation") == "aggregation"
    assert layer_for_component("launcher") == "cognition"
    assert layer_for_component("egress") == "infra"
    assert layer_for_component("writer") == "infra"


def test_layer_for_component_unknown_is_other_not_a_crash() -> None:
    assert layer_for_component("totally-unknown") == "other"
    assert layer_for_component("") == "other"


# --------------------------------------------------------------------------- #
# approx p95 from cumulative histogram buckets (design §4.4/§4.5)
# --------------------------------------------------------------------------- #


def test_approx_quantile_interpolates_within_a_bucket() -> None:
    # Cumulative (le, count): total 6 observations; 0.95*6 = 5.7 lands in (1.0, 2.5].
    buckets = [(0.5, 1.0), (1.0, 5.0), (2.5, 6.0), (5.0, 6.0)]
    p95 = approx_quantile(buckets, count=6.0, q=0.95)
    assert p95 is not None
    assert 1.0 < p95 <= 2.5


def test_approx_quantile_empty_histogram_is_none() -> None:
    assert approx_quantile([(0.5, 0.0), (1.0, 0.0)], count=0.0, q=0.95) is None


def test_approx_quantile_beyond_last_finite_bucket_returns_last_bound() -> None:
    # Everything sits in the implicit +Inf bucket (count 4 > last cumulative 1).
    buckets = [(0.5, 1.0), (1.0, 1.0)]
    p95 = approx_quantile(buckets, count=4.0, q=0.95)
    assert p95 == 1.0


# --------------------------------------------------------------------------- #
# NOW section — live from the registry (fail-soft on missing metrics → n/a)
# --------------------------------------------------------------------------- #


def test_now_on_bare_registry_is_all_na_not_a_crash() -> None:
    # The CoreLoop instrumentation (bead 7.4) may be unmerged: NO operational
    # metric is registered. NOW must still render, with n/a for every field.
    text = "\n".join(render_now(MetricRegistry()))
    assert "n/a" in text
    assert "tick_lag" in text
    assert "tick_duration" in text
    assert "accepts_signals" in text


def test_now_renders_seeded_registry() -> None:
    reg = MetricRegistry()
    reg.gauge("lifemodel_tick_lag_seconds").set(12.5)
    hist = reg.histogram("lifemodel_tick_duration_seconds")
    hist.observe(0.4)
    hist.observe(0.6)  # mean 0.5 over count 2
    reg.gauge("lifemodel_trace_writer_dropped_records").set(3.0)
    reg.gauge("lifemodel_trace_writer_write_errors").set(1.0)
    accepts = reg.gauge("lifemodel_layer_accepts_signals", label_keys=("layer",))
    accepts.set(1.0, layer="autonomic")
    accepts.set(0.0, layer="cognition")
    runs = reg.counter("lifemodel_component_runs_total", label_keys=("component", "layer"))
    runs.inc(5.0, component="contact_sensor", layer="autonomic")
    runs.inc(2.0, component="aggregation", layer="aggregation")  # total 7
    reg.counter("lifemodel_suppressions_total", label_keys=("reason",)).inc(
        3.0, reason="quiet_hours"
    )
    reg.counter("lifemodel_signals_intake_total", label_keys=("outcome",)).inc(9.0, outcome="kept")

    text = "\n".join(render_now(reg))

    assert "12.5" in text  # tick_lag
    assert "0.50" in text  # tick_duration mean
    assert "autonomic" in text and "yes" in text  # accepts_signals rollup
    assert "cognition" in text and "no" in text
    assert "3" in text  # writer dropped
    assert "7" in text  # component runs total
    assert "n/a" not in text  # every field had data


# --------------------------------------------------------------------------- #
# WINDOW section — history from metrics.sqlite (rate within one run_id, §4.4)
# --------------------------------------------------------------------------- #


def test_window_empty_is_friendly_not_a_crash() -> None:
    text = "\n".join(render_window([], last_n=30))
    assert "WINDOW" in text
    assert "no" in text.lower()  # a friendly "no samples" note


def test_window_rate_of_suppressions_by_reason() -> None:
    # A counter climbing 5 -> 15 over 60s = 10 / 60s = 10/min, within one run.
    samples = [
        _sample(1000, "lifemodel_suppressions_total", 5.0, reason="quiet_hours"),
        _sample(1060, "lifemodel_suppressions_total", 15.0, reason="quiet_hours"),
    ]
    text = "\n".join(render_window(samples, last_n=30))
    assert "quiet_hours" in text
    assert "10" in text  # 10 suppressions/min over the window


def test_window_rate_does_not_glue_across_run_ids() -> None:
    # An OLD run left a huge counter value; the NEW run starts fresh. The window
    # must compute the NEW run only — never (12 - 999) nor gluing the two.
    samples = [
        _sample(10, "lifemodel_suppressions_total", 999.0, run_id="OLD", reason="quiet_hours"),
        _sample(1000, "lifemodel_suppressions_total", 2.0, run_id="NEW", reason="quiet_hours"),
        _sample(1060, "lifemodel_suppressions_total", 12.0, run_id="NEW", reason="quiet_hours"),
    ]
    text = "\n".join(render_window(samples, last_n=30))
    assert "quiet_hours" in text
    assert "10" in text  # (12 - 2) / 60s = 10/min, NEW run only
    assert "999" not in text
    assert "-987" not in text  # never a negative glued delta


def test_window_shedding_sums_shed_control_and_shed_sensor() -> None:
    # The emitter folds the lane into the value (``shed_control``/``shed_sensor``);
    # the WINDOW must recognise BOTH as shedding (by the ``shed`` prefix) and sum
    # them — the old exact ``shed`` match never fired, so shedding was always n/a.
    samples = [
        _sample(1000, "lifemodel_signals_intake_total", 2.0, outcome="shed_control"),
        _sample(1060, "lifemodel_signals_intake_total", 8.0, outcome="shed_control"),
        _sample(1000, "lifemodel_signals_intake_total", 1.0, outcome="shed_sensor"),
        _sample(1060, "lifemodel_signals_intake_total", 5.0, outcome="shed_sensor"),
    ]
    text = "\n".join(render_window(samples, last_n=30))
    shed_line = next(line for line in text.splitlines() if "shedding" in line)
    assert "/min" in shed_line  # a real rate, NOT n/a
    assert "10" in shed_line  # (8-2) + (5-1) = 10 shed / 60s = 10/min


def test_window_p95_from_histogram_buckets() -> None:
    # Baseline (all zero) at t0; at t1 the cumulative buckets describe 6 obs whose
    # p95 (0.95*6 = 5.7) lands in (1.0, 2.5]. Windowed = t1 - t0.
    zero = [
        _sample(1000, "lifemodel_tick_duration_seconds_bucket", 0.0, le="0.5"),
        _sample(1000, "lifemodel_tick_duration_seconds_bucket", 0.0, le="1.0"),
        _sample(1000, "lifemodel_tick_duration_seconds_bucket", 0.0, le="2.5"),
        _sample(1000, "lifemodel_tick_duration_seconds_count", 0.0),
    ]
    grown = [
        _sample(1060, "lifemodel_tick_duration_seconds_bucket", 1.0, le="0.5"),
        _sample(1060, "lifemodel_tick_duration_seconds_bucket", 5.0, le="1.0"),
        _sample(1060, "lifemodel_tick_duration_seconds_bucket", 6.0, le="2.5"),
        _sample(1060, "lifemodel_tick_duration_seconds_count", 6.0),
    ]
    text = "\n".join(render_window(zero + grown, last_n=30))
    assert "p95" in text
    # p95 interpolates to ~2.05s (between the 1.0 and 2.5 bounds).
    assert "2.0" in text or "2.1" in text


# --------------------------------------------------------------------------- #
# stats_for_dir — the command entrypoint (read-only, fail-soft on every source)
# --------------------------------------------------------------------------- #


def test_stats_missing_db_still_renders_now_not_a_crash(tmp_path: Path) -> None:
    out = stats_for_dir(tmp_path, "")
    assert "NOW" in out
    assert "WINDOW" in out
    assert "n/a" in out  # nothing instrumented → NOW is all n/a
    assert "no metrics history" in out.lower()  # WINDOW degrades friendly


def test_stats_corrupt_db_is_friendly_not_a_crash(tmp_path: Path) -> None:
    metrics_db_path(tmp_path).write_bytes(b"this is not a sqlite database")
    out = stats_for_dir(tmp_path, "last 5")
    assert "NOW" in out  # NOW still renders from the registry
    assert "unreadable" in out.lower()  # WINDOW degrades, no exception


def test_stats_bad_args_returns_usage(tmp_path: Path) -> None:
    assert "usage" in stats_for_dir(tmp_path, "last notanumber").lower()


def test_stats_end_to_end_registry_now_plus_sqlite_window(tmp_path: Path) -> None:
    reg = get_metric_registry(tmp_path)  # the SAME singleton stats_for_dir reads
    reg.gauge("lifemodel_tick_lag_seconds").set(4.0)
    sup = reg.counter("lifemodel_suppressions_total", label_keys=("reason",))

    sampler = MetricsSampler(reg, metrics_db_path(tmp_path), run_id="R", heartbeat_every=1)
    sup.inc(2.0, reason="quiet_hours")
    sampler.sample_once(now=_dt(1000))
    sup.inc(10.0, reason="quiet_hours")  # total 12; +10 over 60s = 10/min
    sampler.sample_once(now=_dt(1060))
    sampler.close()

    out = stats_for_dir(tmp_path, "last 30")

    assert "NOW" in out and "WINDOW" in out
    assert "4s" in out  # live tick_lag from the registry
    assert "quiet_hours" in out
    assert "10" in out  # 10 suppressions/min over the window

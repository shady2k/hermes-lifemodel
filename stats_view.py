"""``/lifemodel stats`` — the read-only telemetry view (design §4.5).

The chat-facing half of telemetry-core: a two-section operational readout of how
loaded the being is, rendered from the two sources telemetry-core keeps.

* **NOW** — live from the process-local
  :class:`~lifemodel.core.metrics.MetricRegistry` (via
  :func:`~lifemodel.core.metrics.get_metric_registry`): current ``tick_lag``, the
  latest ``tick_duration``, ``trace_writer`` drops/errors, per-layer
  ``accepts_signals``, and the current counters. The CoreLoop instrumentation
  that populates these is a *sibling* bead (7.4) that may not be merged yet, so
  every metric read is defensive: an absent metric renders ``n/a``, never a crash.
* **WINDOW** — history from ``metrics.sqlite`` (the bead 7.6 sampler), read back
  through :func:`~lifemodel.state.metrics_store.read_samples`. Over the last ``N``
  samples it derives event / suppression rates (per ``reason``), shedding, and an
  approximate ``p95`` from the histogram buckets — all *within one* ``run_id`` so
  a restart is never glued across (design §4.4).

Read-only and **fail-soft** on every source (mirrors
:mod:`~lifemodel.trace_view`): a missing / locked / corrupt ``metrics.sqlite``,
or an un-instrumented registry, degrades to a friendly message — losing telemetry
never changes the being's behaviour, so a ``stats`` call must never raise.

Component→layer rollup uses the static
:data:`~lifemodel.core.component.LAYER_BY_TYPE` map (design §4.5); an unknown
component folds to ``"other"`` rather than failing. All stdlib.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .core.component import LAYER_BY_TYPE
from .core.metrics import Counter, Gauge, Histogram, MetricRegistry, get_metric_registry

if TYPE_CHECKING:
    # Imported lazily at call time inside ``_safe_window`` (NOT at module load):
    # ``state.metrics_store`` is only needed when the command actually reads the
    # WINDOW history, and deferring it keeps plugin import free of that subtree —
    # so a missing/unloadable metrics store degrades this ONE section fail-soft
    # rather than breaking the whole plugin's import (the Hermes namespaced load).
    from .state.metrics_store import MetricSample

#: The layer bucket an unknown / unmapped component folds into (design §4.5).
OTHER_LAYER = "other"

# --------------------------------------------------------------------------- #
# Metric names — taken verbatim from the CoreLoop instrumentation (design §4.2).
# Bead 7.4 registers these; they may be ABSENT here (unmerged), so every read is
# defensive and an absent metric renders ``n/a``.
# --------------------------------------------------------------------------- #

TICK_DURATION = "lifemodel_tick_duration_seconds"
TICK_LAG = "lifemodel_tick_lag_seconds"
COMPONENT_RUNS = "lifemodel_component_runs_total"
SIGNALS_INTAKE = "lifemodel_signals_intake_total"
LAYER_ACCEPTS = "lifemodel_layer_accepts_signals"
WRITER_DROPPED = "lifemodel_trace_writer_dropped_records"
WRITER_ERRORS = "lifemodel_trace_writer_write_errors"
SUPPRESSIONS = "lifemodel_suppressions_total"

_NA = "n/a"

#: Default WINDOW width (samples) when ``stats`` is called with no ``last N``.
DEFAULT_LAST_N = 30
#: Hard cap so ``stats last 99999`` can never scan an unbounded history slice.
MAX_LAST_N = 500

#: The quantile the WINDOW section approximates from the histogram buckets (§4.5).
WINDOW_QUANTILE = 0.95
#: The intake ``result`` value that means a signal was dropped under backpressure.
_SHED = "shed"


def layer_for_component(component: str) -> str:
    """Fold a *component* identifier up to its architectural layer (design §4.5).

    Uses the static :data:`~lifemodel.core.component.LAYER_BY_TYPE` rollup; an
    unknown component maps to :data:`OTHER_LAYER` (``"other"``) rather than
    raising, so a new/mislabelled component can never crash the readout.
    """
    layer = LAYER_BY_TYPE.get(component)
    return str(layer) if layer is not None else OTHER_LAYER


def approx_quantile(buckets: Sequence[tuple[float, float]], count: float, q: float) -> float | None:
    """Approximate the *q*-quantile from cumulative histogram buckets (design §4.5).

    *buckets* is ``(le_bound, cumulative_count)`` for each finite bound, ordered
    ascending (Prometheus histogram shape); *count* is the total including the
    implicit ``+Inf`` bucket. Returns ``None`` for an empty histogram. Within the
    bucket that first reaches the target rank the estimate is linearly
    interpolated between bounds; a rank that only the ``+Inf`` bucket satisfies
    returns the last finite bound (a lower estimate — we never invent ``+Inf``).
    """
    if count <= 0:
        return None
    rank = q * count
    prev_bound = 0.0
    prev_cum = 0.0
    for bound, cumulative in buckets:
        if cumulative >= rank:
            if cumulative == prev_cum:
                return bound
            frac = (rank - prev_cum) / (cumulative - prev_cum)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound, prev_cum = bound, cumulative
    return prev_bound


# --------------------------------------------------------------------------- #
# NOW section — live reads off the registry (every read fail-soft → None → n/a)
# --------------------------------------------------------------------------- #


def _fmt_num(value: float) -> str:
    """Whole numbers as ints, else 2 d.p. — matches the debug-dump house style."""
    return str(int(value)) if value == int(value) else f"{value:.2f}"


def _read_scalar(registry: MetricRegistry, name: str) -> float | None:
    """Sum every series of counter/gauge *name*, or ``None`` if unregistered/empty.

    ``None`` (→ ``n/a``) covers both "metric not registered yet" (bead 7.4
    unmerged) and "registered but never emitted", so the readout never asserts a
    value telemetry does not actually have.
    """
    metric = registry.get(name)
    if not isinstance(metric, (Counter, Gauge)):
        return None
    items = metric.items()
    if not items:
        return None
    return sum(value for _lk, _labels, value in items)


def _read_histogram_agg(registry: MetricRegistry, name: str) -> tuple[float, int] | None:
    """Aggregate ``(sum, count)`` over every series of histogram *name*, or ``None``."""
    metric = registry.get(name)
    if not isinstance(metric, Histogram):
        return None
    total_sum = 0.0
    total_count = 0
    for _lk, _labels, snap in metric.items():
        total_sum += snap.sum
        total_count += snap.count
    if total_count == 0:
        return None
    return total_sum, total_count


def _layer_accepts(registry: MetricRegistry) -> list[tuple[str, bool]]:
    """``(layer, accepts?)`` for each ``lifemodel_layer_accepts_signals`` series."""
    metric = registry.get(LAYER_ACCEPTS)
    if not isinstance(metric, Gauge):
        return []
    rows = [
        (labels.get("layer", OTHER_LAYER), value >= 0.5) for _lk, labels, value in metric.items()
    ]
    return sorted(rows)


def render_now(registry: MetricRegistry) -> list[str]:
    """Render the NOW section from live registry state (design §4.5), fail-soft.

    Every field degrades to ``n/a`` when its metric is absent (the CoreLoop
    instrumentation of bead 7.4 may be unmerged) — the section always renders.
    """
    lines = ["**NOW** (live)"]

    lag = _read_scalar(registry, TICK_LAG)
    lines.append(f"**tick_lag:** {_NA if lag is None else _fmt_num(lag) + 's'}")

    dur = _read_histogram_agg(registry, TICK_DURATION)
    if dur is None:
        lines.append(f"**tick_duration:** {_NA}")
    else:
        total_sum, count = dur
        lines.append(f"**tick_duration:** ~{_fmt_num(total_sum / count)}s avg (n={count})")

    dropped = _read_scalar(registry, WRITER_DROPPED)
    errors = _read_scalar(registry, WRITER_ERRORS)
    lines.append(
        f"**trace_writer:** dropped {_NA if dropped is None else _fmt_num(dropped)} · "
        f"errors {_NA if errors is None else _fmt_num(errors)}"
    )

    accepts = _layer_accepts(registry)
    if not accepts:
        lines.append(f"**accepts_signals:** {_NA}")
    else:
        rendered = ", ".join(f"{layer}: {'yes' if ok else 'no'}" for layer, ok in accepts)
        lines.append(f"**accepts_signals:** {rendered}")

    runs = _read_scalar(registry, COMPONENT_RUNS)
    intake = _read_scalar(registry, SIGNALS_INTAKE)
    suppressions = _read_scalar(registry, SUPPRESSIONS)
    lines.append(f"**component_runs:** {_NA if runs is None else _fmt_num(runs)}")
    lines.append(f"**signals_intake:** {_NA if intake is None else _fmt_num(intake)}")
    lines.append(f"**suppressions:** {_NA if suppressions is None else _fmt_num(suppressions)}")

    return lines


# --------------------------------------------------------------------------- #
# WINDOW section — history from metrics.sqlite (rate WITHIN one run_id, §4.4)
# --------------------------------------------------------------------------- #


class _Window:
    """Indexed read over the latest ``run_id``'s samples for a ``[t0, t1]`` window.

    Because the sampler skips unchanged series (design §4.4), the read-side takes
    the LAST point ``<= t`` for each border rather than an exact match. Rates are
    derived only within this single run — a restart mints a new ``run_id`` and is
    never glued across (a negative delta reads as a reset → ``None``).
    """

    def __init__(self, samples: Sequence[MetricSample], last_n: int) -> None:
        run = max(samples, key=lambda s: s.ts).run_id
        run_samples = [s for s in samples if s.run_id == run]
        self.run_id = run

        self._ts: dict[tuple[str, str], list[int]] = {}
        self._val: dict[tuple[str, str], list[float]] = {}
        self.labels: dict[tuple[str, str], dict[str, str]] = {}
        by_key: dict[tuple[str, str], list[tuple[int, float]]] = {}
        ts_seen: set[int] = set()
        for s in run_samples:
            key = (s.name, s.label_key)
            by_key.setdefault(key, []).append((s.ts, s.value))
            self.labels.setdefault(key, s.labels)
            ts_seen.add(s.ts)
        for key, points in by_key.items():
            points.sort()
            self._ts[key] = [ts for ts, _ in points]
            self._val[key] = [val for _, val in points]

        ts_list = sorted(ts_seen)
        self.t1 = ts_list[-1]
        self.t0 = ts_list[max(0, len(ts_list) - 1 - last_n)]
        self.dt = self.t1 - self.t0

    def keys_named(self, name: str) -> list[tuple[str, str]]:
        return [key for key in self._ts if key[0] == name]

    def _value_at(self, key: tuple[str, str], border: int) -> float | None:
        stamps = self._ts.get(key)
        if not stamps:
            return None
        idx = bisect.bisect_right(stamps, border) - 1
        return self._val[key][idx] if idx >= 0 else None

    def delta(self, key: tuple[str, str]) -> float | None:
        """Windowed increase of a counter series, or ``None`` on reset/absence.

        The baseline is the value at ``t0`` (or ``0`` for a zero-width window, so
        a single snapshot still yields the full since-boot cumulative for p95). A
        negative delta means the counter reset inside the window → ``None``.
        """
        current = self._value_at(key, self.t1)
        if current is None:
            return None
        base = self._value_at(key, self.t0) if self.dt > 0 else 0.0
        if base is None:
            base = 0.0
        change = current - base
        return change if change >= 0 else None

    def per_min(self, delta: float | None) -> float | None:
        """A windowed delta as a per-minute rate, or ``None`` for a zero window."""
        if delta is None or self.dt <= 0:
            return None
        return delta / (self.dt / 60.0)


def _rate_line(label: str, rate: float | None) -> str:
    return f"**{label}:** {_NA if rate is None else _fmt_num(rate) + '/min'}"


def render_window(samples: Sequence[MetricSample], *, last_n: int) -> list[str]:
    """Render the WINDOW section from ``metrics.sqlite`` history (design §4.5).

    Over the last *last_n* samples of the latest ``run_id`` it reports throughput,
    suppression and shedding rates (per minute), and an approximate tick ``p95``
    from the histogram buckets. Pure over *samples*; fail-soft is the caller's job
    (an empty list renders a friendly note, never a crash).
    """
    last_n = max(1, last_n)
    if not samples:
        return ["**WINDOW**", "(no metrics history recorded yet)"]

    win = _Window(samples, last_n)
    lines = [f"**WINDOW** (last {last_n} samples · run {win.run_id[:8]})"]

    # Throughput — total component runs across the window.
    run_deltas = [
        d for d in (win.delta(k) for k in win.keys_named(COMPONENT_RUNS)) if d is not None
    ]
    throughput = win.per_min(sum(run_deltas)) if run_deltas else None
    lines.append(_rate_line("throughput", throughput))

    # Suppressions, per reason.
    sup_rows: list[tuple[str, float]] = []
    for key in win.keys_named(SUPPRESSIONS):
        rate = win.per_min(win.delta(key))
        if rate is not None:
            sup_rows.append((win.labels[key].get("reason", "?"), rate))
    if sup_rows:
        rendered = ", ".join(f"{reason} {_fmt_num(rate)}" for reason, rate in sorted(sup_rows))
        lines.append(f"**suppressions/min:** {rendered}")
    else:
        lines.append(f"**suppressions/min:** {_NA}")

    # Shedding — intake series whose result is 'shed' (backpressure drops).
    shed_deltas: list[float] = []
    for key in win.keys_named(SIGNALS_INTAKE):
        if _SHED not in win.labels[key].values():
            continue
        delta = win.delta(key)
        if delta is not None:
            shed_deltas.append(delta)
    shedding = win.per_min(sum(shed_deltas)) if shed_deltas else None
    lines.append(_rate_line("shedding", shedding))

    # Approx tick p95 from the windowed histogram buckets.
    p95 = _window_p95(win)
    lines.append(f"**tick p95:** {_NA if p95 is None else _fmt_num(p95) + 's'}")

    return lines


def _window_p95(win: _Window) -> float | None:
    """Approximate tick-duration p95 over the window from the histogram buckets."""
    count_delta = win.delta((TICK_DURATION + "_count", ""))
    if not count_delta or count_delta <= 0:
        return None
    buckets: list[tuple[float, float]] = []
    for key in win.keys_named(TICK_DURATION + "_bucket"):
        le = win.labels[key].get("le")
        if le is None:
            continue
        try:
            bound = float(le)
        except ValueError:
            continue
        cumulative = win.delta(key)
        if cumulative is not None:
            buckets.append((bound, cumulative))
    buckets.sort()
    return approx_quantile(buckets, count_delta, WINDOW_QUANTILE)


# --------------------------------------------------------------------------- #
# Command entrypoint (read-only, fail-soft on every source — mirrors trace_view)
# --------------------------------------------------------------------------- #

_USAGE = (
    "usage: /lifemodel stats [last N]\n"
    "  (NOW: live tick/writer/counters from the registry · "
    "WINDOW: rates + approx p95 from metrics.sqlite over the last N samples)\n"
)


def _parse_args(raw_args: str) -> tuple[str, int]:
    """Return ``("window", N)`` or ``("usage", 0)`` — mirrors ``trace``'s parser."""
    parts = raw_args.strip().split()
    if not parts:
        return ("window", DEFAULT_LAST_N)
    if parts[0].lower() != "last":
        return ("usage", 0)
    if len(parts) == 1:
        return ("window", DEFAULT_LAST_N)
    try:
        n = int(parts[1])
    except ValueError:
        return ("usage", 0)
    return ("window", max(1, min(n, MAX_LAST_N)))


def _safe_now(base_dir: Path, registry: MetricRegistry | None) -> list[str]:
    """Render NOW, degrading to a friendly note if the registry is unreachable."""
    try:
        reg = registry if registry is not None else get_metric_registry(base_dir)
        return render_now(reg)
    except Exception as exc:  # noqa: BLE001 - a read-only view must never crash (§7)
        return ["**NOW** (live)", f"(registry unavailable: {exc})"]


def _safe_window(base_dir: Path, last_n: int) -> list[str]:
    """Render WINDOW, degrading to a friendly note on a missing/locked/corrupt DB.

    The ``state.metrics_store`` import is deferred to here (not module top): the
    WINDOW source is only touched when the command runs, so any trouble reaching
    it — a missing store, an unreadable file, even an import hiccup — degrades
    this section alone rather than the whole read-only view (design §7).
    """
    try:
        from .state.metrics_store import metrics_db_path, read_samples
    except Exception as exc:  # noqa: BLE001 - never let the view crash on this source
        return ["**WINDOW**", f"(metrics history unavailable: {exc})"]

    db_path = metrics_db_path(base_dir)
    if not db_path.exists():
        return ["**WINDOW**", "(no metrics history yet — metrics.sqlite not created)"]
    try:
        samples = read_samples(db_path)
    except Exception as exc:  # noqa: BLE001 - missing/locked/corrupt store → friendly
        return ["**WINDOW**", f"(metrics history unreadable: {exc})"]
    return render_window(samples, last_n=last_n)


def stats_for_dir(base_dir: Path, raw_args: str, *, registry: MetricRegistry | None = None) -> str:
    """Answer ``/lifemodel stats [last N]`` — read-only, fail-soft (design §4.5).

    Two independently fail-soft sections: **NOW** from the live
    :class:`~lifemodel.core.metrics.MetricRegistry` (the singleton for *base_dir*,
    or an injected *registry* in tests) and **WINDOW** from ``metrics.sqlite``. A
    missing / locked / corrupt store, or an un-instrumented registry, degrades
    that one section to a friendly line — the command never raises.
    """
    kind, last_n = _parse_args(raw_args)
    if kind == "usage":
        return _USAGE

    header = ["lifemodel stats  (read-only)", "=" * 30, ""]
    body = _safe_now(base_dir, registry) + [""] + _safe_window(base_dir, last_n)
    return "\n".join(header + body) + "\n"

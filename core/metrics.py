"""``MetricRegistry`` + the stdlib metric types — the source of CURRENT metric state.

The foundation of telemetry-core (design §4.1): a process-local, thread-safe
registry of the being's live operational metrics — tick/component latency, event
rates, drive levels — that later phases sample into ``metrics.sqlite`` (bead 7.6)
and render in ``/lifemodel stats`` (bead 7.7). It is NOT durable truth (that stays
in ``observability.sqlite``); it holds the *current* value of each metric.

Two modes, deliberately split in the API (design §4.3/§7):

* **Declaration is fail-FAST.** Building a :class:`MetricSpec` or registering a
  metric validates eagerly and raises :class:`MetricSpecError` on anything
  malformed — an unknown kind, an empty name, non-increasing histogram buckets,
  or a label key outside the closed low-cardinality set. This is a build/test-time
  contract (§3 invariant), so a bad declaration cannot reach production.
* **Runtime emission is fail-OPEN.** :meth:`MetricRegistry.inc` /
  :meth:`~MetricRegistry.set` / :meth:`~MetricRegistry.observe` NEVER raise: an
  unknown metric, a kind mismatch, an undeclared label, or a non-finite value is
  a no-op that bumps the self-registered
  :data:`EMIT_ERRORS_METRIC` (``lifemodel_metrics_emit_errors_total{reason}``).
  A tick must never die because a component emitted a bad metric.

The metric *type* objects (:class:`Counter`/:class:`Gauge`/:class:`Histogram`)
are the strict layer underneath: their direct methods validate and raise
:class:`MetricEmitError`; the registry's runtime methods catch that and convert
it to the fail-open no-op + error count above.

**Low cardinality is a hard rule.** Only the closed set
:data:`ALLOWED_LABEL_KEYS` — ``component``, ``layer``, ``phase``, ``reason``,
``outcome``, ``model`` — may appear as a metric label. Never ``trace_id``, a
prompt, a message body, or any free-form string.

Lifecycle mirrors :func:`~lifemodel.state.trace_store.acquire_trace_writer`:
:func:`get_metric_registry` is a **singleton per base_dir**, idempotent — a
repeated call (a plugin ``register()`` runs more than once and has no teardown)
returns the SAME instance so tick, hooks, and ``/lifemodel stats`` all read one
registry. Unlike the trace writer there is no refcount/teardown and, crucially,
**no thread is started here** — the periodic sampler is bead 7.6's concern.

All stdlib (``threading``/``bisect``/``math``) — the plugin runs inside Hermes'
own interpreter, no third-party deps.
"""

from __future__ import annotations

import bisect
import contextlib
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeVar

# --------------------------------------------------------------------------- #
# Vocabulary (design §4.1)
# --------------------------------------------------------------------------- #

#: The kind of a metric — matches the Prometheus data model we render into later.
MetricKind = Literal["counter", "gauge", "histogram"]

#: The CLOSED set of label keys a metric may carry (design §4.1). Low cardinality
#: is enforced at declaration: a spec naming any other key fails fast. NEVER put a
#: ``trace_id``, prompt, message text, or arbitrary string on a metric.
ALLOWED_LABEL_KEYS: Final[frozenset[str]] = frozenset(
    {"component", "layer", "phase", "reason", "outcome", "model"}
)

#: Default histogram bucket bounds in seconds (design §4.1). Coarse on purpose —
#: a ~60s tick doesn't need fine buckets, but exposing latency as a histogram lets
#: a standard Prometheus consumer (a later phase) read it natively.
DEFAULT_BUCKETS: Final[tuple[float, ...]] = (
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

#: The self-registered counter that fail-open runtime emission bumps instead of
#: raising (design §4.3/§7). Its ``reason`` values are a closed internal set:
#: ``unknown_metric`` · ``wrong_kind`` · ``unknown_label`` · ``bad_value``.
EMIT_ERRORS_METRIC: Final = "lifemodel_metrics_emit_errors_total"


class MetricSpecError(ValueError):
    """A metric declaration is malformed — the fail-FAST half of the API (§4.3).

    Raised at build/test time by :class:`MetricSpec` construction or
    :meth:`MetricRegistry.register`. It never reaches the tick path: runtime
    emission is fail-open (see :class:`MetricEmitError` / the registry).
    """


class MetricEmitError(Exception):
    """A single emission is invalid (bad label / kind / value) — the strict signal.

    Raised by the metric *type* objects' direct methods; the registry's runtime
    :meth:`~MetricRegistry.inc`/:meth:`~MetricRegistry.set`/
    :meth:`~MetricRegistry.observe` catch it and convert it to a fail-open no-op
    plus a bump of :data:`EMIT_ERRORS_METRIC`. :attr:`reason` names the closed
    failure category recorded on that counter.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Canonical label key (design §4.4 — used by the sampler / stats, beads 7.6/7.7)
# --------------------------------------------------------------------------- #


def label_key(labels: Mapping[str, str]) -> str:
    """Return a deterministic, order-independent key for a label set (§4.4).

    A stable canonical string built from the sorted ``key=value`` pairs, so the
    same labels always collapse to the same key regardless of insertion order —
    the join column the sampler (bead 7.6) and ``/lifemodel stats`` (bead 7.7)
    group series on without re-parsing JSON per row. An empty label set maps to
    the empty string. Values are already low-cardinality closed-set enums / ids
    (never free-form), so a readable canonical form is preferred over an opaque
    hash — it stays debuggable.
    """
    return ",".join(f"{key}={labels[key]}" for key in sorted(labels))


# --------------------------------------------------------------------------- #
# MetricSpec — declared once, then emitted against (fail-fast validation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricSpec:
    """The declarative shape of a metric — validated eagerly (fail-fast, §4.1/§4.3).

    ``label_keys`` are normalised to a sorted, de-duplicated tuple so two specs
    that differ only in label order compare equal (idempotent re-registration).
    ``buckets`` is meaningful only for ``kind == "histogram"`` and defaults to
    :data:`DEFAULT_BUCKETS`. ``export`` is the sampler whitelist flag (design
    §4.4): when ``False`` the metric stays live in the registry but is NOT
    written to ``metrics.sqlite`` by the periodic sampler (bead 7.6) — its
    ``metric_defs`` row still records ``export=0``. Construction raises
    :class:`MetricSpecError` on:

    * an empty ``name`` or an unknown ``kind``;
    * a label key outside :data:`ALLOWED_LABEL_KEYS` (the low-cardinality rule);
    * histogram buckets that are empty, non-finite, or not strictly increasing.
    """

    name: str
    kind: MetricKind
    unit: str = ""
    help: str = ""
    label_keys: tuple[str, ...] = ()
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    #: Sampler whitelist (design §4.4). ``True`` → sample into ``metrics.sqlite``;
    #: ``False`` → live-only (the def is still recorded, flagged ``export=0``).
    export: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise MetricSpecError("metric name must be a non-empty string")
        if self.kind not in ("counter", "gauge", "histogram"):
            raise MetricSpecError(f"unknown metric kind: {self.kind!r}")

        unknown = frozenset(self.label_keys) - ALLOWED_LABEL_KEYS
        if unknown:
            raise MetricSpecError(
                f"label keys {sorted(unknown)} are outside the closed set "
                f"{sorted(ALLOWED_LABEL_KEYS)} (low-cardinality rule, §4.1)"
            )
        # Normalise to a sorted, unique tuple → order-independent spec equality.
        object.__setattr__(self, "label_keys", tuple(sorted(set(self.label_keys))))

        if self.kind == "histogram":
            self._validate_buckets()
        else:
            # Buckets are inert for non-histograms; drop to () so equality/serialisation
            # of a counter/gauge spec doesn't hinge on an irrelevant default.
            object.__setattr__(self, "buckets", ())

    def _validate_buckets(self) -> None:
        buckets = self.buckets
        if not buckets:
            raise MetricSpecError("a histogram needs at least one bucket bound")
        if any(not math.isfinite(b) for b in buckets):
            raise MetricSpecError("histogram bucket bounds must be finite")
        if any(hi <= lo for lo, hi in zip(buckets, buckets[1:], strict=False)):
            raise MetricSpecError(f"histogram buckets must be strictly increasing: {buckets}")


# --------------------------------------------------------------------------- #
# Metric types — the strict value store (Counter / Gauge / Histogram)
# --------------------------------------------------------------------------- #


class Metric:
    """Common base for the metric types — holds the spec + a per-series lock.

    Each metric keeps one *series* per distinct label set, keyed by
    :func:`label_key`. Direct mutation is the STRICT layer: it validates labels /
    values and raises :class:`MetricEmitError` (which the registry converts to
    fail-open). Reads (``value``/``snapshot``/``items``) are for the sampler
    (bead 7.6) and ``/lifemodel stats`` (bead 7.7).
    """

    kind: ClassVar[MetricKind]

    def __init__(self, spec: MetricSpec) -> None:
        if spec.kind != self.kind:
            raise MetricSpecError(
                f"{type(self).__name__} requires a {self.kind!r} spec, got {spec.kind!r}"
            )
        self.spec = spec
        self._declared: frozenset[str] = frozenset(spec.label_keys)
        self._lock = threading.Lock()
        self._labels: dict[str, dict[str, str]] = {}

    def _key(self, labels: Mapping[str, str]) -> str:
        """Validate labels against the declared set and return the canonical key.

        Any key not declared on this metric's spec is an ``unknown_label`` — the
        closed-set rule is enforced per-metric, not just globally.
        """
        undeclared = frozenset(labels) - self._declared
        if undeclared:
            raise MetricEmitError("unknown_label")
        return label_key(labels)

    @staticmethod
    def _finite(value: float) -> float:
        if not math.isfinite(value):
            raise MetricEmitError("bad_value")
        return value

    def _remember(self, key: str, labels: Mapping[str, str]) -> None:
        if key not in self._labels:
            self._labels[key] = dict(labels)

    def label_sets(self) -> list[tuple[str, dict[str, str]]]:
        """Return ``(label_key, labels)`` for every observed series — sampler glue."""
        with self._lock:
            return [(key, dict(labels)) for key, labels in self._labels.items()]


class Counter(Metric):
    """A monotonic cumulative counter — :meth:`inc` only ever adds (design §4.1)."""

    kind: ClassVar[MetricKind] = "counter"

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._values: dict[str, float] = {}

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Add *value* (default ``1``) to the series for *labels*. STRICT: raises
        :class:`MetricEmitError` on an undeclared label, a non-finite value, or a
        negative increment (a counter never decreases)."""
        value = self._finite(value)
        if value < 0:
            raise MetricEmitError("bad_value")
        key = self._key(labels)
        with self._lock:
            self._remember(key, labels)
            self._values[key] = self._values.get(key, 0.0) + value

    def value(self, **labels: str) -> float:
        """Current value for *labels* (``0.0`` if that series was never touched)."""
        key = label_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def items(self) -> list[tuple[str, dict[str, str], float]]:
        """Every ``(label_key, labels, value)`` series — for the sampler (7.6)."""
        with self._lock:
            return [(key, dict(self._labels[key]), val) for key, val in self._values.items()]


class Gauge(Metric):
    """A settable current value — :meth:`set` replaces (design §4.1)."""

    kind: ClassVar[MetricKind] = "gauge"

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._values: dict[str, float] = {}

    def set(self, value: float, **labels: str) -> None:
        """Replace the series value for *labels*. STRICT (raises
        :class:`MetricEmitError` on an undeclared label or non-finite value)."""
        value = self._finite(value)
        key = self._key(labels)
        with self._lock:
            self._remember(key, labels)
            self._values[key] = value

    def value(self, **labels: str) -> float:
        """Current value for *labels* (``0.0`` if never set)."""
        key = label_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def items(self) -> list[tuple[str, dict[str, str], float]]:
        """Every ``(label_key, labels, value)`` series — for the sampler (7.6)."""
        with self._lock:
            return [(key, dict(self._labels[key]), val) for key, val in self._values.items()]


@dataclass(frozen=True)
class HistogramSnapshot:
    """A read of one histogram series: cumulative buckets + ``count`` + ``sum``.

    ``buckets`` is ``(le, cumulative_count)`` for each finite bound, where the
    count is the number of observations ``<= le`` (Prometheus semantics). The
    implicit ``+Inf`` bucket equals :attr:`count`; observations above the last
    bound land only there (they still add to :attr:`count` / :attr:`sum`).
    """

    buckets: tuple[tuple[float, int], ...]
    count: int
    sum: float


class Histogram(Metric):
    """Fixed-bucket histogram — :meth:`observe` records into buckets + count + sum."""

    kind: ClassVar[MetricKind] = "histogram"

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._bounds: tuple[float, ...] = spec.buckets
        # Non-cumulative per-bound counts; overflow (> last bound) is recovered as
        # count - sum(bucket_counts) and lives only in the implicit +Inf bucket.
        self._bucket_counts: dict[str, list[int]] = {}
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record one *value* into the series for *labels*. STRICT (raises
        :class:`MetricEmitError` on an undeclared label or non-finite value)."""
        value = self._finite(value)
        key = self._key(labels)
        idx = bisect.bisect_left(self._bounds, value)
        with self._lock:
            self._remember(key, labels)
            counts = self._bucket_counts.get(key)
            if counts is None:
                counts = [0] * len(self._bounds)
                self._bucket_counts[key] = counts
            if idx < len(self._bounds):
                counts[idx] += 1
            self._counts[key] = self._counts.get(key, 0) + 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def snapshot(self, **labels: str) -> HistogramSnapshot:
        """Cumulative buckets + ``count`` + ``sum`` for *labels* (empty if unseen)."""
        key = label_key(labels)
        with self._lock:
            counts = list(self._bucket_counts.get(key, [0] * len(self._bounds)))
            total = self._counts.get(key, 0)
            total_sum = self._sums.get(key, 0.0)
        cumulative: list[tuple[float, int]] = []
        running = 0
        for bound, count in zip(self._bounds, counts, strict=True):
            running += count
            cumulative.append((bound, running))
        return HistogramSnapshot(buckets=tuple(cumulative), count=total, sum=total_sum)

    def items(self) -> list[tuple[str, dict[str, str], HistogramSnapshot]]:
        """Every ``(label_key, labels, snapshot)`` series — for the sampler (7.6)."""
        return [(key, labels, self.snapshot(**labels)) for key, labels in self.label_sets()]


_MetricT = TypeVar("_MetricT", bound=Metric)

_KIND_TO_TYPE: Final[dict[MetricKind, type[Metric]]] = {
    "counter": Counter,
    "gauge": Gauge,
    "histogram": Histogram,
}


def _make_metric(spec: MetricSpec) -> Metric:
    return _KIND_TO_TYPE[spec.kind](spec)


# --------------------------------------------------------------------------- #
# MetricRegistry — declares metrics (fail-fast) and emits into them (fail-open)
# --------------------------------------------------------------------------- #


class MetricRegistry:
    """The process-local, thread-safe source of CURRENT metric state (design §4.1).

    Declaration is fail-fast (:meth:`register` and the ``counter``/``gauge``/
    ``histogram`` helpers raise :class:`MetricSpecError`); runtime emission
    (:meth:`inc`/:meth:`set`/:meth:`observe`) is fail-open and NEVER raises. Reads
    (:meth:`get`/:meth:`metrics`/:meth:`specs`) back the sampler (bead 7.6) and
    ``/lifemodel stats`` (bead 7.7). Prefer :func:`get_metric_registry` over
    constructing directly — it enforces the singleton-per-base_dir lifecycle.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path | None = Path(base_dir) if base_dir is not None else None
        self._lock = threading.Lock()
        self._metrics: dict[str, Metric] = {}
        # The self-registered fail-open error counter (design §4.3/§7). Emitting a
        # bad metric bumps this instead of raising, so a tick can't die on telemetry.
        self._emit_errors = Counter(
            MetricSpec(
                name=EMIT_ERRORS_METRIC,
                kind="counter",
                unit="",
                help="Runtime metric emissions rejected fail-open (by reason).",
                label_keys=("reason",),
            )
        )
        self._metrics[EMIT_ERRORS_METRIC] = self._emit_errors

    # ---- declaration (fail-fast) ---------------------------------------- #

    def register(self, spec: MetricSpec) -> Metric:
        """Declare a metric and return its :class:`Metric` object (fail-fast).

        Idempotent for an identical spec (returns the existing metric — a plugin
        ``register()`` runs more than once). A DIFFERENT spec under the same name
        is a declaration bug and raises :class:`MetricSpecError`.
        """
        with self._lock:
            existing = self._metrics.get(spec.name)
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise MetricSpecError(
                    f"metric {spec.name!r} already registered with a different spec"
                )
            metric = _make_metric(spec)
            self._metrics[spec.name] = metric
            return metric

    def counter(
        self, name: str, *, unit: str = "", help: str = "", label_keys: tuple[str, ...] = ()
    ) -> Counter:
        """Declare (or fetch) a :class:`Counter` — fail-fast convenience over :meth:`register`."""
        metric = self.register(
            MetricSpec(name=name, kind="counter", unit=unit, help=help, label_keys=label_keys)
        )
        assert isinstance(metric, Counter)  # register enforces the kind
        return metric

    def gauge(
        self, name: str, *, unit: str = "", help: str = "", label_keys: tuple[str, ...] = ()
    ) -> Gauge:
        """Declare (or fetch) a :class:`Gauge` — fail-fast convenience over :meth:`register`."""
        metric = self.register(
            MetricSpec(name=name, kind="gauge", unit=unit, help=help, label_keys=label_keys)
        )
        assert isinstance(metric, Gauge)
        return metric

    def histogram(
        self,
        name: str,
        *,
        unit: str = "",
        help: str = "",
        label_keys: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> Histogram:
        """Declare (or fetch) a :class:`Histogram` — fail-fast convenience over :meth:`register`."""
        metric = self.register(
            MetricSpec(
                name=name,
                kind="histogram",
                unit=unit,
                help=help,
                label_keys=label_keys,
                buckets=buckets,
            )
        )
        assert isinstance(metric, Histogram)
        return metric

    # ---- runtime emission (fail-open — NEVER raises) -------------------- #

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to counter *name* for *labels*. Fail-open: an unknown metric, wrong
        kind, undeclared label, or bad value is a no-op + an emit-error bump."""
        metric = self._for_emit(name, Counter)
        if metric is None:
            return
        self._guarded(lambda: metric.inc(value, **labels))

    def set(self, name: str, value: float, **labels: str) -> None:
        """Set gauge *name* for *labels*. Fail-open (see :meth:`inc`)."""
        metric = self._for_emit(name, Gauge)
        if metric is None:
            return
        self._guarded(lambda: metric.set(value, **labels))

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record into histogram *name* for *labels*. Fail-open (see :meth:`inc`)."""
        metric = self._for_emit(name, Histogram)
        if metric is None:
            return
        self._guarded(lambda: metric.observe(value, **labels))

    # ---- reads (sampler / stats) ---------------------------------------- #

    def get(self, name: str) -> Metric | None:
        """The registered :class:`Metric` for *name*, or ``None``."""
        with self._lock:
            return self._metrics.get(name)

    def metrics(self) -> list[Metric]:
        """Every registered metric (includes the self-registered error counter)."""
        with self._lock:
            return list(self._metrics.values())

    def specs(self) -> list[MetricSpec]:
        """Every registered metric's spec — the ``metric_defs`` source (bead 7.6)."""
        with self._lock:
            return [metric.spec for metric in self._metrics.values()]

    # ---- fail-open plumbing --------------------------------------------- #

    def _for_emit(self, name: str, expected: type[_MetricT]) -> _MetricT | None:
        with self._lock:
            metric = self._metrics.get(name)
        if metric is None:
            self._record_emit_error("unknown_metric")
            return None
        if not isinstance(metric, expected):
            self._record_emit_error("wrong_kind")
            return None
        return metric

    def _guarded(self, emit: Callable[[], None]) -> None:
        try:
            emit()
        except MetricEmitError as exc:
            self._record_emit_error(exc.reason)
        except Exception:
            # Truly unexpected — still never propagate onto the tick path (§7).
            self._record_emit_error("bad_value")

    def _record_emit_error(self, reason: str) -> None:
        # Counting an error must itself never raise (it would defeat fail-open).
        with contextlib.suppress(Exception):
            self._emit_errors.inc(1.0, reason=reason)


# --------------------------------------------------------------------------- #
# Singleton per base_dir (design §4.1 — idempotent acquire, NO thread started)
# --------------------------------------------------------------------------- #

_registry_lock = threading.Lock()
_registries: dict[str, MetricRegistry] = {}


def _registry_key(base_dir: Path) -> str:
    return str(Path(base_dir).resolve())


def get_metric_registry(base_dir: Path) -> MetricRegistry:
    """Return the process-local :class:`MetricRegistry` for *base_dir* (design §4.1).

    Singleton per resolved *base_dir* and idempotent: the FIRST call constructs
    it, later calls return the SAME instance — so tick, hooks, and
    ``/lifemodel stats`` (all in the one gateway process) read a single registry
    and never diverge. Unlike
    :func:`~lifemodel.state.trace_store.acquire_trace_writer` there is no
    refcount / teardown (a plugin ``register()`` has none) and — crucially — **no
    thread is started here**: the periodic ``metrics.sqlite`` sampler is bead
    7.6's concern.
    """
    key = _registry_key(base_dir)
    with _registry_lock:
        registry = _registries.get(key)
        if registry is None:
            registry = MetricRegistry(base_dir=Path(base_dir))
            _registries[key] = registry
        return registry

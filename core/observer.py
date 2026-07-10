"""``ComponentObserver`` — the per-component domain-metric channel (telemetry-core §4.3).

Universal metrics are snapped BY the harness (the CoreLoop wrapper, bead 7.4) so a
component can never run unmeasured. This is the OTHER half: the channel a component
emits the metrics only IT knows — a drive level, token counts — through
``ctx.observe`` (design §4.3). A thin, typed handle bound to ONE component's
DECLARED ``metric_surface`` plus the shared :class:`MetricRegistry`.

Two fail-open guarantees keep a bad emission off the tick path (design §4.3 / §7):

* the registry's own :meth:`~MetricRegistry.inc`/:meth:`~MetricRegistry.set`/
  :meth:`~MetricRegistry.observe` never raise (unknown metric / wrong kind /
  undeclared label / bad value → no-op + ``lifemodel_metrics_emit_errors_total``);
  this handle simply DELEGATES to them — it does not re-invent fail-open.
* additionally, a ``name`` NOT in the component's own ``metric_surface`` is itself
  fail-open here: a no-op that bumps that same error counter under the
  :data:`UNDECLARED_SURFACE` reason. The surface is the component's honest, enforced
  contract (§3) — a component cannot smuggle out a metric it never declared, and
  trying does not crash the tick.

Stdlib-only — the registry it delegates to already is.
"""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import EMIT_ERRORS_METRIC, MetricRegistry, MetricSpec

#: The ``reason`` recorded on ``lifemodel_metrics_emit_errors_total`` when a
#: component emits through ``ctx.observe`` a metric outside its declared
#: ``metric_surface`` (design §4.3). Extends the registry's internal reason set
#: (``unknown_metric``/``wrong_kind``/``unknown_label``/``bad_value``) with the one
#: failure only this channel can see: the name is real but not THIS component's.
UNDECLARED_SURFACE = "undeclared_surface"


def surface_metric_names(metric_surface: Sequence[MetricSpec | str] | None) -> frozenset[str]:
    """The declared metric NAMES of a ``metric_surface`` (a spec → its ``name``, a
    bare name as-is). ``None`` (an undeclared surface, which
    :meth:`~lifemodel.core.registry.ComponentRegistry.register` already rejects)
    and an empty surface both yield the empty set."""
    return frozenset(
        entry.name if isinstance(entry, MetricSpec) else entry for entry in (metric_surface or ())
    )


class ComponentObserver:
    """A component's typed handle onto its declared domain metrics (design §4.3)."""

    def __init__(self, registry: MetricRegistry, surface: frozenset[str]) -> None:
        self._registry = registry
        self._surface = surface

    @classmethod
    def bind(
        cls, registry: MetricRegistry, metric_surface: Sequence[MetricSpec | str] | None
    ) -> ComponentObserver:
        """Bind an observer to *registry* and the names declared in *metric_surface*."""
        return cls(registry, surface_metric_names(metric_surface))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to declared counter *name* for *labels* (fail-open, see module docs)."""
        if self._declared(name):
            self._registry.inc(name, value, **labels)

    def set(self, name: str, value: float, **labels: str) -> None:
        """Set declared gauge *name* for *labels* (fail-open, see module docs)."""
        if self._declared(name):
            self._registry.set(name, value, **labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record into declared histogram *name* for *labels* (fail-open, see module docs)."""
        if self._declared(name):
            self._registry.observe(name, value, **labels)

    def _declared(self, name: str) -> bool:
        """Whether *name* is in this component's surface; if not, count + drop it.

        The error bump routes through the registry's own fail-open ``inc`` (the
        error counter is a registered counter with a ``reason`` label), so counting
        a rejected emission can itself never raise onto the tick path."""
        if name in self._surface:
            return True
        self._registry.inc(EMIT_ERRORS_METRIC, 1.0, reason=UNDECLARED_SURFACE)
        return False

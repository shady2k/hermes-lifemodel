"""ComponentRegistry — the self-registration seam (spec §15).

Components register themselves (via a DI callback from the composition root),
each with an internal :class:`ComponentManifest`. The registry can enable/disable
by id and yields the enabled components in *registration order* so scheduling is
deterministic. External plugin discovery/loading is deferred (registration now,
loading later); this holds only the in-process registration half.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .component import Component, ComponentLayer

if TYPE_CHECKING:
    from .metrics import MetricSpec


class DuplicateComponent(ValueError):
    """A second component tried to register under an already-used id."""


class UnknownComponent(KeyError):
    """A toggle/lookup referenced an id that was never registered."""


class IncompleteManifest(ValueError):
    """A manifest reached :meth:`ComponentRegistry.register` without declaring how
    it reports statistics — the second lock of the instrumentation invariant
    (telemetry-core §3). A component may declare an EMPTY ``metric_surface``
    ("emits no domain metrics"), but it may not leave ``layer`` or
    ``metric_surface`` *undeclared*.
    """


@dataclass(frozen=True)
class ComponentManifest:
    """Internal descriptor for a registered component.

    Beyond identity (``id``/``type``/``version``/``enabled``/``config``) the
    manifest DECLARES the component's telemetry contract (telemetry-core §3):

    * ``layer`` — the closed :class:`~lifemodel.core.component.ComponentLayer` it
      belongs to (the low-cardinality ``layer`` metric label). **Required** by
      :meth:`ComponentRegistry.register`; ``None`` means "undeclared" and is
      rejected fail-fast.
    * ``metric_surface`` — the domain metrics the component emits, each a
      :class:`~lifemodel.core.metrics.MetricSpec` or its bare name. **Required** to
      be declared, but may be an empty tuple: a component that emits no domain
      metrics must still say so. ``None`` means "undeclared" and is rejected.
    * ``phase`` — an optional finer-grained pipeline phase label (§4). Descriptive
      only; not enforced.
    * ``accepts_signals`` — whether the component consumes ``ctx.signals`` (§4.2).
      Registry knowledge the harness later exposes as
      ``lifemodel_layer_accepts_signals``; defaults ``False``.

    A list passed as ``metric_surface`` is normalised to a tuple so the frozen
    manifest stays immutable.
    """

    id: str
    type: str
    enabled: bool = True
    version: str = "0.0.0"
    config: Mapping[str, Any] = field(default_factory=dict)
    layer: ComponentLayer | None = None
    phase: str | None = None
    metric_surface: Sequence[MetricSpec | str] | None = None
    accepts_signals: bool = False

    def __post_init__(self) -> None:
        # Accept any sequence of specs/names but store an immutable tuple; leave the
        # "undeclared" sentinel (None) untouched so register() can reject it.
        if self.metric_surface is not None and not isinstance(self.metric_surface, tuple):
            object.__setattr__(self, "metric_surface", tuple(self.metric_surface))


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._manifests: dict[str, ComponentManifest] = {}
        self._order: list[str] = []

    def register(self, component: Component, manifest: ComponentManifest) -> None:
        if manifest.id in self._components:
            raise DuplicateComponent(manifest.id)
        # The second lock of the instrumentation invariant (telemetry-core §3): a
        # component cannot be registered without declaring how it reports stats.
        if manifest.layer is None:
            raise IncompleteManifest(f"{manifest.id!r} registered without a layer")
        if manifest.metric_surface is None:
            raise IncompleteManifest(f"{manifest.id!r} registered without a metric_surface")
        self._components[manifest.id] = component
        self._manifests[manifest.id] = manifest
        self._order.append(manifest.id)

    def enable(self, component_id: str) -> None:
        self._set_enabled(component_id, True)

    def disable(self, component_id: str) -> None:
        self._set_enabled(component_id, False)

    def _set_enabled(self, component_id: str, value: bool) -> None:
        manifest = self._require(component_id)
        self._manifests[component_id] = replace(manifest, enabled=value)

    def manifest(self, component_id: str) -> ComponentManifest:
        return self._require(component_id)

    def manifests(self) -> tuple[ComponentManifest, ...]:
        """Every registered manifest in registration order (enabled or not).

        The seam the §3 enforcement test and later telemetry (the sampler / stats
        rollup) walk to inspect *all* components' declared layer + metric surface,
        not just the currently-enabled ones.
        """
        return tuple(self._manifests[cid] for cid in self._order)

    def _require(self, component_id: str) -> ComponentManifest:
        try:
            return self._manifests[component_id]
        except KeyError:
            raise UnknownComponent(component_id) from None

    def enabled(self) -> tuple[Component, ...]:
        return tuple(self._components[cid] for cid in self._order if self._manifests[cid].enabled)

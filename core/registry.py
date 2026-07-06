"""ComponentRegistry — the self-registration seam (spec §15).

Components register themselves (via a DI callback from the composition root),
each with an internal :class:`ComponentManifest`. The registry can enable/disable
by id and yields the enabled components in *registration order* so scheduling is
deterministic. External plugin discovery/loading is deferred (registration now,
loading later); this holds only the in-process registration half.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .component import Component


class DuplicateComponent(ValueError):
    """A second component tried to register under an already-used id."""


class UnknownComponent(KeyError):
    """A toggle/lookup referenced an id that was never registered."""


@dataclass(frozen=True)
class ComponentManifest:
    """Internal descriptor for a registered component."""

    id: str
    type: str
    enabled: bool = True
    version: str = "0.0.0"
    config: Mapping[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._manifests: dict[str, ComponentManifest] = {}
        self._order: list[str] = []

    def register(self, component: Component, manifest: ComponentManifest) -> None:
        if manifest.id in self._components:
            raise DuplicateComponent(manifest.id)
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

    def _require(self, component_id: str) -> ComponentManifest:
        try:
            return self._manifests[component_id]
        except KeyError:
            raise UnknownComponent(component_id) from None

    def enabled(self) -> tuple[Component, ...]:
        return tuple(self._components[cid] for cid in self._order if self._manifests[cid].enabled)

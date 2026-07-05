from __future__ import annotations

from collections.abc import Sequence

import pytest

from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent
from lifemodel.core.registry import (
    ComponentManifest,
    ComponentRegistry,
    DuplicateComponent,
    UnknownComponent,
)


class Stub:
    def __init__(self, cid: str) -> None:
        self.id = cid

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return []


def _manifest(cid: str, *, enabled: bool = True) -> ComponentManifest:
    return ComponentManifest(id=cid, type="neuron", enabled=enabled)


def test_register_then_enabled_returns_in_registration_order() -> None:
    reg = ComponentRegistry()
    a, b, c = Stub("a"), Stub("b"), Stub("c")
    reg.register(a, _manifest("a"))
    reg.register(b, _manifest("b"))
    reg.register(c, _manifest("c"))
    assert [comp.id for comp in reg.enabled()] == ["a", "b", "c"]


def test_disabled_component_excluded_from_enabled() -> None:
    reg = ComponentRegistry()
    reg.register(Stub("a"), _manifest("a"))
    reg.register(Stub("b"), _manifest("b", enabled=False))
    assert [comp.id for comp in reg.enabled()] == ["a"]


def test_enable_and_disable_toggle_membership() -> None:
    reg = ComponentRegistry()
    reg.register(Stub("a"), _manifest("a"))
    reg.disable("a")
    assert reg.enabled() == ()
    reg.enable("a")
    assert [comp.id for comp in reg.enabled()] == ["a"]


def test_duplicate_id_rejected() -> None:
    reg = ComponentRegistry()
    reg.register(Stub("a"), _manifest("a"))
    with pytest.raises(DuplicateComponent):
        reg.register(Stub("a"), _manifest("a"))


def test_toggle_or_manifest_of_unknown_id_raises() -> None:
    reg = ComponentRegistry()
    with pytest.raises(UnknownComponent):
        reg.enable("ghost")
    with pytest.raises(UnknownComponent):
        reg.manifest("ghost")


def test_manifest_reflects_enabled_flag() -> None:
    reg = ComponentRegistry()
    reg.register(Stub("a"), _manifest("a"))
    reg.disable("a")
    assert reg.manifest("a").enabled is False

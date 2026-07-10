"""The instrumentation invariant, second lock (telemetry-core design §3).

A component cannot be registered without declaring how it reports statistics:
``ComponentManifest`` carries a ``layer`` + a ``metric_surface`` and
:meth:`ComponentRegistry.register` fails FAST when either is undeclared. Plus the
enforcement test over the real composition root — every wired component declares
both — and the ``type → ComponentLayer`` rollup mapping (§4.2 / §4.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.component import (
    LAYER_BY_TYPE,
    ComponentLayer,
    TickContext,
    layer_for_type,
)
from lifemodel.core.intents import Intent
from lifemodel.core.metrics import MetricSpec
from lifemodel.core.registry import (
    ComponentManifest,
    ComponentRegistry,
    IncompleteManifest,
)


class Stub:
    def __init__(self, cid: str) -> None:
        self.id = cid

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return []


# --------------------------------------------------------------------------- #
# ComponentLayer — a CLOSED four-member enum (§4.2)
# --------------------------------------------------------------------------- #


def test_component_layer_is_a_closed_four_member_enum() -> None:
    assert {layer.name for layer in ComponentLayer} == {
        "AUTONOMIC",
        "AGGREGATION",
        "COGNITION",
        "INFRA",
    }


# --------------------------------------------------------------------------- #
# type → ComponentLayer mapping (§4.2 / §4.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        ("personality", ComponentLayer.AUTONOMIC),
        ("neuron", ComponentLayer.AUTONOMIC),
        ("drive", ComponentLayer.AUTONOMIC),
        ("aggregation", ComponentLayer.AGGREGATION),
        ("launcher", ComponentLayer.COGNITION),
        ("cognition", ComponentLayer.COGNITION),
        ("proactive", ComponentLayer.INFRA),
        ("egress", ComponentLayer.INFRA),
        ("trace-writer", ComponentLayer.INFRA),
    ],
)
def test_layer_for_type_maps_per_spec(type_: str, expected: ComponentLayer) -> None:
    assert layer_for_type(type_) is expected
    assert LAYER_BY_TYPE[type_] is expected


def test_egress_is_infra_not_cognition() -> None:
    # §4.2: egress is an INFRA concern — do NOT fold it into COGNITION.
    assert layer_for_type("egress") is ComponentLayer.INFRA


def test_layer_for_type_unknown_is_none() -> None:
    assert layer_for_type("no-such-type") is None


# --------------------------------------------------------------------------- #
# register() — FAIL-FAST without layer / metric_surface (§3, second lock)
# --------------------------------------------------------------------------- #


def test_register_rejects_manifest_without_layer() -> None:
    reg = ComponentRegistry()
    manifest = ComponentManifest(id="x", type="neuron", metric_surface=())
    with pytest.raises(IncompleteManifest):
        reg.register(Stub("x"), manifest)


def test_register_rejects_manifest_without_metric_surface() -> None:
    reg = ComponentRegistry()
    manifest = ComponentManifest(id="x", type="neuron", layer=ComponentLayer.AUTONOMIC)
    with pytest.raises(IncompleteManifest):
        reg.register(Stub("x"), manifest)


def test_register_rejects_manifest_without_either() -> None:
    reg = ComponentRegistry()
    with pytest.raises(IncompleteManifest):
        reg.register(Stub("x"), ComponentManifest(id="x", type="neuron"))


def test_register_accepts_complete_manifest_with_empty_declared_surface() -> None:
    # An EMPTY metric_surface is a valid declaration ("emits no domain metrics").
    # Only an UNDECLARED (None) surface fails — declared-empty is the invariant's
    # floor, not a violation.
    reg = ComponentRegistry()
    manifest = ComponentManifest(
        id="x", type="neuron", layer=ComponentLayer.AUTONOMIC, metric_surface=()
    )
    reg.register(Stub("x"), manifest)
    assert reg.manifest("x").layer is ComponentLayer.AUTONOMIC


# --------------------------------------------------------------------------- #
# ComponentManifest field shapes
# --------------------------------------------------------------------------- #


def test_metric_surface_accepts_specs_and_names_normalised_to_tuple() -> None:
    spec = MetricSpec(name="lifemodel_contact_drive_u", kind="gauge")
    manifest = ComponentManifest(
        id="x",
        type="drive",
        layer=ComponentLayer.AUTONOMIC,
        metric_surface=[spec, "lifemodel_solitude_pressure"],
    )
    assert isinstance(manifest.metric_surface, tuple)
    assert manifest.metric_surface == (spec, "lifemodel_solitude_pressure")


def test_accepts_signals_defaults_false_and_is_settable() -> None:
    off = ComponentManifest(
        id="a", type="neuron", layer=ComponentLayer.AUTONOMIC, metric_surface=()
    )
    on = ComponentManifest(
        id="b",
        type="neuron",
        layer=ComponentLayer.AUTONOMIC,
        metric_surface=(),
        accepts_signals=True,
    )
    assert off.accepts_signals is False
    assert on.accepts_signals is True


def test_manifests_returns_all_registered_in_order() -> None:
    reg = ComponentRegistry()
    for cid in ("a", "b", "c"):
        reg.register(
            Stub(cid),
            ComponentManifest(
                id=cid, type="neuron", layer=ComponentLayer.AUTONOMIC, metric_surface=()
            ),
        )
    assert [m.id for m in reg.manifests()] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Enforcement over the REAL composition root (§3 invariant / §8)
# --------------------------------------------------------------------------- #


def test_every_composed_component_declares_layer_and_metric_surface(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    manifests = lm.registry.manifests()
    assert manifests, "composition root registered no components"
    for manifest in manifests:
        assert manifest.layer is not None, f"{manifest.id} declares no layer"
        assert manifest.metric_surface is not None, f"{manifest.id} declares no metric_surface"


def test_composition_assigns_expected_layers(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    by_id = {m.id: m for m in lm.registry.manifests()}
    assert by_id["personality"].layer is ComponentLayer.AUTONOMIC
    assert by_id["contact"].layer is ComponentLayer.AUTONOMIC
    assert by_id["solitude-drive"].layer is ComponentLayer.AUTONOMIC
    assert by_id["contact-aggregation"].layer is ComponentLayer.AGGREGATION
    assert by_id["cognition-launcher"].layer is ComponentLayer.COGNITION


def test_composition_declares_accepts_signals_per_component(tmp_path: Path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    by_id = {m.id: m for m in lm.registry.manifests()}
    # The signal-reading components (they consult ctx.signals in step()).
    assert by_id["contact"].accepts_signals is True
    assert by_id["solitude-drive"].accepts_signals is True
    assert by_id["contact-aggregation"].accepts_signals is True
    # These two never read ctx.signals.
    assert by_id["personality"].accepts_signals is False
    assert by_id["cognition-launcher"].accepts_signals is False

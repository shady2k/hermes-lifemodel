"""``ctx.observe`` — the per-component domain-metric channel (telemetry-core §4.3, bead 7.5).

A component emits the metrics ONLY it knows (drive levels, token counts) through a
thin :class:`~lifemodel.core.observer.ComponentObserver` bound to its DECLARED
``metric_surface`` + the shared registry. Two guarantees, both fail-open (§7): an
emission of a metric outside the component's surface is a no-op that bumps
``lifemodel_metrics_emit_errors_total`` (the tick never dies), and a bare context
with no observer (no graph) is simply ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.core.component import TickContext, layer_for_type
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.intents import Intent
from lifemodel.core.metrics import EMIT_ERRORS_METRIC, MetricRegistry, MetricSpec
from lifemodel.core.observer import UNDECLARED_SURFACE, ComponentObserver
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.solitude_drive import CONTACT_DRIVE_U, CONTACT_DRIVE_U_SPEC, SolitudeDrive
from lifemodel.core.state_actor import StateActor
from lifemodel.state.model import State
from lifemodel.testing import FakeTracer

# --------------------------------------------------------------------------- #
# ComponentObserver — the typed handle, unit level
# --------------------------------------------------------------------------- #


def test_observe_set_emits_declared_gauge() -> None:
    spec = MetricSpec(name="lifemodel_contact_drive_u", kind="gauge")
    reg = MetricRegistry()
    reg.register(spec)
    observer = ComponentObserver.bind(reg, (spec,))

    observer.set("lifemodel_contact_drive_u", 7.0)

    gauge = reg.get("lifemodel_contact_drive_u")
    assert gauge is not None
    assert gauge.value() == 7.0  # type: ignore[attr-defined]


def test_observe_inc_emits_declared_counter() -> None:
    spec = MetricSpec(name="lifemodel_widget_total", kind="counter", label_keys=("reason",))
    reg = MetricRegistry()
    reg.register(spec)
    observer = ComponentObserver.bind(reg, (spec,))

    observer.inc("lifemodel_widget_total", reason="x")
    observer.inc("lifemodel_widget_total", reason="x")

    counter = reg.get("lifemodel_widget_total")
    assert counter is not None
    assert counter.value(reason="x") == 2.0  # type: ignore[attr-defined]


def test_observe_observe_emits_declared_histogram() -> None:
    spec = MetricSpec(name="lifemodel_widget_seconds", kind="histogram")
    reg = MetricRegistry()
    reg.register(spec)
    observer = ComponentObserver.bind(reg, (spec,))

    observer.observe("lifemodel_widget_seconds", 0.2)

    hist = reg.get("lifemodel_widget_seconds")
    assert hist is not None
    assert hist.snapshot().count == 1  # type: ignore[attr-defined]


def test_observe_surface_accepts_bare_string_names() -> None:
    # A surface entry may be a bare metric NAME, not only a full spec.
    reg = MetricRegistry()
    reg.register(MetricSpec(name="lifemodel_b", kind="gauge"))
    observer = ComponentObserver.bind(reg, ("lifemodel_b",))

    observer.set("lifemodel_b", 3.0)

    gauge = reg.get("lifemodel_b")
    assert gauge is not None
    assert gauge.value() == 3.0  # type: ignore[attr-defined]


def test_observe_undeclared_name_is_noop_and_counts_error() -> None:
    # A metric REGISTERED in the registry but NOT in the component's declared
    # surface must not be emittable through observe: fail-open no-op + error bump.
    declared = MetricSpec(name="lifemodel_declared", kind="gauge")
    other = MetricSpec(name="lifemodel_other", kind="gauge")
    reg = MetricRegistry()
    reg.register(declared)
    reg.register(other)
    observer = ComponentObserver.bind(reg, (declared,))

    observer.set("lifemodel_other", 5.0)

    other_gauge = reg.get("lifemodel_other")
    assert other_gauge is not None
    assert other_gauge.value() == 0.0  # untouched  # type: ignore[attr-defined]
    errors = reg.get(EMIT_ERRORS_METRIC)
    assert errors is not None
    assert errors.value(reason=UNDECLARED_SURFACE) == 1.0  # type: ignore[attr-defined]


def test_observe_undeclared_never_raises_for_every_kind() -> None:
    # inc/set/observe on an undeclared name are all no-ops that never raise.
    reg = MetricRegistry()
    observer = ComponentObserver.bind(reg, ())
    observer.inc("nope")
    observer.set("nope", 1.0)
    observer.observe("nope", 1.0)
    errors = reg.get(EMIT_ERRORS_METRIC)
    assert errors is not None
    assert errors.value(reason=UNDECLARED_SURFACE) == 3.0  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# CoreLoop wiring + the first live domain example (SolitudeDrive → drive u)
# --------------------------------------------------------------------------- #


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class RecordingStore:
    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)

    def commit_tick(self, state: State | None, mutations: Sequence[object]) -> None:
        if state is not None:
            self.commit(state)


class Ticking:
    """A monotonic source that advances a fixed step on every read (dt > 0)."""

    def __init__(self, step: float = 0.01) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class RogueObserver:
    """A component that emits a metric it never declared in its surface."""

    id = "rogue"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        assert ctx.observe is not None  # the harness always wires one in-tick
        ctx.observe.set("lifemodel_not_declared", 1.0)
        return []


def _drive_manifest() -> ComponentManifest:
    return ComponentManifest(
        id="solitude-drive",
        type="drive",
        layer=layer_for_type("drive"),
        metric_surface=(CONTACT_DRIVE_U_SPEC,),
        accepts_signals=True,
    )


def _loop(registry: ComponentRegistry, store: RecordingStore, metrics: MetricRegistry) -> CoreLoop:
    return CoreLoop(
        registry=registry,
        state_actor=StateActor(store),
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        tracer=FakeTracer(),
        metrics=metrics,
        monotonic=Ticking(),
    )


def test_coreloop_wires_observer_and_drive_emits_contact_drive_u(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(SolitudeDrive(alpha=1.0 / 240.0, beta=1.0, u_max=100.0), _drive_manifest())
    metrics = MetricRegistry()
    # No ContactSensor → no contact_presence reading → the drive holds u; it still
    # publishes its computed level through ctx.observe. Start-of-tick u = 5.0.
    store = RecordingStore(State(u=5.0))

    _loop(reg, store, metrics).tick()

    gauge = metrics.get(CONTACT_DRIVE_U)
    assert gauge is not None
    assert gauge.value() == 5.0  # type: ignore[attr-defined]


def test_ctx_observe_undeclared_in_component_does_not_kill_tick(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(
        RogueObserver(),
        ComponentManifest(
            id="rogue", type="neuron", layer=layer_for_type("neuron"), metric_surface=()
        ),
    )
    metrics = MetricRegistry()

    report = _loop(reg, RecordingStore(), metrics).tick()

    # The tick survived and the rogue component ran to completion.
    assert report.ran == ("rogue",)
    errors = metrics.get(EMIT_ERRORS_METRIC)
    assert errors is not None
    assert errors.value(reason=UNDECLARED_SURFACE) == 1.0  # type: ignore[attr-defined]


def test_bare_context_has_no_observer_and_drive_step_survives(tmp_path) -> None:
    # A bare TickContext (no graph / no harness) carries observe=None; a component
    # guards on it and simply skips domain emission — the step still runs.
    drive = SolitudeDrive(alpha=1.0 / 240.0, beta=1.0, u_max=100.0)
    ctx = TickContext(
        state=State(u=3.0),
        now=datetime(2026, 7, 6, 12, 0, tzinfo=UTC),
        trace=FakeTracer().start_root(),
    )
    assert ctx.observe is None

    intents = drive.step(ctx)

    assert intents  # UpdateState + EmitSignal still produced, no crash

"""Tests for :class:`~lifemodel.core.intents.LaunchInternalCognition` (lm-705.6).

Mirrors ``LaunchProactive``'s own coreloop-collection test
(``tests/test_frame_acceptance.py``/``tests/test_coreloop.py``): a fake component
emits the intent, and the CoreLoop must surface it in its OWN report field
(``TickReport.internal_launches``) — never the proactive delivery channel
(``TickReport.launches``) — and it carries no State/memory mutation of its own
(only the ordinary tick bookkeeping the frame always commits).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from lifemodel.core.component import TickContext, layer_for_type
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.intents import Intent, LaunchInternalCognition, LaunchProactive
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.domain.memory import MemoryMutation
from lifemodel.state.model import State
from lifemodel.testing import FakeTracer

_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def test_launch_internal_cognition_carries_no_delivery_fields() -> None:
    intent = LaunchInternalCognition(
        prompt="notice something", correlation_id="internal-1", origin_traceparent=_ORIGIN_TP
    )
    assert isinstance(intent, Intent)
    assert intent.prompt == "notice something"
    assert intent.correlation_id == "internal-1"
    assert intent.origin_traceparent == _ORIGIN_TP
    # No reserved_energy / delivery-outcome field exists on this intent — the
    # internal path never touches the being's energy vitals nor the egress.
    assert not hasattr(intent, "reserved_energy")


def test_launch_internal_cognition_is_frozen() -> None:
    intent = LaunchInternalCognition(prompt="p", correlation_id="c", origin_traceparent=_ORIGIN_TP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.prompt = "changed"  # type: ignore[misc]


class RecordingStore:
    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)

    def reset(self) -> State:
        self._state = State()
        self.commits.append(self._state)
        return self._state

    def commit_tick(self, state: State | None, mutations: Sequence[MemoryMutation]) -> None:
        if state is not None:
            self.commit(state)


class InternalLauncher:
    """A fake component emitting a LaunchInternalCognition (lm-705.6)."""

    id = "internal-launcher"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [
            LaunchInternalCognition(
                prompt="notice", correlation_id="internal-1", origin_traceparent=_ORIGIN_TP
            )
        ]


class ProactiveLauncher:
    """A fake component emitting an (unrelated) LaunchProactive."""

    id = "proactive-launcher"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [LaunchProactive(prompt="hi", correlation_id="c-1", origin_traceparent=_ORIGIN_TP)]


def _loop(registry: ComponentRegistry, store: RecordingStore) -> CoreLoop:
    return CoreLoop(
        registry=registry,
        state_actor=StateActor(store),
        clock=_FixedClock(datetime(2026, 7, 16, 12, 0, tzinfo=UTC)),
        tracer=FakeTracer(),
    )


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


def test_launch_internal_cognition_surfaces_in_its_own_report_field() -> None:
    reg = ComponentRegistry()
    reg.register(
        InternalLauncher(),
        ComponentManifest(
            id="internal-launcher",
            type="cognition",
            layer=layer_for_type("cognition"),
            metric_surface=(),
        ),
    )
    store = RecordingStore()
    report = _loop(reg, store).tick()

    assert report.launches == ()  # NOT the proactive delivery channel
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]
    assert launch.correlation_id == "internal-1"
    assert launch.prompt == "notice"
    # No State mutation beyond the ordinary tick bookkeeping the frame always
    # commits (tick_count/last_tick_at) — the launch itself is not applied.
    assert store.commits[-1].tick_count == 1
    assert store.commits[-1].pending_internal_id is None


def test_launch_internal_cognition_and_launch_proactive_surface_independently() -> None:
    reg = ComponentRegistry()
    reg.register(
        ProactiveLauncher(),
        ComponentManifest(
            id="proactive-launcher",
            type="cognition",
            layer=layer_for_type("cognition"),
            metric_surface=(),
        ),
    )
    reg.register(
        InternalLauncher(),
        ComponentManifest(
            id="internal-launcher",
            type="cognition",
            layer=layer_for_type("cognition"),
            metric_surface=(),
        ),
    )
    report = _loop(reg, RecordingStore()).tick()

    assert len(report.launches) == 1
    assert report.launches[0].correlation_id == "c-1"
    assert len(report.internal_launches) == 1
    assert report.internal_launches[0].correlation_id == "internal-1"


def test_no_internal_launch_means_empty_tuple() -> None:
    report = _loop(ComponentRegistry(), RecordingStore()).tick()
    assert report.internal_launches == ()

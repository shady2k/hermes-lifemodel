# Core Rebuild — Phase A: Skeleton (Intents · State-Actor · Registry · CoreLoop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the layered-core *skeleton* — the single state-actor, the `Intent` mutation path, the self-registering component registry (DI), and the bulletproof `CoreLoop` scheduler with fault isolation — as an **isolated, fully-tested library that does NOT yet replace the live decision path**.

**Architecture:** New modules under `core/` implement the spec's mutation seam (`docs/superpowers/specs/2026-07-06-core-engine-design.md` §3, §6, §7, §15). Components (neurons/layers) return `Intent`s; **only** the `StateActor` mutates `State` and calls `StatePort.commit` (§2.3, §6). The `CoreLoop` runs enabled components each tick wrapped in fault isolation + circuit-breaker so a component fault can never crash the heart (§7.2). Phase A wires all of this through the composition root with an **empty registry** — the existing cron/egress/`decide_reachout` path is untouched, so the plugin keeps working and every current test stays green. The contact drive migrates *into* this skeleton in Phase B.

**Tech Stack:** Python 3.11 (stdlib only for the core — runtime deps are empty), frozen dataclasses, `typing.Protocol` (`@runtime_checkable`), `dataclasses.replace`. Tooling: `uv run ruff format/check`, `uv run mypy -p lifemodel` (strict), `uv run pytest`.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from the spec / repo conventions.

- **Flat root-layout package.** The repo dir *is* the `lifemodel` package. Import first-party as `from lifemodel.x import Y` in tests; use **relative** imports inside the package (e.g. `from ..state.model import State`), matching existing modules.
- **Core imports no Hermes.** Nothing under `core/` may import a Hermes module. Pure stdlib + intra-package.
- **Law — single state-actor (spec §2.3, §6):** "Модельное состояние мутирует **только** state-actor. Все остальные (нейроны, слои, хуки Hermes, крон) лишь кладут события/интенты в очередь. Никаких прямых записей." In Phase A the `StateActor` is the *only* new writer; do **not** remove existing `state.commit` callers yet (that cutover is Phase B). Do not add any new direct `StatePort.commit` call outside `StateActor`.
- **Law — layers don't mutate (spec §2.4):** components return `Intent`s; the actor applies them atomically at end of tick.
- **Law — CoreLoop неубиваем (spec §7):** every component call is wrapped; an exception skips that component (never propagates); repeated failures open a circuit-breaker ("живёт без органа"). A component fault must never abort a tick or prevent the checkpoint.
- **Law — всё числовое ограничено и воспроизводимо (spec §2.8):** no wall-clock/random inside pure logic; time comes from `ClockPort.now()`, injected.
- **Checkpoint is implicit-on-mutation (spec §6, §15):** the state-actor commits (checkpoints) atomically **iff** a mutation occurred this batch; a no-mutation batch performs **no** commit.
- **`mypy -p lifemodel` is strict.** Full type annotations; no `Any` leakage except the explicitly-typed `Mapping[str, Any]` state-patch payload.
- **Definition of done for the phase:** `make check` fully green (`ruff format --check`, `ruff check`, `mypy -p lifemodel`, `pytest`). No change to any existing test's expectations except the additive `LifeModel`-fields update in Task 6.

**Do NOT in Phase A:** touch `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, or `hooks.py`; wire `CoreLoop.tick()` into any live loop; add energy costing, neurons, aggregation, cognition, or UserModel (those are Phases B–D). The skeleton is exercised only by its own unit tests.

---

## File Structure

- Create `core/intents.py` — the sealed `Intent` value types (Task 1).
- Create `core/state_actor.py` — the single mutator + implicit checkpoint (Task 2).
- Create `core/component.py` — the `Component` protocol + `TickContext` (Task 3).
- Create `core/registry.py` — `ComponentManifest` + `ComponentRegistry` (enable/disable/order) (Task 4).
- Create `core/coreloop.py` — the scheduler: fault isolation, circuit-breaker, intent collection → state-actor (Task 5).
- Modify `composition.py` — wire registry + state-actor + coreloop into `LifeModel`/`build_lifemodel`, empty registry (Task 6).
- Modify `core/__init__.py` — re-export the new public names (Tasks 1–5, folded into each task's commit).
- Tests: `tests/test_intents.py`, `tests/test_state_actor.py`, `tests/test_component.py`, `tests/test_registry.py`, `tests/test_coreloop.py`; extend `tests/test_composition.py`.

**Interfaces produced by this phase (Phase B consumes these — exact names):**
- `Intent` (marker base); `UpdateState(changes: Mapping[str, Any])`, `EmitSignal(signal: Signal)`, `CheckpointState()`.
- `StateActor(store: StatePort, *, state: State | None = None, logger: EventLogger | None = None)` with `.state -> State` and `.apply(intents: Sequence[Intent]) -> State`.
- `TickContext(state: State, now: datetime, bus: SignalBus)`; `Component(Protocol)` with `id: str` and `step(ctx: TickContext) -> Sequence[Intent]`.
- `ComponentManifest(id, type, enabled=True, version="0.0.0", config={})`; `ComponentRegistry` with `register/enable/disable/enabled()/manifest()`.
- `CoreLoop(*, registry, state_actor, bus, clock, logger=None, breaker_threshold=3)` with `.tick() -> TickReport`; `TickReport(tick, ran, skipped_broken, failed, committed)`.

---

### Task 1: Intent value types

**Files:**
- Create: `core/intents.py`
- Modify: `core/__init__.py` (re-export)
- Test: `tests/test_intents.py`

**Interfaces:**
- Consumes: `lifemodel.domain.signal.Signal`.
- Produces: `Intent`, `UpdateState`, `EmitSignal`, `CheckpointState`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intents.py
from __future__ import annotations

import dataclasses

import pytest

from lifemodel.core.intents import CheckpointState, EmitSignal, Intent, UpdateState
from lifemodel.domain.signal import Signal


def test_update_state_carries_changes() -> None:
    intent = UpdateState({"u": 0.5, "tick_count": 3})
    assert isinstance(intent, Intent)
    assert intent.changes == {"u": 0.5, "tick_count": 3}


def test_update_state_is_frozen() -> None:
    intent = UpdateState({"u": 0.5})
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.changes = {"u": 0.9}  # type: ignore[misc]


def test_emit_signal_wraps_a_signal() -> None:
    sig = Signal(origin_id="n1", kind="contact")
    intent = EmitSignal(sig)
    assert isinstance(intent, Intent)
    assert intent.signal is sig


def test_checkpoint_state_is_a_marker_intent() -> None:
    intent = CheckpointState()
    assert isinstance(intent, Intent)
    assert CheckpointState() == CheckpointState()


def test_intents_are_equal_by_value() -> None:
    assert UpdateState({"u": 1.0}) == UpdateState({"u": 1.0})
    assert UpdateState({"u": 1.0}) != UpdateState({"u": 2.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_intents.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.intents'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/intents.py
"""Intents — the sole channel for state mutation (spec §6).

Layers, neurons and Hermes hooks never write state directly; they return (or
enqueue) `Intent`s, and the single :class:`~lifemodel.core.state_actor.StateActor`
applies them atomically at end of tick. Intents are immutable value objects.

Phase A defines the subset the skeleton actually routes: `UpdateState` (a
validated patch on :class:`~lifemodel.state.model.State`), `EmitSignal` (append
to the durable bus), and `CheckpointState` (the marker the actor emits for
observability when it commits). The energy / cognition / user-model intents
from spec §6 arrive in their own phases against this same base.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.signal import Signal


class Intent:
    """Marker base for every intent. Carries no fields of its own."""

    __slots__ = ()


@dataclass(frozen=True)
class UpdateState(Intent):
    """Patch the model state. ``changes`` maps ``State`` field names to new
    values; the state-actor validates the field names and applies the merge."""

    changes: Mapping[str, Any]


@dataclass(frozen=True)
class EmitSignal(Intent):
    """Append a signal to the durable bus (handled by the CoreLoop, not the
    state-actor — bus writes are immediate, state mutation is end-of-tick,
    spec §7.4)."""

    signal: Signal


@dataclass(frozen=True)
class CheckpointState(Intent):
    """Observability marker for a committed checkpoint. The state-actor emits
    it implicitly on mutation (spec §6); callers need not construct it."""
```

Then add to `core/__init__.py` (follow the existing re-export style): import `CheckpointState, EmitSignal, Intent, UpdateState` from `.intents` and add them to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_intents.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/intents.py core/__init__.py tests/test_intents.py
uv run ruff check core/intents.py core/__init__.py tests/test_intents.py
uv run mypy -p lifemodel
git add core/intents.py core/__init__.py tests/test_intents.py
git commit -m "feat(core): Intent value types — the sole state-mutation channel (spec §6)"
```

---

### Task 2: StateActor — the single mutator + implicit checkpoint

**Files:**
- Create: `core/state_actor.py`
- Modify: `core/__init__.py` (re-export)
- Test: `tests/test_state_actor.py`

**Interfaces:**
- Consumes: `State` (`state/model.py`), `StatePort` (`state/port.py`), `Intent`/`UpdateState`/`CheckpointState` (Task 1), `EventLogger` (`log.py`).
- Produces: `StateActor`, `UnknownStateField`.

**Behavior (spec §6, §7.1, §15):**
- Loads `State` once (or accepts an injected initial `State`) and owns it in memory.
- `apply(intents)` merges every `UpdateState.changes` into one patch, validates each field name against `State`'s dataclass fields (unknown → `UnknownStateField`), and applies via `dataclasses.replace`.
- Commits (checkpoints) **exactly once, iff** the merged patch is non-empty; a no-op batch does **not** touch the store. On commit, emits a `state_checkpoint` event with a monotonic `checkpoint_id`.
- Ignores intents it doesn't own (`EmitSignal` etc.) — those are handled by the CoreLoop.
- Validation is all-or-nothing: an unknown field raises **before** any commit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_actor.py
from __future__ import annotations

import pytest

from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.state_actor import StateActor, UnknownStateField
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


class RecordingStore:
    """Minimal StatePort double that counts commits."""

    def __init__(self, initial: State | None = None) -> None:
        self._state = initial if initial is not None else State()
        self.commits: list[State] = []

    def load(self) -> State:
        return self._state

    def commit(self, state: State) -> None:
        self._state = state
        self.commits.append(state)


def test_apply_merges_updates_and_commits_once() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    result = actor.apply([UpdateState({"u": 0.5}), UpdateState({"tick_count": 2})])
    assert result.u == 0.5
    assert result.tick_count == 2
    assert actor.state is result
    assert len(store.commits) == 1


def test_apply_without_state_changes_does_not_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    before = actor.state
    result = actor.apply([EmitSignal(Signal(origin_id="n1", kind="contact"))])
    assert result is before
    assert store.commits == []


def test_apply_empty_batch_does_not_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    result = actor.apply([])
    assert result is actor.state
    assert store.commits == []


def test_unknown_field_raises_before_commit() -> None:
    store = RecordingStore()
    actor = StateActor(store)
    with pytest.raises(UnknownStateField):
        actor.apply([UpdateState({"u": 0.5}), UpdateState({"not_a_field": 1})])
    assert store.commits == []  # all-or-nothing: nothing committed


def test_actor_loads_initial_state_from_store() -> None:
    store = RecordingStore(State(u=0.9, tick_count=7))
    actor = StateActor(store)
    assert actor.state.u == 0.9
    assert actor.state.tick_count == 7


def test_injected_state_overrides_store_load() -> None:
    store = RecordingStore(State(u=0.1))
    actor = StateActor(store, state=State(u=0.4))
    assert actor.state.u == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_state_actor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.state_actor'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/state_actor.py
"""StateActor — the single owner of model-state mutation (spec §6, §7.1, §15).

Every other producer (neurons, layers, Hermes hooks, cron) only *returns* or
*enqueues* intents; this is the one place that mutates :class:`State` and calls
:meth:`StatePort.commit`. It merges a batch of :class:`UpdateState` intents into
one patch, applies it atomically with :func:`dataclasses.replace`, and commits
(checkpoints) exactly once — and only if something actually changed (spec §6:
"Checkpoint — это интент, который state-actor генерирует сам в конце тика, если
были мутации"). Intents it does not own (e.g. ``EmitSignal``) are ignored here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from ..log import EventLogger
from ..state.model import State
from ..state.port import StatePort
from .intents import Intent, UpdateState

_STATE_FIELDS = frozenset(f.name for f in fields(State))


class UnknownStateField(KeyError):
    """An ``UpdateState`` intent named a field that ``State`` does not declare."""


class StateActor:
    def __init__(
        self,
        store: StatePort,
        *,
        state: State | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self._store = store
        self._state = state if state is not None else store.load()
        self._log = logger
        self._checkpoint_id = 0

    @property
    def state(self) -> State:
        """The current in-memory state (last committed, or the initial load)."""
        return self._state

    def apply(self, intents: Sequence[Intent]) -> State:
        """Apply a batch atomically. Commits once iff the merged patch is
        non-empty; validates all field names *before* committing."""
        patch: dict[str, Any] = {}
        for intent in intents:
            if isinstance(intent, UpdateState):
                for name, value in intent.changes.items():
                    if name not in _STATE_FIELDS:
                        raise UnknownStateField(name)
                    patch[name] = value
        if not patch:
            return self._state

        new_state = replace(self._state, **patch)
        self._store.commit(new_state)
        self._state = new_state
        self._checkpoint_id += 1
        if self._log is not None:
            self._log.info(
                "state_checkpoint",
                checkpoint_id=self._checkpoint_id,
                fields=sorted(patch),
            )
        return new_state
```

Then re-export `StateActor, UnknownStateField` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_state_actor.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/state_actor.py core/__init__.py tests/test_state_actor.py
uv run ruff check core/state_actor.py core/__init__.py tests/test_state_actor.py
uv run mypy -p lifemodel
git add core/state_actor.py core/__init__.py tests/test_state_actor.py
git commit -m "feat(core): StateActor — sole state mutator, implicit checkpoint-on-mutation (spec §6/§7.1)"
```

---

### Task 3: Component protocol + TickContext

**Files:**
- Create: `core/component.py`
- Modify: `core/__init__.py` (re-export)
- Test: `tests/test_component.py`

**Interfaces:**
- Consumes: `State`, `SignalBus` (`core/signal_bus.py`), `Intent` (Task 1), `datetime`.
- Produces: `TickContext`, `Component`.

**Design note:** this is the unified per-tick seam that neurons (Phase B), aggregation (Phase B), personality (Phase C) and cognition (Phase D) all implement. It is intentionally narrower than the legacy `core/neuron.py::Neuron.tick(state) -> list[Signal]`; the legacy ABCs stay untouched and are migrated onto this seam in Phase B.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_component.py
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import Component, TickContext
from lifemodel.core.intents import Intent, UpdateState
from lifemodel.state.model import State


class Ticker:
    id = "ticker"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [UpdateState({"tick_count": ctx.state.tick_count + 1})]


def test_tick_context_exposes_state_now_bus(tmp_path) -> None:
    bus = FileSignalBus(tmp_path)
    now = datetime(2026, 7, 6, tzinfo=UTC)
    ctx = TickContext(state=State(tick_count=4), now=now, bus=bus)
    assert ctx.state.tick_count == 4
    assert ctx.now is now
    assert ctx.bus is bus


def test_component_protocol_is_satisfied_structurally(tmp_path) -> None:
    ticker = Ticker()
    assert isinstance(ticker, Component)
    ctx = TickContext(state=State(tick_count=4), now=datetime(2026, 7, 6, tzinfo=UTC), bus=FileSignalBus(tmp_path))
    (intent,) = ticker.step(ctx)
    assert isinstance(intent, UpdateState)
    assert intent.changes == {"tick_count": 5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_component.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.component'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/component.py
"""The per-tick component seam (spec §3, §7.2).

A component is anything the CoreLoop schedules on a tick — a neuron, an
aggregation stage, the personality, cognition. It reads an immutable
:class:`TickContext` (state snapshot + clock + bus) and returns intents; it
never mutates state. Kept deliberately minimal so every layer can implement it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..state.model import State
from .intents import Intent
from .signal_bus import SignalBus


@dataclass(frozen=True)
class TickContext:
    """Read-only inputs handed to every component on a tick."""

    state: State
    now: datetime
    bus: SignalBus


@runtime_checkable
class Component(Protocol):
    """A schedulable unit. ``id`` is stable and unique within a registry."""

    id: str

    def step(self, ctx: TickContext) -> Sequence[Intent]: ...
```

Then re-export `Component, TickContext` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_component.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/component.py core/__init__.py tests/test_component.py
uv run ruff check core/component.py core/__init__.py tests/test_component.py
uv run mypy -p lifemodel
git add core/component.py core/__init__.py tests/test_component.py
git commit -m "feat(core): Component protocol + TickContext — unified per-tick seam (spec §3/§7.2)"
```

---

### Task 4: ComponentRegistry + manifest (self-registration, enable/disable)

**Files:**
- Create: `core/registry.py`
- Modify: `core/__init__.py` (re-export)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `Component` (Task 3).
- Produces: `ComponentManifest`, `ComponentRegistry`, `DuplicateComponent`, `UnknownComponent`.

**Behavior (spec §15 plugin seam):** components self-register via a DI callback from the composition root; each carries a manifest `{id, type, enabled, version, config}`; the registry can enable/disable by id and yields the enabled components **in registration order** (stable scheduling). External discovery/loading is deferred — this is registration only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/registry.py
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
        return tuple(
            self._components[cid] for cid in self._order if self._manifests[cid].enabled
        )
```

Then re-export `ComponentManifest, ComponentRegistry, DuplicateComponent, UnknownComponent` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/registry.py core/__init__.py tests/test_registry.py
uv run ruff check core/registry.py core/__init__.py tests/test_registry.py
uv run mypy -p lifemodel
git add core/registry.py core/__init__.py tests/test_registry.py
git commit -m "feat(core): ComponentRegistry + manifest — self-registration, enable/disable (spec §15)"
```

---

### Task 5: CoreLoop — scheduler with fault isolation + circuit-breaker

**Files:**
- Create: `core/coreloop.py`
- Modify: `core/__init__.py` (re-export)
- Test: `tests/test_coreloop.py`

**Interfaces:**
- Consumes: `ComponentRegistry` (Task 4), `StateActor` (Task 2), `SignalBus` (`core/signal_bus.py`), `ClockPort` (`ports/clock.py`), `EventLogger`, `Intent`/`UpdateState`/`EmitSignal` (Task 1), `TickContext` (Task 3).
- Produces: `CoreLoop`, `TickReport`.

**Behavior (spec §7):**
1. Read `now = clock.now()` and the actor's current `state`; build one `TickContext`.
2. For each enabled component **not** already broken: call `step(ctx)` inside a `try/except Exception`.
   - On exception: increment its consecutive-failure counter, emit `component_failed`, and if the counter reaches `breaker_threshold`, add it to `broken` and emit `circuit_breaker_open`. **Never re-raise.**
   - On success: reset its failure counter; **publish** each returned `EmitSignal.signal` to the bus immediately (spec §7.4); collect the remaining intents.
3. Append the tick's own bookkeeping `UpdateState({"tick_count": state.tick_count + 1, "last_tick_at": now.isoformat()})`.
4. `state_actor.apply(collected_intents)` — one atomic checkpoint.
5. Return a `TickReport`. A component fault must not abort the tick nor prevent the checkpoint.

**Note:** energy budgeting (which gates the *expensive* layer) is **not** in Phase A — every enabled component runs each tick. The energy gate arrives in Phase C and slots in at step 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coreloop.py
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.coreloop import CoreLoop, TickReport
from lifemodel.core.intents import EmitSignal, Intent, UpdateState
from lifemodel.core.registry import ComponentManifest, ComponentRegistry
from lifemodel.core.state_actor import StateActor
from lifemodel.domain.signal import Signal
from lifemodel.state.model import State


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


class Healthy:
    id = "healthy"

    def __init__(self) -> None:
        self.calls = 0

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.calls += 1
        return [UpdateState({"u": 0.42})]


class Emitter:
    id = "emitter"

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        return [EmitSignal(Signal(origin_id="emitter-1", kind="contact"))]


class Broken:
    id = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        self.calls += 1
        raise RuntimeError("boom")


def _loop(registry: ComponentRegistry, store: RecordingStore, bus: FileSignalBus, *, breaker_threshold: int = 3) -> CoreLoop:
    return CoreLoop(
        registry=registry,
        state_actor=StateActor(store),
        bus=bus,
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        breaker_threshold=breaker_threshold,
    )


def test_healthy_component_intents_reach_state_and_tick_bumps(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(Healthy(), ComponentManifest(id="healthy", type="neuron"))
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    report = loop.tick()
    assert isinstance(report, TickReport)
    assert store.commits[-1].u == 0.42
    assert store.commits[-1].tick_count == 1
    assert store.commits[-1].last_tick_at is not None
    assert report.ran == ("healthy",)


def test_emit_signal_is_published_to_bus(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(Emitter(), ComponentManifest(id="emitter", type="neuron"))
    bus = FileSignalBus(tmp_path)
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    published = bus.peek_unprocessed()
    assert [s.origin_id for s in published] == ["emitter-1"]


def test_failing_component_is_isolated_and_others_still_run(tmp_path) -> None:
    reg = ComponentRegistry()
    healthy = Healthy()
    reg.register(Broken(), ComponentManifest(id="broken", type="neuron"))
    reg.register(healthy, ComponentManifest(id="healthy", type="neuron"))
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    report = loop.tick()  # must not raise
    assert healthy.calls == 1
    assert store.commits[-1].u == 0.42  # tick still checkpointed
    assert store.commits[-1].tick_count == 1
    assert "broken" in report.failed


def test_repeated_failures_open_breaker_and_skip_component(tmp_path) -> None:
    reg = ComponentRegistry()
    broken = Broken()
    reg.register(broken, ComponentManifest(id="broken", type="neuron"))
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path), breaker_threshold=3)
    for _ in range(3):
        loop.tick()
    assert broken.calls == 3  # tripped after the 3rd failure
    report = loop.tick()
    assert broken.calls == 3  # not called again — breaker open
    assert "broken" in report.skipped_broken


def test_tick_count_increments_each_tick(tmp_path) -> None:
    reg = ComponentRegistry()
    store = RecordingStore()
    loop = _loop(reg, store, FileSignalBus(tmp_path))
    loop.tick()
    loop.tick()
    assert store.commits[-1].tick_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.coreloop'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/coreloop.py
"""CoreLoop — the heart/scheduler (spec §7).

Runs the enabled components each tick, isolated so no component fault can crash
the heart: every ``step`` call is wrapped; an exception skips that component and
counts toward a per-component circuit-breaker ("живёт без органа"). Successful
components' signals are published to the durable bus immediately (spec §7.4);
their state intents are collected and handed — together with the tick's own
bookkeeping — to the single :class:`StateActor` for one atomic checkpoint.

Phase A runs *every* enabled component each tick. Energy budgeting (which gates
the expensive cognition layer) slots into the per-component loop in Phase C.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..log import EventLogger
from ..ports.clock import ClockPort
from .component import TickContext
from .intents import EmitSignal, Intent, UpdateState
from .registry import ComponentRegistry
from .signal_bus import SignalBus
from .state_actor import StateActor


@dataclass(frozen=True)
class TickReport:
    """What happened on one tick — for observability/tests."""

    tick: int
    ran: tuple[str, ...]
    skipped_broken: tuple[str, ...]
    failed: tuple[str, ...]
    committed: bool


class CoreLoop:
    def __init__(
        self,
        *,
        registry: ComponentRegistry,
        state_actor: StateActor,
        bus: SignalBus,
        clock: ClockPort,
        logger: EventLogger | None = None,
        breaker_threshold: int = 3,
    ) -> None:
        self._registry = registry
        self._state_actor = state_actor
        self._bus = bus
        self._clock = clock
        self._log = logger
        self._breaker_threshold = breaker_threshold
        self._failures: dict[str, int] = {}
        self._broken: set[str] = set()

    def tick(self) -> TickReport:
        now = self._clock.now()
        state = self._state_actor.state
        ctx = TickContext(state=state, now=now, bus=self._bus)

        intents: list[Intent] = []
        ran: list[str] = []
        failed: list[str] = []

        for component in self._registry.enabled():
            if component.id in self._broken:
                continue
            try:
                produced = component.step(ctx)
            except Exception as exc:  # isolation: the heart never dies
                self._record_failure(component.id, exc)
                failed.append(component.id)
                continue
            self._failures[component.id] = 0
            for intent in produced:
                if isinstance(intent, EmitSignal):
                    self._bus.publish(intent.signal)
                else:
                    intents.append(intent)
            ran.append(component.id)

        intents.append(
            UpdateState({"tick_count": state.tick_count + 1, "last_tick_at": now.isoformat()})
        )
        new_state = self._state_actor.apply(intents)

        return TickReport(
            tick=new_state.tick_count,
            ran=tuple(ran),
            skipped_broken=tuple(sorted(self._broken)),
            failed=tuple(failed),
            committed=new_state is not state,
        )

    def _record_failure(self, component_id: str, exc: Exception) -> None:
        count = self._failures.get(component_id, 0) + 1
        self._failures[component_id] = count
        if self._log is not None:
            self._log.info("component_failed", component=component_id, error=repr(exc), consecutive=count)
        if count >= self._breaker_threshold and component_id not in self._broken:
            self._broken.add(component_id)
            if self._log is not None:
                self._log.info("circuit_breaker_open", component=component_id, after=count)
```

Then re-export `CoreLoop, TickReport` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/coreloop.py core/__init__.py tests/test_coreloop.py
uv run ruff check core/coreloop.py core/__init__.py tests/test_coreloop.py
uv run mypy -p lifemodel
git add core/coreloop.py core/__init__.py tests/test_coreloop.py
git commit -m "feat(core): CoreLoop scheduler — fault isolation + circuit-breaker (spec §7)"
```

---

### Task 6: Wire the skeleton through the composition root (empty registry, no live cutover)

**Files:**
- Modify: `composition.py`
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `LifeModel.registry`, `LifeModel.state_actor`, `LifeModel.coreloop`; `build_lifemodel(..., registry=None)` param.

**Behavior:** `build_lifemodel` additionally constructs a `ComponentRegistry` (empty by default — no neurons in Phase A), a `StateActor` over the same `StatePort`, and a `CoreLoop` wired to them. These become optional frozen fields on `LifeModel`. **Nothing calls `coreloop.tick()` on any live path** — the cron/egress path is untouched — so behavior is unchanged and all existing tests keep passing. This proves the DI seam end-to-end and gives Phase B a `CoreLoop` to populate and cut over to.

Read `composition.py` first (it is the frozen `LifeModel` dataclass + `build_lifemodel(*, base_dir, state, bus, clock, delivery, aggregator, neurons, logger)` factory). Add fields/params consistently with the existing `None`-means-default style.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_composition.py
from lifemodel.core.coreloop import CoreLoop
from lifemodel.core.registry import ComponentRegistry
from lifemodel.core.state_actor import StateActor


def test_build_wires_registry_state_actor_and_coreloop(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    assert isinstance(lm.registry, ComponentRegistry)
    assert isinstance(lm.state_actor, StateActor)
    assert isinstance(lm.coreloop, CoreLoop)


def test_default_registry_is_empty(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    assert lm.registry.enabled() == ()


def test_coreloop_tick_is_inert_but_bookkeeps(tmp_path) -> None:
    # Empty registry: a tick runs no components but still checkpoints the
    # bookkeeping bump — proves the wired seam works without touching the
    # live path.
    lm = build_lifemodel(base_dir=tmp_path)
    report = lm.coreloop.tick()
    assert report.ran == ()
    assert lm.state.load().tick_count == 1
```

(Use the existing `build_lifemodel` import already at the top of `tests/test_composition.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py -q`
Expected: FAIL — `AttributeError: 'LifeModel' object has no attribute 'registry'` (and the new tests error).

- [ ] **Step 3: Write minimal implementation**

In `composition.py`:

1. Add imports:
```python
from .core.coreloop import CoreLoop
from .core.registry import ComponentRegistry
from .core.state_actor import StateActor
```

2. Add three fields to the frozen `LifeModel` dataclass (after `neurons`), each defaulted so construction stays backward-compatible:
```python
    registry: ComponentRegistry = field(default_factory=ComponentRegistry)
    state_actor: StateActor | None = None
    coreloop: CoreLoop | None = None
```
(Import `field` from `dataclasses` if not already imported. `state_actor`/`coreloop` are typed optional only to keep the dataclass default-constructible; `build_lifemodel` always populates them — see below.)

3. In `build_lifemodel`, after the existing collaborators are resolved (state/bus/clock/delivery/aggregator/neurons) and before constructing `LifeModel`, build the skeleton:
```python
    registry = registry if registry is not None else ComponentRegistry()
    state_actor = StateActor(state, logger=logger)
    coreloop = CoreLoop(
        registry=registry,
        state_actor=state_actor,
        bus=bus,
        clock=clock,
        logger=logger,
    )
```
and add `registry: ComponentRegistry | None = None` to the `build_lifemodel` signature (mirroring the other optional collaborators), then pass `registry=registry, state_actor=state_actor, coreloop=coreloop` into the `LifeModel(...)` constructor.

- [ ] **Step 4: Run the full suite to verify green (nothing regressed)**

Run: `uv run pytest -q`
Expected: PASS — the three new composition tests pass and every pre-existing test still passes (the live path is unchanged).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format composition.py tests/test_composition.py
uv run ruff check composition.py tests/test_composition.py
uv run mypy -p lifemodel
git add composition.py tests/test_composition.py
git commit -m "feat(core): wire registry/state-actor/CoreLoop through composition root (empty registry, no cutover)"
```

---

## Phase-A Definition of Done

- [ ] `make check` fully green — paste the tail (ruff format check, ruff check, `mypy -p lifemodel` success, pytest summary with the new counts).
- [ ] Six commits on the phase branch (one per task).
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `CoreLoop.tick()` is reachable via `build_lifemodel(...).coreloop` but is **not** invoked by any live loop.
- [ ] Do **not** push, merge, or touch `main`. Work stays on the phase branch. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` when done (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §6 Intents → Task 1; §6/§7.1/§15 single state-actor + implicit checkpoint → Task 2; §3/§7.2 component seam → Task 3; §15 plugin registration/enable-disable → Task 4; §7 CoreLoop fault isolation + circuit-breaker → Task 5; §7.5/DI wiring → Task 6. **Deferred to later phases (by design, per §19):** durable-bus consumed-offset-in-blob & `PRUNE_BUS` (§15) → Phase D; energy gate in the scheduler (§8) → Phase C; neurons/aggregation (§11/§12) → Phase B; async cognition (§7.3), UserModel (§10), backstop (§14), tick-discipline dt-split & dithering (§17), observability API (§16) → Phases C/D. Phase A is deliberately the inert skeleton.
- **Type consistency:** `StateActor(store, *, state=None, logger=None)` / `.apply(Sequence[Intent]) -> State` / `.state` used identically in Tasks 2, 5, 6. `CoreLoop(*, registry, state_actor, bus, clock, logger=None, breaker_threshold=3)` / `.tick() -> TickReport` identical in Tasks 5, 6. `Component.step(ctx) -> Sequence[Intent]` identical in Tasks 3, 4, 5. `ComponentManifest(id, type, enabled, version, config)` identical in Tasks 4, 5, 6.
- **No placeholders:** every step ships real code and an exact command with expected output.

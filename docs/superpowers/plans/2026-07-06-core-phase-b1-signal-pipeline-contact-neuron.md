# Core Rebuild — Phase B1: Signal Pipeline + AUTONOMIC Contact Neuron Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase-A CoreLoop into a real **dataflow pipeline** and add the first AUTONOMIC neuron — `ContactNeuron` — which owns the contact drive `u` (rise in silence, satiate on a real exchange) and emits a transient `contact` signal for the aggregation layer to read. Still **no live cutover**: `core/decision.py` and the cron/egress path are untouched; the new components run only under the CoreLoop, exercised by their own tests.

**Architecture:** Phase A left `CoreLoop` publishing `EmitSignal` straight to the durable bus. Phase B1 corrects the signal model into two distinct roles (spec §3, §7.4, §15 blackboard):
- **Durable external inputs** (`exchange`, later `verdict`) are written to the `SignalBus` by producers (Hermes hooks in Phase E; tests here). The CoreLoop **consumes** the bus once at tick start and threads those signals into the tick.
- **Transient intra-tick signals** (`contact`, neuron→aggregation) are threaded through the `TickContext` to *later* components in the **same** tick and are **never persisted** — they are recomputed from state every tick, so persisting them would re-feed them next tick and double-count.

`ContactNeuron` is a dumb sensor (spec §2.1, §11): it measures deprivation (`Drive.rise`), resets on a genuine exchange (`Drive.satiate`), and emits raw `{value, delta}`. It computes **no** salience, thresholds, or gates — those belong to AGGREGATION (Phase B2).

**Tech Stack:** Python 3.11 stdlib-only core; frozen dataclasses; the certified `sim.drive.Drive` / `sim.quality.quality_of` primitives (reused, never reimplemented). Tooling: `uv run ruff format/check`, `uv run mypy -p lifemodel` (strict), `uv run pytest`.

## Global Constraints

- **Flat root-layout package:** tests import `from lifemodel.x import Y`; package-internal code uses **relative** imports (`from ..sim.drive import Drive`).
- **Core imports no Hermes.** Pure stdlib + intra-package.
- **Reuse the certified sim, do not reimplement it.** The drive math is `lifemodel.sim.drive.Drive` (`rise(*, dt)`, `satiate(*, q)`); exchange quality is `lifemodel.sim.quality.quality_of(*, actor, label)`. Bootstrap constants match `core/decision.py`: `alpha = 1/240`, `beta = 1.0`, `u_max = 100.0` (θ=1.0 is θ_u, used only in B2).
- **Single writer per state field (spec §6):** the `StateActor` (Phase A) is the sole mutator; components only return intents. In B1, `ContactNeuron` is the **sole writer of `u`** (rise+satiate). Do not have any other component write `u`.
- **Law — layers don't mutate; CoreLoop неубиваем; всё числовое ограничено** (spec §2). Time comes from `ClockPort.now()`; no wall-clock/random in pure logic.
- **`mypy -p lifemodel` strict.** Full annotations.
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. All existing tests (378) must still pass, except the **one** Phase-A `EmitSignal` test that Task 2 deliberately re-specs (documented there).
- **Branch:** work on `core/rebuild` (already checked out). One commit per task.

## File Structure

- Create `core/taxonomy.py` — signal-kind constants + typed payload builders/readers (Task 1).
- Modify `core/component.py` — add `signals` to `TickContext` (Task 2).
- Modify `core/coreloop.py` — consume-at-tick-start + transient intra-tick threading (Task 2).
- Modify `tests/test_coreloop.py` — re-spec the `EmitSignal` behavior test (Task 2).
- Create `core/timeutil.py` — the defensive `minutes_between` clock helper (Task 3).
- Create `core/contact_neuron.py` — the `ContactNeuron` component (Task 3).
- Modify `composition.py` — register `ContactNeuron` as an enabled component; expose the drive constants (Task 4).
- Modify `core/__init__.py` — re-export new public names (each task).
- Tests: `tests/test_taxonomy.py`, `tests/test_timeutil.py`, `tests/test_contact_neuron.py`, extend `tests/test_coreloop.py`, `tests/test_composition.py`.

**Interfaces produced (Phase B2 consumes these):**
- `core/taxonomy.py`: `KIND_CONTACT = "contact"`, `KIND_EXCHANGE = "exchange"`; `contact_signal(*, origin_id, value, delta, timestamp) -> Signal`; `exchange_signal(*, origin_id, actor, label, timestamp) -> Signal`; `read_exchange(signal) -> tuple[Actor, Label]`; `is_kind(signal, kind) -> bool`.
- `core/component.py`: `TickContext(state, now, bus, signals: tuple[Signal, ...] = ())`.
- `core/coreloop.py`: unchanged public surface (`CoreLoop.tick() -> TickReport`), new internal dataflow.
- `core/timeutil.py`: `minutes_between(a_iso: str | None, b: datetime) -> float`.
- `core/contact_neuron.py`: `ContactNeuron(*, alpha, beta, u_max, id="contact")` implementing `Component`; consumes `KIND_EXCHANGE` from `ctx.signals`, emits `KIND_CONTACT`, writes `u`.

---

### Task 1: Signal taxonomy (contact + exchange kinds)

**Files:**
- Create: `core/taxonomy.py`
- Modify: `core/__init__.py`
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Consumes: `lifemodel.domain.signal.Signal`, `lifemodel.sim.quality.{Actor, Label}`.
- Produces: `KIND_CONTACT`, `KIND_EXCHANGE`, `contact_signal`, `exchange_signal`, `read_exchange`, `is_kind`.

**Behavior (spec §4):** a small typed vocabulary for the two Phase-B1 signal kinds. `contact` is the neuron's transient output (unipolar drive value + delta). `exchange` is a durable external input describing a real lane event (actor + label, per `sim.quality`). Builders keep payloads JSON-native and consistent; readers validate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_taxonomy.py
from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_CONTACT,
    KIND_EXCHANGE,
    contact_signal,
    exchange_signal,
    is_kind,
    read_exchange,
)


def test_contact_signal_carries_value_and_delta() -> None:
    sig = contact_signal(origin_id="c-1", value=1.25, delta=0.02, timestamp="2026-07-06T00:00:00+00:00")
    assert sig.kind == KIND_CONTACT
    assert sig.origin_id == "c-1"
    assert sig.payload["value"] == 1.25
    assert sig.payload["delta"] == 0.02
    assert is_kind(sig, KIND_CONTACT)
    assert not is_kind(sig, KIND_EXCHANGE)


def test_exchange_signal_roundtrips_actor_label() -> None:
    sig = exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)
    assert sig.kind == KIND_EXCHANGE
    assert read_exchange(sig) == ("user", "two_way")


def test_read_exchange_rejects_wrong_kind() -> None:
    sig = contact_signal(origin_id="c-2", value=0.0, delta=0.0, timestamp=None)
    with pytest.raises(ValueError):
        read_exchange(sig)


def test_read_exchange_rejects_bad_payload() -> None:
    from lifemodel.domain.signal import Signal

    sig = Signal(origin_id="e-2", kind=KIND_EXCHANGE, payload={"actor": "user"})  # missing label
    with pytest.raises(ValueError):
        read_exchange(sig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.taxonomy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/taxonomy.py
"""Signal taxonomy — the typed vocabulary of the pipeline (spec §4).

Phase B1 defines two kinds:
- ``contact`` — the neuron's *transient* intra-tick output: the unipolar drive
  value ``[0..u_max]`` plus its per-tick ``delta``. Never persisted.
- ``exchange`` — a *durable* external input: a real lane event (actor + label,
  per :mod:`lifemodel.sim.quality`) that the neuron reads to satiate the drive.

Builders keep payloads JSON-native and uniform; readers validate on the way out.
"""

from __future__ import annotations

from typing import cast

from ..domain.signal import Signal
from ..sim.quality import Actor, Label

KIND_CONTACT = "contact"
KIND_EXCHANGE = "exchange"

_ACTORS: frozenset[str] = frozenset({"user", "assistant", "proactive_internal"})
_LABELS: frozenset[str] = frozenset({"two_way", "ack", "monologue", "rejection"})


def is_kind(signal: Signal, kind: str) -> bool:
    return signal.kind == kind


def contact_signal(*, origin_id: str, value: float, delta: float, timestamp: str | None) -> Signal:
    """Build a transient contact signal carrying the drive value and its delta."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_CONTACT,
        payload={"value": float(value), "delta": float(delta)},
        timestamp=timestamp,
    )


def exchange_signal(*, origin_id: str, actor: Actor, label: Label, timestamp: str | None) -> Signal:
    """Build a durable exchange-input signal from a lane event."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_EXCHANGE,
        payload={"actor": actor, "label": label},
        timestamp=timestamp,
    )


def read_exchange(signal: Signal) -> tuple[Actor, Label]:
    """Validate and extract ``(actor, label)`` from an exchange signal."""
    if signal.kind != KIND_EXCHANGE:
        raise ValueError(f"not an exchange signal: kind={signal.kind!r}")
    actor = signal.payload.get("actor")
    label = signal.payload.get("label")
    if actor not in _ACTORS or label not in _LABELS:
        raise ValueError(f"invalid exchange payload: {signal.payload!r}")
    return cast(Actor, actor), cast(Label, label)
```

Re-export `KIND_CONTACT, KIND_EXCHANGE, contact_signal, exchange_signal, is_kind, read_exchange` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run ruff check core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run mypy -p lifemodel
git add core/taxonomy.py core/__init__.py tests/test_taxonomy.py
git commit -m "feat(core): signal taxonomy — contact (transient) + exchange (durable) kinds (spec §4)"
```

---

### Task 2: CoreLoop dataflow — consume-at-tick-start + transient intra-tick threading

**Files:**
- Modify: `core/component.py` (add `signals` to `TickContext`)
- Modify: `core/coreloop.py` (dataflow)
- Modify: `tests/test_coreloop.py` (re-spec EmitSignal behavior + add dataflow tests)
- Test: `tests/test_coreloop.py`

**Interfaces:**
- Consumes: `SignalBus.consume_unprocessed()` (`adapters/signal_bus.py` / `core/signal_bus.py`), `TickContext`, `EmitSignal`.
- Produces: `TickContext(state, now, bus, signals=())`; new CoreLoop dataflow.

**Behavior (spec §3, §7.4, §15):**
1. At tick start, `inbound = bus.consume_unprocessed()` — the durable external inputs since last tick. Seed a mutable `available: list[Signal] = list(inbound)`.
2. For each enabled, non-broken component: build `TickContext(state=state, now=now, bus=bus, signals=tuple(available))` and call `step`. This means a component sees the inbound inputs **and** every transient signal emitted by earlier components this tick.
3. On success, for each returned `EmitSignal(sig)`: **append `sig` to `available`** so later components in this tick see it — do **NOT** publish it to the durable bus (transient blackboard, spec §7.4). Collect all non-`EmitSignal` intents.
4. Bookkeeping `UpdateState({tick_count, last_tick_at})` and `state_actor.apply(...)` as before.

**Why the change from Phase A:** Phase A published `EmitSignal` to the durable bus. A `contact` signal recomputed every tick would then be re-consumed next tick and double-counted. Transient threading fixes this; durable traffic is reserved for external inputs written by producers (hooks/tests) directly to the bus.

- [ ] **Step 1: Update `TickContext` (add `signals`)**

In `core/component.py`, add a `signals` field to `TickContext` (default empty so existing constructions keep working):

```python
@dataclass(frozen=True)
class TickContext:
    """Read-only inputs handed to every component on a tick."""

    state: State
    now: datetime
    bus: SignalBus
    signals: tuple[Signal, ...] = ()
```

Add the import `from ..domain.signal import Signal` to `core/component.py`.

- [ ] **Step 2: Write the failing tests (re-spec EmitSignal + dataflow)**

In `tests/test_coreloop.py`: **replace** the existing `test_emit_signal_is_published_to_bus` test body with the transient-threading spec below, and **add** the two new dataflow tests. (Keep all other Phase-A coreloop tests unchanged.)

```python
# --- Phase B1: signal dataflow ---
from collections.abc import Sequence as _Seq

from lifemodel.core.component import TickContext as _TickContext
from lifemodel.core.taxonomy import KIND_CONTACT, contact_signal
from lifemodel.domain.signal import Signal as _Signal


class SeenRecorder:
    """Records what signals it saw in ctx.signals."""

    id = "seen"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def step(self, ctx: _TickContext) -> _Seq[Intent]:
        self.seen = [s.origin_id for s in ctx.signals]
        return []


class ContactEmitter:
    id = "emitter"

    def step(self, ctx: _TickContext) -> _Seq[Intent]:
        return [EmitSignal(contact_signal(origin_id="c-tick", value=1.0, delta=0.0, timestamp=None))]


def test_emit_signal_is_transient_not_durable(tmp_path) -> None:
    # EmitSignal threads to later components in-tick; it is NOT written to the bus.
    reg = ComponentRegistry()
    reg.register(ContactEmitter(), ComponentManifest(id="emitter", type="neuron"))
    bus = FileSignalBus(tmp_path)
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    assert bus.peek_unprocessed() == []  # transient — nothing persisted


def test_later_component_sees_earlier_components_emitted_signal(tmp_path) -> None:
    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(ContactEmitter(), ComponentManifest(id="emitter", type="neuron"))
    reg.register(seen, ComponentManifest(id="seen", type="aggregation"))
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path))
    loop.tick()
    assert "c-tick" in seen.seen  # aggregation saw the neuron's transient contact signal


def test_durable_inbound_signal_is_consumed_once_and_threaded(tmp_path) -> None:
    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(seen, ComponentManifest(id="seen", type="aggregation"))
    bus = FileSignalBus(tmp_path)
    bus.publish(_Signal(origin_id="ext-1", kind="exchange", payload={"actor": "user", "label": "two_way"}))
    loop = _loop(reg, RecordingStore(), bus)
    loop.tick()
    assert seen.seen == ["ext-1"]  # inbound external input threaded in
    seen.seen = []
    loop.tick()
    assert seen.seen == []  # consumed once — not re-served next tick
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: FAIL — the new tests error (no `signals` threading yet) and `test_emit_signal_is_transient_not_durable` fails because Phase-A code still publishes to the bus.

- [ ] **Step 4: Implement the dataflow in `core/coreloop.py`**

Replace the body of `CoreLoop.tick` so it consumes the bus once and threads signals; `EmitSignal` becomes transient:

```python
    def tick(self) -> TickReport:
        now = self._clock.now()
        state = self._state_actor.state
        available: list[Signal] = list(self._bus.consume_unprocessed())

        intents: list[Intent] = []
        ran: list[str] = []
        failed: list[str] = []

        for component in self._registry.enabled():
            if component.id in self._broken:
                continue
            ctx = TickContext(state=state, now=now, bus=self._bus, signals=tuple(available))
            try:
                produced = component.step(ctx)
            except Exception as exc:  # isolation: the heart never dies
                self._record_failure(component.id, exc)
                failed.append(component.id)
                continue
            self._failures[component.id] = 0
            for intent in produced:
                if isinstance(intent, EmitSignal):
                    available.append(intent.signal)  # transient — visible to later components this tick
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
```

Add the import `from ..domain.signal import Signal` to `core/coreloop.py` and remove the now-unused `SignalBus`-publish usage (the `self._bus` field stays — it is used for `consume_unprocessed`).

- [ ] **Step 5: Run the coreloop suite to verify green**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: PASS (all coreloop tests, including the 3 new/updated dataflow tests).

- [ ] **Step 6: Format, type-check, commit**

```bash
uv run ruff format core/component.py core/coreloop.py tests/test_coreloop.py
uv run ruff check core/component.py core/coreloop.py tests/test_coreloop.py
uv run mypy -p lifemodel
git add core/component.py core/coreloop.py tests/test_coreloop.py
git commit -m "feat(core): CoreLoop dataflow — consume bus once, thread transient intra-tick signals (spec §7.4)"
```

---

### Task 3: ContactNeuron + defensive time helper

**Files:**
- Create: `core/timeutil.py`
- Create: `core/contact_neuron.py`
- Modify: `core/__init__.py`
- Test: `tests/test_timeutil.py`, `tests/test_contact_neuron.py`

**Interfaces:**
- Consumes: `Drive` (`sim/drive.py`), `quality_of` (`sim/quality.py`), `taxonomy` (Task 1), `TickContext`/`Component` (Phase A), `UpdateState`/`EmitSignal` (Phase A), `State`.
- Produces: `minutes_between`; `ContactNeuron`.

**Behavior (spec §2.1, §11):**
- `minutes_between(a_iso, b)` mirrors `core/decision.py::_minutes_between`: minutes from an ISO timestamp to `b`; returns `0.0` for `None` / unparseable / tz-naive (defensive — a malformed `last_tick_at` must not crash the tick).
- `ContactNeuron.step(ctx)`:
  1. `dt = minutes_between(ctx.state.last_tick_at, ctx.now)`.
  2. Reconstruct `drive = Drive(alpha, beta, u_max, u=ctx.state.u)`; if `dt > 0`, `drive.rise(dt=dt)`.
  3. For each `exchange` signal in `ctx.signals`, satiate by its quality: `drive.satiate(q=quality_of(actor, label))`. (`proactive_internal` yields `q=0` → no-op, per `sim.quality` — the being's own nudge never satiates.)
  4. `delta = drive.u - ctx.state.u`.
  5. Return `[UpdateState({"u": drive.u}), EmitSignal(contact_signal(origin_id=f"contact-{ctx.now.isoformat()}", value=drive.u, delta=delta, timestamp=ctx.now.isoformat()))]`.
- The neuron writes **only** `u`. It does not touch `last_exchange_at`, `desire_status`, gates, or thresholds — those are AGGREGATION's (Phase B2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_timeutil.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.timeutil import minutes_between


def test_minutes_between_counts_forward() -> None:
    a = "2026-07-06T00:00:00+00:00"
    b = datetime(2026, 7, 6, 0, 30, tzinfo=UTC)
    assert minutes_between(a, b) == 30.0


def test_minutes_between_none_is_zero() -> None:
    assert minutes_between(None, datetime(2026, 7, 6, tzinfo=UTC)) == 0.0


def test_minutes_between_unparseable_is_zero() -> None:
    assert minutes_between("not-a-date", datetime(2026, 7, 6, tzinfo=UTC)) == 0.0


def test_minutes_between_naive_is_zero() -> None:
    assert minutes_between("2026-07-06T00:00:00", datetime(2026, 7, 6, 1, 0, tzinfo=UTC)) == 0.0
```

```python
# tests/test_contact_neuron.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.contact_neuron import ContactNeuron
from lifemodel.core.component import TickContext
from lifemodel.core.intents import EmitSignal, UpdateState
from lifemodel.core.taxonomy import KIND_CONTACT, exchange_signal
from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.state.model import State

ALPHA = 1.0 / 240.0


def _neuron() -> ContactNeuron:
    return ContactNeuron(alpha=ALPHA, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals))


def test_rises_by_elapsed_silence(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min → +1.0
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert abs(update.changes["u"] - 1.0) < 1e-9


def test_emits_contact_signal_with_value_and_delta(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    emit = next(i for i in intents if isinstance(i, EmitSignal))
    assert emit.signal.kind == KIND_CONTACT
    assert abs(emit.signal.payload["value"] - 1.0) < 1e-9
    assert abs(emit.signal.payload["delta"] - 1.0) < 1e-9


def test_exchange_satiates_the_drive(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)  # dt=0 → no rise
    ex = exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None)  # q=1.0
    intents = _neuron().step(_ctx(state, now, [ex], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 0.0  # 1.0 - beta*1.0


def test_own_impulse_does_not_satiate(tmp_path) -> None:
    state = State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
    own = exchange_signal(origin_id="e-2", actor="proactive_internal", label="two_way", timestamp=None)
    intents = _neuron().step(_ctx(state, now, [own], tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert update.changes["u"] == 1.0  # proactive_internal → q=0 → unchanged


def test_neuron_writes_only_u(tmp_path) -> None:
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
    intents = _neuron().step(_ctx(state, now, tmp_path=tmp_path))
    update = next(i for i in intents if isinstance(i, UpdateState))
    assert set(update.changes) == {"u"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_timeutil.py tests/test_contact_neuron.py -q`
Expected: FAIL — `ModuleNotFoundError` for `lifemodel.core.timeutil` / `lifemodel.core.contact_neuron`.

- [ ] **Step 3: Write minimal implementations**

```python
# core/timeutil.py
"""Defensive clock arithmetic shared by the pipeline (spec §17).

``minutes_between`` returns elapsed minutes from an ISO-8601 timestamp to a
``datetime``, and — like ``core/decision.py``'s private helper it generalises —
returns ``0.0`` ("no elapsed rise") for ``None``, an unparseable string, or a
tz-naive value, so a malformed ``last_tick_at`` never crashes a tick.
"""

from __future__ import annotations

from datetime import datetime


def minutes_between(a_iso: str | None, b: datetime) -> float:
    if a_iso is None:
        return 0.0
    try:
        a = datetime.fromisoformat(a_iso)
    except ValueError:
        return 0.0
    if a.tzinfo is None or a.utcoffset() is None:
        return 0.0
    return (b - a).total_seconds() / 60.0
```

```python
# core/contact_neuron.py
"""ContactNeuron — the AUTONOMIC contact sensor (spec §2.1, §11).

A dumb sensor: it measures contact deprivation by accumulating the certified
drive ``u`` over elapsed silence (``sim.drive.Drive.rise``) and resets it on a
real exchange (``Drive.satiate`` with ``sim.quality.quality_of``). It emits the
raw ``{value, delta}`` as a transient ``contact`` signal and writes only ``u``.
It computes no salience, thresholds, or gates — those are AGGREGATION's job
(the lower layer is never smarter than the layer above it).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.drive import Drive
from ..sim.quality import quality_of
from .component import TickContext
from .intents import EmitSignal, Intent, UpdateState
from .taxonomy import KIND_EXCHANGE, contact_signal, is_kind, read_exchange


class ContactNeuron:
    """The v1 first neuron: contact-deprivation sensor. Sole writer of ``u``."""

    def __init__(self, *, alpha: float, beta: float, u_max: float, id: str = "contact") -> None:
        self.id = id
        self._alpha = alpha
        self._beta = beta
        self._u_max = u_max

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        from .timeutil import minutes_between

        dt = minutes_between(ctx.state.last_tick_at, ctx.now)
        drive = Drive(alpha=self._alpha, beta=self._beta, u_max=self._u_max, u=ctx.state.u)
        if dt > 0:
            drive.rise(dt=dt)
        for signal in ctx.signals:
            if is_kind(signal, KIND_EXCHANGE):
                actor, label = read_exchange(signal)
                drive.satiate(q=quality_of(actor=actor, label=label))

        delta = drive.u - ctx.state.u
        emit = contact_signal(
            origin_id=f"contact-{ctx.now.isoformat()}",
            value=drive.u,
            delta=delta,
            timestamp=ctx.now.isoformat(),
        )
        return [UpdateState({"u": drive.u}), EmitSignal(emit)]
```

(Move the `minutes_between` import to the module top if the type checker prefers; the local import above avoids any import-cycle risk and is acceptable.) Re-export `ContactNeuron` and `minutes_between` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timeutil.py tests/test_contact_neuron.py -q`
Expected: PASS (4 + 5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/timeutil.py core/contact_neuron.py core/__init__.py tests/test_timeutil.py tests/test_contact_neuron.py
uv run ruff check core/timeutil.py core/contact_neuron.py core/__init__.py tests/test_timeutil.py tests/test_contact_neuron.py
uv run mypy -p lifemodel
git add core/timeutil.py core/contact_neuron.py core/__init__.py tests/test_timeutil.py tests/test_contact_neuron.py
git commit -m "feat(core): ContactNeuron — AUTONOMIC drive sensor (rise/satiate/emit, sole writer of u) (spec §11)"
```

---

### Task 4: Register ContactNeuron in the composition root + pipeline integration test

**Files:**
- Modify: `composition.py`
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Consumes: `ContactNeuron` (Task 3), `ComponentManifest`/`ComponentRegistry` (Phase A), `CoreLoop` (Phase A).
- Produces: a `build_lifemodel(...)` whose `registry` contains an enabled `contact` neuron and whose `coreloop.tick()` rises `u` and satiates on an inbound exchange signal.

**Behavior:** `build_lifemodel` constructs a `ContactNeuron` with the bootstrap constants (`alpha=1/240`, `beta=1.0`, `u_max=100.0`) and `register`s it into the `ComponentRegistry` (enabled) **before** building the `CoreLoop`. Still no live cutover — the neuron runs only when something calls `coreloop.tick()` (tests here; the cron/egress path is untouched). Expose the constants as module-level names in `composition.py` for reuse (do not duplicate literals).

Read `composition.py` first (Task 6 of Phase A added `registry`/`state_actor`/`coreloop` wiring). Insert the neuron registration between building the empty `registry` and building the `coreloop`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_composition.py
from datetime import UTC, datetime

from lifemodel.core.contact_neuron import ContactNeuron
from lifemodel.core.taxonomy import KIND_CONTACT, exchange_signal


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._m = moment

    def now(self) -> datetime:
        return self._m


def test_contact_neuron_is_registered_enabled(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert "contact" in ids
    assert any(isinstance(c, ContactNeuron) for c in lm.registry.enabled())


def test_pipeline_tick_rises_u_and_persists(tmp_path) -> None:
    # Seed last_tick_at 240 min before the clock; one tick should rise u to ~1.0.
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State

    store = JsonStateStore(tmp_path)
    store.commit(State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC)))
    lm.coreloop.tick()
    assert abs(store.load().u - 1.0) < 1e-9


def test_pipeline_tick_satiates_on_inbound_exchange(tmp_path) -> None:
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State

    store = JsonStateStore(tmp_path)
    store.commit(State(u=1.0, last_tick_at="2026-07-06T00:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 0, 0, tzinfo=UTC)))
    lm.bus.publish(exchange_signal(origin_id="e-1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    assert store.load().u == 0.0  # satiated by the two_way exchange
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py -q`
Expected: FAIL — `contact` not registered; `enabled()` has no `ContactNeuron`.

- [ ] **Step 3: Implement the wiring in `composition.py`**

Add module constants near the top (after imports):
```python
CONTACT_ALPHA = 1.0 / 240.0
CONTACT_BETA = 1.0
CONTACT_U_MAX = 100.0
```
Add imports:
```python
from .core.contact_neuron import ContactNeuron
from .core.registry import ComponentManifest
```
In `build_lifemodel`, after `registry = registry if registry is not None else ComponentRegistry()` and before constructing `coreloop`, register the neuron:
```python
    contact = ContactNeuron(alpha=CONTACT_ALPHA, beta=CONTACT_BETA, u_max=CONTACT_U_MAX)
    registry.register(contact, ComponentManifest(id=contact.id, type="neuron"))
```
(Guard against double-registration if a caller passes a registry that already has `contact`: only register when `contact` is absent — `if "contact" not in [m for ...]`; simplest is to check `registry.enabled()`/manifests. Use a small helper: attempt `registry.manifest("contact")` in a `try/except UnknownComponent` and register only on `UnknownComponent`.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — the new composition tests pass and every pre-existing test still passes (live path unchanged).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format composition.py tests/test_composition.py
uv run ruff check composition.py tests/test_composition.py
uv run mypy -p lifemodel
git add composition.py tests/test_composition.py
git commit -m "feat(core): register ContactNeuron in composition root (enabled, no live cutover)"
```

---

## Phase-B1 Definition of Done

- [ ] `make check` fully green — paste the tail (mypy Success + pytest summary).
- [ ] Four commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `contact` neuron runs only via `coreloop.tick()`; the cron/egress path is untouched; all prior behavior intact.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` when done (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §4 taxonomy → Task 1; §3/§7.4 pipeline dataflow (durable-in / transient-intra-tick) → Task 2; §2.1/§11 dumb-sensor neuron, drive rise/satiate, sole-writer-of-`u` → Task 3; DI registration/enable → Task 4. **Deferred to B2 (by design):** salience (§5), duration_over_theta, wake gates + desire lifecycle + verdict/exchange policy (§12), certified-scenario replay. **Deferred to C:** latent/effective pressure, ActionPending (§9). **Deferred to E:** live cutover, hooks→intents, delete `decision.py`.
- **Type consistency:** `TickContext(state, now, bus, signals=())` used identically in Tasks 2, 3, 4. `contact_signal(*, origin_id, value, delta, timestamp)` / `exchange_signal(*, origin_id, actor, label, timestamp)` / `read_exchange` identical in Tasks 1, 3, 4. `ContactNeuron(*, alpha, beta, u_max, id="contact")` identical in Tasks 3, 4.
- **No placeholders:** every step ships real code + an exact command with expected output.
- **Key risk documented:** transient vs durable signal split (Task 2) prevents the recompute-and-re-consume double-count — the reason the CoreLoop consumes once and threads `EmitSignal` transiently.

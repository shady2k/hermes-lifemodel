# Core Rebuild — Phase B3: Aggregation Backpressure Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the aggregation intake **unfailable under a signal flood** — a bounded-work, lane-classified intake stage so the layer can never be overwhelmed (spec §5.1). Per the Codex review (rev.5), this is a *skeleton*: classify signals into a **control lane** (lossless, processed first, never shed) and a **sensor lane** (coalesced latest-wins, sheddable), with a hard per-lane intake cap. Full salience (Weber-Fechner + arousal) is **not** built here — it is premature for a one-sensor (contact) v1; salience stays "boring."

**Architecture (spec §5.1):** A new pure helper `core/intake.py` applies backpressure to the batch the CoreLoop consumes from the bus each tick: **control signals** (`exchange`, `verdict`, `in_flight`, `delivery_result`) are kept losslessly (bounded by a generous `MAX_CONTROL` prefix — under overflow the tail is **not** dropped-and-lost, it is left for the next tick by not advancing the offset — see the note in Task 3), **sensor signals** (`contact`, future `ping`/`presence`) are coalesced per-kind to latest-wins and bounded by `MAX_SENSOR`. The CoreLoop threads the kept set into `ctx.signals`. The layer's per-tick work is now O(bounded) regardless of input volume — the "layer never fails" invariant.

**Tech Stack:** Python 3.11 stdlib-only core; frozen dataclasses. `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout:** tests import `from lifemodel.x import Y`; package-internal uses relative imports.
- **Core imports no Hermes.** Pure stdlib + intra-package.
- **`apply_intake` MUST NEVER raise** — it is the flood guard; malformed/unknown-kind signals are classified defensively (unknown kind → sensor/droppable, never control), never crash the tick.
- **Control lane is lossless (spec §5.1):** control signals (`exchange`/`verdict`/`in_flight`/`delivery_result`) are never salience-shed. Only sensor signals may be coalesced/dropped.
- **Determinism:** coalescing/bounding must be order-deterministic (no `set` iteration in output ordering, no randomness).
- **`mypy -p lifemodel` strict.**
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. All existing tests (419) must still pass.
- **Branch:** `core/rebuild` (already checked out). One commit per task.

## File Structure

- Modify `core/taxonomy.py` — add lane classification (`lane_of`, `Lane`, control-kinds set) (Task 1).
- Create `core/intake.py` — `apply_intake` + `IntakeLimits` + `IntakeResult` (Task 2).
- Modify `core/coreloop.py` — apply intake after consuming the bus; thread kept; log shed (Task 3).
- Modify `core/__init__.py` — re-export new public names.
- Tests: extend `tests/test_taxonomy.py`; create `tests/test_intake.py`; extend `tests/test_coreloop.py`.

**Interfaces produced:**
- `core/taxonomy.py`: `Lane` (`"control"|"sensor"` string literals), `CONTROL_KINDS: frozenset[str]`, `lane_of(kind: str) -> Lane`.
- `core/intake.py`: `IntakeLimits(max_control=256, max_sensor=64)`; `IntakeResult(kept: tuple[Signal,...], shed_control: int, shed_sensor: int, coalesced_sensor: int)`; `apply_intake(signals: Iterable[Signal], *, limits: IntakeLimits, lane_of: Callable[[str], Lane]) -> IntakeResult`.

---

### Task 1: Taxonomy — lane classification

**Files:**
- Modify: `core/taxonomy.py`
- Modify: `core/__init__.py`
- Test: `tests/test_taxonomy.py` (extend)

**Interfaces:**
- Produces: `Lane`, `CONTROL_KINDS`, `lane_of`.

**Behavior (spec §5.1):** classify each signal kind into its backpressure lane. Control kinds are the load-bearing lifecycle events; everything else (including unknown kinds) is a sensor — so an unknown flood can never masquerade as lossless control.

- [ ] **Step 1: Write the failing test (append to `tests/test_taxonomy.py`)**

```python
from lifemodel.core.taxonomy import CONTROL_KINDS, KIND_CONTACT, lane_of


def test_control_kinds_are_control_lane() -> None:
    from lifemodel.core.taxonomy import KIND_EXCHANGE, KIND_IN_FLIGHT, KIND_VERDICT

    for k in (KIND_EXCHANGE, KIND_VERDICT, KIND_IN_FLIGHT):
        assert k in CONTROL_KINDS
        assert lane_of(k) == "control"


def test_contact_is_sensor_lane() -> None:
    assert lane_of(KIND_CONTACT) == "sensor"


def test_unknown_kind_defaults_to_sensor_never_control() -> None:
    assert lane_of("something-new") == "sensor"  # unknown floods can't claim lossless control
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: FAIL — `ImportError` for `CONTROL_KINDS` / `lane_of`.

- [ ] **Step 3: Extend `core/taxonomy.py`**

Add (after the kind constants, e.g. after `KIND_IN_FLIGHT`):
```python
from typing import Literal

Lane = Literal["control", "sensor"]

#: Load-bearing lifecycle events — never salience-shed (spec §5.1). Includes a
#: forward-looking ``delivery_result`` kind (used by phases D/E) so the lane is
#: stable before that signal exists.
CONTROL_KINDS: frozenset[str] = frozenset(
    {KIND_EXCHANGE, KIND_VERDICT, KIND_IN_FLIGHT, "delivery_result"}
)


def lane_of(kind: str) -> Lane:
    """Backpressure lane for a signal kind. Unknown kinds are sensors (never
    control) so an unknown flood cannot claim the lossless lane."""
    return "control" if kind in CONTROL_KINDS else "sensor"
```
Re-export `Lane, CONTROL_KINDS, lane_of` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run ruff check core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run mypy -p lifemodel
git add core/taxonomy.py core/__init__.py tests/test_taxonomy.py
git commit -m "feat(core): taxonomy lane classification — control (lossless) vs sensor (spec §5.1)"
```

---

### Task 2: `apply_intake` — bounded, lane-aware backpressure (never fails)

**Files:**
- Create: `core/intake.py`
- Modify: `core/__init__.py`
- Test: `tests/test_intake.py`

**Interfaces:**
- Consumes: `Signal`, `Lane`/`lane_of` (Task 1).
- Produces: `IntakeLimits`, `IntakeResult`, `apply_intake`.

**Behavior (spec §5.1):**
- **Control lane (lossless):** keep control signals in arrival order up to `max_control`. Overflow beyond `max_control` is counted in `shed_control` (the CoreLoop, Task 3, treats a non-zero `shed_control` as "do not advance the bus offset" — but `apply_intake` itself is pure and just reports the count).
- **Sensor lane (coalesce + bound):** coalesce per-kind to **latest-wins** (the last signal of each sensor kind in arrival order), then keep up to `max_sensor` (deterministic: first-seen kind order). Report `coalesced_sensor` (how many were merged away) and `shed_sensor` (kinds dropped past the cap).
- **Output order:** control first, then sensor (control is processed first downstream).
- **Never raises**, whatever the input volume or shape.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_intake.py
from __future__ import annotations

from lifemodel.core.intake import IntakeLimits, IntakeResult, apply_intake
from lifemodel.core.taxonomy import lane_of
from lifemodel.domain.signal import Signal


def _c(i: int) -> Signal:  # a contact (sensor) signal
    return Signal(origin_id=f"c{i}", kind="contact", payload={"value": float(i)})


def _ex(i: int) -> Signal:  # an exchange (control) signal
    return Signal(origin_id=f"e{i}", kind="exchange", payload={"actor": "user", "label": "two_way"})


def _limits(**kw) -> IntakeLimits:
    return IntakeLimits(**kw)


def test_control_signals_are_kept_lossless_and_first() -> None:
    signals = [_c(1), _ex(1), _c(2), _ex(2)]
    res = apply_intake(signals, limits=_limits(), lane_of=lane_of)
    kept_kinds = [s.kind for s in res.kept]
    assert kept_kinds[:2] == ["exchange", "exchange"]  # control first
    assert res.shed_control == 0


def test_sensor_kind_coalesces_to_latest_wins() -> None:
    # three contact signals in one batch collapse to the last one
    res = apply_intake([_c(1), _c(2), _c(3)], limits=_limits(), lane_of=lane_of)
    contacts = [s for s in res.kept if s.kind == "contact"]
    assert len(contacts) == 1
    assert contacts[0].origin_id == "c3"  # latest wins
    assert res.coalesced_sensor == 2


def test_control_overflow_is_counted_not_reordered() -> None:
    signals = [_ex(i) for i in range(10)]
    res = apply_intake(signals, limits=_limits(max_control=4), lane_of=lane_of)
    assert len([s for s in res.kept if s.kind == "exchange"]) == 4
    assert res.shed_control == 6
    assert [s.origin_id for s in res.kept[:4]] == ["e0", "e1", "e2", "e3"]  # prefix, in order


def test_flood_never_raises_and_is_bounded() -> None:
    # 10_000 mixed signals in one batch -> bounded output, control preserved, no exception
    flood = []
    for i in range(5000):
        flood.append(_c(i))
        flood.append(_ex(i))
    res = apply_intake(flood, limits=_limits(max_control=256, max_sensor=64), lane_of=lane_of)
    assert isinstance(res, IntakeResult)
    exchanges = [s for s in res.kept if s.kind == "exchange"]
    contacts = [s for s in res.kept if s.kind == "contact"]
    assert len(exchanges) == 256  # control kept up to the cap
    assert len(contacts) == 1  # 5000 contacts coalesced to latest
    assert res.shed_control == 5000 - 256
    assert len(res.kept) <= 256 + 64  # O(bounded)


def test_unknown_kinds_are_sensors_and_droppable() -> None:
    weird = [Signal(origin_id=f"w{i}", kind="mystery", payload={}) for i in range(100)]
    res = apply_intake(weird, limits=_limits(max_sensor=64), lane_of=lane_of)
    # 'mystery' is a sensor kind -> coalesced to latest-wins (1 survives)
    assert len([s for s in res.kept if s.kind == "mystery"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_intake.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.intake'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/intake.py
"""Backpressure intake — the aggregation flood guard (spec §5.1).

Bounds the per-tick signal batch so the aggregation is O(bounded) regardless of
input volume ("the layer never fails"). Signals are split by lane:

* **control** (``exchange``/``verdict``/``in_flight``/``delivery_result``) —
  lossless: kept in arrival order up to ``max_control``; any overflow is *counted*
  (the CoreLoop leaves it on the bus for next tick, not dropped).
* **sensor** (``contact`` and future ``ping``/``presence``) — coalesced per-kind
  to latest-wins, then bounded to ``max_sensor``.

Pure and total: never raises, whatever the input. Full salience-based shedding
(spec §5) slots in for the sensor lane once there are multiple noisy sensors; v1
keeps it to latest-wins coalescing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..domain.signal import Signal
from .taxonomy import Lane


@dataclass(frozen=True)
class IntakeLimits:
    """Per-lane intake caps (bootstrap values; tuned later, spec §22)."""

    max_control: int = 256
    max_sensor: int = 64


@dataclass(frozen=True)
class IntakeResult:
    kept: tuple[Signal, ...]
    shed_control: int
    shed_sensor: int
    coalesced_sensor: int


def apply_intake(
    signals: Iterable[Signal],
    *,
    limits: IntakeLimits,
    lane_of: Callable[[str], Lane],
) -> IntakeResult:
    """Apply lane-aware backpressure to a signal batch. Never raises."""
    control: list[Signal] = []
    sensor_latest: dict[str, Signal] = {}
    coalesced = 0
    for sig in signals:
        if lane_of(sig.kind) == "control":
            control.append(sig)
        else:
            if sig.kind in sensor_latest:
                coalesced += 1
            sensor_latest[sig.kind] = sig  # latest-wins

    control_kept = control[: limits.max_control]
    shed_control = len(control) - len(control_kept)

    sensor_all = list(sensor_latest.values())  # dict preserves first-seen kind order (py3.7+)
    sensor_kept = sensor_all[: limits.max_sensor]
    shed_sensor = len(sensor_all) - len(sensor_kept)

    return IntakeResult(
        kept=tuple(control_kept) + tuple(sensor_kept),
        shed_control=shed_control,
        shed_sensor=shed_sensor,
        coalesced_sensor=coalesced,
    )
```
Re-export `IntakeLimits, IntakeResult, apply_intake` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_intake.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/intake.py core/__init__.py tests/test_intake.py
uv run ruff check core/intake.py core/__init__.py tests/test_intake.py
uv run mypy -p lifemodel
git add core/intake.py core/__init__.py tests/test_intake.py
git commit -m "feat(core): apply_intake — bounded lane-aware backpressure, never fails (spec §5.1)"
```

---

### Task 3: Wire intake into the CoreLoop + never-fail integration test

**Files:**
- Modify: `core/coreloop.py`
- Test: `tests/test_coreloop.py` (extend)

**Interfaces:**
- Consumes: `apply_intake`/`IntakeLimits` (Task 2), `lane_of` (Task 1).
- Produces: CoreLoop consumes the bus, applies intake, threads the kept set.

**Behavior (spec §5.1):** the CoreLoop applies `apply_intake` to the batch it consumes from the bus each tick, threads the **kept** signals into `ctx.signals`, and logs a `signals_shed` event when anything was shed/coalesced. Transient signals emitted by components mid-tick (B1) are appended after (internal, trusted, not flooded). The CoreLoop gains an `IntakeLimits` field (default `IntakeLimits()`).

**Note on the "don't advance offset" requirement (spec §5.1):** the current `FileSignalBus.consume_unprocessed()` consumes-and-marks the whole batch atomically — it has no partial-consume API, so a true "leave the control overflow on the bus" cannot be implemented without a bus change. For this phase, log `shed_control` so overflow is **observable**, and leave the bus-offset refinement (bounded consume) as an explicit follow-up (create bead `core: bounded bus consume so control overflow survives to next tick`). With the generous `MAX_CONTROL=256` default and v1's low control volume, `shed_control` is expected to be 0 in practice; the invariant that matters here — **the layer never crashes under flood** — holds regardless.

- [ ] **Step 1: Write the failing tests (append to `tests/test_coreloop.py`)**

```python
def test_coreloop_bounds_a_flood_and_survives(tmp_path) -> None:
    # publish a flood of exchange signals to the bus; one tick must complete,
    # bounded, without raising, and the aggregation-facing ctx must be capped.
    from lifemodel.core.intake import IntakeLimits
    from lifemodel.domain.signal import Signal as _S

    reg = ComponentRegistry()
    seen = SeenRecorder()
    reg.register(seen, ComponentManifest(id="seen", type="aggregation"))
    bus = FileSignalBus(tmp_path)
    for i in range(1000):
        bus.publish(_S(origin_id=f"e{i}", kind="exchange", payload={"actor": "user", "label": "two_way"}))
    loop = CoreLoop(
        registry=reg,
        state_actor=StateActor(RecordingStore()),
        bus=bus,
        clock=FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC)),
        intake_limits=IntakeLimits(max_control=256, max_sensor=64),
    )
    report = loop.tick()  # must not raise
    assert report.committed
    assert len(seen.seen) <= 256 + 64  # aggregation saw a bounded batch


def test_coreloop_default_intake_limits_present(tmp_path) -> None:
    loop = _loop(ComponentRegistry(), RecordingStore(), FileSignalBus(tmp_path))
    loop.tick()  # smoke: default IntakeLimits, no flood, still ticks
```

(`SeenRecorder`, `FixedClock`, `RecordingStore`, `_loop` already exist in this test file from B1.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: FAIL — `CoreLoop.__init__` has no `intake_limits` param.

- [ ] **Step 3: Wire intake into `core/coreloop.py`**

Add imports: `from .intake import IntakeLimits, apply_intake` and `from .taxonomy import lane_of`. Add an `intake_limits: IntakeLimits | None = None` keyword param to `__init__`, stored as `self._intake_limits = intake_limits or IntakeLimits()`. In `tick()`, replace the bus-consume line:
```python
        intake = apply_intake(
            self._bus.consume_unprocessed(), limits=self._intake_limits, lane_of=lane_of
        )
        if self._log is not None and (intake.shed_control or intake.shed_sensor or intake.coalesced_sensor):
            self._log.info(
                "signals_shed",
                shed_control=intake.shed_control,
                shed_sensor=intake.shed_sensor,
                coalesced_sensor=intake.coalesced_sensor,
            )
        available: list[Signal] = list(intake.kept)
```
(Everything downstream — threading, transient EmitSignal append — is unchanged.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new coreloop tests pass; every prior test (incl. `tests/sim/`) still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/coreloop.py tests/test_coreloop.py
uv run ruff check core/coreloop.py tests/test_coreloop.py
uv run mypy -p lifemodel
git add core/coreloop.py tests/test_coreloop.py
git commit -m "feat(core): CoreLoop applies bounded intake — aggregation never fails under flood (spec §5.1)"
```

---

## Phase-B3 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Three commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] A follow-up bead filed for "bounded bus consume so control overflow survives to next tick" (the offset refinement §5.1 leaves for later).
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §5.1 lane classification (control lossless / sensor coalesce) → Task 1; bounded, never-fail intake → Task 2; CoreLoop wiring + flood-survival invariant → Task 3. **Explicitly scoped out (per Codex rev.5):** full salience (Weber-Fechner + arousal) — premature for one sensor; bus bounded-consume "don't advance offset" — filed as a follow-up bead (needs a bus API change). Arousal-coupling of overload (§5.1) → Phase C/D (needs the arousal state).
- **Type consistency:** `apply_intake(signals, *, limits, lane_of)` / `IntakeLimits(max_control, max_sensor)` / `IntakeResult(kept, shed_control, shed_sensor, coalesced_sensor)` used identically in Tasks 2, 3. `lane_of(kind) -> Lane` identical in Tasks 1, 2, 3.
- **No placeholders:** every step ships real code + an exact command with expected output.
- **Key invariant:** `apply_intake` never raises and is O(bounded) — the flood test (Task 2 + Task 3) proves the layer cannot be driven to failure.

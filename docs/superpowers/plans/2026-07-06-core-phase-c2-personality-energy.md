# Core Rebuild — Phase C2: Personality Component + Energy (E/S/C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the **Personality** component that holds the being's physiology — the energy **battery `E`**, the **fatigue debt `S`**, and the **circadian rhythm `C`** — and the **energy reserve/settle** mechanism that will gate the (expensive) cognition layer. Energy recovers during rest (idle) and faster at night; fatigue decays at rest; thinking will cost more when tired (`cost·(1+α·S)`). **No sleep state** (per user: the digital human simply rests during idle; a real sleep/dreaming feature with a consolidation job comes later). Still **no live cutover**; the cognition gate is wired in Phase D (its only consumer).

**Architecture (spec §8, §9):** A new `Personality` component (a cheap, 0-energy layer that runs every tick) evolves `E` (recover, clamped to `E_MAX`) and `S` (decay) from elapsed time, with recovery boosted when the circadian `C` is low (night rest). `C(t) = 0.5 + 0.5·cos(2π(t−φ)/24h)` is a pure function of absolute wall-clock (§8). A pure `core/energy.py` provides `cost_real = cost_base·(1+α·S)` and a `reserve → settle` lifecycle (gate on affordability; refund the unused estimate; overspend is allowed → `E` may go negative → natural recovery). The **"don't petrify"** invariant (the cheapest act at max fatigue stays affordable) is a calibrated test. The **spend** side (cognition deducting `E` via reserve/settle and *raising* `S`) and the **circadian gating** of cognition are wired in Phase D — here we build the battery + recovery + the mechanism, tested with synthetic costs (like `apply_intake` before real floods).

**Tech Stack:** Python 3.11 stdlib-only (`math`); frozen dataclasses. `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.**
- **Only cognition pays energy (spec §3, §8 — coma-fix):** cheap layers (neurons, aggregation, personality) cost **0**. The `Personality` component only *recovers* `E`/*decays* `S`; the *spend* comes from cognition in Phase D. Never add an `if energy < X` shutdown — progressive shutoff is emergent from the budget (Phase D).
- **`E` may go negative** (overspend → forced recovery); `E` is clamped **above** at `E_MAX` only.
- **`S ∈ [0,1]`, `C ∈ [0,1]`;** all numeric finite/clamped.
- **Don't petrify (spec §8):** `cost_real(cost_fast, S=1.0, α) < E_MIN_AFFORDABLE` — α is calibrated so the cheapest thought is always affordable at maximum fatigue.
- **Bootstrap constants (spec §22, one-line tunable):** `E_MAX=1.0`, `ENERGY_RECOVERY_PER_MIN=0.01`, `NIGHT_RECOVERY_BOOST=0.5`, `FATIGUE_DECAY_PER_MIN=0.002`, `COST_ALPHA=2.0`, `CIRCADIAN_PEAK_UTC_HOUR=13.0` (peak alertness 16:00 MSK, trough 04:00 MSK), prices `COST_FAST=0.02`, `COST_SMART=0.08`, `COST_SEND=0.03`, `E_MIN_AFFORDABLE=0.1`.
- **`mypy -p lifemodel` strict.**
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `state/model.py` — add `fatigue: float = 0.0` (Task 1).
- Create `core/circadian.py` — `circadian(now, *, peak_hour_utc)` (Task 1).
- Create `core/energy.py` — `cost_real`, `Reservation`, `reserve`, `settle`, `can_afford` (Task 2).
- Create `core/personality.py` — the `Personality` component (Task 3).
- Modify `composition.py` — register `Personality` (Task 4).
- Modify `core/__init__.py` — re-export new names.
- Tests: extend `tests/test_state_model.py`; create `tests/test_circadian.py`, `tests/test_energy.py`, `tests/test_personality.py`; extend `tests/test_composition.py`.

**Interfaces produced (Phase D consumes):**
- `state.model.State.fatigue: float`.
- `core/circadian.py`: `circadian(now: datetime, *, peak_hour_utc: float) -> float`.
- `core/energy.py`: `cost_real(cost_base: float, s: float, *, alpha: float) -> float`; `can_afford(energy: float, cost: float) -> bool`; `Reservation(reserved: float)`; `reserve(energy: float, estimate: float) -> tuple[float, Reservation] | None`; `settle(energy: float, reservation: Reservation, actual: float) -> float`.
- `core/personality.py`: `Personality(*, e_max, recovery_per_min, night_boost, fatigue_decay_per_min, peak_hour_utc, id="personality")`.

---

### Task 1: State field `fatigue` + circadian rhythm

**Files:**
- Modify: `state/model.py`
- Create: `core/circadian.py`
- Modify: `core/__init__.py`
- Test: `tests/test_state_model.py` (extend), `tests/test_circadian.py`

**Interfaces:**
- Produces: `State.fatigue`; `circadian`.

**Behavior (spec §8):** `fatigue` (S) is the two-process homeostatic sleep-pressure debt, `[0,1]`, default `0.0`. `circadian(now, *, peak_hour_utc)` is the 24-h alertness wave from wall-clock time-of-day (UTC): `0.5 + 0.5·cos(2π(h − peak_hour_utc)/24)` where `h` is the UTC hour-of-day. Peak (C=1) at `peak_hour_utc`, trough (C=0) 12 h later.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_circadian.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.circadian import circadian

PEAK = 13.0  # 16:00 MSK peak, 04:00 MSK trough


def test_peak_at_peak_hour() -> None:
    c = circadian(datetime(2026, 7, 6, 13, 0, tzinfo=UTC), peak_hour_utc=PEAK)
    assert abs(c - 1.0) < 1e-9


def test_trough_twelve_hours_later() -> None:
    c = circadian(datetime(2026, 7, 6, 1, 0, tzinfo=UTC), peak_hour_utc=PEAK)  # 01:00 UTC = 04:00 MSK
    assert abs(c - 0.0) < 1e-9


def test_midpoint_at_quarter_phase() -> None:
    c = circadian(datetime(2026, 7, 6, 19, 0, tzinfo=UTC), peak_hour_utc=PEAK)  # +6h from peak
    assert abs(c - 0.5) < 1e-9


def test_always_in_unit_interval() -> None:
    for hour in range(24):
        c = circadian(datetime(2026, 7, 6, hour, 0, tzinfo=UTC), peak_hour_utc=PEAK)
        assert 0.0 <= c <= 1.0
```

```python
# append to tests/test_state_model.py
def test_fatigue_defaults_zero_and_roundtrips() -> None:
    assert State().fatigue == 0.0
    assert State.from_dict({}).fatigue == 0.0  # additive
    assert State.from_dict(State(fatigue=0.4).to_dict()).fatigue == 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_circadian.py tests/test_state_model.py -q`
Expected: FAIL — `ModuleNotFoundError` for `lifemodel.core.circadian`; `State` has no `fatigue`.

- [ ] **Step 3: Implement**

In `state/model.py`, add the field to `State` (near `energy`):
```python
    fatigue: float = 0.0
```
and in `from_dict`, add (mirroring `energy`'s `_as_float`):
```python
        fatigue=_as_float(data, "fatigue", State.fatigue),
```
(Match the exact call shape the existing `energy=_as_float(...)` line uses in this file — reuse the same validator/signature.)

```python
# core/circadian.py
"""Circadian rhythm C(t) — the 24-hour alertness wave (spec §8).

A pure function of absolute wall-clock time (not ticks): ``0.5 + 0.5·cos(2π(h −
peak)/24)`` over the UTC hour-of-day ``h``. Peak alertness (C=1) at
``peak_hour_utc``; trough (C=0) twelve hours later. Part of the two-process sleep
model (Borbély): C is the circadian process; S (fatigue) is the homeostatic one.
"""

from __future__ import annotations

import math
from datetime import datetime


def circadian(now: datetime, *, peak_hour_utc: float) -> float:
    h = now.hour + now.minute / 60.0 + now.second / 3600.0
    return 0.5 + 0.5 * math.cos(2.0 * math.pi * (h - peak_hour_utc) / 24.0)
```
Re-export `circadian` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_circadian.py tests/test_state_model.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format state/model.py core/circadian.py core/__init__.py tests/test_circadian.py tests/test_state_model.py
uv run ruff check state/model.py core/circadian.py core/__init__.py tests/test_circadian.py tests/test_state_model.py
uv run mypy -p lifemodel
git add state/model.py core/circadian.py core/__init__.py tests/test_circadian.py tests/test_state_model.py
git commit -m "feat(core): fatigue (S) state field + circadian rhythm C(t) (spec §8)"
```

---

### Task 2: Energy cost model + reserve/settle mechanism

**Files:**
- Create: `core/energy.py`
- Modify: `core/__init__.py`
- Test: `tests/test_energy.py`

**Interfaces:**
- Produces: `cost_real`, `can_afford`, `Reservation`, `reserve`, `settle`.

**Behavior (spec §8):** `cost_real(cost_base, s, *, alpha) = cost_base·(1+α·s)` — fatigue makes an act cost more. `reserve(energy, estimate)` gates on affordability (`energy ≥ estimate`) and, if affordable, returns the energy **after** holding the estimate plus a `Reservation`; returns `None` if unaffordable. `settle(energy_after_reserve, reservation, actual)` refunds the **unused** estimate (`+ (reserved − actual)`) — so the net deduction is the actual cost, and an **overspend** (`actual > reserved`) pushes energy **negative** (allowed → forced recovery). The **don't-petrify** invariant is asserted as a test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_energy.py
from __future__ import annotations

from lifemodel.core.energy import Reservation, can_afford, cost_real, reserve, settle

COST_FAST = 0.02
ALPHA = 2.0
E_MIN_AFFORDABLE = 0.1


def test_cost_inflates_with_fatigue() -> None:
    assert cost_real(0.02, 0.0, alpha=ALPHA) == 0.02
    assert abs(cost_real(0.02, 1.0, alpha=ALPHA) - 0.06) < 1e-9  # 0.02*(1+2*1)


def test_dont_petrify_cheapest_act_affordable_at_max_fatigue() -> None:
    # the calibrated invariant: cheapest thought at S=1 stays under the affordability floor
    assert cost_real(COST_FAST, 1.0, alpha=ALPHA) < E_MIN_AFFORDABLE


def test_reserve_gates_on_affordability() -> None:
    assert reserve(0.05, 0.10) is None  # can't afford
    result = reserve(0.30, 0.10)
    assert result is not None
    energy_after, res = result
    assert abs(energy_after - 0.20) < 1e-9  # estimate held
    assert res == Reservation(reserved=0.10)


def test_settle_refunds_unused_estimate() -> None:
    energy_after, res = reserve(0.30, 0.10)  # energy 0.30 -> 0.20, reserved 0.10
    final = settle(energy_after, res, actual=0.04)  # only spent 0.04
    assert abs(final - 0.26) < 1e-9  # 0.20 + (0.10 - 0.04) => net -0.04 from 0.30


def test_overspend_pushes_energy_negative() -> None:
    energy_after, res = reserve(0.05, 0.05)  # energy -> 0.0, reserved 0.05
    final = settle(energy_after, res, actual=0.12)  # blew the estimate
    assert final < 0.0  # 0.0 + (0.05 - 0.12) = -0.07 — allowed, forces recovery


def test_can_afford() -> None:
    assert can_afford(0.10, 0.06) is True
    assert can_afford(0.05, 0.06) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_energy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.energy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/energy.py
"""Energy budget: cost model + reserve/settle lifecycle (spec §8).

Only cognition pays. Before an expensive act the CoreLoop (Phase D) reserves the
*expected* cost (gate: is it affordable?); afterwards it settles the *actual*
cost and refunds the unused estimate. Fatigue ``S`` inflates cost —
``cost_real = cost_base·(1+α·S)`` — so a tired being naturally drops to reflexes
without any ``if energy < X`` (progressive shutoff is emergent). Overspend is
allowed: energy may go negative and recover. Ego-depletion is NOT modelled.
"""

from __future__ import annotations

from dataclasses import dataclass


def cost_real(cost_base: float, s: float, *, alpha: float) -> float:
    """The fatigue-inflated cost of an act: ``cost_base·(1+α·s)``."""
    return cost_base * (1.0 + alpha * s)


def can_afford(energy: float, cost: float) -> bool:
    return energy >= cost


@dataclass(frozen=True)
class Reservation:
    """A held energy estimate, settled once the act's actual cost is known."""

    reserved: float


def reserve(energy: float, estimate: float) -> tuple[float, Reservation] | None:
    """Gate on affordability and hold ``estimate``. Returns ``(energy_after,
    reservation)`` or ``None`` if unaffordable."""
    if not can_afford(energy, estimate):
        return None
    return energy - estimate, Reservation(reserved=estimate)


def settle(energy: float, reservation: Reservation, actual: float) -> float:
    """Refund the unused estimate; net deduction is the actual cost. Overspend
    (``actual > reserved``) drives energy negative — allowed (forced recovery)."""
    return energy + (reservation.reserved - actual)
```
Re-export `cost_real, can_afford, Reservation, reserve, settle` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_energy.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/energy.py core/__init__.py tests/test_energy.py
uv run ruff check core/energy.py core/__init__.py tests/test_energy.py
uv run mypy -p lifemodel
git add core/energy.py core/__init__.py tests/test_energy.py
git commit -m "feat(core): energy cost model + reserve/settle lifecycle, don't-petrify (spec §8)"
```

---

### Task 3: Personality component — recover E, decay S, circadian-modulated

**Files:**
- Create: `core/personality.py`
- Modify: `core/__init__.py`
- Test: `tests/test_personality.py`

**Interfaces:**
- Consumes: `circadian` (Task 1), `timeutil.minutes_between`, `TickContext`, `UpdateState`.
- Produces: `Personality`.

**Behavior (spec §8):** a cheap, 0-energy component that runs every tick and evolves physiology from elapsed time:
- `dt = minutes_between(state.last_tick_at, now)`.
- `c = circadian(now, peak_hour_utc=…)`.
- **Recovery (idle rest):** `E' = min(E_MAX, energy + recovery_per_min·(1 + night_boost·(1−c))·dt)` — recovers faster when `C` is low (night).
- **Fatigue decay (rest):** `S' = max(0.0, fatigue − fatigue_decay_per_min·dt)`.
- Returns `UpdateState({"energy": E', "fatigue": S'})`. (The **spend** that lowers `E` and raises `S` is cognition's, wired in Phase D; here the being only rests.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_personality.py
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent, UpdateState
from lifemodel.core.personality import Personality
from lifemodel.state.model import State

PEAK = 13.0


def _p() -> Personality:
    return Personality(
        e_max=1.0, recovery_per_min=0.01, night_boost=0.5,
        fatigue_decay_per_min=0.002, peak_hour_utc=PEAK,
    )


def _ctx(state: State, now: datetime, *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=())


def _changes(intents: Sequence[Intent]) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def test_energy_recovers_during_idle(tmp_path) -> None:
    state = State(energy=0.5, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 12, 10, tzinfo=UTC)  # dt=10 min, near peak -> ~1x recovery
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["energy"] > 0.5  # recovered
    assert changes["energy"] <= 1.0


def test_energy_clamped_at_max(tmp_path) -> None:
    state = State(energy=0.99, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 18, 0, tzinfo=UTC)  # long dt
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["energy"] == 1.0


def test_fatigue_decays_during_rest(tmp_path) -> None:
    state = State(fatigue=0.5, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 13, 0, tzinfo=UTC)  # dt=60 min -> -0.12
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert abs(changes["fatigue"] - 0.38) < 1e-9


def test_fatigue_never_negative(tmp_path) -> None:
    state = State(fatigue=0.01, last_tick_at="2026-07-06T12:00:00+00:00")
    now = datetime(2026, 7, 6, 18, 0, tzinfo=UTC)  # long dt
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["fatigue"] == 0.0


def test_night_recovers_faster_than_day(tmp_path) -> None:
    # same dt, but at circadian trough (01:00 UTC) recovery is boosted vs peak (13:00 UTC)
    day = State(energy=0.5, last_tick_at="2026-07-06T13:00:00+00:00")
    day_changes = _changes(_p().step(_ctx(day, datetime(2026, 7, 6, 13, 10, tzinfo=UTC), tmp_path=tmp_path)))
    night = State(energy=0.5, last_tick_at="2026-07-06T01:00:00+00:00")
    night_changes = _changes(_p().step(_ctx(night, datetime(2026, 7, 6, 1, 10, tzinfo=UTC), tmp_path=tmp_path)))
    assert night_changes["energy"] > day_changes["energy"]  # night rest recovers faster
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_personality.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.personality'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/personality.py
"""Personality — the being's physiology component (spec §8, §9).

A cheap (0-energy) layer that runs every tick and evolves the battery ``E`` and
the fatigue debt ``S`` from elapsed rest, modulated by the circadian rhythm
``C``. Here the being only *recovers*: ``E`` climbs back toward ``E_MAX`` (faster
when ``C`` is low — night rest) and ``S`` decays toward 0. The *spend* that
drains ``E`` and raises ``S`` is cognition's, wired in Phase D (only cognition
pays energy — the coma-fix). No sleep state: the digital human simply rests
during idle.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..state.model import State  # noqa: F401  (documents the fields this reads/writes)
from .circadian import circadian
from .component import TickContext
from .intents import Intent, UpdateState
from .timeutil import minutes_between


class Personality:
    """Holds and evolves physiology (energy, fatigue) against the circadian clock."""

    def __init__(
        self,
        *,
        e_max: float,
        recovery_per_min: float,
        night_boost: float,
        fatigue_decay_per_min: float,
        peak_hour_utc: float,
        id: str = "personality",
    ) -> None:
        self.id = id
        self._e_max = e_max
        self._recovery_per_min = recovery_per_min
        self._night_boost = night_boost
        self._fatigue_decay_per_min = fatigue_decay_per_min
        self._peak_hour_utc = peak_hour_utc

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        dt = minutes_between(state.last_tick_at, ctx.now)
        c = circadian(ctx.now, peak_hour_utc=self._peak_hour_utc)

        recovery = self._recovery_per_min * (1.0 + self._night_boost * (1.0 - c))
        energy = min(self._e_max, state.energy + recovery * dt)
        fatigue = max(0.0, state.fatigue - self._fatigue_decay_per_min * dt)

        return [UpdateState({"energy": energy, "fatigue": fatigue})]
```
(If ruff/mypy flags the `State` import as unused, drop it — it is only there to document intent.) Re-export `Personality` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_personality.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/personality.py core/__init__.py tests/test_personality.py
uv run ruff check core/personality.py core/__init__.py tests/test_personality.py
uv run mypy -p lifemodel
git add core/personality.py core/__init__.py tests/test_personality.py
git commit -m "feat(core): Personality component — recover E, decay S, circadian-modulated (spec §8)"
```

---

### Task 4: Register Personality in the composition root + pipeline integration

**Files:**
- Modify: `composition.py`
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Produces: `build_lifemodel(...)` registers `Personality`; a `coreloop.tick()` recovers `E` and decays `S`.

**Behavior:** construct `Personality` with the bootstrap constants and register it (enabled). Order does not matter for correctness (all components read the same pre-tick state snapshot); register it **first** for semantic clarity (physiology before drives). Still no live cutover.

- [ ] **Step 1: Write the failing test (append to `tests/test_composition.py`)**

```python
from lifemodel.core.personality import Personality


def test_personality_is_registered(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    assert any(isinstance(c, Personality) for c in lm.registry.enabled())


def test_pipeline_tick_recovers_energy_and_decays_fatigue(tmp_path) -> None:
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State

    store = JsonStateStore(tmp_path)
    store.commit(State(energy=0.5, fatigue=0.5, last_tick_at="2026-07-06T12:00:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 12, 30, tzinfo=UTC)))
    lm.coreloop.tick()
    final = store.load()
    assert final.energy > 0.5  # recovered during the idle tick
    assert final.fatigue < 0.5  # decayed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py -q`
Expected: FAIL — no `Personality` in the registry.

- [ ] **Step 3: Implement in `composition.py`**

Add module constants near the other `CONTACT_*` constants:
```python
E_MAX = 1.0
ENERGY_RECOVERY_PER_MIN = 0.01
NIGHT_RECOVERY_BOOST = 0.5
FATIGUE_DECAY_PER_MIN = 0.002
CIRCADIAN_PEAK_UTC_HOUR = 13.0  # peak alertness 16:00 MSK, trough 04:00 MSK
```
Add the import `from .core.personality import Personality`. In `build_lifemodel`, after the registry is resolved and **before** registering the contact neuron, register personality (same `UnknownComponent` double-registration guard used for the others):
```python
    personality = Personality(
        e_max=E_MAX,
        recovery_per_min=ENERGY_RECOVERY_PER_MIN,
        night_boost=NIGHT_RECOVERY_BOOST,
        fatigue_decay_per_min=FATIGUE_DECAY_PER_MIN,
        peak_hour_utc=CIRCADIAN_PEAK_UTC_HOUR,
    )
    try:
        registry.manifest(personality.id)
    except UnknownComponent:
        registry.register(personality, ComponentManifest(id=personality.id, type="personality"))
```

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new tests pass; every prior test (incl. `tests/sim/`) still passes. (Personality only runs via `coreloop.tick()`; the live path is unchanged.)

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format composition.py tests/test_composition.py
uv run ruff check composition.py tests/test_composition.py
uv run mypy -p lifemodel
git add composition.py tests/test_composition.py
git commit -m "feat(core): register Personality component in composition (physiology first) (spec §8)"
```

---

## Phase-C2 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Four commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §8 two-process energy (S fatigue, C circadian, E budget) → Tasks 1+3; `cost_real=cost_base·(1+αS)` + reserve/settle/refund + don't-petrify → Task 2; "energy is part of personality" (§9) → the `Personality` component (Task 3); DI wiring → Task 4. **Deferred (by design):** the *spend* side (cognition deducts E via reserve/settle, raises S) and the **cognition energy gate** + emergent progressive shutoff → Phase D (cognition is energy's only consumer); arousal A + Yerkes-Dodson → Phase D; the §17 two-time dt-cap (laptop-sleep) → Phase D (here recovery uses raw dt — a long idle *should* fully recover, which is correct rest); **no sleep state** (user decision) — recovery is idle+circadian, a real sleep/dreaming consolidation feature is future work.
- **Type consistency:** `circadian(now, *, peak_hour_utc)` identical in Tasks 1, 3, 4. `reserve/settle/cost_real/Reservation` signatures identical in Task 2. `Personality(*, e_max, recovery_per_min, night_boost, fatigue_decay_per_min, peak_hour_utc)` identical in Tasks 3, 4.
- **Coma-safety:** Personality is a cheap 0-energy layer; it always runs and recovers E even at E≤0, so the being is never locked out of recovering — the §3 coma-fix holds.
- **No placeholders:** every step ships real code + an exact command with expected output.

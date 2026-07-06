# Core Rebuild — Phase C1: Latent/Effective Pressure + ActionPending Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the contact drive into **latent** (true deprivation) and **effective** (permission-to-wake) pressure, and add **ActionPending** — a decaying inhibition with a grace-period plateau that suppresses re-waking right after the being reaches out (spec §9.1, §9.2). This makes **send ≠ contact**: a FULFILL no longer satiates the drive; it starts an inhibition window. Only a real user exchange satiates. Still **no live cutover**.

**Architecture (spec §9):** The certified drive `u` (owned by `ContactNeuron`) IS the **latent** pressure — it rises by wall-clock silence and is satiated only by a real exchange (unchanged). `ContactAggregation` now computes an **inhibition** ∈ [0,1] from an `action_pending_since` clock (grace plateau then exponential decay), derives **`effective = latent · (1 − inhibition)`**, and gates the wake decision on **effective** instead of raw `u`. `duration_over_theta` stays on **latent** (§9.1 anti-neglect — a long-ignored need keeps escalating even while inhibited). A verdict `FULFILL` sets `action_pending_since = now` (send happened) and does **not** satiate/stamp-exchange/reset-duration; a real exchange clears `action_pending_since` (contact resolved).

**Why this diverges from the certified sim on FULFILL:** the certified `sim` models the OLD "send = contact" (FULFILL satiates). The spec deliberately overturns that (§2.5, §9.2). So from here the aggregation's FULFILL diverges from `sim`; the `sim` remains the reference only for the *drive math* (rise / satiate-on-real-contact), which we still reuse. `tests/sim/` stays green (it tests the sim directly, not the aggregation).

**Tech Stack:** Python 3.11 stdlib-only core (`math` for the decay); the certified `sim.wake.evaluate_wake` (reused). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout;** package-internal uses relative imports; **core imports no Hermes.**
- **Latent = truth, effective = permission (spec §9.1, Codex rev.5):** `duration_over_theta` and escalation read **latent** (`u`); only the wake threshold reads **effective**. Never let effective become the persisted history.
- **Send ≠ contact (§2.5, §9.2):** FULFILL sets ActionPending, does **not** satiate `u`, does **not** set `last_exchange_at`, does **not** reset `duration_over_theta`. Only a real exchange satiates (`ContactNeuron`, B1) and clears ActionPending.
- **`[SILENT]`/`REJECT` do NOT set ActionPending (Codex):** only FULFILL does. REJECT keeps its existing decline-backoff.
- **Bootstrap constants (spec §22, one-line tunable):** `INHIBITION_I0 = 1.0`, `ACTION_PENDING_GRACE_MIN = 45.0`, `INHIBITION_HALFLIFE_MIN = 60.0` (λ = ln 2 / half-life). During grace effective ≈ 0 (guaranteed quiet window); then inhibition halves each hour, so an *ignored* send lets latent loneliness re-cross the threshold after ~1.5–2 h — not spammy, not neglectful.
- **`mypy -p lifemodel` strict; all numeric clamped/finite.**
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `state/model.py` — add `action_pending_since: str | None = None` (Task 1).
- Create `core/pressure.py` — `inhibition_at`, `effective_pressure` (Task 1).
- Modify `core/aggregation.py` — gate on effective pressure (Task 2); FULFILL→ActionPending, exchange clears it (Task 3).
- Modify `tests/test_aggregation.py` — **replace** the two B2 FULFILL tests with send≠contact versions; add effective/inhibition tests (Tasks 2–3).
- Modify `composition.py` — wire the inhibition constants into `ContactAggregation` (Task 4).
- Modify `core/__init__.py` — re-export new names.
- Tests: `tests/test_pressure.py`, extend `tests/test_state_model.py`, extend `tests/test_aggregation.py`, extend `tests/test_composition.py`.

**Interfaces produced (Phases D/E consume):**
- `state.model.State.action_pending_since: str | None`.
- `core/pressure.py`: `inhibition_at(action_pending_since: str | None, now: datetime, *, i0: float, grace_min: float, halflife_min: float) -> float`; `effective_pressure(latent: float, inhibition: float) -> float`.
- `ContactAggregation(*, params, theta, beta, u_max, i0, grace_min, halflife_min, id="contact-aggregation")`.

---

### Task 1: State field + pressure math (`inhibition_at`, `effective_pressure`)

**Files:**
- Modify: `state/model.py`
- Create: `core/pressure.py`
- Modify: `core/__init__.py`
- Test: `tests/test_state_model.py` (extend), `tests/test_pressure.py`

**Interfaces:**
- Consumes: `timeutil.minutes_between` (Phase B1).
- Produces: `State.action_pending_since`; `inhibition_at`, `effective_pressure`.

**Behavior (spec §9.2):** `inhibition_at` returns `0.0` when there is no ActionPending; `i0` during the grace plateau (`t ≤ grace_min`); then `i0·e^(−λ(t−grace_min))` with `λ = ln2/halflife_min`. Result clamped to `[0,1]`. `effective_pressure(latent, inhibition) = max(0, latent·(1−inhibition))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pressure.py
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from lifemodel.core.pressure import effective_pressure, inhibition_at

BASE = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def test_no_action_pending_is_zero_inhibition() -> None:
    assert inhibition_at(None, BASE, i0=1.0, grace_min=45.0, halflife_min=60.0) == 0.0


def test_grace_plateau_holds_full_inhibition() -> None:
    since = BASE.isoformat()
    assert inhibition_at(since, _at(0), i0=1.0, grace_min=45.0, halflife_min=60.0) == 1.0
    assert inhibition_at(since, _at(44), i0=1.0, grace_min=45.0, halflife_min=60.0) == 1.0


def test_decays_by_halflife_after_grace() -> None:
    since = BASE.isoformat()
    # one half-life (60 min) after the grace end (45 min) -> ~0.5
    val = inhibition_at(since, _at(45 + 60), i0=1.0, grace_min=45.0, halflife_min=60.0)
    assert abs(val - 0.5) < 1e-9


def test_inhibition_clamped_to_unit_interval() -> None:
    since = BASE.isoformat()
    v = inhibition_at(since, _at(10_000), i0=1.0, grace_min=45.0, halflife_min=60.0)
    assert 0.0 <= v <= 1.0
    assert v < 1e-3  # long after: essentially gone


def test_effective_pressure_suppressed_by_inhibition() -> None:
    assert effective_pressure(2.0, 1.0) == 0.0  # fully inhibited
    assert effective_pressure(2.0, 0.0) == 2.0  # no inhibition
    assert abs(effective_pressure(2.0, 0.5) - 1.0) < 1e-9
    assert effective_pressure(0.0, 0.0) == 0.0
```

```python
# append to tests/test_state_model.py
def test_action_pending_since_roundtrips() -> None:
    s = State(action_pending_since="2026-07-06T12:00:00+00:00")
    assert State.from_dict(s.to_dict()).action_pending_since == "2026-07-06T12:00:00+00:00"


def test_action_pending_since_defaults_none() -> None:
    assert State().action_pending_since is None
    assert State.from_dict({}).action_pending_since is None  # additive: missing key is fine
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pressure.py tests/test_state_model.py -q`
Expected: FAIL — `ModuleNotFoundError` for `lifemodel.core.pressure`; `State` has no `action_pending_since`.

- [ ] **Step 3: Implement**

In `state/model.py`, add the field to the `State` dataclass — place it **next to `pending_proactive_since`** (they are different clocks: `pending_proactive_since` tracks an in-flight proactive turn awaiting a verdict; `action_pending_since` is the ActionPending inhibition clock started when the being reaches out). Follow the existing `_as_opt_iso` validation pattern used by the other timestamp fields in `from_dict`:
```python
    action_pending_since: str | None = None
```
and in `from_dict`, add (mirroring the other optional-ISO fields):
```python
        action_pending_since=_as_opt_iso(data, "action_pending_since"),
```
(Confirm `to_dict` is `asdict`-based so the new field serializes automatically; the existing tests for round-trip will exercise it.)

```python
# core/pressure.py
"""Latent vs effective pressure + ActionPending inhibition (spec §9.1, §9.2).

The certified drive ``u`` is the *latent* pressure (true deprivation; rises by
silence, satiated only by real contact). After the being reaches out we suppress
*effective* pressure — the permission to wake — with an ``inhibition`` that holds
at ``i0`` for a grace plateau (a guaranteed quiet window) then decays
exponentially. ``effective = latent · (1 − inhibition)``. Escalation/duration read
latent; only the wake threshold reads effective.
"""

from __future__ import annotations

import math
from datetime import datetime

from .timeutil import minutes_between


def inhibition_at(
    action_pending_since: str | None,
    now: datetime,
    *,
    i0: float,
    grace_min: float,
    halflife_min: float,
) -> float:
    """Inhibition ∈ [0,1] at ``now`` for an ActionPending started at
    ``action_pending_since`` (ISO). ``None`` → 0. Grace plateau then half-life
    decay."""
    if action_pending_since is None:
        return 0.0
    t = minutes_between(action_pending_since, now)
    if t <= grace_min:
        value = i0
    else:
        lam = math.log(2.0) / halflife_min
        value = i0 * math.exp(-lam * (t - grace_min))
    return max(0.0, min(1.0, value))


def effective_pressure(latent: float, inhibition: float) -> float:
    """Permission-to-wake pressure: ``max(0, latent·(1−inhibition))``."""
    return max(0.0, latent * (1.0 - inhibition))
```
Re-export `inhibition_at, effective_pressure` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pressure.py tests/test_state_model.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format state/model.py core/pressure.py core/__init__.py tests/test_pressure.py tests/test_state_model.py
uv run ruff check state/model.py core/pressure.py core/__init__.py tests/test_pressure.py tests/test_state_model.py
uv run mypy -p lifemodel
git add state/model.py core/pressure.py core/__init__.py tests/test_pressure.py tests/test_state_model.py
git commit -m "feat(core): latent/effective pressure + ActionPending inhibition math (spec §9.1/§9.2)"
```

---

### Task 2: Aggregation gates on effective pressure

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py` (extend)

**Interfaces:**
- Consumes: `pressure.{inhibition_at, effective_pressure}` (Task 1).
- Produces: `ContactAggregation` now takes `i0`, `grace_min`, `halflife_min` and gates on effective.

**Behavior (spec §9.1):** the wake evaluation uses **effective** pressure; `duration_over_theta` stays on **latent** (`u_now`). Add the three inhibition constructor params. Compute `inhibition = inhibition_at(state.action_pending_since, now, ...)` and `effective = effective_pressure(u_now, inhibition)`, and pass `u=effective` to `evaluate_wake` (the threshold gate). Everything else (silence-window, in-flight, decline-backoff, lifecycle) is unchanged.

- [ ] **Step 1: Write the failing tests (append)**

The existing `_agg()` helper keeps working — Task 2 adds the inhibition params **with defaults** (`i0=1.0, grace_min=45.0, halflife_min=60.0`), so `_agg()` (which omits them) gets exactly those bootstrap values. Use `_agg()` for the new tests.

```python
def test_action_pending_grace_suppresses_wake_despite_high_latent(tmp_path) -> None:
    # latent u=3 (>= theta) but a send 10 min ago (within 45-min grace) -> effective ~0 -> no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0, desire_status="none",
        action_pending_since="2026-07-06T03:50:00+00:00",  # 10 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # inhibited during grace


def test_pressure_recovers_after_grace_and_decay(tmp_path) -> None:
    # send ~3h ago: grace(45m)+ ~2 half-lives -> inhibition ~0.06 -> effective ~ u -> wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0, desire_status="none",
        action_pending_since="2026-07-06T01:00:00+00:00",  # 180 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"  # ignored long enough -> loneliness returns


def test_duration_over_theta_uses_latent_not_effective(tmp_path) -> None:
    # even fully inhibited (effective 0), latent u>=theta so duration keeps accruing
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5
    state = State(
        u=2.0, desire_status="none", duration_over_theta=10.0,
        action_pending_since="2026-07-06T00:04:00+00:00",  # in grace -> inhibition 1
        last_tick_at="2026-07-06T00:00:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert abs(changes["duration_over_theta"] - 15.0) < 1e-9  # latent-based, accrues under inhibition
    assert changes["desire_status"] == "none"  # but no wake (effective suppressed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — `ContactAggregation` has no `i0`/`grace_min`/`halflife_min` params.

- [ ] **Step 3: Implement in `core/aggregation.py`**

Add the import `from .pressure import effective_pressure, inhibition_at`. Extend `__init__` with `i0: float = 1.0`, `grace_min: float = 45.0`, `halflife_min: float = 60.0` (keyword-only, **with these bootstrap defaults** so existing construction — the `_agg()` helper and `build_lifemodel`, wired authoritatively in Task 4 — keeps working without a signature break mid-plan), stored on `self`. In `step`, after computing `u_now` and before the wake gates, derive effective:
```python
        inhibition = inhibition_at(
            state.action_pending_since, now,
            i0=self._i0, grace_min=self._grace_min, halflife_min=self._halflife_min,
        )
        effective = effective_pressure(u_now, inhibition)
```
Change the wake evaluation to use effective (leave `duration` on `u_now`):
```python
        outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=self._params)
```
(The `duration` computation still reads `u_now >= self._theta` — latent — unchanged.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — all prior B2 tests still pass (with `action_pending_since=None` → inhibition 0 → effective == u, so the gate is unchanged), the 3 new effective-pressure tests pass, and `build_lifemodel`/composition still work (the new params default to the bootstrap values until Task 4 wires them explicitly).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): aggregation gates on effective pressure (latent·(1−inhibition)) (spec §9.1)"
```

---

### Task 3: FULFILL sets ActionPending (send ≠ contact); exchange clears it

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py` (**replace** the two old FULFILL tests + add new)

**Interfaces:**
- Produces: FULFILL → `action_pending_since`; exchange → clears `action_pending_since`.

**Behavior (spec §2.5, §9.2):** a FULFILL verdict means the being **reached out** — it starts the ActionPending inhibition and records the outreach, but it is **not** contact: it does **not** satiate `u`, does **not** set `last_exchange_at`, does **not** reset `duration_over_theta`. A real exchange (already resets clocks in B2) additionally **clears** `action_pending_since` (contact resolved the pull). REJECT/DEFER are unchanged and never set ActionPending.

- [ ] **Step 1: Update the tests**

In `tests/test_aggregation.py`, **replace** `test_fulfill_satiates_u_and_stamps_contact` and `test_fulfill_resets_duration_even_when_u_stays_high` (old "send = contact" model) with:

```python
def test_fulfill_starts_action_pending_and_does_not_satiate(tmp_path) -> None:
    # send != contact: FULFILL sets action_pending_since, leaves u and duration alone
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.5, desire_status="active", duration_over_theta=99.0,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # desire resolved
    assert changes["action_pending_since"] == now.isoformat()  # ActionPending started
    assert "u" not in changes  # NOT satiated — send is not contact
    assert changes["last_exchange_at"] is None  # send does not count as an exchange
    assert changes["last_contact_at"] == now.isoformat()  # our outreach is recorded (observability)
    assert changes["duration_over_theta"] != 0.0  # latent duration NOT reset by a mere send


def test_exchange_clears_action_pending(tmp_path) -> None:
    # a real reply resolves the pull: clears ActionPending (neuron satiates u separately)
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.0, desire_status="active",
        action_pending_since="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.0, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["action_pending_since"] is None  # contact resolved the pull
    assert changes["desire_status"] == "none"


def test_reject_does_not_set_action_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["action_pending_since"] is None  # REJECT never inhibits
    assert changes["decline_count"] == 1  # existing backoff bookkeeping intact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — the new FULFILL/exchange assertions fail (old FULFILL still satiates; exchange doesn't clear ActionPending; `action_pending_since` never written).

- [ ] **Step 3: Implement in `core/aggregation.py`**

Add a local accumulator near the top of `step` (with the other locals): `action_pending_since = state.action_pending_since`.

In the **exchange** block, also clear ActionPending:
```python
                if actor != "proactive_internal":
                    last_exchange_at = now.isoformat()
                    declined_at = None
                    decline_count = 0
                    action_pending_since = None  # real contact resolves the pull
                    agg.on_exchange()
```

**Replace** the FULFILL branch of the verdict block (send ≠ contact — no satiate, no exchange stamp, no duration reset; start ActionPending + record outreach):
```python
                if verdict is Verdict.FULFILL:
                    action_pending_since = now.isoformat()  # send happened -> inhibition starts
                    last_contact_at = now.isoformat()  # record our outreach (observability only)
```
Remove the old FULFILL body (the `Drive(...).satiate`, `u_now`/`u_out` assignment, `fulfilled = True`, `last_exchange_at = now`). Drop the now-unused `fulfilled` local and the `if fulfilled:` branch of the duration computation, restoring it to:
```python
        dt = minutes_between(state.last_tick_at, now)
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0
```
(`u_out` stays declared as `None` and is now never set, so `u` is never in `changes` — the neuron is the sole writer of `u` again. You may drop `u_out`/the `Drive` import if unused; keep `Drive` only if still referenced elsewhere — it is not, so remove that import.)

Finally add `action_pending_since` to the returned `changes` dict:
```python
        changes: dict[str, object] = {
            "desire_status": agg.status.value,
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
            "action_pending_since": action_pending_since,
        }
        return [UpdateState(changes)]
```
(The `if u_out is not None` block is removed since `u_out` is never set now.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — updated FULFILL/exchange tests pass; every prior test (incl. `tests/sim/`) still passes. (The Task-2 effective-pressure tests still hold; `action_pending_since=None` in older tests → inhibition 0 → behavior unchanged.)

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): FULFILL sets ActionPending, not satiation — send≠contact; exchange clears it (spec §2.5/§9.2)"
```

---

### Task 4: Wire inhibition constants into the composition root + pipeline integration

**Files:**
- Modify: `composition.py`
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Produces: `build_lifemodel` constructs `ContactAggregation` with the inhibition bootstrap constants.

**Behavior:** pass the bootstrap inhibition constants into `ContactAggregation`. Define them once as module constants (tunable in one place).

- [ ] **Step 1: Write the failing test (append to `tests/test_composition.py`)**

```python
def test_pipeline_send_suppresses_then_recovers(tmp_path) -> None:
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State

    store = JsonStateStore(tmp_path)
    # high latent, a send 10 min ago -> within grace -> no wake this tick
    store.commit(State(
        u=3.0, desire_status="none",
        action_pending_since="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    ))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC)))
    lm.coreloop.tick()
    assert store.load().desire_status == "none"  # grace suppresses the wake end-to-end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py -q`
Expected: FAIL — `ContactAggregation(...)` in `build_lifemodel` is missing the new required params (`TypeError`).

- [ ] **Step 3: Implement in `composition.py`**

Add module constants near `CONTACT_PARAMS`:
```python
CONTACT_I0 = 1.0
CONTACT_GRACE_MIN = 45.0
CONTACT_INHIBITION_HALFLIFE_MIN = 60.0
```
Update the `ContactAggregation(...)` construction to pass them:
```python
    aggregation = ContactAggregation(
        params=CONTACT_PARAMS,
        theta=CONTACT_PARAMS.theta_u,
        beta=CONTACT_BETA,
        u_max=CONTACT_U_MAX,
        i0=CONTACT_I0,
        grace_min=CONTACT_GRACE_MIN,
        halflife_min=CONTACT_INHIBITION_HALFLIFE_MIN,
    )
```

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — the new integration test passes; every prior test still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format composition.py tests/test_composition.py
uv run ruff check composition.py tests/test_composition.py
uv run mypy -p lifemodel
git add composition.py tests/test_composition.py
git commit -m "feat(core): wire ActionPending inhibition constants into composition (spec §9.2)"
```

---

## Phase-C1 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Four commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §9.1 latent vs effective (wake on effective, duration on latent) → Tasks 1–2; §9.2 ActionPending grace-plateau-then-decay → Task 1 (math) + Task 2 (gate) + Task 3 (set on FULFILL, clear on exchange); §2.5 send≠contact (FULFILL no longer satiates) → Task 3; DI wiring → Task 4. **Codex invariants honored:** `[SILENT]`/`REJECT` don't inhibit (Task 3 test); latent is the truth (duration/escalation read `u`, not effective). **Deferred:** "ActionPending only after *confirmed delivery*" (Codex) needs the real send/delivery path → Phase E cutover (here FULFILL stands in for the send); learned set-points → v2; arousal-coupled inhibition → later.
- **Type consistency:** `inhibition_at(action_pending_since, now, *, i0, grace_min, halflife_min)` / `effective_pressure(latent, inhibition)` used identically in Tasks 1, 2. `ContactAggregation(*, params, theta, beta, u_max, i0, grace_min, halflife_min)` identical in Tasks 2, 3, 4 (and the `_agg()`/`_agg_i()` test helpers).
- **Behavior-preservation check:** with `action_pending_since=None` (all pre-C1 states), inhibition is 0 → effective == latent → the wake gate is identical to B2, so only the FULFILL semantics (Task 3, intentionally) and the new ActionPending paths change. `tests/sim/` untouched.
- **No placeholders:** every step ships real code + an exact command with expected output.

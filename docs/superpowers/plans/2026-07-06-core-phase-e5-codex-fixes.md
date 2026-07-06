# Core Rebuild — Phase E5: Codex Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three real runtime bugs the whole-branch Codex review (`019f3795`) found, before merge: (1) **negative-`dt` corrupts physiology** (a clock going backward reduces energy / raises fatigue / shrinks duration); (2) **energy reservation leak** (a proactive launch deducts energy even when the backstop blocks it or the launch fails — nothing refunds); (3) **`/lifemodel debug` under-reports wake readiness** (it reads the persisted `u`, but the live tick first rises `u`). The other findings are documented follow-up beads (`lm-s56`, `lm-2gi`, `lm-l79`).

**Tech Stack:** Python 3.11. `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.** `mypy -p lifemodel` strict. `tests/sim/` must stay green.
- **Do NOT** push/merge/touch `main`. **Do NOT** modify `tick.py`, `heartbeat.py`.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `core/personality.py`, `core/aggregation.py` — clamp `dt ≥ 0` (Task 1).
- Modify `core/intents.py` (`LaunchProactive.reserved_energy`), `core/cognition.py`, `egress_service.py` — refund on blocked/failed launch (Task 2).
- Modify `core/introspect.py` (`DebugConfig` + rise `u` for display), `debug.py` — accurate debug drive (Task 3).
- Tests: extend `tests/test_personality.py`, `tests/test_aggregation.py`, `tests/test_cognition.py`, `tests/test_egress_service_tick.py`, `tests/test_introspect.py`.

---

### Task 1: Clamp `dt ≥ 0` in physiology + aggregation (Codex #6)

**Files:** `core/personality.py`, `core/aggregation.py`; `tests/test_personality.py`, `tests/test_aggregation.py`.

**Behavior:** a `last_tick_at` in the future (clock skew) makes `minutes_between` negative; integrating a negative `dt` runs physiology backward. Clamp `dt = max(0.0, …)` wherever it drives dynamics. (The contact neuron already guards `dt > 0`.)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_personality.py
def test_negative_dt_does_not_run_physiology_backward(tmp_path) -> None:
    # last_tick_at is in the FUTURE relative to now -> negative dt
    state = State(energy=0.5, fatigue=0.5, last_tick_at="2026-07-06T12:10:00+00:00")
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)  # 10 min BEFORE last_tick
    changes = _changes(_p().step(_ctx(state, now, tmp_path=tmp_path)))
    assert changes["energy"] == 0.5  # not reduced
    assert changes["fatigue"] == 0.5  # not increased
```

```python
# append to tests/test_aggregation.py
def test_negative_dt_does_not_shrink_duration(tmp_path) -> None:
    state = State(u=2.0, desire_status="none", duration_over_theta=30.0, last_tick_at="2026-07-06T12:10:00+00:00")
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)  # before last_tick
    c = contact_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["duration_over_theta"] == 30.0  # unchanged (dt clamped to 0), not reduced
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_personality.py tests/test_aggregation.py -q -k "negative_dt"`
Expected: FAIL — physiology/duration move on negative dt.

- [ ] **Step 3: Implement**

In `core/personality.py::step`, change `dt = minutes_between(state.last_tick_at, ctx.now)` to:
```python
        dt = max(0.0, minutes_between(state.last_tick_at, ctx.now))
```
In `core/aggregation.py::step`, change `dt = minutes_between(state.last_tick_at, now)` to:
```python
        dt = max(0.0, minutes_between(state.last_tick_at, now))
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_personality.py tests/test_aggregation.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/personality.py core/aggregation.py tests/test_personality.py tests/test_aggregation.py
uv run ruff check core/personality.py core/aggregation.py tests/test_personality.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/personality.py core/aggregation.py tests/test_personality.py tests/test_aggregation.py
git commit -m "fix(core): clamp dt>=0 so a backward clock can't run physiology in reverse (Codex #6)"
```

---

### Task 2: Refund the energy reservation on blocked/failed launch (Codex #3)

**Files:** `core/intents.py`, `core/cognition.py`, `egress_service.py`; `tests/test_cognition.py`, `tests/test_egress_service_tick.py`.

**Behavior:** `Cognition` reserves (deducts) the proactive-turn energy at launch. If the backstop then **blocks** the send, or the launch **fails** to reach out, no native turn runs — so that reservation must be **refunded**. The `LaunchProactive` intent carries the reserved amount so the egress can refund it in its reconciliation commit. (A `DELIVERED` launch keeps the cost — the turn ran. A `[SILENT]` turn also ran, so it legitimately keeps the cost; the exact per-verdict settle is a documented follow-up, `lm-l79`.)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_cognition.py
def test_launch_carries_the_reserved_energy(tmp_path) -> None:
    state = State(desire_status="active", u=2.0, energy=1.0, fatigue=0.0)
    launch = _launch(_cog().step(_ctx(state, tmp_path=tmp_path)))
    # estimate = (0.02+0.03)*(1+2*0) = 0.05
    assert abs(launch.reserved_energy - 0.05) < 1e-9
```

```python
# append to tests/test_egress_service_tick.py
def test_backstop_block_refunds_the_reservation(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    log = ["2026-07-06T11:00:00+00:00", "2026-07-06T10:00:00+00:00", "2026-07-06T09:00:00+00:00"]
    state = State(desire_status="active", u=2.0, energy=1.0, proactive_send_log=log, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    run_proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    # cognition deducted 0.05, then the block refunded it -> back to ~1.0 (minus any recovery)
    assert lm.state.load().energy >= 0.99


def test_failed_launch_refunds_the_reservation(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="active", u=2.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    run_proactive_tick(lm, FakeEgress(outcome=ReachOutcome.UNAVAILABLE), TARGET, logger=get_logger("t"))
    assert lm.state.load().energy >= 0.99  # refunded — no turn ran


def test_delivered_launch_keeps_the_cost(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="active", u=2.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    run_proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    assert lm.state.load().energy < 1.0  # the turn ran -> energy spent, not refunded
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cognition.py tests/test_egress_service_tick.py -q`
Expected: FAIL — `LaunchProactive` has no `reserved_energy`; egress does not refund.

- [ ] **Step 3: Implement**

In `core/intents.py`, add the field to `LaunchProactive`:
```python
@dataclass(frozen=True)
class LaunchProactive(Intent):
    prompt: str
    correlation_id: str
    reserved_energy: float = 0.0
```
In `core/cognition.py`, pass the estimate into the intent:
```python
        return [
            LaunchProactive(
                prompt=packet.prompt, correlation_id=correlation_id, reserved_energy=estimate
            ),
            UpdateState(
                {
                    "energy": energy_after,
                    "pending_proactive_id": correlation_id,
                    "pending_proactive_since": ctx.now.isoformat(),
                }
            ),
        ]
```
In `egress_service.py::run_proactive_tick`, track a refund and apply it in the reconciliation commit. Replace the launch block + reconciliation with:
```python
    outcome = ReachOutcome.SKIPPED_BUSY
    rollback_status: str | None = None
    refund_energy = 0.0
    if report.launches:
        state = lm.state.load()
        launch = report.launches[0]
        if not allow_send(state.proactive_send_log, now):
            rollback_status = "deferred"  # backstop: hold, send nothing (spec §14)
            refund_energy = launch.reserved_energy  # no turn ran -> refund the reservation
            logger.info("proactive_backstop_blocked")
        else:
            outcome = egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)
            if outcome is not ReachOutcome.DELIVERED:
                rollback_status = "active"  # launch failed — keep active to retry
                refund_energy = launch.reserved_energy  # no turn ran -> refund
                logger.info("proactive_launch_failed", outcome=outcome.value)

    # one reconciliation commit: liveness + optional pending rollback + reservation refund
    state = lm.state.load()
    state.egress_service_alive_at = now.isoformat()
    if rollback_status is not None:
        state.pending_proactive_id = None
        state.pending_proactive_since = None
        state.desire_status = rollback_status
        state.energy += refund_energy
    lm.state.commit(state)
    logger.info("proactive_tick", launches=len(report.launches), outcome=outcome.value)
    return outcome
```
Add one sentence to the `run_proactive_tick` docstring: it **assumes a fresh `LifeModel` per call** (the supervised loop builds one each tick); the reconciliation commit is a host concern outside the state-actor, safe under that invariant (Codex #2).

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — cognition + egress refund tests pass; `tests/sim/` green.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/intents.py core/cognition.py egress_service.py tests/test_cognition.py tests/test_egress_service_tick.py
uv run ruff check core/intents.py core/cognition.py egress_service.py tests/test_cognition.py tests/test_egress_service_tick.py
uv run mypy -p lifemodel
git add core/intents.py core/cognition.py egress_service.py tests/test_cognition.py tests/test_egress_service_tick.py
git commit -m "fix(core): refund the energy reservation when a launch is blocked/failed (Codex #3)"
```

---

### Task 3: `/lifemodel debug` rises `u` for display (Codex #7)

**Files:** `core/introspect.py`, `debug.py`; `tests/test_introspect.py`.

**Behavior:** the live tick rises `u` in the neuron *before* aggregation reads it, so the debug view — which reads persisted `u` — under-reports the drive. Rise `u` by the elapsed-since-`last_tick` deprivation for display (clamped to `u_max`), matching what the next real tick will see. Add `alpha`/`u_max` to `DebugConfig`.

- [ ] **Step 1: Write the failing test (append to `tests/test_introspect.py`)**

```python
def test_drive_is_risen_as_of_now_for_display() -> None:
    # persisted u=0, but 240 min elapsed at alpha=1/240 -> should display ~1.0
    cfg = DebugConfig(
        params=GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0),
        theta=1.0, i0=1.0, grace_min=45.0, halflife_min=60.0,
        peak_hour_utc=13.0, max_per_day=3, min_interval_min=60.0,
        alpha=1.0 / 240.0, u_max=100.0,
    )
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min later
    r = compute_readings(state, now=now, cfg=cfg)
    assert abs(r.u - 1.0) < 1e-6  # risen as of now
    assert r.would_wake is True  # and the debug reflects the imminent wake
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_introspect.py -q -k risen`
Expected: FAIL — `DebugConfig` has no `alpha`/`u_max`; `u` is read raw.

- [ ] **Step 3: Implement**

In `core/introspect.py`, add `alpha: float` and `u_max: float` to `DebugConfig`. In `compute_readings`, replace `u = state.u` with a risen-for-display value:
```python
    dt = max(0.0, minutes_between(state.last_tick_at, now))
    u = min(cfg.u_max, state.u + dt * cfg.alpha)
```
(Everything downstream already reads the local `u`.)

In `debug.py::_cfg`, add the two fields from the composition constants:
```python
        alpha=composition.CONTACT_ALPHA,
        u_max=composition.CONTACT_U_MAX,
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/introspect.py debug.py tests/test_introspect.py
uv run ruff check core/introspect.py debug.py tests/test_introspect.py
uv run mypy -p lifemodel
git add core/introspect.py debug.py tests/test_introspect.py
git commit -m "fix(debug): rise u as-of-now so /lifemodel debug matches the next tick (Codex #7)"
```

---

## Phase-E5 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Three commits on `core/rebuild`, one per task.
- [ ] `tick.py`, `heartbeat.py` unmodified. `tests/sim/` green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review

- **Codex findings addressed:** #6 negative-dt clamp → Task 1; #3 reservation refund on blocked/failed → Task 2 (+ #2 fresh-`lm` invariant documented); #7 debug drive risen-as-of-now → Task 3. **Deferred with beads:** #1 control-losslessness (`lm-s56`), #5 delivery-confirmation (`lm-2gi`), #4 delta-summed energy (`lm-l79`) — all documented, low practical risk at v1 bootstrap values.
- **No placeholders:** every step ships real code + an exact command with expected output.

# Wire the certified desire model into the live plugin — Implementation Plan (Phase 1: drum-killer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implementer model: **sonnet** (per user). Review is batched at the END of the phase (Codex), not per task.

**Goal:** Replace the live plugin's old proactive-contact decision layer (global `State.pressure` + `StubTimerNeuron` + `ThresholdAggregator` + drain-on-wake + fixed 30-min cooldown = the 2026-07-04 drumming bug) with the certified desire model already in `src/lifemodel/sim/`, so the being hears the user, never drums, and can say "nothing to add" (NO_REPLY→reject→growing backoff).

**Architecture:** The **in-process egress service** (`run_proactive_tick`) becomes the SOLE decision brain; the cron `run_tick` is demoted to a silent watchdog (fail-closed, no proactive fallback). A thin live adapter `core/decision.py` reconstructs the certified `Drive`/`LaneState`/`Aggregator` from persisted `State` each tick, rises by *elapsed minutes*, evaluates the gates, and mutates `State` — **reusing `src/lifemodel/sim/*` directly as the single source of truth** (no reimplementation). A wake launches a proactive turn and records a *pending* desire; the verdict comes from the **final LLM output** via a `post_llm_call` observer (NO_REPLY→reject, real text→fulfill), correlated by a pending id — NOT from the egress `ReachOutcome` (which only means "turn launched"). Inbound user messages are observed via a `pre_gateway_dispatch` hook (fallback `pre_llm_call`, ignoring the lifemodel impulse prefix) → satiate + stamp `last_exchange_at` + clear reject record + resolve any live desire.

**Tech Stack:** Python 3.11, stdlib only in the plugin core (json + dataclasses); pytest / ruff / mypy-strict; reuse `lifemodel.sim.{drive,wake,aggregation,quality}`.

## Global Constraints

- **Stdlib-only in the plugin core.** The plugin loads inside Hermes' own interpreter, which may lack our deps. Import only stdlib + `lifemodel.*` + lazily-guarded Hermes host APIs (mirror `src/lifemodel/logging.py`, `heartbeat.py`). `lifemodel.sim.*` is stdlib-only and safe to import live.
- **No backward-compat, no dead code.** The old plugin was a feasibility skeleton. Remove the old decision path outright; do not keep migration shims. `State.from_dict` must *tolerate* (ignore) unknown legacy keys so an old `state.json` loads with new fields defaulted — but no field-level migration.
- **Reuse `sim/` as the single source of truth.** Do not reimplement drive/gates/lifecycle in `core/`. `core/decision.py` only *adapts* State↔primitives + elapsed-time.
- **The in-proc service is the only brain.** Cron never wakes proactively. Both must never disagree, so cron simply never decides.
- **DEFER is not triggered live in Phase 1** (no availability signal until v2, bead `lm-ocx`). The defer code path stays (exercised by sim tests), just unreachable live. Anti-drum rests on REJECT + growing backoff + desire dedup + the active-silence window.
- **BASE prior** (from `tests/sim/test_sim_scenarios.py::BASE`, spec §9): `α=1/240` (per-minute), `θ=1`, `β=1`, `W=15` min, `r0=30`, `k=2`, `r_max=1440`. Phase 1 hardcodes these as module defaults; disk hot-reload is Phase 2.
- **TDD.** Every new function gets a failing test first. Pure units tested with fakes (a frozen `FakeClock`, an in-memory `State` store) — no Hermes required.

## File Structure

- Create `src/lifemodel/core/decision.py` — the live decision adapter (State ↔ sim primitives + elapsed-time). One responsibility: given a `State` + `now` + `busy`, evolve the drive and return whether to wake + apply a verdict.
- Create `tests/test_decision.py` — unit tests for the adapter.
- Modify `src/lifemodel/state/model.py` — add lifecycle fields; remove `pressure`/`cooldown_until`; tolerate unknown keys.
- Modify `src/lifemodel/egress_service.py` — `run_proactive_tick` calls `core/decision`; records the pending proactive desire; no old drain/cooldown.
- Modify `src/lifemodel/tick.py` — demote `run_tick` to a silent watchdog.
- Modify `src/lifemodel/composition.py` — drop the `ThresholdAggregator`/`StubTimerNeuron` decision defaults from the live path.
- Create `src/lifemodel/hooks.py` — the `pre_gateway_dispatch`/`pre_llm_call` inbound observer + the `post_llm_call` verdict observer (both host-API-guarded, mirroring `gateway_core.py`).
- Modify `src/lifemodel/__init__.py` — register the two new hooks in `register(ctx)`.
- Modify `src/lifemodel/adapters/reachin.py` — remove the stale `runner._running_agents` busy-skip (centralize busy in the gate).

---

## Task 1: State lifecycle fields

**Files:**
- Modify: `src/lifemodel/state/model.py`
- Test: `tests/test_state_model.py`

**Interfaces:**
- Produces: `State` dataclass with new fields — `u: float = 0.0`, `duration_over_theta: float = 0.0`, `last_exchange_at: str | None = None`, `desire_status: str = "none"`, `declined_at: str | None = None`, `decline_count: int = 0`, `pending_proactive_id: str | None = None`, `pending_proactive_since: str | None = None`. Removes `pressure`, `cooldown_until`. Keeps `schema_version`, `tick_count`, `energy`, `last_tick_at`, `last_contact_at`, `egress_service_alive_at`.
- `last_exchange_at`, `declined_at`, `pending_proactive_since` are tz-aware ISO strings (validated via `_as_opt_iso`, since they are compared to `now`). `desire_status ∈ {"none","active","deferred"}` (plain `_as_opt_str`-style, non-null). `u`/`duration_over_theta` via `_as_float`; `decline_count` via `_as_int`.

- [ ] **Step 1: Write failing tests** in `tests/test_state_model.py` (add to the existing file):

```python
def test_state_has_lifecycle_fields_with_defaults():
    s = State()
    assert s.u == 0.0
    assert s.duration_over_theta == 0.0
    assert s.last_exchange_at is None
    assert s.desire_status == "none"
    assert s.declined_at is None
    assert s.decline_count == 0
    assert s.pending_proactive_id is None
    assert s.pending_proactive_since is None

def test_state_roundtrips_lifecycle_fields():
    s = State(u=42.0, duration_over_theta=7.0, last_exchange_at="2026-07-05T10:00:00+00:00",
              desire_status="active", declined_at="2026-07-05T09:00:00+00:00", decline_count=3,
              pending_proactive_id="p-1", pending_proactive_since="2026-07-05T10:01:00+00:00")
    assert State.from_dict(s.to_dict()) == s

def test_from_dict_ignores_unknown_legacy_keys():
    # Old state.json carried pressure/cooldown_until; they must be dropped, not crash.
    data = {"schema_version": 1, "pressure": 5.0, "cooldown_until": "2026-01-01T00:00:00+00:00", "u": 3.0}
    s = State.from_dict(data)
    assert s.u == 3.0
    assert not hasattr(s, "pressure")

def test_naive_lifecycle_timestamp_is_corruption():
    import pytest
    from lifemodel.state.errors import StateCorruptError
    with pytest.raises(StateCorruptError):
        State.from_dict({"schema_version": 1, "last_exchange_at": "2026-07-05T10:00:00"})  # no tz
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `.venv/bin/python -m pytest tests/test_state_model.py -q`
Expected: FAIL (unexpected keyword / attribute errors — fields don't exist).

- [ ] **Step 3: Implement** in `src/lifemodel/state/model.py`: remove the `pressure` and `cooldown_until` fields; add the eight new fields in serialization order after `energy`; route `last_exchange_at`/`declined_at`/`pending_proactive_since` through `_as_opt_iso`, `desire_status` through a non-null string validator (default `"none"`), `u`/`duration_over_theta` through `_as_float`, `decline_count` through `_as_int`, `pending_proactive_id` through `_as_opt_str`. In `from_dict`, iterate only over known field names (ignore unknown keys) instead of `**data`, so legacy keys are dropped. Do NOT bump `SCHEMA_VERSION` (still 1; additive+tolerant).

- [ ] **Step 4: Run tests, verify PASS** — `.venv/bin/python -m pytest tests/test_state_model.py -q`

- [ ] **Step 5: Update dependents that referenced the removed fields.** Grep `git grep -n "\.pressure\|cooldown_until" src/` and fix each non-test reference (debug renderer `src/lifemodel/debug.py`, any status command). For the debug renderer, render the new fields (`u`, `duration_over_theta`, `desire_status`, `last_exchange_at`, `decline_count`). Re-run `.venv/bin/python -m pytest tests/test_debug.py tests/test_state_model.py -q`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(state): lifecycle fields (u, duration, exchange, desire, reject); drop pressure/cooldown"`

---

## Task 2: `core/decision.py` — live decision adapter (reuse sim primitives)

**Files:**
- Create: `src/lifemodel/core/decision.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Consumes: `State` (Task 1); `lifemodel.sim.drive.Drive`, `lifemodel.sim.wake.{evaluate_wake,GateParams,LaneState,WakeOutcome}`, `lifemodel.sim.aggregation.{Aggregator,DesireStatus,Verdict}`, `lifemodel.sim.quality.quality_of`.
- Produces:
  - `BASE_PARAMS: GateParams` and module drive constants `ALPHA=1/240`, `BETA=1.0`, `U_MAX=100.0`, `THETA=1.0` (the BASE prior).
  - `@dataclass(frozen=True) ReachoutDecision: wake: bool; reason: str`.
  - `decide_reachout(state: State, *, now: datetime, busy: bool) -> ReachoutDecision` — rises the drive by elapsed minutes since `state.last_tick_at`, updates `duration_over_theta`, evaluates the gates, creates ONE `active` desire on a clean URGE, and **mutates `state` in place** (`u`, `duration_over_theta`, `desire_status`, `last_tick_at`). Never creates a second desire while one is live (dedup).
  - `observe_exchange(state: State, *, actor: str, label: str, now: datetime) -> None` — satiate `u` by `β·quality_of(actor,label)` for positive q, set `last_exchange_at=now`, clear reject record (`declined_at=None`, `decline_count=0`), resolve any live desire (`desire_status="none"`). Internal `proactive_internal` actor is a no-op (never satiates, never touches the clock).
  - `apply_verdict(state: State, verdict: Verdict, *, now: datetime) -> None` — FULFILL: satiate `u` by `β·1`, `duration_over_theta=0`, `desire_status="none"`, `last_exchange_at=now`, `last_contact_at=now`, clear pending. REJECT: `desire_status="none"`, `declined_at=now`, `decline_count+=1`, clear pending (no satiation). (DEFER: `desire_status="deferred"` — kept for completeness, unreachable live.)
- Helper `_minutes_between(a_iso: str | None, b: datetime) -> float` (0.0 if `a_iso` is None).

- [ ] **Step 1: Write failing tests** in `tests/test_decision.py`:

```python
from datetime import datetime, timedelta, timezone
from lifemodel.state.model import State
from lifemodel.sim.aggregation import Verdict
from lifemodel.core.decision import decide_reachout, observe_exchange, apply_verdict, THETA

UTC = timezone.utc
def at(mins): return datetime(2026, 7, 5, 0, 0, tzinfo=UTC) + timedelta(minutes=mins)

def test_rises_in_silence_and_wakes_after_urge_matures():
    # BASE α=1/240 → ~240 min of silence to cross θ=1. No prior exchange, no reject.
    s = State(last_tick_at=at(0).isoformat())
    d = decide_reachout(s, now=at(239), busy=False)
    assert d.wake is False and s.u < THETA
    d = decide_reachout(s, now=at(240), busy=False)
    assert d.wake is True and s.desire_status == "active"

def test_dedup_no_second_wake_while_desire_active():
    s = State(last_tick_at=at(0).isoformat())
    decide_reachout(s, now=at(240), busy=False)          # active
    d = decide_reachout(s, now=at(300), busy=False)      # still active
    assert d.wake is False and s.desire_status == "active"

def test_no_wake_within_active_silence_window():
    s = State(u=50.0, last_tick_at=at(0).isoformat(), last_exchange_at=at(0).isoformat())
    d = decide_reachout(s, now=at(10), busy=False)       # 10 < W=15
    assert d.wake is False and d.reason == "no_wake_silence_window"

def test_no_wake_while_busy():
    s = State(u=50.0, last_tick_at=at(0).isoformat())
    d = decide_reachout(s, now=at(30), busy=True)
    assert d.wake is False and d.reason == "no_wake_in_flight"

def test_user_exchange_satiates_and_clears_desire_and_reject():
    s = State(u=99.0, desire_status="active", declined_at=at(0).isoformat(), decline_count=3,
              last_tick_at=at(0).isoformat())
    observe_exchange(s, actor="user", label="two_way", now=at(100))
    assert s.u < 99.0 and s.desire_status == "none" and s.decline_count == 0
    assert s.declined_at is None and s.last_exchange_at == at(100).isoformat()

def test_internal_impulse_never_satiates_or_touches_clock():
    s = State(u=50.0, last_tick_at=at(0).isoformat())
    observe_exchange(s, actor="proactive_internal", label="monologue", now=at(5))
    assert s.u == 50.0 and s.last_exchange_at is None

def test_fulfill_satiates_resets_and_clears_pending():
    s = State(u=99.0, duration_over_theta=40.0, desire_status="active",
              pending_proactive_id="p1", last_tick_at=at(0).isoformat())
    apply_verdict(s, Verdict.FULFILL, now=at(50))
    assert s.desire_status == "none" and s.duration_over_theta == 0.0
    assert s.u < 99.0 and s.pending_proactive_id is None and s.last_contact_at == at(50).isoformat()

def test_reject_records_growing_backoff_no_satiation():
    s = State(u=99.0, desire_status="active", decline_count=1, pending_proactive_id="p1",
              last_tick_at=at(0).isoformat())
    apply_verdict(s, Verdict.REJECT, now=at(50))
    assert s.desire_status == "none" and s.decline_count == 2
    assert s.u == 99.0 and s.declined_at == at(50).isoformat() and s.pending_proactive_id is None

def test_reject_backoff_suppresses_then_releases():
    # After a reject at t, evaluate_wake's growing backoff must veto within R then wake after.
    s = State(u=99.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1)
    assert decide_reachout(s, now=at(20), busy=False).wake is False   # 20 < r0=30
    s2 = State(u=99.0, last_tick_at=at(0).isoformat(), declined_at=at(0).isoformat(), decline_count=1)
    assert decide_reachout(s2, now=at(31), busy=False).wake is True   # 31 > 30
```

- [ ] **Step 2: Run tests, verify they FAIL** — `.venv/bin/python -m pytest tests/test_decision.py -q` → ImportError (module missing).

- [ ] **Step 3: Implement** `src/lifemodel/core/decision.py`. Reconstruct primitives from `State` each call: `Drive(alpha=ALPHA, beta=BETA, u_max=U_MAX, u=state.u)`; `LaneState(last_exchange_at=_iso_to_min?, in_flight=busy, declined_at=..., decline_count=state.decline_count)`; `Aggregator(status=DesireStatus(state.desire_status))`. **Time unit:** the sim uses abstract minutes; live uses `datetime`. Convert all timestamps the gate compares to *minutes since an epoch* consistently — simplest: work in `datetime` for gates by using a local `evaluate_wake`-equivalent, OR convert `now`, `last_exchange_at`, `declined_at` to "minutes since `now - large`" via `_minutes_between`. RECOMMENDED: pass gate times as *minutes-since-`last_tick`-epoch* is fragile; instead compute the three gate quantities directly in `decide_reachout` and call `evaluate_wake(u=drive.u, now=NOW_MIN, state=LaneState(last_exchange_at=EXCH_MIN, in_flight=busy, declined_at=DECL_MIN, decline_count=...), params=BASE_PARAMS)` where `NOW_MIN=0.0`, `EXCH_MIN = -_minutes_between(state.last_exchange_at, now)`, `DECL_MIN = -_minutes_between(state.declined_at, now)` (i.e. express everything as minutes relative to `now`, so `now - last_exchange = _minutes_between`). Rise: `dt = _minutes_between(state.last_tick_at, now)`; if `dt>0: drive.rise(dt=dt)`. Update `duration_over_theta`: `state.duration_over_theta = state.duration_over_theta + dt if drive.u >= THETA else 0.0`. Then the wake branch mirrors harness step 4: if `DesireStatus(state.desire_status) is NONE` and `evaluate_wake(...).is_urge` → `agg.on_urge()` → wake, `state.desire_status="active"`. Map `WakeOutcome` value to `reason`. Write back `state.u=drive.u`, `state.last_tick_at=now.isoformat()`. `observe_exchange`/`apply_verdict` per the Interfaces block, reusing `quality_of` and `Drive.satiate`.

- [ ] **Step 4: Run tests, verify PASS** — `.venv/bin/python -m pytest tests/test_decision.py -q`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): live decision adapter reusing certified sim primitives"`

---

## Task 3: rewire `run_proactive_tick` onto `core/decision`

**Files:**
- Modify: `src/lifemodel/egress_service.py`
- Test: `tests/test_egress_service_tick.py`

**Interfaces:**
- Consumes: `decide_reachout`, `apply_verdict` (Task 2); existing `egress.reach_out(target, impulse) -> ReachOutcome`; `compose_impulse`.
- Produces: a `run_proactive_tick(lm, egress, target, *, logger, busy=False) -> ReachOutcome` that no longer sums `salience`/drains/uses `cooldown`. Flow: `state=lm.state.load()`, `now=lm.clock.now()`, `d=decide_reachout(state, now=now, busy=busy)`; if `d.wake`: build a `pending_proactive_id` (e.g. `f"p-{state.tick_count}-{now.isoformat()}"`), set `state.pending_proactive_id`/`state.pending_proactive_since=now.isoformat()`, then `outcome=egress.reach_out(target, compose_impulse(...))`; if `outcome` is not `DELIVERED` (launch failed/unavailable/busy) → roll back: `state.desire_status="none"`, clear pending, do NOT reject. If not `d.wake`: `outcome=ReachOutcome.SKIPPED_BUSY`-or-a-noop sentinel. Always stamp `state.egress_service_alive_at=now.isoformat()` and `commit`. **The verdict (fulfill/reject) is applied later by the post_llm_call observer (Task 5), NOT here** — `DELIVERED` only means the turn launched.

- [ ] **Step 1: Write failing tests** in `tests/test_egress_service_tick.py` (rewrite the pressure-based ones):

```python
# Uses existing fakes: a FakeClock, an in-memory state store, a FakeEgress recording reach_out calls.
def test_no_reach_out_below_threshold(make_lm, fake_egress):
    lm = make_lm(last_tick_at="epoch")  # helper seeds a fresh state
    out = run_proactive_tick(lm, fake_egress, target="t", logger=NULL_LOGGER, busy=False)
    assert fake_egress.calls == []            # urge not matured → no reach-out
    assert lm.state.load().egress_service_alive_at is not None  # liveness always stamped

def test_wake_launches_turn_records_pending_and_does_not_apply_verdict(make_lm_high_u, fake_egress):
    lm = make_lm_high_u()                      # u high, past W, no reject
    out = run_proactive_tick(lm, fake_egress, target="t", logger=NULL_LOGGER, busy=False)
    assert len(fake_egress.calls) == 1
    s = lm.state.load()
    assert s.desire_status == "active" and s.pending_proactive_id is not None
    assert s.decline_count == 0                # verdict NOT applied here

def test_busy_gate_blocks_reach_out(make_lm_high_u, fake_egress):
    lm = make_lm_high_u()
    run_proactive_tick(lm, fake_egress, target="t", logger=NULL_LOGGER, busy=True)
    assert fake_egress.calls == []

def test_failed_launch_rolls_back_desire(make_lm_high_u, fake_egress_failing):
    lm = make_lm_high_u()
    run_proactive_tick(lm, fake_egress_failing, target="t", logger=NULL_LOGGER, busy=False)
    s = lm.state.load()
    assert s.desire_status == "none" and s.pending_proactive_id is None  # rolled back, no reject
```

- [ ] **Step 2: Run tests, verify they FAIL** (old signature still sums pressure). Run: `.venv/bin/python -m pytest tests/test_egress_service_tick.py -q`

- [ ] **Step 3: Implement** the new `run_proactive_tick` per Interfaces; delete the `state.pressure += ...`, `aggregator.decide`, `in_cooldown`, drain, and `cooldown` param. Keep the liveness stamp + single `commit`.

- [ ] **Step 4: Run tests, verify PASS**, plus the full existing egress suite: `.venv/bin/python -m pytest tests/test_egress_service_tick.py tests/test_tick_defer.py tests/test_tick_wake_cooldown.py -q` (fix or delete now-obsolete cooldown/pressure tests — they assert removed behavior).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(egress): run_proactive_tick decides via core/decision; verdict deferred to post_llm_call"`

---

## Task 4: demote cron `run_tick` to a silent watchdog

**Files:**
- Modify: `src/lifemodel/tick.py`, `src/lifemodel/composition.py`
- Test: `tests/test_tick.py`

**Interfaces:**
- Produces: `run_tick(lm, *, logger) -> WakeDecision` that ALWAYS returns `WakeDecision.stay_asleep()` (the being never proactively wakes from cron), still stamping `tick_count`/`last_tick_at` and committing, and still emitting the `{"wakeAgent": false}` gate line via `wake_gate_line`. Remove the neuron loop, `state.pressure` accumulation, `aggregator.decide`, and the drain/cooldown block. `composition.build_lifemodel` no longer wires a decision aggregator/neuron into the cron path (drop `ThresholdAggregator()`/`StubTimerNeuron` defaults, or leave them only if other code needs the types — grep first).

- [ ] **Step 1: Write failing tests** in `tests/test_tick.py`:

```python
def test_cron_tick_never_wakes(make_lm_high_u):
    lm = make_lm_high_u()                      # even with a high urge...
    d = run_tick(lm, logger=NULL_LOGGER)
    assert d.wake is False                     # cron is a silent watchdog
    assert lm.state.load().last_tick_at is not None  # still ticks bookkeeping

def test_cron_gate_line_is_stay_asleep(make_lm_high_u):
    from lifemodel.tick import wake_gate_line
    assert wake_gate_line(run_tick(make_lm_high_u(), logger=NULL_LOGGER)) == '{"wakeAgent": false}'
```

- [ ] **Step 2: Run tests, verify they FAIL** — `.venv/bin/python -m pytest tests/test_tick.py -q`

- [ ] **Step 3: Implement** the watchdog `run_tick`; delete the decision block; update `composition.py` (grep `ThresholdAggregator`/`StubTimerNeuron`/`SilentAggregator` usage — keep the classes if still referenced by tests, but the live `build_lifemodel` default aggregator becomes `SilentAggregator()` since cron never decides and the egress path uses `core/decision` directly).

- [ ] **Step 4: Run tests, verify PASS** — `.venv/bin/python -m pytest tests/test_tick.py tests/test_composition.py -q`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(tick): demote cron run_tick to a silent watchdog (in-proc service is the only brain)"`

---

## Task 5: verdict feedback via `post_llm_call` (NO_REPLY→reject, text→fulfill)

**Files:**
- Create: `src/lifemodel/hooks.py`
- Modify: `src/lifemodel/__init__.py`
- Test: `tests/test_hooks.py`

**SPIKE FIRST (host API):** Read the Hermes host to confirm the `post_llm_call` hook signature and payload — what it receives (final assistant text, session/turn id, whether the turn was internal/proactive), and how to correlate it with the pending proactive turn (via the `internal=True` MessageEvent injected by `inject_proactive_turn`, a session id, or a tag we set on `pending_proactive_id`). Also confirm how Hermes' existing silence-marker filter treats `NO_REPLY`/`[SILENT]` (does the marker get suppressed from delivery, and does streaming leak it before suppression?). Grep the host and `src/lifemodel/gateway_core.py` (`inject_proactive_turn`), `impulse.py`. Record findings in the commit body. **If `post_llm_call` cannot correlate a proactive turn without upstreaming, STOP and file a bead + bark the user** (this is the one identified potential upstream blocker).

**Interfaces:**
- Produces: `make_post_llm_observer(lm) -> Callable` — a hook handler that, when a *proactive* turn (matching `state.pending_proactive_id`) finishes, reads the final assistant text and calls `apply_verdict(state, Verdict.REJECT if _is_no_reply(text) else Verdict.FULFILL, now=lm.clock.now())` then commits. Non-proactive turns are ignored. `_is_no_reply(text: str) -> bool` matches the exact final markers `{"NO_REPLY","NO REPLY","[SILENT]","SILENT"}` (case-insensitive, stripped).

- [ ] **Step 1: Write failing tests** in `tests/test_hooks.py` (pure — inject a fake payload, no Hermes):

```python
def test_no_reply_maps_to_reject(make_lm_pending):
    lm = make_lm_pending(pending_id="p1")            # state has a live active desire p1
    obs = make_post_llm_observer(lm)
    obs(_fake_payload(pending_id="p1", text="NO_REPLY"))
    s = lm.state.load()
    assert s.desire_status == "none" and s.decline_count == 1 and s.declined_at is not None

def test_real_text_maps_to_fulfill(make_lm_pending):
    lm = make_lm_pending(pending_id="p1")
    obs = make_post_llm_observer(lm)
    obs(_fake_payload(pending_id="p1", text="Привет! Как ты?"))
    s = lm.state.load()
    assert s.desire_status == "none" and s.decline_count == 0 and s.last_contact_at is not None

def test_non_proactive_turn_ignored(make_lm_pending):
    lm = make_lm_pending(pending_id="p1")
    make_post_llm_observer(lm)(_fake_payload(pending_id="OTHER", text="hi"))
    assert lm.state.load().desire_status == "active"   # untouched
```

- [ ] **Step 2: Run tests, verify they FAIL** — `.venv/bin/python -m pytest tests/test_hooks.py -q`
- [ ] **Step 3: Implement** `_is_no_reply`, `make_post_llm_observer`; the correlation key uses whatever the spike found (adapt `_fake_payload` to the real payload shape).
- [ ] **Step 4: Run tests, verify PASS.**
- [ ] **Step 5: Register** in `__init__.py::register(ctx)`: `ctx.register_hook("post_llm_call", make_post_llm_observer(lm))` (host-guarded, best-effort like the existing hooks). Add a smoke test that `register` wires it without a real host (fake ctx records hooks).
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(hooks): post_llm_call verdict — NO_REPLY→reject, text→fulfill (kills the drum)"`

---

## Task 6: inbound observation (`pre_gateway_dispatch` / `pre_llm_call`)

**Files:**
- Modify: `src/lifemodel/hooks.py`, `src/lifemodel/__init__.py`
- Test: `tests/test_hooks.py`

**SPIKE FIRST:** Confirm whether `pre_gateway_dispatch` is in Hermes `VALID_HOOKS` (preferred — sees genuine user input, not internal impulses). If yes, use it; if not, use `pre_llm_call` and **explicitly skip turns whose text starts with the lifemodel impulse prefix** (from `impulse.py`) so the being never satiates on its own nudge. Record which hook was used.

**Interfaces:**
- Produces: `make_inbound_observer(lm) -> Callable` — on a genuine inbound user message, `observe_exchange(state, actor="user", label="two_way", now=lm.clock.now())` then commit. Ignores internal/proactive turns (impulse-prefix check).

- [ ] **Step 1: Write failing tests** in `tests/test_hooks.py`:

```python
def test_inbound_user_message_satiates_and_stamps(make_lm_high_u):
    lm = make_lm_high_u()
    make_inbound_observer(lm)(_fake_inbound(text="привет"))
    s = lm.state.load()
    assert s.last_exchange_at is not None and s.u < make_lm_high_u().state.load().u

def test_inbound_ignores_own_impulse(make_lm_high_u):
    lm = make_lm_high_u()
    from lifemodel.impulse import compose_impulse  # or the known prefix
    make_inbound_observer(lm)(_fake_inbound(text=<impulse-prefixed text>))
    assert lm.state.load().last_exchange_at is None   # own nudge is not user contact
```

- [ ] **Step 2–4: RED → implement → GREEN** (`.venv/bin/python -m pytest tests/test_hooks.py -q`).
- [ ] **Step 5: Register** the inbound hook in `register(ctx)` (host-guarded). Smoke-test wiring via fake ctx.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(hooks): inbound observer satiates drive + resets silence clock (the being hears the user)"`

---

## Task 7: centralize `busy` (remove stale `runner._running_agents` skip)

**Files:**
- Modify: `src/lifemodel/adapters/reachin.py`, `src/lifemodel/egress_service.py`
- Test: `tests/test_reachin_adapter.py`, `tests/test_egress_service_tick.py`

**Interfaces:** `ReachInEgress.reach_out` no longer returns `SKIPPED_BUSY` based on `runner._running_agents` (that veto now lives in the ONE place: the `busy` argument to `decide_reachout`, computed once by the service loop from the accurate runner state and passed down). The service loop computes `busy` from the runner and passes it to `run_proactive_tick(..., busy=busy)`.

- [ ] **Step 1: Write failing test** in `tests/test_reachin_adapter.py`:

```python
def test_reach_out_does_not_self_skip_on_running_agents(fake_runner_busy):
    # Busy ownership is the caller's gate now; the adapter must not second-guess it.
    egress = ReachInEgress(runner_accessor=lambda: fake_runner_busy)
    out = egress.reach_out(target="t", impulse="hi")
    assert out is not ReachOutcome.SKIPPED_BUSY
```

- [ ] **Step 2–4: RED → remove the `_running_agents` skip in `reachin.py`; compute `busy` in `proactive_service_loop` from the runner and thread it into `run_proactive_tick` → GREEN.** Run: `.venv/bin/python -m pytest tests/test_reachin_adapter.py tests/test_egress_service_tick.py -q`
- [ ] **Step 5: Commit** — `git add -A && git commit -m "fix(egress): centralize busy ownership in the wake gate (drop stale reachin self-skip)"`

---

## Phase 2 (follow-up plan, NOT this phase — list only)

Filed as follow-up tasks after the drum is dead and field-tested: (8) load BASE from disk config + hot-reload (BRD NFR5); (9) **disable streaming for internal proactive events** — investigate whether a NO_REPLY marker leaks visibly under streaming; if it does, that is the one identified upstream dependency → file a hermes-agent bead; (10) refresh `/lifemodel` status + debug renderer for the new fields; (11) delete any now-orphaned code (`StubTimerNeuron`/`ThresholdAggregator`/timer-pressure Signal path) once confirmed unused.

## Self-Review

- **Spec coverage:** Task 1 = spec §10 per-neuron bounded state; Task 2 = §5 drive + §7 gates + §8 lifecycle (reused from certified sim); Task 3+5 = §5 fulfill/reject + the async verdict; Task 6 = §4/§6 inbound satiation (RC1); Task 4 = "in-proc is the only brain"; Task 7 = RC2 busy. DEFER (§5) intentionally unreachable live (Global Constraints). NO_REPLY suppression-under-streaming = Phase 2/possible upstream (flagged).
- **Placeholder scan:** the two SPIKE steps (Tasks 5, 6) are host-API investigations with fully-specified target behavior + fake-payload tests — not placeholders; the implementer reads the real host signature (the plan cannot hardcode a signature that lives outside this repo).
- **Type consistency:** `desire_status` strings `{"none","active","deferred"}` ↔ `DesireStatus` enum values (they match the sim enum `.value`s); `Verdict.{FULFILL,REJECT,DEFER}` used consistently; `pending_proactive_id` threaded Task 3→5.

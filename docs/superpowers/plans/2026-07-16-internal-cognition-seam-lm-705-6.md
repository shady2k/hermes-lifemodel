# Internal-Cognition Seam (lm-705.6) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A **non-delivered, async, off-lock** internal-cognition seam — the being can run a cheap side-model pass that reads state and writes results (thoughts, later), delivering nothing to the human. It is the shared foundation for noticing (lm-705.5) and processing (lm-705.2).

**Architecture:** Mirror the proactive path but deliver nothing. A tick emits a `LaunchInternalCognition` intent (like `LaunchProactive`); an **adapter-owned `InternalCognitionRunner`** on the gateway asyncio loop awaits a `LlmPort` call **off the state-actor lock**, then runs an `ASYNC_COMPLETION` frame whose launches are dispatched by a **shared launch-dispatcher** (so an internal frame can never strand a proactive launch). A **durable FR20 call quota** in `State` gates it.

**Tech Stack:** Python 3.11 stdlib-only runtime; `uv`/`ruff`/`mypy --strict`/`pytest`. Hermes host: `ctx.llm.acomplete_structured(...)` (`agent/plugin_llm.py:823`), `ctx.register_auxiliary_task(key, *, display_name, description, defaults)` (`hermes_cli/plugins.py:1047`), the gateway asyncio loop (`runner._gateway_loop`), the proactive pattern (`core/proactive.py:54`, `gateway_core.py:378`).

## Global Constraints

- **bd:** lm-705.6 (Phase 5a / lm-705). Spec: `docs/superpowers/specs/2026-07-16-waking-mind-noticing-internal-cognition-design.md` §3. This bead BLOCKS lm-705.5 (noticing) and lm-705.2 (processing).
- **Non-delivery is structural** — the internal path calls the `LlmPort`, never `egress.reach_out`/`inject_proactive_turn`; there is no `post_llm` outcome for it.
- **The aux call runs OFF the state-actor lock.** Only the FR20 reservation (a frame) and the typed-result application (a frame) are serialized. `run_frame` takes `_STATE_ACTOR_LOCK` (`core/frame.py:129`); never hold it across an `await`.
- **The completion frame's launches MUST be dispatched.** A frame runs *every* enabled component regardless of trigger (`core/coreloop.py:301`), and `CoreLoop` returns `LaunchProactive` separately (`coreloop.py:360`, `TickReport.launches`). The completion path routes launches through the shared dispatcher — never ignores them (else `pending_proactive_id` is set with nothing injected → real outreach blocked).
- **Correlation `pending_internal_id` is separate** from `pending_proactive_id` — never read back as `[SILENT]`/`SENT`, never occupies the proactive in-flight gate.
- **FR20 is a durable quota WE enforce** — a daily call ceiling in `State`, atomically reserved *before* the task is created. (The aux slot is only model routing.)
- **State fields are additive** — `SQLiteRuntimeStore` persists `State` as one JSON blob with forward-compat load (`state/sqlite_store.py:564`); adding a field needs **no migration**.
- **This bead delivers no user-facing behavior** — it is the seam + a trivial internal pass exercised only by tests (noticing/processing are the real consumers). Prove it with unit/sim tests + one **isolated-`HERMES_HOME`** host-integration test — **never the live being**.
- **Every step ends green:** `make check`.

## File Structure

- **Modify** `state/model.py` — add `pending_internal_id: str | None`, `internal_calls_today: int`, `internal_calls_day: str` to `State` (+ `to_dict`/`from_dict`).
- **Create** `core/budget.py` — pure FR20 quota helpers (`reserve_internal_call`, day-rollover).
- **Modify** `core/intents.py` — add `LaunchInternalCognition`.
- **Modify** `core/coreloop.py` — collect `LaunchInternalCognition` into `TickReport` (a new `internal_launches` tuple) alongside `launches`.
- **Create** `core/llm_port.py` — the `LlmPort` Protocol + `InternalCognitionRequest`/`Result` types.
- **Create** `testing/llm.py` — `FakeLlmPort`.
- **Modify** `core/proactive.py` — extract `dispatch_launches(...)` (shared by proactive + internal completion).
- **Create** `core/internal_cognition.py` — `run_internal_completion(...)` (the completion-frame body: apply result + dispatch launches + clear pending).
- **Create** `adapters/internal_runner.py` — `InternalCognitionRunner` (gateway-loop task lifecycle).
- **Create** `adapters/plugin_llm_adapter.py` — the real `LlmPort` over `ctx.llm.acomplete_structured`.
- **Modify** `adapters/being_platform.py` — own the runner (start/stale-recovery in `connect`, cancel in `disconnect`); drive internal launches from the tick.
- **Modify** `__init__.py` — `ctx.register_auxiliary_task("lifemodel_internal", ...)`; inject the `LlmPort`.
- Tests: `tests/test_budget.py`, `tests/test_internal_intent.py`, `tests/test_llm_port.py`, `tests/test_dispatch_launches.py`, `tests/test_internal_cognition.py`, `tests/test_internal_runner.py`, `tests/hermes_internal_cognition_integration.py`.

---

## Task 1: State fields + FR20 quota helper

**Files:** Modify `state/model.py`; Create `core/budget.py`, `tests/test_budget.py`.

**Interfaces:**
- Produces: `State.pending_internal_id: str | None = None`, `State.internal_calls_today: int = 0`, `State.internal_calls_day: str = ""`; `reserve_internal_call(state: State, *, now: datetime, daily_ceiling: int) -> State | None` (returns a new `State` with the counter incremented + `pending_internal_id` unchanged, or `None` if the ceiling is reached; rolls the day over when `now`'s date ≠ `internal_calls_day`).

- [ ] **Step 1: Failing test** (`tests/test_budget.py`)

```python
from datetime import datetime, timezone
from lifemodel.core.budget import reserve_internal_call
from lifemodel.state.model import State

def _now(day="2026-07-16"): return datetime.fromisoformat(f"{day}T12:00:00+00:00")

def test_reserve_increments_and_rolls_day():
    s0 = State(internal_calls_today=0, internal_calls_day="")
    s1 = reserve_internal_call(s0, now=_now(), daily_ceiling=3)
    assert s1 is not None and s1.internal_calls_today == 1 and s1.internal_calls_day == "2026-07-16"

def test_reserve_denies_at_ceiling():
    s = State(internal_calls_today=3, internal_calls_day="2026-07-16")
    assert reserve_internal_call(s, now=_now(), daily_ceiling=3) is None

def test_new_day_resets_counter():
    s = State(internal_calls_today=3, internal_calls_day="2026-07-15")
    s2 = reserve_internal_call(s, now=_now("2026-07-16"), daily_ceiling=3)
    assert s2 is not None and s2.internal_calls_today == 1 and s2.internal_calls_day == "2026-07-16"
```

- [ ] **Step 2: Run → fail** (`ImportError`).
- [ ] **Step 3:** Add the three fields to `State` (with defaults; thread through `to_dict`/`from_dict` exactly like the existing scalar fields — mirror `unanswered_outbound_count`). Create `core/budget.py`:

```python
"""FR20 — a durable daily ceiling on expensive internal-cognition calls (spec §3.4)."""
from __future__ import annotations
from datetime import datetime
from ..state.model import State

def _day(now: datetime) -> str:
    return now.date().isoformat()

def reserve_internal_call(state: State, *, now: datetime, daily_ceiling: int) -> State | None:
    """Atomically-in-`State` reserve one internal-cognition call, or None if over the
    daily ceiling. Caller commits the returned State via an UpdateState in a frame."""
    today = _day(now)
    used = state.internal_calls_today if state.internal_calls_day == today else 0
    if used >= daily_ceiling:
        return None
    import dataclasses
    return dataclasses.replace(state, internal_calls_today=used + 1, internal_calls_day=today)
```

- [ ] **Step 4:** Run → pass. `make check`.
- [ ] **Step 5:** Commit `feat(internal-cognition): State pending_internal_id + FR20 daily call quota (lm-705.6)`.

---

## Task 2: `LaunchInternalCognition` intent + coreloop collection

**Files:** Modify `core/intents.py`, `core/coreloop.py`; Test `tests/test_internal_intent.py`.

**Interfaces:**
- Produces: `LaunchInternalCognition(Intent)` with `prompt: str`, `correlation_id: str`, `origin_traceparent: str` (mirror `LaunchProactive`, minus delivery fields); `TickReport.internal_launches: tuple[LaunchInternalCognition, ...] = ()`.
- Consumes: the coreloop launch-sorting (`coreloop.py:360-368`).

- [ ] **Step 1: Failing test** — a fake component emitting `LaunchInternalCognition` → it appears in `report.internal_launches`, NOT `report.launches`, and is not committed as a State mutation. (Mirror `tests/test_frame_acceptance.py`'s launch assertions.)
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Add `LaunchInternalCognition` to `core/intents.py`. In `coreloop.py:360-368`, add an `elif isinstance(intent, LaunchInternalCognition): internal_launches.append(intent)` branch, thread `internal_launches` into `TickReport` (Task-1-style additive field).
- [ ] **Step 4:** Run → pass. `make check`.
- [ ] **Step 5:** Commit `feat(internal-cognition): LaunchInternalCognition intent, collected separately from proactive launches (lm-705.6)`.

---

## Task 3: `LlmPort` + `FakeLlmPort`

**Files:** Create `core/llm_port.py`, `testing/llm.py`, `tests/test_llm_port.py`.

**Interfaces:**
- Produces: `InternalCognitionRequest` (`instructions: str`, `input_text: str`, `json_schema: dict | None`), `InternalCognitionResult` (`raw: str`, `parsed: dict | None`); `LlmPort` Protocol with `async def complete_structured(self, req: InternalCognitionRequest) -> InternalCognitionResult`; `FakeLlmPort(result: InternalCognitionResult | Exception)`.

- [ ] **Step 1: Failing test** — `FakeLlmPort` returns the scripted result; a scripted `Exception` propagates from `await complete_structured(...)`.
- [ ] **Step 2–4:** Create the Protocol + types (frozen dataclasses) + `FakeLlmPort` (records the request; returns/raises the script). The core stays Hermes-free — the Protocol names no Hermes type. `make check`.
- [ ] **Step 5:** Commit `feat(internal-cognition): LlmPort protocol + FakeLlmPort (lm-705.6)`.

---

## Task 4: Extract `dispatch_launches` (the strand fix — codex #2)

**Files:** Modify `core/proactive.py`; Test `tests/test_dispatch_launches.py`.

**Interfaces:**
- Produces: `dispatch_launches(lm, report, egress, target, *, voice=None) -> ReachOutcome | None` — the existing `proactive_tick` body from `if not report.launches` onward (backstop → voice → egress → rollback/delivery span), extracted so **any** frame's `report.launches` get dispatched. `proactive_tick` now calls it.

- [ ] **Step 1: Failing test** — build a report with a `LaunchProactive` (via a fake component or a constructed `TickReport`); `dispatch_launches` with a fake egress delivers it (egress.reach_out called), and a held backstop rolls back. **Regression for the strand:** a report from a *non-proactive* completion frame that carries a `LaunchProactive` is still dispatched (not dropped).
- [ ] **Step 2: Run → fail** (`dispatch_launches` not defined).
- [ ] **Step 3:** Move `proactive_tick`'s post-`run_frame` body into `dispatch_launches(lm, report, egress, target, *, voice=None)`; `proactive_tick` becomes `report = run_frame(...); return dispatch_launches(lm, report, egress, target, voice=voice)`. Behavior-identical for the proactive path (existing `proactive_tick` callers/tests stay green).
- [ ] **Step 4:** Run → pass; existing proactive tests green. `make check`.
- [ ] **Step 5:** Commit `refactor(internal-cognition): extract dispatch_launches so any frame's launches are dispatched (lm-705.6, codex #2)`.

---

## Task 5: `run_internal_completion` (apply result + dispatch launches + clear pending)

**Files:** Create `core/internal_cognition.py`; Test `tests/test_internal_cognition.py`.

**Interfaces:**
- Consumes: `run_frame`/`FrameTrigger.ASYNC_COMPLETION`, `dispatch_launches` (Task 4), the intent bus.
- Produces: `run_internal_completion(lm, egress, target, *, correlation_id, result, apply)` — runs an `ASYNC_COMPLETION` frame seeded with a result signal (a new `internal_result_signal`, taxonomy), whose result-applying component (`apply`, injected) turns `result` into intents (for lm-705.6: a no-op/echo component; noticing supplies the real one), commits under the lock, **then dispatches any returned launches** via `dispatch_launches`, and clears `pending_internal_id` (an `UpdateState`).

- [ ] **Step 1: Failing test** — a completion frame with a fake apply-component that emits a `PutRecord` commits it AND clears `pending_internal_id`; **a completion frame whose (unrelated) `CognitionLauncher` returns a `LaunchProactive` dispatches it** (fake egress called) — the codex-#2 regression, end-to-end.
- [ ] **Step 2–4:** Implement over the real `run_frame` + `dispatch_launches`. Use the fake-port harness (`testing/harness.py`). `make check`.
- [ ] **Step 5:** Commit `feat(internal-cognition): completion frame applies result, dispatches launches, clears pending (lm-705.6)`.

---

## Task 6: `InternalCognitionRunner` (adapter-owned, gateway-loop lifecycle)

**Files:** Create `adapters/internal_runner.py`; Test `tests/test_internal_runner.py`.

**Interfaces:**
- Consumes: the `LlmPort` (Task 3), `run_internal_completion` (Task 5), `reserve_internal_call` (Task 1), the gateway loop + a `build_lm` callable.
- Produces: `InternalCognitionRunner(build_lm, llm, egress, target, *, daily_ceiling, gateway_loop, apply)`:
  - `launch(request, correlation_id) -> bool` — reserve FR20 (a frame under the lock); if denied, return False; else set `pending_internal_id` (a frame), create + **retain** an asyncio task on `gateway_loop` running `_run(request, correlation_id)`.
  - `async _run(...)` — `await self.llm.complete_structured(req)` **off the lock**; on success/timeout/exception build a fresh `lm` and call `run_internal_completion(...)` (a *failure* still clears `pending_internal_id`); remove the task from the set.
  - `recover_stale(lm)` — at connect: if `pending_internal_id` is set with no live task, clear it (a frame) so the next launch isn't blocked.
  - `cancel_all()` — cancel + await the tracked task set (called from `disconnect`).

- [ ] **Step 1: Failing test** — inject a `FakeLlmPort` + a fake loop (`asyncio` event loop in the test): `launch` over budget returns False and creates no task; within budget it sets `pending_internal_id`, the task runs, `run_internal_completion` applies the fake result, `pending_internal_id` clears; a `FakeLlmPort` that raises still clears pending (no strand); `cancel_all` cancels a pending task.
- [ ] **Step 2–4:** Implement. **Never hold `_STATE_ACTOR_LOCK` across the `await`** (reservation + completion are separate frames). Guard the whole `_run` body fail-loud (a throw clears pending + logs, never crashes the loop). `make check`.
- [ ] **Step 5:** Commit `feat(internal-cognition): InternalCognitionRunner — off-lock async, lifecycle, stale recovery (lm-705.6)`.

---

## Task 7: Wire it — composition + being_platform + aux task

**Files:** Modify `adapters/being_platform.py`, `__init__.py`, `composition.py`, `adapters/plugin_llm_adapter.py` (new). Test: extend `tests/test_composition.py`.

- [ ] **Step 1:** `adapters/plugin_llm_adapter.py` — `PluginLlmPort(ctx_llm)` implementing `LlmPort.complete_structured` over `ctx.llm.acomplete_structured(instructions=..., input=[...], json_schema=..., model=..., timeout=...)`, mapping the host result → `InternalCognitionResult`. (Thin; the real host call — no unit test beyond a construction/shape test; it's exercised by Task 8.)
- [ ] **Step 2:** `being_platform.connect()` — build the `InternalCognitionRunner` (with `gateway_loop`, the `PluginLlmPort`, the egress, `build_lm`), call `recover_stale(...)` before the loop's first tick, and in the tick drive `report.internal_launches` → `runner.launch(...)`. `disconnect()` → `runner.cancel_all()`.
- [ ] **Step 3:** `__init__.py` `register(ctx)` — `ctx.register_auxiliary_task("lifemodel_internal", display_name="Lifemodel inner cognition", description="The being's private, non-delivered thinking (noticing/rumination).", defaults=...)`; pass the aux `task="lifemodel_internal"` through to the `PluginLlmPort` call so it routes `auxiliary.lifemodel_internal`.
- [ ] **Step 4:** Run the suite + `make check`. Commit `feat(internal-cognition): wire runner + LlmPort + aux task lifemodel_internal (lm-705.6)`.

---

## Task 8: Host-integration test (isolated `HERMES_HOME`)

**Files:** `tests/hermes_internal_cognition_integration.py`.

- [ ] Prove against a **real host with an isolated `HERMES_HOME`** (never the live being): a `LaunchInternalCognition` → the runner calls the aux model via `auxiliary.lifemodel_internal` routing, **delivers nothing** (no gateway turn, no message on any lane), the typed result applies in a completion frame, `pending_internal_id` clears; a simulated timeout/failure clears pending (no strand); the FR20 ceiling denies the N+1 call that day; `disconnect()` cancels an in-flight task cleanly. Mark it appropriately (a slow/integration marker) so `make check` can gate it or run it separately; document the `HERMES_HOME` setup in the test module docstring.
- [ ] Commit `test(internal-cognition): isolated-HERMES_HOME host-integration for the non-delivered seam (lm-705.6)`.

---

## Self-Review (after execution)

- **Spec §3 coverage:** InternalCognitionRunner off-lock (T6) ✓ · generic launch-dispatch (T4/T5) ✓ · distinct `pending_internal_id` (T1/T6) ✓ · LlmPort over `ctx.llm`/aux slot (T3/T7) ✓ · durable FR20 quota (T1/T6) ✓ · host-integration test (T8) ✓.
- **Non-delivery:** the internal path never calls `egress.reach_out`/`inject_proactive_turn` (only `dispatch_launches` for any *proactive* launch a completion frame incidentally returns).
- **The strand fix:** T5's regression proves a completion frame's `LaunchProactive` is dispatched.
- **Deferred (not here):** the noticing buffer/trigger/pass (lm-705.5) supplies the real `apply` + the `LaunchInternalCognition` emitter; processing (lm-705.2) reuses the runner; sharing the FR20 quota with the *proactive* path is a follow-up (here the quota gates internal calls; the field is shaped to extend).
- **Confirm-before-code:** the exact `ctx.llm.acomplete_structured` result shape (`agent/plugin_llm.py:823`) and how `runner._gateway_loop` is reached from the adapter (`gateway_core.py` / `inject_proactive_turn`'s `schedule`).

# Being-as-Platform-Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the fragile self-spawned in-process egress service (and the neutered cron heartbeat) with the being hosted as a gateway-supervised platform adapter, so the proactive brain can never silently die.

**Architecture:** All decision/loop/supervision logic lives in Hermes-free `core/` modules that unit-test with fakes. A thin Hermes shell (`adapters/being_platform.py`, subclasses `BasePlatformAdapter`) wires those into the gateway: `connect()` runs a supervised periodic loop that drives the Hermes-free proactive tick; loop death is converted to `_set_fatal_error(retryable=True)` + `_notify_fatal_error()` so the gateway reconnect watcher restarts it. Delivery stays the existing reach-in into the user's Telegram lane; the adapter's own `send()` is a no-op.

**Tech Stack:** Python 3, asyncio, pytest, ruff, mypy. Hermes plugin API (`ctx.register_platform`, `BasePlatformAdapter`).

## Global Constraints

- **Hermes stays behind an abstract boundary.** Core (`core/*`, `domain/*`, `state/*`, `ports/*`, `paths.py`, `log.py`, `composition.py`) imports NO Hermes and unit-tests with fakes. All Hermes coupling lives only in `adapters/` + `__init__.py`. Dependency direction: `core → ports ← adapters → Hermes`.
- **The test venv cannot import `gateway`.** Any module with a top-level `from gateway...` import is NOT importable by the off-host test suite. Therefore no `core/`/tested module may import Hermes (directly or transitively), and the adapter shell is verified at runtime, not by off-host unit tests.
- **No backward compatibility, no dead code.** Delete replaced modules outright.
- **KEEP `last_tick_at`** — it is the core dt clock (`core/aggregation.py`, `core/personality.py`, `core/contact_neuron.py`, stamped by `core/coreloop.py`). It doubles as the liveness signal. Only `egress_service_alive_at` and cron machinery are removed.
- **TDD, RED-first.** Every core function gets a failing test first. Frequent commits. Work on branch `shady2k/lm-being-platform`, never `main`.
- Reference adapter patterns: `~/.hermes/hermes-agent/plugins/platforms/irc/adapter.py` (correct fatal-on-loop-exit: `:387-391`), `plugins/platforms/homeassistant/adapter.py` (connect starts a background loop: `:131`).

---

### Task 0: Branch

- [ ] **Step 1:** `git checkout -b shady2k/lm-being-platform`
- [ ] **Step 2:** Confirm `pytest -q` is green on the untouched tree (baseline). If pre-existing failures exist, note them.

---

### Task 1: Hermes-free supervised loop (`core/supervised_loop.py`)

The load-bearing new unit: a periodic loop that DETECTS its own death and reports it via an injected callback. This is the behavior whose absence caused the silent outage.

**Files:**
- Create: `core/supervised_loop.py`
- Test: `tests/test_supervised_loop.py`

**Interfaces:**
- Produces:
  ```python
  class SupervisedLoop:
      def __init__(self, *, tick: Callable[[], None], interval_sec: float,
                   on_death: Callable[[BaseException | None], None],
                   sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None: ...
      async def run(self) -> None: ...   # loops tick()+sleep until stop()/cancel/exception
      def stop(self) -> None: ...        # request clean exit (no on_death)
  ```
  Contract: `run()` calls `tick()` then `await sleep(interval_sec)` while alive. If `tick()` raises, call `on_death(exc)` exactly once and return (no re-raise). On `stop()`, exit cleanly WITHOUT `on_death`. On `asyncio.CancelledError`, re-raise WITHOUT `on_death` (clean shutdown). Never calls `on_death` more than once.

- [ ] **Step 1: Failing tests** (`tests/test_supervised_loop.py`) — use a fake async `sleep` that yields control and a manual clock/counter:
  - `test_tick_called_each_interval`: after N fake-sleep cycles, tick called N(+1) times.
  - `test_tick_exception_calls_on_death_once_and_stops`: tick raises `ValueError` → `on_death` called once with that exc; loop returns; tick not called again.
  - `test_stop_exits_without_on_death`: stop() before/between ticks → run() returns, on_death never called.
  - `test_cancel_is_clean`: cancelling the `run()` task raises `CancelledError` out and never calls on_death.
- [ ] **Step 2:** Run `pytest tests/test_supervised_loop.py -v` → expect FAIL (module missing).
- [ ] **Step 3:** Implement `core/supervised_loop.py` per the contract (stdlib only; `asyncio` imported for the default sleep + CancelledError).
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** `ruff check` + `mypy core/supervised_loop.py`; commit `feat(core): supervised loop with self-death detection`.

---

### Task 2: Hermes-free proactive tick (`core/proactive.py`)

Extract the decision+delivery tick from `egress_service.run_proactive_tick`, dropping (a) the `egress_service_alive_at` liveness stamp and (b) the `gateway_core.reachin_available` import (the Hermes leak). Pure core, port-only.

**Files:**
- Create: `core/proactive.py`
- Test: `tests/test_core_proactive.py`
- Reference (delete later): `egress_service.py:38-84`

**Interfaces:**
- Consumes: `composition.LifeModel`, `ports.proactive.ProactiveEgressPort`, `core.backstop.allow_send`, `core.wake_packet.IMPULSE_LABEL_PREFIX`, `domain.egress.ReachOutcome`.
- Produces:
  ```python
  def proactive_tick(lm: LifeModel, egress: ProactiveEgressPort,
                     target: Mapping[str, str | None], *, logger: EventLogger) -> ReachOutcome: ...
  ```
  Behavior (verbatim from current logic, minus the alive stamp): run `lm.coreloop.tick()`; if a launch surfaces, apply `allow_send` (backstop) → blocked ⇒ hold desire `deferred` + refund reserved energy; else `egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)` → non-DELIVERED ⇒ pending back to `active` + refund. One reconciliation commit for rollback/refund (NO `egress_service_alive_at`). `last_tick_at` is stamped by `coreloop.tick()` — do not touch it here.

- [ ] **Step 1: Failing tests** (mirror existing egress-service tests; use `build_lifemodel` with a fake clock + a fake `ProactiveEgressPort` recording calls; drive state so `coreloop.tick()` surfaces a launch):
  - `test_launch_delivered_returns_delivered_and_clears_pending`
  - `test_backstop_block_defers_and_refunds_energy`
  - `test_failed_delivery_rolls_pending_active_and_refunds`
  - `test_no_launch_is_noop_skip`
  - `test_does_not_write_egress_service_alive_at` (assert the field stays None/absent after a tick)
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `core/proactive.py` (copy the body of `run_proactive_tick`, remove the alive stamp lines and the `reachin_available` import; keep the refund/rollback reconciliation).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** `ruff` + `mypy`; commit `feat(core): Hermes-free proactive tick (no liveness stamp, no reachin leak)`.

---

### Task 3: Honest HEALTH readings (derive liveness from `last_tick_at`)

The debug command reads from disk and has no live-adapter handle, so liveness is derived from `last_tick_at` freshness (the loop stamps it every tick via `coreloop.tick()`). Replace the `egress_service_alive_at` readings.

**Files:**
- Modify: `core/introspect.py` (Readings fields `egress_service_alive_at`/`egress_service_ago_min` at `:77-78,190-193`; add `brain_alive`, keep `last_tick_ago_min`)
- Modify: `debug.py` (TIMING section `:96-100` → HEALTH)
- Modify: `tests/test_debug.py` and any introspect test

**Interfaces:**
- Produces (Readings additions): `brain_alive: bool` (True iff `last_tick_ago_min is not None and <= BRAIN_STALE_MIN`), constant `BRAIN_STALE_MIN = 2.0`.
- Removes: `egress_service_alive_at`, `egress_service_ago_min`.

- [ ] **Step 1: Failing test** (`tests/test_debug.py` or introspect test): given a state whose `last_tick_at` is 30s before `now`, `compute_readings(...).brain_alive is True`; 15 min before ⇒ `False`. And `render_debug_dump` contains a `HEALTH` line reading `brain alive` / `brain STALE` + `last tick Xm ago`, and no longer references `service_alive`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement: add `brain_alive` to `Readings` + compute it in `compute_readings`; drop the two egress fields; rewrite the debug `TIMING` block into `HEALTH` (`brain <alive|STALE> (last tick Xm ago) · would_wake … · sends today N/cap`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** `ruff` + `mypy`; commit `feat(debug): HEALTH view — brain liveness from last_tick freshness`.

---

### Task 4: Relocate the home-origin resolver into the Hermes boundary

`_resolve_home_origin` currently lives in `heartbeat.py` (to be deleted). Move it into `adapters/origin.py`.

**Files:**
- Create: `adapters/origin.py` with `resolve_home_origin() -> dict[str, str | None] | None` (copy body from `heartbeat.py:_resolve_home_origin`).
- Modify: any importer (`__init__.py`, `ports/proactive.py` docstring reference).

- [ ] **Step 1:** Copy the function verbatim into `adapters/origin.py` (keep its lazy Hermes import inside the function so the module stays importable off-host, matching `adapters/reachin.py`). If it has an existing test, repoint it.
- [ ] **Step 2:** `ruff` + `mypy adapters/origin.py`; commit `refactor(adapters): relocate home-origin resolver out of heartbeat`.

---

### Task 5: The being platform adapter (thin Hermes shell)

**Files:**
- Create: `adapters/being_platform.py`
- Modify: `__init__.py` (register the platform; remove egress-service + heartbeat wiring)

**Interfaces (shell — not off-host unit-tested; verified at runtime):**
```python
# adapters/being_platform.py
class BeingAdapter(BasePlatformAdapter):     # from gateway.platforms.base
    def __init__(self, config, *, base_dir: Path, target, logger, interval_sec: float = 60.0): ...
    async def connect(self, *, is_reconnect: bool = False) -> bool: ...
    async def disconnect(self) -> None: ...
    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult: ...  # no-op failure

def register_being_platform(ctx, *, base_dir: Path, target, logger) -> None: ...
```

- [ ] **Step 1:** Implement `BeingAdapter`:
  - `__init__`: `super().__init__(config, Platform("lifemodel"))`; store base_dir/target/logger/interval; `self._loop_task = None`; `self._shutting_down = False`.
  - `connect`: build egress `ReachInEgress(runner_accessor=default_runner_accessor, logger=self._log)`; build `SupervisedLoop(tick=lambda: proactive_tick(build_lifemodel(base_dir=self._base_dir, logger=self._log), egress, self._target, logger=self._log), interval_sec=self._interval, on_death=self._on_loop_death)`; `self._loop_task = asyncio.create_task(loop.run())`; `self._mark_connected()`; return `True`. (One `LifeModel` per tick — matches today's `build_lm()` per-tick invariant.)
  - `_on_loop_death(exc)`: if `self._shutting_down`: return; `self._set_fatal_error("brain_loop_exited", f"{exc!r}", retryable=True)`; schedule `self._notify_fatal_error()` on the running loop (`asyncio.get_running_loop().create_task(...)` — the callback is sync).
  - `disconnect`: `self._shutting_down = True`; `self._loop.stop()`; cancel `self._loop_task` and await-suppress `CancelledError`.
  - `send`: return a `SendResult` failure/no-op (the being never receives replies on its own lane; delivery is reach-in into Telegram).
- [ ] **Step 2:** Implement `register_being_platform(ctx, ...)`: `ctx.register_platform("lifemodel", label="Life Model", adapter_factory=lambda cfg: BeingAdapter(cfg, base_dir=base_dir, target=target, logger=logger), check_fn=lambda: True)`. Add `emoji`/`validate_config` if needed so `platforms.lifemodel.enabled: true` needs no fake tokens.
- [ ] **Step 3:** Modify `__init__.py`: DELETE the egress-service wiring block (`origin`/`ReachInEgress`/`register_gateway_service`/deferred `on_session_start` arm) and the `register_heartbeat` call and their imports; ADD `register_being_platform(ctx, base_dir=sdir, target=resolve_home_origin(), logger=logger)` (best-effort try/except, never break load). Keep the `/lifemodel` command + the two observer hooks.
- [ ] **Step 4:** `ruff` (adapter file is import-checked but NOT imported by pytest); commit `feat(adapters): being as a gateway-supervised platform adapter`.

---

### Task 6: Delete the replaced machinery + prune state/tests

**Files:**
- Delete: `egress_service.py`, `heartbeat.py`, `tick.py`
- Modify: `gateway_core.py` — remove `register_gateway_service`, `install_core_shim`, `_spawn_on_loop` (keep `inject_proactive_turn`, `reachin_available`, `_default_*`, `_select_adapter` — the delivery path)
- Modify: `state/model.py` — remove `egress_service_alive_at` field + its parse (`:113,178-179`)
- Delete/rewrite tests: `tests/test_*` covering `tick`, `heartbeat`, `egress_service`, and the `register_gateway_service`/`install_core_shim` paths of `gateway_core`.

- [ ] **Step 1:** `git rm egress_service.py heartbeat.py tick.py` and delete their tests.
- [ ] **Step 2:** Edit `gateway_core.py` to drop the service-spawn functions; verify nothing else imports them (`grep -rn "register_gateway_service\|install_core_shim\|_spawn_on_loop\|proactive_service_loop\|run_proactive_tick\|service_is_alive" --include=*.py .`).
- [ ] **Step 3:** Remove `egress_service_alive_at` from `state/model.py`; fix any reader (introspect already handled in Task 3).
- [ ] **Step 4:** Run full `pytest -q`; fix/rewrite red tests (they should only be the deleted-module tests + introspect/debug already updated).
- [ ] **Step 5:** `ruff check .` + `mypy .` + `pre-commit run -a`; commit `refactor(core): delete self-hosted egress service, cron heartbeat, liveness stamp`.

---

### Task 7: Verify + hand off

- [ ] **Step 1:** Full gates green: `pytest -q`, `ruff check .`, `mypy .`, `pre-commit run -a`. Capture output.
- [ ] **Step 2:** Grep audit: no `core/`/tested module imports Hermes; no references to deleted symbols remain.
- [ ] **Step 3:** Codex final review of the diff (read-only) against this plan + the spec.
- [ ] **Step 4:** Do NOT merge to `main` or restart the gateway. bark the owner: summary + "on branch `shady2k/lm-being-platform`, gates green, codex-reviewed — ready to merge + `hermes gateway restart` to activate." Runtime verification (gateway restart → `/lifemodel debug` shows HEALTH brain alive; observe a proactive reach-out) is the owner's activation step.

---

## Self-Review

- **Spec coverage:** adapter host (Task 5) ✓; supervised loop / fatal-on-death (Task 1 + Task 5 wiring) ✓; Hermes-free tick, reachin leak fixed (Task 2) ✓; delete service/cron/liveness (Task 6) ✓; keep `last_tick_at` (Global Constraints + Task 6) ✓; HEALTH observability (Task 3) ✓; boundary discipline (Global Constraints, enforced by test venv) ✓; delivery via reach-in + send() no-op (Task 5) ✓; enable/config + virtual-platform (Task 5) ✓.
- **Out of scope (unchanged):** `lm-67g` command list; per-layer metrics.
- **Type consistency:** `proactive_tick(lm, egress, target, *, logger)`, `SupervisedLoop(tick, interval_sec, on_death, sleep)`, `resolve_home_origin()`, `Readings.brain_alive`, `register_being_platform(ctx, *, base_dir, target, logger)` — used consistently across tasks.
- **Runtime-only risk:** the adapter shell (Task 5) is not off-host unit-tested (test venv lacks Hermes). Mitigation: all logic delegated to tested core units; shell kept thin; owner runtime-verifies at activation.

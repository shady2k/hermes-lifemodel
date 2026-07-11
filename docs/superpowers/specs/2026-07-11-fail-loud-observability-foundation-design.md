# Fail-Loud & Load Observability — Foundation

**Status:** design (codex-reviewed, revised) → implementation
**Bead:** lm-fib.9 (epic)
**Trigger:** live incident 2026-07-11 — the being's brain silently never started after a deploy while the plugin reported "enabled".

## 1. The incident (concrete)

After deploying `origin/main` to the live being:
- `/lifemodel` command **enabled**, gateway healthy, **no error in the main log**.
- But the being (platform adapter) **never connected** — the autonomic brain was dead. The only signal was the *absence* of a `being_connected` log line.

Cause, found only by manual reproduction:
```
INFO lifemodel: being_platform_registration_skipped error=ModuleNotFoundError: No module named 'lifemodel'
```
`state/metrics_store.py:52` did `from lifemodel.core.metrics import …` (absolute). Hermes imports directory plugins as **`hermes_plugins.<slug>`** (`hermes_cli/plugins.py:_load_directory_module`), so `lifemodel` is not a top-level module at load → `ModuleNotFoundError`. `register_being_platform` raised; `register()` caught it as `except → _LOG.info("…_skipped")` (INFO, **no traceback**) and continued. The plugin "loaded" as a brain-dead shell.

## 2. Why our safety nets failed (the philosophy is the bug)

1. **`make check` was GREEN.** `conftest.py` inserts the package parent on `sys.path`, so `lifemodel` resolves *in tests*. The harness makes absolute self-imports work — the exact thing fatal in prod. **The green check lied.**
2. **The failure degraded silently.** Load-bearing wiring (the brain) used the same `except → INFO "skipped"; never break load` pattern as genuinely-optional wiring. A fatal error was logged as benign, at a lost level, **without a traceback**. Silence became ambiguous: healthy vs brain-dead.

Past observability epics built *infrastructure* (traces, spans, `metrics.sqlite`) but never changed the **default at failure boundaries**. Infrastructure does not help when the failure path writes nothing to it.

**Invariant this spec establishes:**
> Silence means healthy. Any failure of a **load-bearing** part is loud (ERROR + traceback) and observable as a **health state** — enforced by tests, not discipline.

## 3. Hermes facts (validated against `~/.hermes/hermes-agent/`)

- Directory plugins import as `hermes_plugins.<slug>` (`/`→`__`, `-`→`_`); ours is `hermes_plugins.lifemodel`. Runtime code must therefore use **relative** imports (`from ..core.metrics`), never `lifemodel.…` nor `hermes_plugins.lifemodel.…`.
- `_load_plugin` (`hermes_cli/plugins.py:1746-1828`) wraps *import + `register(ctx)`* in one `try/except`: on exception it sets `loaded.error`, leaves `loaded.enabled=False`, logs `WARNING "Failed to load plugin"` with `exc_info=_PLUGINS_DEBUG` (traceback only when `HERMES_PLUGINS_DEBUG` set). So a **re-raise makes Hermes mark us not-enabled + log it** — that is the "loud" channel.
- `ctx.register_hook(name, cb)` does **not** fail on an unknown hook name — it stores + warns. So an exception while wiring `post_llm_call`/`pre_gateway_dispatch` is **our bug**, not "host lacks the hook".
- `adapters/being_platform.py` imports `gateway.*` (e.g. `BasePlatformAdapter`) at **module import time**. Dev `make check` runs under `uv` **without** the `gateway` package — so any test that imports `being_platform` must stub `gateway.*` in `sys.modules` or run in a Hermes env.

## 4. The complete fix

### 4.1 Kill the error class (static + faithful-load guards)
- **Fix the import:** `state/metrics_store.py:52` → `from ..core.metrics import …`. (Only absolute self-import in runtime dirs; `testing/` is test-only.)
- **Linter (AST test, in `make check`)** — `tests/test_no_absolute_self_imports.py`. Ban, in runtime dirs (`core domain state adapters ports sim` + root `*.py`):
  - `from lifemodel…` / `import lifemodel` **and** `from hermes_plugins.lifemodel…` / `import hermes_plugins.lifemodel` (self-binding to the loader namespace is also wrong);
  - literal-string dynamic imports: `importlib.import_module("lifemodel…"|"hermes_plugins.lifemodel…")` and `__import__("lifemodel…")`.
  - Catches imports anywhere in the AST (module level, inside functions, under `TYPE_CHECKING`).
- **Faithful-load smoke test** — `tests/test_real_loader_import.py`. Reproduce Hermes' loader so an absolute self-import CANNOT resolve via the conftest shim. It MUST:
  1. ensure a `hermes_plugins` namespace package in `sys.modules`;
  2. `spec_from_file_location("hermes_plugins.lifemodel", <pkg>/__init__.py, submodule_search_locations=[<pkg>])`;
  3. insert `sys.modules["hermes_plugins.lifemodel"] = module` **before** `exec_module`; set `__package__`/`__path__`;
  4. **remove the checkout parent from `sys.path` and delete `lifemodel` + `lifemodel.*` from `sys.modules`** for the duration (so a stray absolute import fails as in prod);
  5. import the **Hermes-free runtime surface** under this namespace — every module `build_lifemodel` touches **plus `state.metrics_store` directly** (this is what catches the incident without needing `gateway`); do NOT import `adapters.being_platform` here (it needs `gateway`);
  6. restore `sys.path` and purge all `hermes_plugins.lifemodel*` from `sys.modules` in a `finally`.
  - Acceptance: this test **fails on the pre-fix tree** (prove by reverting the import fix during review) and passes after.
- **register()/adapter smoke test (gateway-stubbed)** — a separate test installs minimal `gateway.*` stubs in `sys.modules` (`BasePlatformAdapter` and whatever `being_platform` imports at module level), then imports `being_platform` and calls `register(fake_ctx)` under the isolated namespace, asserting the platform-import path is exercised. If stubbing proves fragile, mark this test `@pytest.mark.integration` to run only in a Hermes env, and **log that it was skipped** (no silent coverage gap).

### 4.2 One shared brain-health object (the backbone)
Introduce a process-local `BrainHealth` (a small dataclass/singleton per base_dir) that is the single source of truth, written by `register()`, `BeingAdapter.connect()`, the tick path, `_on_loop_death()`, and the observer hooks; read by `check_fn` and `/lifemodel status`:
- `state`: `never_started | connecting | connected | loop_dead | boot_failed`
- `boot_error`, `last_loop_death` (message + traceback ref), `death_count`
- `last_tick_at`, `ticks_total` (advanced by the tick path — the **durable, primary** liveness signal)
- `last_observer_error` (per observer)
This replaces `check_fn=lambda: True`: `check_fn` returns `False` with a reason whenever `state != connected` or the last tick is stale beyond a threshold.

### 4.3 Fail loud on load-bearing wiring
- **One helper** `wire(step_name, *, required)` (context manager) used at every boundary in `register()` and `connect()`:
  - DEBUG on start/success; on exception **ERROR with `exc_info=True`** (full traceback, always — independent of `HERMES_PLUGINS_DEBUG`);
  - `required=True` → record `boot_failed` in `BrainHealth` (persist a durable boot-health record), then **re-raise**;
  - `required=False` → skip (only for a capability the host genuinely lacks, proven by an explicit check — NOT a blanket catch).
- **"Both" strategy for the diagnostic lever** (codex CRITICAL-2): in `register()`, **register the `/lifemodel` command FIRST**, before any load-bearing wiring, so that even when the brain fails and we re-raise, the owner keeps the diagnostic command *if Hermes retains partial registration*, and `/lifemodel status` can report `boot_failed: <reason>` from the persisted boot-health record. Net owner signal: `hermes plugins list` shows **not-enabled** + an ERROR traceback in the log + `/lifemodel status` shows the reason.
- **Classification:**
  - `register_being_platform` (brain) → **required**.
  - `post_llm_call` / `pre_gateway_dispatch` observers → **required on a supporting host**; `optional` only after an explicit capability check proves the host lacks the hook API. A throw from *our* builder/import is required-loud.
- **`connect()` coverage** (codex MAJOR-6): brain-loop startup = **required**; metrics sampler = optional/degraded (set health, WARNING+traceback, keep the brain alive); trace writer = required-for-observability (its whole job is making failure visible) → fail loud. All paths set `BrainHealth` + log with traceback.
- **Loop death** (codex MAJOR-7): `_on_loop_death()` logs **ERROR `exc_info=True`** on an exception death, sets `loop_dead`, stores `last_loop_death` + `death_count`; status/`check_fn` reflect it until a clean reconnect clears it. A failed fatal-notify is also ERROR, not INFO.
- **Runtime observer failures** (codex MAJOR-4): wrap each observer body (the `build_lifemodel()`+`run_frame()` lambdas in `hooks.py`) in plugin-owned handling: **ERROR `exc_info=True`**, record `last_observer_error` in `BrainHealth`, bump a failure metric. Do not rely on Hermes' hook wrapper for observability.

### 4.4 Positive health signal
- `check_fn` returns `BrainHealth`-derived status (not `True`).
- `/lifemodel status` gains a **brain-liveness block**: `state`, `last_tick_at`, `ticks_total`, `loop_alive?`, `death_count`, last error(s). The owner reads liveness in the surface they already use.
- **Heartbeat** metric per tick into `metrics.sqlite` — **supporting evidence only** (codex MAJOR-8): the *primary* liveness is the durable `last_tick_at`/`ticks_total` in `BrainHealth`, so a dead metrics sampler cannot re-introduce ambiguity.

## 5. Acceptance criteria
1. `state/metrics_store.py` relative; zero absolute self-imports in runtime dirs (linter green).
2. Linter fails on a deliberately-added `from lifemodel.x import y` / `importlib.import_module("lifemodel.x")` / `hermes_plugins.lifemodel` self-ref (prove by temporary edit).
3. Faithful-load smoke test **fails on the pre-fix tree** and passes after; it resolves NO `lifemodel` top-level module (sys.path/sys.modules scrubbed).
4. Fail-loud: a forced exception in a `required` step makes `register()`/`connect()` **raise** and logs ERROR **with traceback**; an `optional` step failure logs WARNING **with traceback** and continues. Asserted with a fake ctx + `BrainHealth` inspection.
5. `check_fn` returns unhealthy-with-reason when the brain isn't connected / last tick is stale / loop died. `/lifemodel status` shows the brain-liveness block; a simulated loop death and a simulated boot failure are both visible in it.
6. Tick advances `last_tick_at`/`ticks_total` and emits a heartbeat metric.
7. `make check` green. On the live being after deploy: `being_connected` logged **and** `ticks_total` advanced within one interval; forcing brain wiring to fail makes `hermes plugins list` show lifemodel **not-enabled** with the error (loud), not "enabled".

## 6. Non-goals
- Not redesigning the brain/engine. Not a broader alerting system. `testing/`'s absolute imports (test-only) are an optional follow-up, tracked separately.

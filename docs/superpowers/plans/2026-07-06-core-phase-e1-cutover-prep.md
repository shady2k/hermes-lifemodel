# Core Rebuild — Phase E1: Cutover Prep (surface LaunchProactive + verdict correlation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two small, safe, **still-isolated** changes that the live cutover (Phase E2) needs: (1) the `CoreLoop` must **surface** the `LaunchProactive` intents its cognition layer emits (today they are silently dropped) so the egress host can act on them; and (2) the `verdict` signal must carry a **`correlation_id`** so a returning verdict can be matched to the exact proactive turn it resolves (feeding the async invalidation check). **No live cutover yet** — the cron/egress path and the hooks are untouched here; this only prepares the seams.

**Architecture:** `CoreLoop.tick()` currently collects every non-`EmitSignal` intent and hands it to the state-actor, which ignores anything that is not an `UpdateState` — so a `LaunchProactive` is lost. This phase routes `LaunchProactive` intents into the returned `TickReport` instead (a pure additive change to the report), so Phase E2's `run_proactive_tick` can consume them and call `egress.reach_out`. Separately, `verdict_signal`/`read_verdict` gain an optional `correlation_id` (defaulting to `""`) that Phase E2's `post_llm` hook fills from `pending_proactive_id`, and Phase E2's aggregation staleness gate reads.

**Tech Stack:** Python 3.11 stdlib-only. `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.**
- **Additive only — no live-path change.** Do NOT modify `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **Backward compatibility of the `verdict` signal:** `correlation_id` is **optional** (default `""`) so existing B2 verdict tests keep passing unchanged.
- **`mypy -p lifemodel` strict.**
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `core/coreloop.py` — add `TickReport.launches`; route `LaunchProactive` into it (Task 1).
- Modify `core/taxonomy.py` — `verdict_signal` + `read_verdict` gain `correlation_id`; add `read_verdict_correlation` (Task 2).
- Modify `core/__init__.py` — re-export `read_verdict_correlation`.
- Tests: extend `tests/test_coreloop.py`, `tests/test_taxonomy.py`.

**Interfaces produced (Phase E2 consumes):**
- `core/coreloop.py`: `TickReport(..., launches: tuple[LaunchProactive, ...])`.
- `core/taxonomy.py`: `verdict_signal(*, origin_id, verdict, timestamp, correlation_id: str = "")`; `read_verdict_correlation(signal) -> str`.

---

### Task 1: CoreLoop surfaces `LaunchProactive` in `TickReport`

**Files:**
- Modify: `core/coreloop.py`
- Test: `tests/test_coreloop.py` (extend)

**Interfaces:**
- Consumes: `LaunchProactive` (`core/intents.py`, D1).
- Produces: `TickReport.launches: tuple[LaunchProactive, ...]`.

**Behavior:** during the per-component loop, a returned `LaunchProactive` intent is collected into a `launches` list (instead of being appended to `intents` and dropped by the state-actor). `EmitSignal` still threads transiently; `UpdateState` still goes to the state-actor. `TickReport` gains a `launches` field. This lets Phase E2's host consume the launch and reach out.

- [ ] **Step 1: Write the failing test (append to `tests/test_coreloop.py`)**

```python
from lifemodel.core.intents import LaunchProactive


class Launcher:
    id = "launcher"

    def step(self, ctx) -> list:
        return [LaunchProactive(prompt="hi", correlation_id="c-1")]


def test_launch_proactive_is_surfaced_in_report(tmp_path) -> None:
    reg = ComponentRegistry()
    reg.register(Launcher(), ComponentManifest(id="launcher", type="cognition"))
    loop = _loop(reg, RecordingStore(), FileSignalBus(tmp_path))
    report = loop.tick()
    assert len(report.launches) == 1
    assert report.launches[0].correlation_id == "c-1"
    assert report.launches[0].prompt == "hi"


def test_no_launch_means_empty_tuple(tmp_path) -> None:
    loop = _loop(ComponentRegistry(), RecordingStore(), FileSignalBus(tmp_path))
    assert loop.tick().launches == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: FAIL — `TickReport` has no `launches`.

- [ ] **Step 3: Implement in `core/coreloop.py`**

Add the import `from .intents import EmitSignal, Intent, LaunchProactive, UpdateState` (extend the existing intents import). Add a field to `TickReport`:
```python
    launches: tuple[LaunchProactive, ...] = ()
```
In `tick()`, add a `launches: list[LaunchProactive] = []` accumulator next to `intents`, and in the per-component intent loop route `LaunchProactive` into it:
```python
            for intent in produced:
                if isinstance(intent, EmitSignal):
                    available.append(intent.signal)
                elif isinstance(intent, LaunchProactive):
                    launches.append(intent)
                else:
                    intents.append(intent)
```
Include `launches=tuple(launches)` in the returned `TickReport(...)`.

- [ ] **Step 4: Run the coreloop suite to verify green**

Run: `uv run pytest tests/test_coreloop.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/coreloop.py tests/test_coreloop.py
uv run ruff check core/coreloop.py tests/test_coreloop.py
uv run mypy -p lifemodel
git add core/coreloop.py tests/test_coreloop.py
git commit -m "feat(core): CoreLoop surfaces LaunchProactive in TickReport (cutover prep)"
```

---

### Task 2: `verdict` signal carries a `correlation_id`

**Files:**
- Modify: `core/taxonomy.py`, `core/__init__.py`
- Test: `tests/test_taxonomy.py` (extend)

**Interfaces:**
- Produces: `verdict_signal(*, origin_id, verdict, timestamp, correlation_id: str = "")`; `read_verdict_correlation(signal) -> str`.

**Behavior:** the verdict signal gains an optional `correlation_id` in its payload (default `""`), matched in Phase E2 against `pending_proactive_id` by the async invalidation check. `read_verdict` is unchanged; `read_verdict_correlation` extracts the id (or `""`). Existing B2 verdict tests (which omit `correlation_id`) keep passing.

- [ ] **Step 1: Write the failing tests (append to `tests/test_taxonomy.py`)**

```python
from lifemodel.core.taxonomy import read_verdict_correlation


def test_verdict_signal_carries_correlation_id() -> None:
    from lifemodel.sim.aggregation import Verdict

    sig = verdict_signal(origin_id="v9", verdict=Verdict.FULFILL, timestamp=None, correlation_id="proactive-X")
    assert read_verdict(sig) is Verdict.FULFILL
    assert read_verdict_correlation(sig) == "proactive-X"


def test_verdict_correlation_defaults_empty() -> None:
    from lifemodel.sim.aggregation import Verdict

    sig = verdict_signal(origin_id="v10", verdict=Verdict.REJECT, timestamp=None)
    assert read_verdict_correlation(sig) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: FAIL — `verdict_signal` has no `correlation_id`; no `read_verdict_correlation`.

- [ ] **Step 3: Implement in `core/taxonomy.py`**

Change `verdict_signal` to accept and store `correlation_id`:
```python
def verdict_signal(
    *, origin_id: str, verdict: Verdict, timestamp: str | None, correlation_id: str = ""
) -> Signal:
    """Build a durable verdict-input signal (cognition's decision on a desire)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_VERDICT,
        payload={"verdict": verdict.value, "correlation_id": correlation_id},
        timestamp=timestamp,
    )
```
Add the reader:
```python
def read_verdict_correlation(signal: Signal) -> str:
    """The correlation id a verdict resolves (``""`` if absent)."""
    if signal.kind != KIND_VERDICT:
        raise ValueError(f"not a verdict signal: kind={signal.kind!r}")
    raw = signal.payload.get("correlation_id", "")
    return raw if isinstance(raw, str) else ""
```
Re-export `read_verdict_correlation` from `core/__init__.py`.

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new taxonomy tests pass; every prior test (incl. B2 verdict tests that omit `correlation_id`, and `tests/sim/`) still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run ruff check core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run mypy -p lifemodel
git add core/taxonomy.py core/__init__.py tests/test_taxonomy.py
git commit -m "feat(core): verdict signal carries correlation_id for async invalidation (cutover prep)"
```

---

## Phase-E1 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Two commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** surfaces the cognition→host `LaunchProactive` seam (§13 model A) → Task 1; correlation id for the async verdict match (§7.3) → Task 2. Both are additive prep — the actual wiring (egress consumes launches, hook fills correlation, aggregation gates on staleness) is Phase E2.
- **Backward compatibility preserved:** `launches` defaults to `()`; `correlation_id` defaults to `""` — no existing test changes.
- **No placeholders:** every step ships real code + an exact command with expected output.

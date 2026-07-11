# Phase-2 Close: Last-Wake-Outcome + Adapter Smoke Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Phase 2 (`lm-fib`) by shipping the two remaining blockers: a compact "last wake outcome" line in `/lifemodel debug` (`lm-9zj`), and a cheap pre-deploy adapter smoke probe (`lm-dte`).

**Architecture:** `lm-9zj` adds a read-only reader over the existing trace store (`observability.sqlite`) that finds the most recent proactive terminal outcome and renders it in the debug dump; the model's full reasoning already lives in `/lifemodel trace` and is untouched. `lm-dte` adds `lifemodel/smoke.py` + a `make smoke` target run by the Hermes venv interpreter that asserts the adapter has no unimplemented abstract methods and constructs cleanly.

**Tech Stack:** Python stdlib only (runtime), `sqlite3`, `dataclasses`; `ruff`/`mypy`/`pytest` under `uv` for the dev gate; GNU Make.

## Global Constraints

- **Runtime code is stdlib-only** and uses **relative imports** (`.foo`, `..bar`) — it loads inside Hermes' interpreter, not a pip install. No third-party runtime imports.
- **The gate is `make check`**: `ruff format --check .`, `ruff check .`, `mypy -p lifemodel`, `pytest`. Every task ends green.
- **Conservative git profile (CLAUDE.md):** do NOT commit or push unless the owner has authorized it this session. Where a step says "Commit", run it only under that authorization; otherwise stage the change and report it. Never `git push` / `make deploy` here.
- **Do not change `/lifemodel trace` output** — `render_trace` in `trace_view.py` stays byte-identical; we only ADD functions.
- **Never touch the live being** (`~/.hermes`): the smoke probe constructs against a `tempfile.mkdtemp()` directory only.
- **House style for the debug dump:** one datum per line, `**label:** value` (single space, no column padding), timestamps via `_local(...)`.

---

## GROUP A — `lm-9zj`: last wake outcome in `/lifemodel debug`

Foundation already shipped (verified — do NOT rebuild): the turn's reasoning is durable in `trace_events` (`_log_proactive_reasoning`, `hooks.py:200`; persistence is level-independent, `log.py:195`) and rendered by `/lifemodel trace` (`render_trace`, `trace_view.py:237`). This group only adds a compact *decision* line to `/lifemodel debug`.

**Taxonomy (verified in code):** both terminal markers are `trace_events` rows —
- a delivery → `event = "proactive_delivery"` (`core/proactive.py:200`);
- a suppression → `event = "suppression"` (`EVENT_SUPPRESSION`, `core/suppression.py:40`) with `fields["reason"]` a `SuppressionReason` value (`core/suppression.py:73-84`).

"Last **wake** outcome" = the newest marker that is a delivery OR a *post-wake* suppression reason. Pre-wake resting gates (`below_threshold`, `silence_window`, `in_flight`, `pending_proactive`, `decline_backoff`) are excluded — the existing DRIVE section already explains the resting state.

### Task A1: Pure last-wake selector + value type

**Files:**
- Modify: `trace_view.py` (add `LastWakeOutcome`, `POST_WAKE_REASONS`, `pick_last_wake_outcome`, after the `_Event` dataclass ~line 67)
- Test: `tests/test_trace_view.py`

**Interfaces:**
- Consumes: the existing `_Event` dataclass (`trace_view.py:56`).
- Produces:
  - `POST_WAKE_REASONS: frozenset[str]` = `{"act_gate_silent","backstop_rate_limited","egress_unavailable","egress_failed","energy_unaffordable","repeat_pure_longing"}`
  - `@dataclass(frozen=True) class LastWakeOutcome: outcome: str; ts: str; trace_id: str`
  - `pick_last_wake_outcome(events: Sequence[_Event]) -> LastWakeOutcome | None` — returns the matching event with the greatest `(ts, record_id)` (ts is fixed-width ISO-8601 UTC, so lexical compare == chronological), or `None` if none match.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trace_view.py`:

```python
from lifemodel.trace_view import (
    LastWakeOutcome,
    _Event,
    pick_last_wake_outcome,
)


def _ev(record_id: int, event: str, ts: str, **fields) -> _Event:
    return _Event(
        record_id=record_id, trace_id=f"t{record_id}", span_id="s",
        tick=record_id, event=event, ts=ts, fields=fields,
    )


def test_pick_last_wake_prefers_newest_post_wake_marker() -> None:
    events = [
        _ev(1, "suppression", "2026-07-11T10:00:00+00:00", reason="below_threshold"),
        _ev(2, "proactive_delivery", "2026-07-11T10:03:00+00:00", outcome="delivered"),
        _ev(3, "suppression", "2026-07-11T10:05:00+00:00", reason="act_gate_silent"),
    ]
    result = pick_last_wake_outcome(events)
    assert result == LastWakeOutcome(outcome="act_gate_silent", ts="2026-07-11T10:05:00+00:00", trace_id="t3")


def test_pick_last_wake_ignores_pre_wake_gates() -> None:
    events = [
        _ev(1, "proactive_delivery", "2026-07-11T09:00:00+00:00", outcome="delivered"),
        _ev(2, "suppression", "2026-07-11T10:00:00+00:00", reason="below_threshold"),
        _ev(3, "suppression", "2026-07-11T10:01:00+00:00", reason="silence_window"),
    ]
    result = pick_last_wake_outcome(events)
    assert result is not None
    assert result.outcome == "delivered"  # newest *wake* marker, not the later resting gates


def test_pick_last_wake_none_when_only_resting_gates() -> None:
    events = [_ev(1, "suppression", "2026-07-11T10:00:00+00:00", reason="below_threshold")]
    assert pick_last_wake_outcome(events) is None


def test_pick_last_wake_skips_suppression_without_reason() -> None:
    events = [_ev(1, "suppression", "2026-07-11T10:00:00+00:00")]  # malformed: no reason
    assert pick_last_wake_outcome(events) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trace_view.py -k pick_last_wake -o addopts="" -q`
Expected: FAIL with `ImportError: cannot import name 'LastWakeOutcome'` (or `pick_last_wake_outcome`).

- [ ] **Step 3: Implement the selector**

In `trace_view.py`, immediately after the `_Event` dataclass (line 67), add:

```python
#: Suppression reasons that mean the being actually WOKE cognition and then held
#: back — the "why silent" cases. The pre-wake resting gates (below_threshold,
#: silence_window, in_flight, pending_proactive, decline_backoff) are deliberately
#: excluded: the DRIVE section of /lifemodel debug already explains the resting state.
POST_WAKE_REASONS: frozenset[str] = frozenset(
    {
        "act_gate_silent",
        "backstop_rate_limited",
        "egress_unavailable",
        "egress_failed",
        "energy_unaffordable",
        "repeat_pure_longing",
    }
)


@dataclass(frozen=True)
class LastWakeOutcome:
    """The most recent proactive terminal decision, for the /lifemodel debug dump."""

    outcome: str
    ts: str
    trace_id: str


def pick_last_wake_outcome(events: Sequence[_Event]) -> LastWakeOutcome | None:
    """The newest delivery / post-wake-suppression among *events*, or ``None``.

    A delivery is ``event == "proactive_delivery"`` (outcome ``"delivered"``); a
    post-wake suppression is ``event == "suppression"`` whose ``fields["reason"]``
    is in :data:`POST_WAKE_REASONS`. ``ts`` is fixed-width ISO-8601 UTC, so the
    lexical ``(ts, record_id)`` max is the chronological latest.
    """
    best: LastWakeOutcome | None = None
    best_key: tuple[str, int] = ("", -1)
    for e in events:
        if e.event == "proactive_delivery":
            outcome = "delivered"
        elif e.event == "suppression":
            reason = e.fields.get("reason")
            if not isinstance(reason, str) or reason not in POST_WAKE_REASONS:
                continue
            outcome = reason
        else:
            continue
        key = (e.ts, e.record_id)
        if key > best_key:
            best_key = key
            best = LastWakeOutcome(outcome=outcome, ts=e.ts, trace_id=e.trace_id)
    return best
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trace_view.py -k pick_last_wake -o addopts="" -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit** (only if commits are authorized this session — see Global Constraints)

```bash
git add trace_view.py tests/test_trace_view.py
git commit -m "feat(trace_view): pure last-wake-outcome selector (lm-9zj)"
```

### Task A2: sqlite reader `read_last_wake_outcome`

**Files:**
- Modify: `trace_view.py` (add `read_last_wake_outcome`, near `trace_for_dir` ~line 319)
- Test: `tests/test_trace_view.py`

**Interfaces:**
- Consumes: `pick_last_wake_outcome`, `_Event`, `_loads` (Task A1 + existing); `observability_db_path`, `connect` (existing imports, `trace_view.py:33`); `_live_ring_flush` (`trace_view.py:309`).
- Produces: `read_last_wake_outcome(base_dir: Path) -> LastWakeOutcome | None` — fail-soft (missing/unreadable store → `None`).

**Note:** a bounded scan (`LIMIT`) keeps the every-tick `below_threshold` rows from being fully decoded; it covers recent history (a wake older than the window reads as "none"), which is the right semantic for a *debug* view.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trace_view.py` (the file already imports `acquire_trace_writer`, `observability_db_path`, `release_trace_writer` — reuse them):

```python
from lifemodel.trace_view import read_last_wake_outcome


def test_read_last_wake_outcome_from_store(tmp_path) -> None:
    db = observability_db_path(tmp_path)
    writer = acquire_trace_writer(db)
    try:
        writer.submit_event(record_id=1, trace_id="t1", span_id="s", tick=1,
                            event="suppression", ts="2026-07-11T10:00:00+00:00",
                            fields={"reason": "below_threshold"})
        writer.submit_event(record_id=2, trace_id="t2", span_id="s", tick=2,
                            event="proactive_delivery", ts="2026-07-11T10:03:00+00:00",
                            fields={"outcome": "delivered"})
        writer.submit_event(record_id=3, trace_id="t3", span_id="s", tick=3,
                            event="suppression", ts="2026-07-11T10:05:00+00:00",
                            fields={"reason": "act_gate_silent"})
        writer.flush(timeout=5.0)
        result = read_last_wake_outcome(tmp_path)
    finally:
        release_trace_writer(db)
    assert result is not None
    assert result.outcome == "act_gate_silent"
    assert result.trace_id == "t3"


def test_read_last_wake_outcome_missing_store_is_none(tmp_path) -> None:
    # No observability.sqlite created → fail-soft None, never raise.
    assert read_last_wake_outcome(tmp_path) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_trace_view.py -k read_last_wake_outcome -o addopts="" -q`
Expected: FAIL with `ImportError: cannot import name 'read_last_wake_outcome'`.

- [ ] **Step 3: Implement the reader**

In `trace_view.py`, add (after `trace_for_dir`):

```python
#: Bounded scan for the last-wake reader: enough to span days of every-tick
#: below_threshold rows so a genuine recent wake is still found, cheap to decode.
_LAST_WAKE_SCAN_LIMIT = 5000


def read_last_wake_outcome(base_dir: Path) -> LastWakeOutcome | None:
    """The being's most recent proactive terminal decision, or ``None``.

    Read-only + fail-soft (invariant law 3): a missing/locked/corrupt trace DB
    degrades to ``None`` (the debug dump then shows a friendly line), never raises.
    Flushes the live singleton writer first for read-your-writes (§4.2).
    """
    db_path = observability_db_path(base_dir)
    if not db_path.exists():
        return None
    _live_ring_flush(base_dir)
    try:
        with closing(connect(db_path, create_parent=False)) as conn:
            rows = conn.execute(
                "SELECT record_id, trace_id, span_id, tick, event, ts, fields_json "
                "FROM trace_events WHERE event IN ('proactive_delivery', 'suppression') "
                "ORDER BY ts DESC, record_id DESC LIMIT ?",
                (_LAST_WAKE_SCAN_LIMIT,),
            ).fetchall()
    except sqlite3.Error:
        return None
    events = [_Event(r[0], r[1], r[2], r[3], r[4], r[5], _loads(r[6])) for r in rows]
    return pick_last_wake_outcome(events)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_trace_view.py -k read_last_wake_outcome -o addopts="" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Confirm `/lifemodel trace` is unchanged**

Run: `uv run pytest tests/test_trace_view.py -o addopts="" -q`
Expected: PASS (all existing trace tests still green — render_trace untouched).

- [ ] **Step 6: Commit** (only if authorized)

```bash
git add trace_view.py tests/test_trace_view.py
git commit -m "feat(trace_view): read_last_wake_outcome reader over the trace store (lm-9zj)"
```

### Task A3: render the section + wire `render_dump_for_dir`

**Files:**
- Modify: `debug.py` (`render_debug_dump` signature + new section; `render_dump_for_dir` reads + passes)
- Test: `tests/test_debug.py`

**Interfaces:**
- Consumes: `LastWakeOutcome`, `read_last_wake_outcome` (Task A1/A2); `_local` (`debug.py:97`); `_metrics` (`debug.py:161`).
- Produces: `render_debug_dump(*, readings: Readings, last_wake: LastWakeOutcome | None = None) -> str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_debug.py`:

```python
from lifemodel.debug import render_debug_dump, render_dump_for_dir
from lifemodel.trace_view import LastWakeOutcome
# `_readings()` helper: reuse whatever the existing tests use to build a Readings;
# if none exists, build via compute_readings(State(), now=..., cfg=_cfg()).


def test_debug_shows_last_wake_outcome_line() -> None:
    lw = LastWakeOutcome(outcome="act_gate_silent", ts="2026-07-11T10:05:00+00:00", trace_id="abc123")
    out = render_debug_dump(readings=_readings(), last_wake=lw)
    assert "LAST WAKE OUTCOME" in out
    assert "act_gate_silent" in out
    assert "abc123" in out  # trace_id, so the owner can /lifemodel trace abc123 for reasoning


def test_debug_last_wake_none_shows_friendly_line() -> None:
    out = render_debug_dump(readings=_readings(), last_wake=None)
    assert "LAST WAKE OUTCOME" in out
    assert "no wake outcome recorded yet" in out


def test_debug_never_renders_reasoning_text() -> None:
    lw = LastWakeOutcome(outcome="act_gate_silent", ts="2026-07-11T10:05:00+00:00", trace_id="abc123")
    out = render_debug_dump(readings=_readings(), last_wake=lw)
    assert "reasoning" not in out.lower()  # reasoning stays in /lifemodel trace only
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_debug.py -k last_wake -o addopts="" -q`
Expected: FAIL — `render_debug_dump()` got an unexpected keyword argument `last_wake` (or the assertions fail).

- [ ] **Step 3: Implement**

In `debug.py`, add the import at the top (with the other local imports):

```python
from .trace_view import LastWakeOutcome, read_last_wake_outcome
```

Change the `render_debug_dump` signature (line 178) and append the section before the final `return`. First the signature:

```python
def render_debug_dump(*, readings: Readings, last_wake: LastWakeOutcome | None = None) -> str:
```

Then, immediately before `render_debug_dump` returns its joined string, add:

```python
    lines.append("**LAST WAKE OUTCOME**")
    if last_wake is None:
        lines.append("  (no wake outcome recorded yet)")
    else:
        lines += _metrics(
            [
                ("outcome", last_wake.outcome),
                ("when", _local(last_wake.ts)),
                ("trace", f"`{last_wake.trace_id}`  (→ /lifemodel trace {last_wake.trace_id})"),
            ]
        )
    lines.append("")
```

Then wire the reader into `render_dump_for_dir` (line 121). After `now = lm.clock.now()` and before the `return`, read it fail-soft and pass it in:

```python
    last_wake = read_last_wake_outcome(base_dir)  # fail-soft: None when no/unreadable trace store
```

and pass `last_wake=last_wake` into the `render_debug_dump(...)` call.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_debug.py -k last_wake -o addopts="" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Fail-soft integration check**

Add and run:

```python
def test_render_dump_for_dir_no_trace_store_does_not_crash(tmp_path) -> None:
    # A fresh base_dir with no observability.sqlite: the dump renders with the
    # friendly line, never raises.
    out = render_dump_for_dir(tmp_path)
    assert "LAST WAKE OUTCOME" in out
    assert "no wake outcome recorded yet" in out
```

Run: `uv run pytest tests/test_debug.py -k no_trace_store -o addopts="" -q`
Expected: PASS.

- [ ] **Step 6: Full gate**

Run: `make check`
Expected: all green (ruff, mypy, pytest).

- [ ] **Step 7: Commit** (only if authorized)

```bash
git add debug.py tests/test_debug.py
git commit -m "feat(debug): LAST WAKE OUTCOME section in /lifemodel debug (lm-9zj)"
```

---

## GROUP B — `lm-dte`: cheap adapter smoke probe

Runs under the **Hermes venv** (which has `gateway`), not `uv`. The load-bearing check is `__abstractmethods__ == frozenset()` (the version-skew guard); construction is the doc-endorsed secondary check. No `connect()`/loop.

> **Post-implementation deviation (2026-07-11):** the construction step (B2) was **dropped**, per the spec's pre-authorized fallback. `make smoke` revealed that `BeingAdapter.__init__` calls `Platform(PLATFORM_NAME)`, and the real gateway only makes that name valid *after* `register()` runs `ctx.register_platform(...)` (`adapters/being_platform.py:307`) — so standalone construction raises `ValueError("'lifemodel' is not a valid Platform")`. Replicating gateway-internal platform registration is exactly the fragile coupling the spec warned against, so `run_smoke`'s `construct` is now optional and `_main` calls `run_smoke(BeingAdapter)` (import + abstract-method guard only). The version-skew guard — the load-bearing part that catches the `get_chat_info` class of bug — passes green (`SMOKE OK`).

### Task B1: `smoke.py` — `run_smoke` + `SmokeFailure`

**Files:**
- Create: `smoke.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces:
  - `class SmokeFailure(Exception)`
  - `run_smoke(adapter_cls: type, construct: Callable[[], object]) -> None` — raises `SmokeFailure` if `adapter_cls.__abstractmethods__` is non-empty, or if `construct()` raises; returns `None` on success. `construct` is injected so the pure logic is unit-testable without `gateway`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_smoke.py`:

```python
from abc import ABC, abstractmethod

import pytest

from lifemodel.smoke import SmokeFailure, run_smoke


class _Incomplete(ABC):
    @abstractmethod
    def must_impl(self) -> None: ...


class _Complete(_Incomplete):
    def must_impl(self) -> None:
        return None


def test_run_smoke_raises_on_unimplemented_abstractmethods() -> None:
    with pytest.raises(SmokeFailure, match="must_impl"):
        run_smoke(_Incomplete, lambda: None)


def test_run_smoke_passes_on_concrete_class_and_calls_construct() -> None:
    called: list[int] = []
    run_smoke(_Complete, lambda: called.append(1) or object())
    assert called == [1]


def test_run_smoke_wraps_construction_failure() -> None:
    def boom() -> object:
        raise RuntimeError("bad config shape")

    with pytest.raises(SmokeFailure, match="bad config shape"):
        run_smoke(_Complete, boom)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_smoke.py -o addopts="" -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lifemodel.smoke'`.

- [ ] **Step 3: Implement `smoke.py`**

Create `smoke.py`:

```python
"""Pre-deploy smoke probe for the platform-adapter shell (bead lm-dte).

`make check` runs under `uv`, whose venv lacks Hermes' `gateway` package, so
`BasePlatformAdapter` is invisible off-host (mypy sees `Any`, pytest can't
instantiate the adapter). This probe closes that blind spot: run under the
Hermes venv, it asserts the adapter implements every abstract method of the
*actually-installed* base (the version-skew guard that catches the get_chat_info
class of failure) and constructs cleanly. The load-bearing check needs no config;
construction is the doc-endorsed secondary check. It never starts the brain loop
and never touches the live being (a throwaway temp dir only).
"""

from __future__ import annotations

from collections.abc import Callable


class SmokeFailure(Exception):
    """A pre-deploy adapter-shell check failed."""


def run_smoke(adapter_cls: type, construct: Callable[[], object]) -> None:
    """Assert *adapter_cls* is fully concrete, then that *construct* succeeds.

    Raises :class:`SmokeFailure` (never a bare AssertionError/arbitrary error) so
    the ``__main__`` entry can print one clean message and exit non-zero.
    """
    missing = getattr(adapter_cls, "__abstractmethods__", frozenset())
    if missing:
        raise SmokeFailure(
            f"{adapter_cls.__name__} has unimplemented abstract methods: "
            f"{sorted(missing)} — the installed gateway base declares abstract methods "
            f"this adapter does not implement (it would fail to instantiate at connect)."
        )
    try:
        construct()
    except Exception as exc:  # noqa: BLE001 - normalize every construction failure
        raise SmokeFailure(f"{adapter_cls.__name__} construction failed: {exc!r}") from exc
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_smoke.py -o addopts="" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit** (only if authorized)

```bash
git add smoke.py tests/test_smoke.py
git commit -m "feat(smoke): run_smoke adapter-shell probe core (lm-dte)"
```

### Task B2: `smoke.py` `__main__` entry

**Files:**
- Modify: `smoke.py` (add `_main` + `if __name__ == "__main__"`)

**Interfaces:**
- Consumes: `run_smoke`, `SmokeFailure` (Task B1); `BeingAdapter` (`adapters/being_platform.py:73`).
- Produces: `_main() -> int` (0 ok, 1 fail).

**Note:** this entry imports `gateway` (via `BeingAdapter`), so it is exercised by `make smoke` against the Hermes venv, NOT by `pytest`. Keep it thin — all testable logic is in `run_smoke`.

- [ ] **Step 1: Implement `_main`**

Append to `smoke.py`:

```python
def _main() -> int:
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="lifemodel-smoke-"))
    try:
        # Import here (not at module top) so an import/shell regression is caught
        # as a smoke failure, and so `run_smoke`'s unit tests never import gateway.
        from .adapters.being_platform import BeingAdapter

        run_smoke(
            BeingAdapter,
            # config=None matches tests/test_being_platform_fail_loud.py's construction;
            # if the installed base rejects None, replace with a minimal stub config.
            lambda: BeingAdapter(config=None, base_dir=tmp, target=None),
        )
    except SmokeFailure as exc:
        print(f"SMOKE FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - e.g. an import error is also a smoke failure
        print(f"SMOKE FAIL: import/setup error: {exc!r}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 2: Verify the dev gate still passes (entry is import-safe under uv)**

Run: `uv run python -c "import lifemodel.smoke"`
Expected: no output, exit 0 (module imports without pulling gateway — the `BeingAdapter` import is inside `_main`).

Run: `make check`
Expected: all green.

- [ ] **Step 3: Commit** (only if authorized)

```bash
git add smoke.py
git commit -m "feat(smoke): __main__ entry constructing BeingAdapter in a temp dir (lm-dte)"
```

### Task B3: Makefile `smoke` target + `deploy` pre-flight

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the target and pre-flight dependency**

In `Makefile`, update the `.PHONY` line to include `smoke`:

```makefile
.PHONY: help check fmt test smoke deploy
```

Add near the top (after `.DEFAULT_GOAL`), the overridable venv python:

```makefile
# The Hermes runtime interpreter — the ONLY venv with the `gateway` package.
# Override if your being lives elsewhere: `make smoke HERMES_VENV_PY=/path/to/python`.
HERMES_VENV_PY ?= $(HOME)/.hermes/hermes-agent/venv/bin/python
```

Add the `smoke` target (after `test:`):

```makefile
smoke:  ## Adapter-shell smoke check against the Hermes venv (pre-deploy, needs gateway)
	@test -x "$(HERMES_VENV_PY)" || { echo "!! Hermes venv python not found at $(HERMES_VENV_PY) — set HERMES_VENV_PY=/path/to/python"; exit 1; }
	PYTHONPATH=.. "$(HERMES_VENV_PY)" -m lifemodel.smoke
```

Make `deploy` depend on `smoke` (pre-flight before push) — change the `deploy:` line:

```makefile
deploy: smoke  ## Deploy to the live being: smoke, push, pull into ~/.hermes, restart gateway
```

(The recipe body of `deploy` is unchanged; `smoke` runs first and aborts the deploy on failure.)

- [ ] **Step 2: Run the smoke target against the real venv**

Run: `make smoke`
Expected: `SMOKE OK` and exit 0 — **on a machine with the Hermes venv present**.
- If `config=None` is rejected by the installed base, `_main` prints `SMOKE FAIL: ... construction failed: ...`; replace the `construct` lambda in Task B2 with a minimal stub config carrying the attributes the error names, then re-run. (The `__abstractmethods__` guard still ran and is the load-bearing part.)
- If run on a machine without the Hermes venv, the target prints the "not found" hint and exits 1 — expected; run it where the being lives.

- [ ] **Step 3: Confirm `make check` is unaffected**

Run: `make check`
Expected: all green (Makefile change does not touch the `uv`-based gate).

- [ ] **Step 4: Commit** (only if authorized)

```bash
git add Makefile
git commit -m "build(make): smoke target + deploy pre-flight (lm-dte)"
```

---

## Wrap-up (after both groups)

- [ ] Run the full gate once more: `make check` → all green.
- [ ] Run `make smoke` on the being's host → `SMOKE OK`.
- [ ] Report to the owner: changed files, `make check` result, `make smoke` result. Do NOT commit/push or `bd close lm-9zj` / `bd close lm-dte` unless the owner authorizes (conservative profile). Suggested close evidence is captured per task above.

---

## Self-Review

**1. Spec coverage:**
- lm-9zj "last wake outcome block, outcome + ts + trace_id, no reasoning" → Tasks A1–A3. ✓
- lm-9zj "read trace store, reuse trace_view helpers, /lifemodel trace byte-identical" → A2 (adds functions only; A2 Step 5 asserts trace tests still pass). ✓
- lm-9zj "fail-soft missing/unreadable/no-wake" → A2 (`missing store → None`, `sqlite3.Error → None`), A3 Step 5 (dir-level no-crash). ✓
- lm-9zj "reasoning already in /lifemodel trace, don't rebuild" → foundation note, no task touches reasoning capture. ✓
- lm-9zj "reachability dropped" → not implemented (correct); egress_* are outcome values. ✓
- lm-dte "run_smoke factored to take adapter class, dev-venv unit-testable" → B1 (`run_smoke(adapter_cls, construct)`, tested with fakes). ✓
- lm-dte "import + __abstractmethods__ + construct in tempdir, no connect" → B1 (abstractmethods) + B2 (import + construct in `mkdtemp`, no `connect`). ✓
- lm-dte "make smoke (Hermes venv default, overridable, PYTHONPATH, working tree) + deploy pre-flight" → B3. ✓
- Constraints: stdlib-only + relative imports (smoke.py uses `collections.abc`, `.adapters...`; trace_view/debug already stdlib) ✓; make check gate each task ✓; conservative git (commit steps gated) ✓; never touch live being (tempdir) ✓.

**2. Placeholder scan:** One intentional test helper reference — `_readings()` in Task A3 Step 1 — is annotated inline with how to build it (reuse the existing test helper or `compute_readings(State(), now=..., cfg=_cfg())`). No "TBD"/"handle edge cases"/"similar to". All code steps show real code.

**3. Type consistency:** `LastWakeOutcome(outcome, ts, trace_id)` identical across A1/A2/A3; `pick_last_wake_outcome(events) -> LastWakeOutcome | None` and `read_last_wake_outcome(base_dir) -> LastWakeOutcome | None` consistent; `run_smoke(adapter_cls: type, construct: Callable[[], object]) -> None` and `SmokeFailure` consistent across B1/B2; `render_debug_dump(*, readings, last_wake=None)` matches the A3 call site. ✓

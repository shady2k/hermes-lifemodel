# Core Rebuild — Phase D2: Global Backstop + Async Semantic Invalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two *pure, safety-critical* guards the cutover (Phase E) needs: the **global safety backstop** (a hard, fail-closed rate limit on real proactive sends — ≤3/day, ≥60 min apart, independent of the desire model) and the **async semantic invalidation** check (decide whether a verdict that arrived after a proactive turn is still valid, or is stale and must be dropped — so a reply that lands while the being is composing never produces a double message). Both are pure functions unit-tested here; they are applied at the live send/verdict sites in Phase E.

**Architecture (spec §14, §7.3):** The backstop is a pure rate-limiter over a persisted `proactive_send_log` (ISO timestamps of real sends); it fails **closed** (deny on any doubt). The invalidation check is a pure predicate over the state at verdict-application time: a verdict is stale if the desire was already resolved, the verdict's correlation id no longer matches the live pending turn, the user replied after the turn launched, the effective pressure fell below threshold while thinking, or a deadline passed (spec §7.3 — semantic, not version-based). Neither is wired live here (Phase E applies the backstop before a real send and the invalidation before applying a `post_llm` verdict).

**Tech Stack:** Python 3.11 stdlib-only (`datetime`). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.**
- **Backstop fails CLOSED (spec §14):** on a malformed log entry or any ambiguity, deny the send (user protection over the being's urge).
- **Invalidation is semantic, not version-based (spec §7.3, Codex):** stale only on a real semantic condition (desire resolved / correlation mismatch / user replied / pressure satisfied / deadline) — a mere energy/mood tick must NOT invalidate a good verdict.
- **Bootstrap constants (spec §14, §22 — one-line tunable):** `MAX_PROACTIVE_PER_DAY = 3`, `MIN_SEND_INTERVAL_MIN = 60.0`, `SEND_LOG_KEEP = 20`, `VERDICT_DEADLINE_MIN = 30.0`.
- **`mypy -p lifemodel` strict; pure/total functions (never raise on malformed input — classify defensively).**
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `state/model.py` — add `proactive_send_log: list[str]` (Task 1).
- Create `core/backstop.py` — `allow_send`, `record_send` (Task 1).
- Create `core/invalidation.py` — `is_verdict_stale` (Task 2).
- Modify `core/__init__.py` — re-exports.
- Tests: extend `tests/test_state_model.py`; create `tests/test_backstop.py`, `tests/test_invalidation.py`.

**Interfaces produced (Phase E consumes):**
- `state.model.State.proactive_send_log: list[str]`.
- `core/backstop.py`: `allow_send(send_log: Sequence[str], now: datetime, *, max_per_day: int = 3, min_interval_min: float = 60.0) -> bool`; `record_send(send_log: Sequence[str], now: datetime, *, keep: int = 20) -> list[str]`.
- `core/invalidation.py`: `is_verdict_stale(*, desire_status: str, pending_id: str | None, verdict_correlation_id: str, last_exchange_at: str | None, pending_since: str | None, effective: float, threshold: float, now: datetime, deadline_min: float = 30.0) -> tuple[bool, str]`.

---

### Task 1: Global safety backstop (rate limiter, fail-closed)

**Files:**
- Modify: `state/model.py`
- Create: `core/backstop.py`
- Modify: `core/__init__.py`
- Test: `tests/test_state_model.py` (extend), `tests/test_backstop.py`

**Interfaces:**
- Produces: `State.proactive_send_log`; `allow_send`, `record_send`.

**Behavior (spec §14):** `allow_send` returns `True` only if, over the trailing 24 h, fewer than `max_per_day` real sends have occurred **and** the last send was ≥ `min_interval_min` ago. Malformed timestamps in the log are ignored (fail-closed doesn't mean crash — a bad entry simply doesn't count as "no recent send"). `record_send` appends `now` and keeps the last `keep` entries.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backstop.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.backstop import allow_send, record_send

NOW = datetime(2026, 7, 6, 20, 0, tzinfo=UTC)


def _ago(minutes: float) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat()


def test_allows_when_log_empty() -> None:
    assert allow_send([], NOW) is True


def test_denies_when_daily_cap_reached() -> None:
    log = [_ago(600), _ago(400), _ago(200)]  # 3 sends within 24h -> cap (default 3)
    assert allow_send(log, NOW) is False


def test_denies_within_min_interval() -> None:
    log = [_ago(30)]  # last send 30 min ago < 60 min
    assert allow_send(log, NOW) is False


def test_allows_after_min_interval_and_under_cap() -> None:
    log = [_ago(90)]  # 90 min ago, only 1 today
    assert allow_send(log, NOW) is True


def test_old_sends_outside_24h_do_not_count() -> None:
    log = [_ago(60 * 25), _ago(60 * 26), _ago(60 * 27)]  # all >24h ago
    assert allow_send(log, NOW) is True


def test_malformed_entries_are_ignored_not_crashing() -> None:
    assert allow_send(["not-a-date", _ago(90)], NOW) is True  # bad entry skipped, 1 valid @90m
    assert allow_send(["garbage"], NOW) is True  # no valid recent send


def test_record_send_appends_and_bounds() -> None:
    log = [_ago(1000 + i) for i in range(25)]
    new = record_send(log, NOW, keep=20)
    assert len(new) == 20
    assert new[-1] == NOW.isoformat()  # newest last
```

```python
# append to tests/test_state_model.py
def test_proactive_send_log_defaults_empty_and_roundtrips() -> None:
    assert State().proactive_send_log == []
    assert State.from_dict({}).proactive_send_log == []  # additive
    s = State(proactive_send_log=["2026-07-06T20:00:00+00:00"])
    assert State.from_dict(s.to_dict()).proactive_send_log == ["2026-07-06T20:00:00+00:00"]


def test_proactive_send_log_rejects_non_list() -> None:
    import pytest

    from lifemodel.state.errors import StateCorruptError

    with pytest.raises(StateCorruptError):
        State.from_dict({"proactive_send_log": "nope"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backstop.py tests/test_state_model.py -q`
Expected: FAIL — `ModuleNotFoundError` for `lifemodel.core.backstop`; `State` has no `proactive_send_log`.

- [ ] **Step 3: Implement**

In `state/model.py`: add `from dataclasses import field` (if not already imported) and the field to `State`:
```python
    proactive_send_log: list[str] = field(default_factory=list)
```
Add a `_as_str_list` validator near the other `_as_*` helpers and use it in `from_dict`:
```python
def _as_str_list(data: Mapping[str, Any], key: str, default: list[str]) -> list[str]:
    if key not in data:
        return list(default)
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise StateCorruptError(f"'{key}' must be a list[str]")
    return list(value)
```
```python
        proactive_send_log=_as_str_list(data, "proactive_send_log", []),
```
(Match the existing `from_dict` construction style; `StateCorruptError` is already imported/used there.)

```python
# core/backstop.py
"""Global safety backstop — a hard rate limit on real proactive sends (spec §14).

A fail-closed guard *above* the desire model: even if the drive model and the LLM
both misbehave, the being cannot send more than ``max_per_day`` proactive messages
or send twice within ``min_interval_min``. This protects the user from a buggy /
hallucinating cognition; it is NOT the restraint mechanism (that is emergent).
Pure over a persisted log of ISO send timestamps; malformed entries are ignored
(they never count as "no recent send").
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta


def _parse(ts: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return value if value.tzinfo is not None else None


def allow_send(
    send_log: Sequence[str],
    now: datetime,
    *,
    max_per_day: int = 3,
    min_interval_min: float = 60.0,
) -> bool:
    """True only if under the daily cap AND past the minimum interval."""
    day_ago = now - timedelta(hours=24)
    recent = [t for ts in send_log if (t := _parse(ts)) is not None and t >= day_ago]
    if len(recent) >= max_per_day:
        return False
    if recent:
        last = max(recent)
        if (now - last).total_seconds() / 60.0 < min_interval_min:
            return False
    return True


def record_send(send_log: Sequence[str], now: datetime, *, keep: int = 20) -> list[str]:
    """Append this send and keep the most recent ``keep`` entries."""
    return [*send_log, now.isoformat()][-keep:]
```
Re-export `allow_send, record_send` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backstop.py tests/test_state_model.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format state/model.py core/backstop.py core/__init__.py tests/test_backstop.py tests/test_state_model.py
uv run ruff check state/model.py core/backstop.py core/__init__.py tests/test_backstop.py tests/test_state_model.py
uv run mypy -p lifemodel
git add state/model.py core/backstop.py core/__init__.py tests/test_backstop.py tests/test_state_model.py
git commit -m "feat(core): global safety backstop — fail-closed send rate-limit (spec §14)"
```

---

### Task 2: Async semantic invalidation (verdict staleness)

**Files:**
- Create: `core/invalidation.py`
- Modify: `core/__init__.py`
- Test: `tests/test_invalidation.py`

**Interfaces:**
- Consumes: `datetime`.
- Produces: `is_verdict_stale`.

**Behavior (spec §7.3, Codex):** a verdict arriving after a proactive turn is **stale** (must be dropped, not applied) if any of: the desire is no longer active (already resolved); the verdict's `correlation_id` no longer matches the live `pending_id` (it refers to an old/other desire); the user exchanged **after** the turn launched (`last_exchange_at > pending_since` — the reactive path already responded, so applying would double-message); the effective pressure fell below threshold while thinking (need satisfied); or the deadline elapsed. Otherwise it is fresh. Returns `(stale, reason)`. **Semantic only** — a mere energy/mood tick is not a reason.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_invalidation.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.invalidation import is_verdict_stale

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
PENDING = "proactive-2026-07-06T11:55:00+00:00"


def _call(**over) -> tuple[bool, str]:
    kw = dict(
        desire_status="active",
        pending_id=PENDING,
        verdict_correlation_id=PENDING,
        last_exchange_at=None,
        pending_since="2026-07-06T11:55:00+00:00",
        effective=2.0,
        threshold=1.0,
        now=NOW,
        deadline_min=30.0,
    )
    kw.update(over)
    return is_verdict_stale(**kw)  # type: ignore[arg-type]


def test_fresh_verdict_is_applied() -> None:
    stale, reason = _call()
    assert stale is False and reason == "fresh"


def test_resolved_desire_is_stale() -> None:
    assert _call(desire_status="none")[0] is True


def test_correlation_mismatch_is_stale() -> None:
    stale, reason = _call(verdict_correlation_id="proactive-OTHER")
    assert stale is True and reason == "stale_desire_id"


def test_user_reply_after_launch_is_stale() -> None:
    # exchange at 11:58 is after pending_since 11:55 -> reactive path already answered
    stale, reason = _call(last_exchange_at="2026-07-06T11:58:00+00:00")
    assert stale is True and reason == "user_replied"


def test_exchange_before_launch_is_not_stale() -> None:
    # exchange at 11:50 predates the launch -> not a during-think reply
    assert _call(last_exchange_at="2026-07-06T11:50:00+00:00")[0] is False


def test_pressure_satisfied_is_stale() -> None:
    stale, reason = _call(effective=0.5, threshold=1.0)
    assert stale is True and reason == "pressure_satisfied"


def test_deadline_elapsed_is_stale() -> None:
    # pending since 11:00, now 12:00 -> 60 min > 30 min deadline
    stale, reason = _call(pending_since="2026-07-06T11:00:00+00:00", deadline_min=30.0)
    assert stale is True and reason == "deadline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_invalidation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.invalidation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/invalidation.py
"""Async semantic invalidation of a proactive verdict (spec §7.3).

A proactive turn outlives the tick that launched it; when its verdict returns we
must decide whether it is still valid. Invalidation is **semantic, not
version-based** (Codex): a mere energy/mood tick must not drop a good verdict.
A verdict is stale only if the situation it was about has genuinely changed —
the desire was resolved, its correlation no longer matches, the user replied
after the launch (the reactive path already answered — applying would double-
message), the pressure was satisfied while thinking, or the deadline elapsed.
"""

from __future__ import annotations

from datetime import datetime


def _parse(ts: str | None) -> datetime | None:
    if ts is None:
        return None
    try:
        value = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    return value if value.tzinfo is not None else None


def is_verdict_stale(
    *,
    desire_status: str,
    pending_id: str | None,
    verdict_correlation_id: str,
    last_exchange_at: str | None,
    pending_since: str | None,
    effective: float,
    threshold: float,
    now: datetime,
    deadline_min: float = 30.0,
) -> tuple[bool, str]:
    """Return ``(stale, reason)`` for a returning proactive verdict."""
    if desire_status != "active":
        return True, "desire_resolved"
    if verdict_correlation_id != pending_id:
        return True, "stale_desire_id"

    launched = _parse(pending_since)
    exchanged = _parse(last_exchange_at)
    if launched is not None and exchanged is not None and exchanged > launched:
        return True, "user_replied"
    if effective < threshold:
        return True, "pressure_satisfied"
    if launched is not None and (now - launched).total_seconds() / 60.0 > deadline_min:
        return True, "deadline"
    return False, "fresh"
```
Re-export `is_verdict_stale` from `core/__init__.py`.

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new tests pass; every prior test (incl. `tests/sim/`) still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/invalidation.py core/__init__.py tests/test_invalidation.py
uv run ruff check core/invalidation.py core/__init__.py tests/test_invalidation.py
uv run mypy -p lifemodel
git add core/invalidation.py core/__init__.py tests/test_invalidation.py
git commit -m "feat(core): async semantic verdict invalidation — no double-message (spec §7.3)"
```

---

## Phase-D2 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Two commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §14 fail-closed send rate-limit (≤3/day, ≥60 min) → Task 1; §7.3 semantic verdict invalidation (resolved / correlation / user-replied / pressure-satisfied / deadline) → Task 2. **Deferred to Phase E:** applying the backstop before a real send + `record_send` on a confirmed send; applying `is_verdict_stale` before a `post_llm` verdict. **Folded into E / follow-ups:** tick-discipline dt-cap (our model already uses real elapsed for deprivation/recovery, so no spurious spike — the classic §17 concern is largely moot; a small per-tick rise cap can be added at cutover if a long service outage proves jarring), dithering, bus-pruning (`lm-w7c`), observability/debug rewrite.
- **Type consistency:** `allow_send(send_log, now, *, max_per_day, min_interval_min)` / `record_send(send_log, now, *, keep)` / `is_verdict_stale(*, …)` signatures self-consistent and match the interfaces block.
- **Totality:** both modules parse timestamps defensively (never raise on malformed input); backstop fails closed.
- **No placeholders:** every step ships real code + an exact command with expected output.

# Core Rebuild — Phase E3: Live Cutover (hooks→signals, egress→CoreLoop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the live path over to the new layered pipeline. The two Hermes hooks stop mutating state directly and instead **publish signals** (verdict / exchange) to the durable bus; the in-process proactive service stops calling the `core/decision.py` monolith and instead drives **`coreloop.tick()`**, consuming the `LaunchProactive` intents it surfaces to reach out (gated by the global backstop). After this phase the monolith `core/decision.py` is **dead** (no live caller) — it is physically deleted in Phase E4 (which also rewrites `/lifemodel debug`, the last remaining importer).

**Architecture (spec §7.1, §13, §14, model A):** Producers only enqueue (§7.1) — `post_llm_call` publishes a `verdict` signal (correlated by `pending_proactive_id`), `pre_gateway_dispatch` publishes an `exchange` signal; the aggregation consumes them on the next tick (E2 already handles staleness/record_send). The service builds the `LifeModel`, calls `coreloop.tick()` (which runs personality→neuron→aggregation→cognition and commits state via the single state-actor), and for a surfaced `LaunchProactive` applies the **backstop** (`allow_send`) before injecting the being's native proactive turn (prompt = `IMPULSE_LABEL_PREFIX + wake_packet_prompt`, so the hooks recognise it). A blocked/failed launch rolls the pending bookkeeping back. **Output-lint is advisory** here (the native turn self-delivers; the observer can only log a mechanical leak, not block it).

**Tech Stack:** Python 3.11; the plugin's Hermes seams (`ProactiveEgressPort`, hooks, bus). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **This phase MODIFIES the live path** — `egress_service.py` and `hooks.py` are edited here (they are no longer forbidden). **Still forbidden:** `core/decision.py` (leave it in place, now dead — E4 deletes it), `tick.py`, `heartbeat.py`, `impulse.py` (keep `IMPULSE_LABEL_PREFIX` importable; E4 relocates+deletes).
- **Producers only enqueue (spec §7.1):** the hooks publish signals; they do **not** load/mutate/commit `State` for the drive lifecycle. (The service still does the liveness stamp + launch rollback — a host reconciliation, not drive logic.)
- **Backstop fail-closed (spec §14):** a blocked launch holds the desire (`deferred`), never sends, never records.
- **Do NOT push/merge/touch `main`.** `tests/sim/` must stay green. `mypy -p lifemodel` strict.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `hooks.py` — publish `verdict`/`exchange` signals; advisory output-lint (Task 1).
- Modify `egress_service.py` — drive `coreloop.tick()`, consume `LaunchProactive`, apply backstop (Task 2).
- Rewrite `tests/test_hooks.py`, `tests/test_egress_service_tick.py` for the new behavior.

---

### Task 1: Hooks publish signals (verdict + exchange)

**Files:**
- Modify: `hooks.py`
- Test: `tests/test_hooks.py` (rewrite the behavior assertions)

**Behavior:** `make_post_llm_observer` — on a correlated proactive turn (`pending_proactive_id` set AND `user_message` starts with `IMPULSE_LABEL_PREFIX`) whose desire is still active, decide `FULFILL` (any text) vs `REJECT` (a `NO_REPLY`/`[SILENT]` marker), run `lint_proactive` on a FULFILL response and **log** a mechanical leak (advisory), then **publish** a `verdict` signal to `lm.bus` carrying the `correlation_id = pending_proactive_id`. `make_inbound_observer` — on a genuine (non-internal, non-own-impulse) inbound message, **publish** an `exchange` signal to `lm.bus`. Neither mutates `State`.

- [ ] **Step 1: Rewrite `tests/test_hooks.py` behavior assertions**

Replace the state-mutation assertions with signal-publish assertions. Use `build_lifemodel(base_dir=tmp_path)` so the hook and the assertion share the same `FileSignalBus`. Representative tests (keep any pure helper tests like `_is_no_reply`):

```python
# tests/test_hooks.py (behavior tests — adapt existing fixtures/imports)
from __future__ import annotations

from types import SimpleNamespace

from lifemodel.composition import build_lifemodel
from lifemodel.core.taxonomy import KIND_EXCHANGE, KIND_VERDICT, read_verdict, read_verdict_correlation
from lifemodel.hooks import make_inbound_observer, make_post_llm_observer
from lifemodel.impulse import IMPULSE_LABEL_PREFIX
from lifemodel.sim.aggregation import Verdict
from lifemodel.state.model import State


def _lm_with_pending(tmp_path, corr="p-1"):
    lm = build_lifemodel(base_dir=tmp_path)
    lm.state.commit(State(desire_status="active", pending_proactive_id=corr, pending_proactive_since="2026-07-06T00:00:00+00:00"))
    return lm


def test_post_llm_publishes_fulfill_verdict_signal(tmp_path) -> None:
    lm = _lm_with_pending(tmp_path, corr="p-1")
    obs = make_post_llm_observer(lm)
    obs(user_message=f"{IMPULSE_LABEL_PREFIX} внутри тяга...", assistant_response="Саш, привет, скучаю!")
    signals = lm.bus.peek_unprocessed()
    verdicts = [s for s in signals if s.kind == KIND_VERDICT]
    assert len(verdicts) == 1
    assert read_verdict(verdicts[0]) is Verdict.FULFILL
    assert read_verdict_correlation(verdicts[0]) == "p-1"


def test_post_llm_publishes_reject_on_silent(tmp_path) -> None:
    lm = _lm_with_pending(tmp_path)
    obs = make_post_llm_observer(lm)
    obs(user_message=f"{IMPULSE_LABEL_PREFIX} ...", assistant_response="[SILENT]")
    verdicts = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT]
    assert read_verdict(verdicts[0]) is Verdict.REJECT


def test_post_llm_ignores_uncorrelated_turn(tmp_path) -> None:
    lm = _lm_with_pending(tmp_path)
    obs = make_post_llm_observer(lm)
    obs(user_message="just a normal user message", assistant_response="hi")  # not our impulse
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT] == []


def test_post_llm_ignores_when_desire_not_active(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    lm.state.commit(State(desire_status="none", pending_proactive_id="p-1"))
    make_post_llm_observer(lm)(user_message=f"{IMPULSE_LABEL_PREFIX} x", assistant_response="hi")
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_VERDICT] == []


def test_inbound_publishes_exchange_signal(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    event = SimpleNamespace(text="привет!", internal=False, id="m-42")
    make_inbound_observer(lm)(event=event)
    exchanges = [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE]
    assert len(exchanges) == 1


def test_inbound_ignores_internal_and_own_impulse(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    make_inbound_observer(lm)(event=SimpleNamespace(text="x", internal=True, id="a"))
    make_inbound_observer(lm)(event=SimpleNamespace(text=f"{IMPULSE_LABEL_PREFIX} own", internal=False, id="b"))
    assert [s for s in lm.bus.peek_unprocessed() if s.kind == KIND_EXCHANGE] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hooks.py -q`
Expected: FAIL — the observers still call `apply_verdict`/`observe_exchange` (no signals published).

- [ ] **Step 3: Rewrite `hooks.py`**

Replace the imports and the two observer bodies. Remove `from .core.decision import apply_verdict, observe_exchange`. Add:
```python
from .core.taxonomy import exchange_signal, verdict_signal
from .output_lint import lint_proactive
```
Keep `IMPULSE_LABEL_PREFIX`, `Verdict`, `_NO_REPLY_MARKERS`, `_is_no_reply`, `_is_pending_proactive_turn`, `_is_own_impulse` unchanged. Replace the observers:
```python
def make_post_llm_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``post_llm_call`` handler that PUBLISHES a verdict signal (§7.1)."""

    def _observer(*, user_message: str = "", assistant_response: str = "", **_ignored: Any) -> None:
        state = lm.state.load()
        if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
            return
        if state.desire_status != "active":
            return
        verdict = Verdict.REJECT if _is_no_reply(assistant_response) else Verdict.FULFILL
        if verdict is Verdict.FULFILL:
            lint = lint_proactive(assistant_response)
            if not lint.ok:  # advisory only — the native turn already delivered
                lm.bus  # noqa: B018  (kept explicit that we do not block here)
                _log_lint(lm, lint.reason)
        now = lm.clock.now()
        lm.bus.publish(
            verdict_signal(
                origin_id=f"verdict-{state.pending_proactive_id}",
                verdict=verdict,
                timestamp=now.isoformat(),
                correlation_id=state.pending_proactive_id or "",
            )
        )

    return _observer


def make_inbound_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``pre_gateway_dispatch`` handler that PUBLISHES an exchange signal (§7.1)."""

    def _observer(*, event: Any = None, **_ignored: Any) -> None:
        if event is None or getattr(event, "internal", False):
            return
        text = getattr(event, "text", "") or ""
        if _is_own_impulse(text):
            return
        now = lm.clock.now()
        origin = getattr(event, "id", None) or getattr(event, "message_id", None) or f"exchange-{now.isoformat()}"
        lm.bus.publish(
            exchange_signal(origin_id=str(origin), actor="user", label="two_way", timestamp=now.isoformat())
        )

    return _observer
```
Add a small helper near the top (after the constants):
```python
def _log_lint(lm: LifeModel, reason: str) -> None:
    """Advisory: record that a delivered proactive message tripped the output-lint
    (mechanical timer / filler). Model A can't block the native send — this is
    observability feeding future prompt tuning (spec §13)."""
    try:
        from .log import get_logger

        get_logger("lifemodel.hooks").info("proactive_output_lint", reason=reason)
    except Exception:  # noqa: BLE001 - advisory logging must never break a turn
        pass
```
(Simplify the FULFILL lint block to just `if not lint.ok: _log_lint(lm, lint.reason)` — drop the stray `lm.bus` line above; it was illustrative. The final block should read:)
```python
        if verdict is Verdict.FULFILL:
            lint = lint_proactive(assistant_response)
            if not lint.ok:
                _log_lint(lm, lint.reason)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hooks.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format hooks.py tests/test_hooks.py
uv run ruff check hooks.py tests/test_hooks.py
uv run mypy -p lifemodel
git add hooks.py tests/test_hooks.py
git commit -m "feat(cutover): hooks publish verdict/exchange signals instead of mutating state (spec §7.1)"
```

---

### Task 2: Egress service drives the CoreLoop + backstop

**Files:**
- Modify: `egress_service.py`
- Test: `tests/test_egress_service_tick.py` (rewrite)

**Behavior:** `run_proactive_tick` now: (1) `report = lm.coreloop.tick()` (runs the pipeline; state committed by the state-actor); (2) for a surfaced `LaunchProactive`, load state, apply the **backstop** (`allow_send(state.proactive_send_log, now)`) — if blocked, hold the desire (`deferred`) and clear pending; else inject the native turn via `egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)` and, if not `DELIVERED`, roll pending back (keep `active` to retry); (3) always stamp `egress_service_alive_at` in one reconciliation commit. The `busy` parameter is retained but ignored (in-flight is a signal concern now).

- [ ] **Step 1: Rewrite `tests/test_egress_service_tick.py`**

```python
# tests/test_egress_service_tick.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import build_lifemodel
from lifemodel.domain.egress import ReachOutcome
from lifemodel.egress_service import run_proactive_tick
from lifemodel.impulse import IMPULSE_LABEL_PREFIX
from lifemodel.log import get_logger
from lifemodel.state.model import State

TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class FakeEgress:
    def __init__(self, outcome=ReachOutcome.DELIVERED) -> None:
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reach_out(self, target, impulse):
        self.calls.append((target, impulse))
        return self.outcome


class FixedClock:
    def __init__(self, m):
        self._m = m

    def now(self):
        return self._m


def _lm(tmp_path, state: State, now: datetime):
    lm = build_lifemodel(base_dir=tmp_path, clock=FixedClock(now))
    lm.state.commit(state)
    return lm


def test_active_desire_launches_native_turn(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="active", u=2.0, energy=1.0, pending_proactive_id=None, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert len(egress.calls) == 1
    _, impulse = egress.calls[0]
    assert impulse.startswith(IMPULSE_LABEL_PREFIX)  # correlation marker prepended
    assert lm.state.load().pending_proactive_id is not None  # a turn is in flight


def test_no_active_desire_does_not_reach_out(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="none", u=0.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert egress.calls == []


def test_backstop_blocks_when_cap_reached(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    log = ["2026-07-06T11:00:00+00:00", "2026-07-06T10:00:00+00:00", "2026-07-06T09:00:00+00:00"]  # 3 today
    state = State(desire_status="active", u=2.0, energy=1.0, proactive_send_log=log, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress()
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    assert egress.calls == []  # backstop blocked the send
    assert lm.state.load().desire_status == "deferred"  # held, not sent


def test_failed_launch_rolls_back_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="active", u=2.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    egress = FakeEgress(outcome=ReachOutcome.UNAVAILABLE)
    run_proactive_tick(lm, egress, TARGET, logger=get_logger("t"))
    final = lm.state.load()
    assert final.pending_proactive_id is None  # rolled back
    assert final.desire_status == "active"  # kept to retry (not rejected)


def test_liveness_is_stamped(tmp_path) -> None:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    state = State(desire_status="none", last_tick_at="2026-07-06T11:59:00+00:00")
    lm = _lm(tmp_path, state, now)
    run_proactive_tick(lm, FakeEgress(), TARGET, logger=get_logger("t"))
    assert lm.state.load().egress_service_alive_at == now.isoformat()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_egress_service_tick.py -q`
Expected: FAIL — `run_proactive_tick` still uses `decide_reachout`.

- [ ] **Step 3: Rewrite `run_proactive_tick` in `egress_service.py`**

Replace the imports: remove `from .core.decision import THETA, decide_reachout`, `from .domain.wake import WakePacket`, `from .impulse import compose_impulse`, and the unused `datetime`. Add `from .core.backstop import allow_send` and `from .impulse import IMPULSE_LABEL_PREFIX`. Replace `run_proactive_tick`:
```python
def run_proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
    busy: bool = False,  # retained for the loop's call shape; in-flight is a signal now
) -> ReachOutcome:
    """One in-process proactive tick — run the layered pipeline, launch on a
    surfaced desire, gated by the global backstop (spec §13/§14, model A)."""
    report = lm.coreloop.tick()  # pipeline runs + state committed by the state-actor
    now = lm.clock.now()

    outcome = ReachOutcome.SKIPPED_BUSY
    rollback_status: str | None = None
    if report.launches:
        state = lm.state.load()
        launch = report.launches[0]
        if not allow_send(state.proactive_send_log, now):
            rollback_status = "deferred"  # backstop: hold the desire, send nothing (spec §14)
            logger.info("proactive_backstop_blocked")
        else:
            outcome = egress.reach_out(target, IMPULSE_LABEL_PREFIX + launch.prompt)
            if outcome is not ReachOutcome.DELIVERED:
                rollback_status = "active"  # launch failed — keep active to retry
                logger.info("proactive_launch_failed", outcome=outcome.value)

    # one reconciliation commit: liveness stamp + optional pending rollback
    state = lm.state.load()
    state.egress_service_alive_at = now.isoformat()
    if rollback_status is not None:
        state.pending_proactive_id = None
        state.pending_proactive_since = None
        state.desire_status = rollback_status
    lm.state.commit(state)
    logger.info("proactive_tick", launches=len(report.launches), outcome=outcome.value)
    return outcome
```
Leave `proactive_service_loop` unchanged (it still calls `run_proactive_tick(build_lm(), egress, target, logger=logger, busy=busy)`; the `busy` computation stays but is now inert). Update the module docstring's first paragraph to describe driving `coreloop.tick()` (not `decide_reachout`) — keep it brief.

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new egress + hooks tests pass. NOTE: `tests/test_introspect.py` / `tests/test_debug.py` still import `core/decision.py` and pass (decision.py is unchanged, just no longer on the live path); they are rewritten in Phase E4. `tests/sim/` stays green.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format egress_service.py tests/test_egress_service_tick.py
uv run ruff check egress_service.py tests/test_egress_service_tick.py
uv run mypy -p lifemodel
git add egress_service.py tests/test_egress_service_tick.py
git commit -m "feat(cutover): egress drives coreloop.tick() + LaunchProactive + backstop (decision.py now dead)"
```

---

## Phase-E3 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Two commits on `core/rebuild`, one per task.
- [ ] `core/decision.py` unmodified (now dead — no live importer except introspect/debug, deleted in E4). `tick.py`, `heartbeat.py`, `impulse.py` unmodified.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §7.1 hooks only enqueue (publish signals) → Task 1; §13 model-A launch of the native turn via the wake-packet prompt + §14 backstop gate + rollback → Task 2; advisory output-lint (model A can't block the native send) → Task 1. **Deferred to E4:** delete `core/decision.py` + `test_decision.py`; rewrite `introspect.py`/`debug.py` + their tests onto the new model; relocate `IMPULSE_LABEL_PREFIX` and delete `impulse.py`/`compose_impulse` + dead code.
- **Correlation preserved:** the injected prompt keeps `IMPULSE_LABEL_PREFIX`, so `post_llm` correlates and `pre_gateway_dispatch` ignores the being's own nudge.
- **Single-writer nuance:** the drive lifecycle mutates only via the state-actor (inside `coreloop.tick()`); the service's reconciliation commit (liveness + launch rollback) is a host concern, documented, and never touches drive fields beyond the pending rollback.
- **No placeholders:** every step ships real code + an exact command with expected output.

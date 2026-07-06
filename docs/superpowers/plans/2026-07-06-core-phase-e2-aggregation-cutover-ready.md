# Core Rebuild — Phase E2: Aggregation Cutover-Ready (verdict staleness + record_send) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ContactAggregation` ready to receive **real, correlated verdicts** from the live cutover (Phase E3): drop a verdict that is **stale** (async semantic invalidation, §7.3 — desire resolved / correlation mismatch / user replied during the turn / pressure satisfied / deadline), clear the pending-turn bookkeeping when a verdict resolves the desire, and **record each real send** into `proactive_send_log` so the global backstop can count it. Still **no live-path change** — this is internal aggregation surgery, exercised with fakes.

**Architecture (spec §7.3, §12, §14):** The verdict handling now gates each verdict signal through `core.invalidation.is_verdict_stale` using the verdict's `correlation_id` (Phase E1) matched against `pending_proactive_id`, the effective pressure at verdict time, and the exchange/deadline clocks. A fresh `FULFILL` (a real send happened) starts `ActionPending`, stamps `last_contact_at`, **appends the send to `proactive_send_log`** (`core.backstop.record_send`), and clears the pending id/since; a fresh `REJECT` records the decline backoff and clears pending; a stale verdict is dropped (no-op). Because the exchange block runs first and can clear the desire, **exchange dominates a same-tick verdict** (the verdict then sees `desire_status != active` → stale).

**Tech Stack:** Python 3.11 stdlib-only; reuses `is_verdict_stale` (D2), `record_send` (D2), `read_verdict_correlation` (E1). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.**
- **Semantic invalidation only (§7.3):** a verdict is dropped only on a genuine semantic condition — never on an energy/mood tick.
- **`record_send` only on a fresh FULFILL** (a real proactive message was produced). `REJECT`/`[SILENT]`/stale never touch `proactive_send_log`.
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **`mypy -p lifemodel` strict.**
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `core/aggregation.py` — verdict-staleness gate + clear pending (Task 1); `record_send` on FULFILL (Task 2).
- Modify `tests/test_aggregation.py` — update the verdict tests to the correlated model; add staleness + record_send tests.

**Interfaces (unchanged public surface):** `ContactAggregation(*, params, theta, beta, u_max, i0, grace_min, halflife_min, verdict_deadline_min=30.0, id="contact-aggregation")` — one new optional constructor param.

---

### Task 1: Verdict-staleness gate + clear pending (async invalidation)

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py`

**Interfaces:**
- Consumes: `is_verdict_stale` (`core/invalidation.py`), `read_verdict_correlation` (`core/taxonomy.py`).

**Behavior:** replace `ContactAggregation.step` with the version below — it computes the effective pressure at verdict time, gates each verdict through `is_verdict_stale`, applies a fresh verdict (clearing pending on resolution), and keeps everything else identical. Add the `verdict_deadline_min` constructor param.

- [ ] **Step 1: Update the verdict tests to the correlated model, add staleness tests**

In `tests/test_aggregation.py`: the existing verdict tests inject a verdict with no pending/correlation — under the new staleness gate that verdict would be *stale* (correlation mismatch) and dropped. **Replace** `test_fulfill_starts_action_pending_and_does_not_satiate`, `test_reject_records_growing_backoff`, `test_reject_then_backoff_blocks_immediate_rewake`, and `test_defer_holds_desire_and_keeps_pressure` with correlated versions, and add staleness tests:

```python
CORR = "proactive-2026-07-06T03:55:00+00:00"


def _live_pending_state(**over) -> State:
    """A state with a proactive turn in flight, matching CORR."""
    base = dict(
        u=1.5, desire_status="active",
        pending_proactive_id=CORR, pending_proactive_since="2026-07-06T03:55:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    base.update(over)
    return State(**base)


def test_fulfill_starts_action_pending_and_clears_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(duration_over_theta=99.0)
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["action_pending_since"] == now.isoformat()  # send -> ActionPending
    assert "u" not in changes  # not satiated (send != contact)
    assert changes["last_contact_at"] == now.isoformat()
    assert changes["pending_proactive_id"] is None  # turn resolved
    assert changes["pending_proactive_since"] is None


def test_reject_records_backoff_and_clears_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(decline_count=1)
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["decline_count"] == 2
    assert changes["declined_at"] == now.isoformat()
    assert changes["pending_proactive_id"] is None


def test_defer_holds_desire_keeps_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.DEFER, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "deferred"
    assert "u" not in changes


def test_stale_verdict_wrong_correlation_is_dropped(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None, correlation_id="proactive-OTHER")
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"  # verdict dropped — desire untouched
    assert changes["action_pending_since"] is None


def test_exchange_dominates_same_tick_verdict(tmp_path) -> None:
    # a real reply this tick clears the desire; the (now-stale) fulfill is ignored
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # exchange cleared it
    assert changes["action_pending_since"] is None  # fulfill was dropped (desire resolved)
    assert changes["last_exchange_at"] == now.isoformat()
```

(Keep `test_fulfill_resets_duration_...` removed already in C1; `test_reject_does_not_set_action_pending` from C1 stays but update it to `_live_pending_state()` + `correlation_id=CORR` so its REJECT is not stale.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — staleness gate not implemented; `verdict_deadline_min` param missing; pending not cleared.

- [ ] **Step 3: Replace `ContactAggregation.step` (and extend `__init__`)**

Add imports at the top of `core/aggregation.py`:
```python
from .invalidation import is_verdict_stale
from .taxonomy import (
    KIND_EXCHANGE,
    KIND_VERDICT,
    contact_value,
    is_in_flight,
    read_exchange,
    read_verdict,
    read_verdict_correlation,
)
```
Add `verdict_deadline_min: float = 30.0` to `__init__` (keyword-only, stored as `self._verdict_deadline_min`).

Replace the entire `step` method body with:
```python
    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        now = ctx.now
        u_now = contact_value(ctx.signals, default=state.u)

        # working copies of the policy fields (threaded like decision.py)
        last_exchange_at = state.last_exchange_at
        declined_at = state.declined_at
        decline_count = state.decline_count
        agg = Aggregator(status=DesireStatus(state.desire_status))
        last_contact_at = state.last_contact_at
        action_pending_since = state.action_pending_since
        pending_id = state.pending_proactive_id
        pending_since = state.pending_proactive_since
        send_log = state.proactive_send_log

        # effective pressure at verdict time (from persisted inhibition) — staleness input
        effective_now = effective_pressure(
            u_now,
            inhibition_at(
                state.action_pending_since, now,
                i0=self._i0, grace_min=self._grace_min, halflife_min=self._halflife_min,
            ),
        )

        # 1) real exchanges reset clocks, clear the desire and ActionPending (before verdict/wake)
        for sig in ctx.signals:
            if sig.kind == KIND_EXCHANGE:
                actor, _label = read_exchange(sig)
                if actor != "proactive_internal":
                    last_exchange_at = now.isoformat()
                    declined_at = None
                    decline_count = 0
                    action_pending_since = None
                    agg.on_exchange()

        # 2) a verdict resolves the woken desire — dropped if stale (async invalidation §7.3)
        for sig in ctx.signals:
            if sig.kind != KIND_VERDICT:
                continue
            stale, _reason = is_verdict_stale(
                desire_status=agg.status.value,
                pending_id=pending_id,
                verdict_correlation_id=read_verdict_correlation(sig),
                last_exchange_at=last_exchange_at,
                pending_since=pending_since,
                effective=effective_now,
                threshold=self._theta,
                now=now,
                deadline_min=self._verdict_deadline_min,
            )
            if stale:
                continue
            verdict = read_verdict(sig)
            agg.apply_verdict(verdict)
            if verdict is Verdict.FULFILL:
                action_pending_since = now.isoformat()  # send -> inhibition starts
                last_contact_at = now.isoformat()
                pending_id = None
                pending_since = None
            elif verdict is Verdict.REJECT:
                declined_at = now.isoformat()
                decline_count += 1
                pending_id = None
                pending_since = None

        # duration on latent u
        dt = minutes_between(state.last_tick_at, now)
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0

        # effective pressure for the wake gate (post-verdict inhibition)
        effective = effective_pressure(
            u_now,
            inhibition_at(
                action_pending_since, now,
                i0=self._i0, grace_min=self._grace_min, halflife_min=self._halflife_min,
            ),
        )

        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=self._params)
        if outcome.is_urge:
            agg.on_urge()

        changes: dict[str, object] = {
            "desire_status": agg.status.value,
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
            "action_pending_since": action_pending_since,
            "pending_proactive_id": pending_id,
            "pending_proactive_since": pending_since,
            "proactive_send_log": send_log,
        }
        return [UpdateState(changes)]
```
(Remove the now-unused local `inhibition`/`effective` earlier duplicate lines — the method above is the complete replacement.)

- [ ] **Step 4: Run the aggregation suite to verify green**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: PASS — verdict tests pass under the correlated model; staleness + exchange-dominance tests pass; wake/exchange tests (no verdict) unchanged.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): aggregation drops stale verdicts + clears pending (async invalidation §7.3)"
```

---

### Task 2: `record_send` on a fresh FULFILL (backstop counter)

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py`

**Interfaces:**
- Consumes: `record_send` (`core/backstop.py`).

**Behavior:** a fresh `FULFILL` (a real proactive message) appends `now` to `proactive_send_log` via `record_send`, so the global backstop (Phase E3) can enforce the daily/interval cap. Nothing else records a send.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_fulfill_records_a_send(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(proactive_send_log=["2026-07-06T02:00:00+00:00"])
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    log = changes["proactive_send_log"]
    assert log[-1] == now.isoformat()  # this send recorded
    assert len(log) == 2  # appended to the prior one


def test_reject_does_not_record_a_send(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(proactive_send_log=["2026-07-06T02:00:00+00:00"])
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None, correlation_id=CORR)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["proactive_send_log"] == ["2026-07-06T02:00:00+00:00"]  # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — FULFILL does not yet append to `proactive_send_log`.

- [ ] **Step 3: Implement**

Add the import `from .backstop import record_send` to `core/aggregation.py`. In the FULFILL branch of the verdict loop, add `send_log = record_send(send_log, now)`:
```python
            if verdict is Verdict.FULFILL:
                action_pending_since = now.isoformat()
                last_contact_at = now.isoformat()
                send_log = record_send(send_log, now)  # backstop counter (spec §14)
                pending_id = None
                pending_since = None
```
(`send_log` is already threaded into `changes["proactive_send_log"]` from Task 1.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new record_send tests pass; every prior test (incl. `tests/sim/`) still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): aggregation records a send on FULFILL for the backstop counter (spec §14)"
```

---

## Phase-E2 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Two commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §7.3 async invalidation applied at the verdict site (drop stale) → Task 1; §12 exchange-dominates-verdict (exchange runs first → verdict sees resolved desire → stale) → Task 1 test; pending cleared on verdict resolution → Task 1; §14 send recorded for the backstop → Task 2. **Deferred to E3 (live cutover):** hooks publish the verdict/exchange signals; egress drives `coreloop.tick()` and applies the backstop `allow_send` before reach-out; delete `decision.py`; `/lifemodel debug` rewrite.
- **Ordering invariant:** effective-for-staleness uses persisted inhibition (verdict-time), effective-for-wake uses post-verdict inhibition — a FULFILL this tick suppresses a same-tick re-wake.
- **Idempotence:** the changes dict always writes `pending_*`/`proactive_send_log`; cognition runs *after* aggregation in the pipeline, so its pending-stamp wins the merge (a launch this tick is not clobbered).
- **No placeholders:** every step ships real code + an exact command with expected output.

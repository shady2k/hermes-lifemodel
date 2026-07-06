# Core Rebuild — Phase B2: AGGREGATION Layer + Desire Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the AGGREGATION layer — `ContactAggregation` — a CoreLoop component that runs after the `ContactNeuron` and owns the whole contact-**desire lifecycle**: it reads the neuron's transient `contact` value plus durable `exchange`/`verdict`/`in_flight` inputs, evaluates the certified wake gates, and drives the desire state machine (birth → dedup → verdict resolution → exchange clear). This faithfully ports the proven logic of `core/decision.py` onto the layer boundary, **reusing** the certified `sim` primitives. Still **no live cutover**: `core/decision.py` and the cron/egress path are untouched; the new layer runs only under `CoreLoop.tick()`.

**Architecture (spec §7, §12):** The layer is **stateless** — every tick it reconstructs the certified `sim.aggregation.Aggregator` from `state.desire_status` and the `sim.wake` gate inputs from the state's clocks, exactly as `core/decision.py` does today. Within one `step`, events are applied in the order **exchange → verdict → wake**, threaded through local variables (not re-reading state), then emitted as **one** `UpdateState` intent. The neuron remains the sole writer of `u` on rise/exchange-satiation; the aggregation writes `u` **only** on a `FULFILL` verdict (the delivery satiation the certified model requires — spec's send≠contact refinement is Phase C).

**Deliberately deferred:** salience (§5) and `DEFER→release` on availability → **Phase D** (their consumers — act-gate, UserModel — live there). `pending`/stale-recovery, `WakeCognition` intent, live cutover → **Phase D/E**. `latent`/`effective` pressure + ActionPending → **Phase C**.

**Tech Stack:** Python 3.11 stdlib-only core; the certified `sim.aggregation.Aggregator`, `sim.wake.{evaluate_wake, LaneState, GateParams, WakeOutcome}`, `sim.drive.Drive`, `sim.quality.quality_of` (reused, never reimplemented). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout:** tests import `from lifemodel.x import Y`; package-internal code uses relative imports.
- **Core imports no Hermes.** Pure stdlib + intra-package.
- **Reuse the certified sim, do not reimplement it.** Gate math is `sim.wake.evaluate_wake`; lifecycle is `sim.aggregation.Aggregator`; satiation is `sim.drive.Drive.satiate`. The gate constants match `core/decision.py`: `GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)`, `THETA=1.0`, `BETA=1.0`, `U_MAX=100.0`.
- **Time-unit bridge (load-bearing — copy from `core/decision.py`):** every gate quantity is minutes **relative to `now`**, so `now=0.0` and any earlier instant is negative (`-minutes_between(instant, now)`). A `None` clock stays `None` (gate skipped), never `-0.0`.
- **Single writer per field (spec §6):** aggregation writes `desire_status`, `duration_over_theta`, `last_exchange_at`, `declined_at`, `decline_count`, `last_contact_at`, and `u` **only on FULFILL**. It must **not** write `u` on a normal tick (the neuron owns rise/exchange-satiation), `tick_count`, or `last_tick_at` (the CoreLoop owns those).
- **Registration order matters:** `ContactAggregation` must be registered **after** `ContactNeuron` so it sees the neuron's transient `contact` signal in `ctx.signals` this tick.
- **`mypy -p lifemodel` strict.** Full annotations.
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. All existing tests must still pass, and the certified `tests/sim/` scenarios must stay green (they exercise the sim directly — unchanged).
- **Branch:** `core/rebuild` (already checked out). One commit per task.

## File Structure

- Modify `core/taxonomy.py` — add `verdict` + `in_flight` kinds, builders, readers (Task 1).
- Create `core/aggregation.py` — `ContactAggregation` component (Tasks 2–4, built up incrementally).
- Modify `composition.py` — register `ContactAggregation` after the neuron (Task 5).
- Modify `core/__init__.py` — re-export new public names.
- Tests: extend `tests/test_taxonomy.py`; create `tests/test_aggregation.py`; extend `tests/test_composition.py`.

**Interfaces produced (Phases C–E consume these):**
- `core/taxonomy.py`: `KIND_VERDICT`, `KIND_IN_FLIGHT`; `verdict_signal(*, origin_id, verdict, timestamp) -> Signal`; `read_verdict(signal) -> Verdict`; `in_flight_signal(*, origin_id, value, timestamp) -> Signal`; `is_in_flight(signals: Iterable[Signal]) -> bool`; `contact_value(signals, *, default) -> float`.
- `core/aggregation.py`: `ContactAggregation(*, params: GateParams, theta: float, beta: float, u_max: float, id="contact-aggregation")` implementing `Component`.

---

### Task 1: Taxonomy — verdict + in_flight signal kinds

**Files:**
- Modify: `core/taxonomy.py`
- Modify: `core/__init__.py`
- Test: `tests/test_taxonomy.py` (extend)

**Interfaces:**
- Consumes: `Signal`; `sim.aggregation.Verdict`.
- Produces: `KIND_VERDICT`, `KIND_IN_FLIGHT`, `verdict_signal`, `read_verdict`, `in_flight_signal`, `is_in_flight`, `contact_value`.

**Behavior:** `verdict` is a durable input from cognition (Phase D) / tests carrying one of `fulfill|defer|reject`. `in_flight` is a durable input (the gateway is mid-turn) that gates a wake. `contact_value` reads the transient `contact` signal's `value` (falling back to a default when the neuron didn't run).

- [ ] **Step 1: Write the failing test (append to `tests/test_taxonomy.py`)**

```python
from lifemodel.core.taxonomy import (
    KIND_IN_FLIGHT,
    KIND_VERDICT,
    contact_signal as _contact_signal,
    contact_value,
    in_flight_signal,
    is_in_flight,
    read_verdict,
    verdict_signal,
)
from lifemodel.sim.aggregation import Verdict


def test_verdict_signal_roundtrips() -> None:
    sig = verdict_signal(origin_id="v-1", verdict=Verdict.FULFILL, timestamp=None)
    assert sig.kind == KIND_VERDICT
    assert read_verdict(sig) is Verdict.FULFILL


def test_read_verdict_rejects_bad_value() -> None:
    from lifemodel.domain.signal import Signal

    with pytest.raises(ValueError):
        read_verdict(Signal(origin_id="v-2", kind=KIND_VERDICT, payload={"verdict": "nope"}))


def test_in_flight_signal_and_reader() -> None:
    busy = in_flight_signal(origin_id="f-1", value=True, timestamp=None)
    idle = in_flight_signal(origin_id="f-2", value=False, timestamp=None)
    assert busy.kind == KIND_IN_FLIGHT
    assert is_in_flight([idle, busy]) is True
    assert is_in_flight([idle]) is False
    assert is_in_flight([]) is False


def test_contact_value_reads_transient_signal_or_default() -> None:
    c = _contact_signal(origin_id="c-9", value=2.5, delta=0.1, timestamp=None)
    assert contact_value([c], default=0.0) == 2.5
    assert contact_value([], default=1.23) == 1.23
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: FAIL — `ImportError` for the new names.

- [ ] **Step 3: Extend `core/taxonomy.py`**

Add near the top (after the existing kind constants):
```python
from collections.abc import Iterable

from ..sim.aggregation import Verdict

KIND_VERDICT = "verdict"
KIND_IN_FLIGHT = "in_flight"

_VERDICTS: dict[str, Verdict] = {v.value: v for v in Verdict}
```
Add these functions:
```python
def verdict_signal(*, origin_id: str, verdict: Verdict, timestamp: str | None) -> Signal:
    """Build a durable verdict-input signal (cognition's decision on a desire)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_VERDICT,
        payload={"verdict": verdict.value},
        timestamp=timestamp,
    )


def read_verdict(signal: Signal) -> Verdict:
    """Validate and extract the ``Verdict`` from a verdict signal."""
    if signal.kind != KIND_VERDICT:
        raise ValueError(f"not a verdict signal: kind={signal.kind!r}")
    raw = signal.payload.get("verdict")
    if raw not in _VERDICTS:
        raise ValueError(f"invalid verdict payload: {signal.payload!r}")
    return _VERDICTS[raw]


def in_flight_signal(*, origin_id: str, value: bool, timestamp: str | None) -> Signal:
    """Build a durable in-flight input (a turn is running/queued)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_IN_FLIGHT,
        payload={"value": bool(value)},
        timestamp=timestamp,
    )


def is_in_flight(signals: Iterable[Signal]) -> bool:
    """True if any in-flight signal in the batch reports a running turn."""
    return any(
        s.kind == KIND_IN_FLIGHT and bool(s.payload.get("value")) for s in signals
    )


def contact_value(signals: Iterable[Signal], *, default: float) -> float:
    """The most recent transient contact value in the batch, or ``default``."""
    value = default
    for s in signals:
        if s.kind == KIND_CONTACT:
            raw = s.payload.get("value", default)
            value = float(raw) if isinstance(raw, int | float) else default
    return value
```
Re-export `KIND_VERDICT, KIND_IN_FLIGHT, verdict_signal, read_verdict, in_flight_signal, is_in_flight, contact_value` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run ruff check core/taxonomy.py core/__init__.py tests/test_taxonomy.py
uv run mypy -p lifemodel
git add core/taxonomy.py core/__init__.py tests/test_taxonomy.py
git commit -m "feat(core): taxonomy — verdict + in_flight kinds, contact_value reader (spec §4)"
```

---

### Task 2: ContactAggregation — the wake path

**Files:**
- Create: `core/aggregation.py`
- Modify: `core/__init__.py`
- Test: `tests/test_aggregation.py`

**Interfaces:**
- Consumes: `sim.wake.{evaluate_wake, LaneState, GateParams, WakeOutcome}`, `sim.aggregation.{Aggregator, DesireStatus}`, `taxonomy.{contact_value, is_in_flight}`, `timeutil.minutes_between`, `TickContext`, `UpdateState`.
- Produces: `ContactAggregation`.

**Behavior (this task = the wake decision only; Tasks 3–4 add exchange/verdict):**
1. `u_now = contact_value(ctx.signals, default=state.u)`.
2. `dt = minutes_between(state.last_tick_at, now)`; `duration = (state.duration_over_theta + dt) if u_now >= theta else 0.0`.
3. Build `LaneState` in minutes-relative-to-`now` from state's `last_exchange_at`/`declined_at`/`decline_count` and `busy = is_in_flight(ctx.signals)`.
4. `outcome = evaluate_wake(u=u_now, now=0.0, state=lane, params=params)`. Reconstruct `agg = Aggregator(status=DesireStatus(state.desire_status))`; if `outcome.is_urge`, `agg.on_urge()` (NONE→ACTIVE, or dedup).
5. Return one `UpdateState({"desire_status": agg.status.value, "duration_over_theta": duration})`.

This mirrors `core/decision.py::decide_reachout` minus the drive rise (the neuron does that) and minus stale-pending (Phase E).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aggregation.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.component import TickContext
from lifemodel.core.intents import UpdateState
from lifemodel.core.taxonomy import contact_signal, in_flight_signal
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State

PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)


def _agg() -> ContactAggregation:
    return ContactAggregation(params=PARAMS, theta=1.0, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, tmp_path) -> TickContext:
    return TickContext(state=state, now=now, bus=FileSignalBus(tmp_path), signals=tuple(signals))


def _changes(intents) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def test_urge_over_threshold_creates_active_desire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"


def test_below_threshold_stays_none(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=0.5, delta=0.0, timestamp=None)  # < theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_second_urge_is_deduped_no_refire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "active"  # still one desire — dedup


def test_silence_window_suppresses_wake(tmp_path) -> None:
    # exchange 5 min ago (< w=15) → SILENCE_WINDOW, no wake even with high u
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0, desire_status="none",
        last_exchange_at="2026-07-06T03:55:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_in_flight_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    busy = in_flight_signal(origin_id="f1", value=True, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, busy], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"


def test_decline_backoff_suppresses_then_allows(tmp_path) -> None:
    # declined 10 min ago, decline_count=1 → backoff r0=30 min active → no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0, desire_status="none", decline_count=1,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # inside backoff


def test_duration_over_theta_accumulates(tmp_path) -> None:
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5 min
    state = State(
        u=2.0, desire_status="active", duration_over_theta=10.0,
        last_tick_at="2026-07-06T00:00:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)  # >= theta
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert abs(changes["duration_over_theta"] - 15.0) < 1e-9


def test_aggregation_does_not_write_u_on_normal_tick(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="none", last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert "u" not in changes  # neuron owns u; aggregation only writes it on FULFILL (Task 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.aggregation'`.

- [ ] **Step 3: Create `core/aggregation.py` (wake path only)**

```python
# core/aggregation.py
"""ContactAggregation — the AGGREGATION layer for the contact desire (spec §7, §12).

Stateless: every tick it reconstructs the certified ``sim`` primitives from the
persisted state and drives the desire lifecycle. It reads the neuron's transient
``contact`` value plus durable ``exchange``/``verdict``/``in_flight`` inputs,
applies them in the order exchange → verdict → wake (threaded through locals,
like ``core/decision.py``'s functions), and emits one ``UpdateState``.

The neuron owns ``u`` on rise/exchange-satiation; this layer writes ``u`` only on
a ``FULFILL`` verdict (the certified model's delivery satiation). This is the port
of ``core/decision.py`` onto the layer boundary — the wake/lifecycle math is the
reused ``sim`` code, never reimplemented here.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.aggregation import Aggregator, DesireStatus
from ..sim.wake import GateParams, LaneState, evaluate_wake
from .component import TickContext
from .intents import Intent, UpdateState
from .taxonomy import contact_value, is_in_flight
from .timeutil import minutes_between


class ContactAggregation:
    """Owns the contact-desire lifecycle (one desire per lane)."""

    def __init__(
        self,
        *,
        params: GateParams,
        theta: float,
        beta: float,
        u_max: float,
        id: str = "contact-aggregation",
    ) -> None:
        self.id = id
        self._params = params
        self._theta = theta
        self._beta = beta
        self._u_max = u_max

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        now = ctx.now
        u_now = contact_value(ctx.signals, default=state.u)

        # working copies of the policy fields (threaded like decision.py)
        last_exchange_at = state.last_exchange_at
        declined_at = state.declined_at
        decline_count = state.decline_count
        agg = Aggregator(status=DesireStatus(state.desire_status))

        # duration-over-threshold accumulates on the current (risen) u
        dt = minutes_between(state.last_tick_at, now)
        duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0

        # wake gates — every quantity as minutes relative to now (now = 0.0)
        exch_min = -minutes_between(last_exchange_at, now) if last_exchange_at is not None else None
        decl_min = -minutes_between(declined_at, now) if declined_at is not None else None
        lane = LaneState(
            last_exchange_at=exch_min,
            in_flight=is_in_flight(ctx.signals),
            declined_at=decl_min,
            decline_count=decline_count,
        )
        outcome = evaluate_wake(u=u_now, now=0.0, state=lane, params=self._params)
        if outcome.is_urge:
            agg.on_urge()

        return [
            UpdateState(
                {
                    "desire_status": agg.status.value,
                    "duration_over_theta": duration,
                }
            )
        ]
```

Re-export `ContactAggregation` from `core/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py core/__init__.py tests/test_aggregation.py
uv run ruff check core/aggregation.py core/__init__.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py core/__init__.py tests/test_aggregation.py
git commit -m "feat(core): ContactAggregation wake path — gates + desire birth via sim (spec §12)"
```

---

### Task 3: ContactAggregation — exchange resolution

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py` (extend)

**Interfaces:**
- Consumes: `taxonomy.{KIND_EXCHANGE, read_exchange}` (Phase B1).
- Produces: exchange handling in `ContactAggregation.step`.

**Behavior (spec §12; port of `core/decision.py::observe_exchange`):** a real (non-`proactive_internal`) exchange this tick clears the desire and resets the policy clocks **before** the wake evaluation, so the fresh silence window suppresses any wake. The layer sets `last_exchange_at = now`, `declined_at = None`, `decline_count = 0`, and `agg.on_exchange()` (status → NONE). It does **not** satiate `u` — the neuron already did (B1). An `proactive_internal` exchange is ignored (the being's own nudge is not contact).

- [ ] **Step 1: Write the failing tests (append)**

```python
from lifemodel.core.taxonomy import exchange_signal


def test_exchange_clears_desire_and_resets_clocks(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0, desire_status="active", decline_count=2,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # desire cleared
    assert changes["decline_count"] == 0
    assert changes["declined_at"] is None
    assert changes["last_exchange_at"] == now.isoformat()


def test_exchange_this_tick_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # fresh exchange → SILENCE_WINDOW


def test_internal_impulse_is_not_an_exchange(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    own = exchange_signal(origin_id="e1", actor="proactive_internal", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, own], tmp_path=tmp_path)))
    assert changes["last_exchange_at"] is None  # own nudge did not reset the clock
    assert changes["desire_status"] == "active"  # desire not cleared by own nudge
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — the three new tests fail (no exchange handling yet).

- [ ] **Step 3: Add exchange handling to `core/aggregation.py`**

Add the import `from .taxonomy import KIND_EXCHANGE, contact_value, is_in_flight, read_exchange` (extend the existing taxonomy import). Insert this block into `step`, **immediately after** `agg = Aggregator(...)` and **before** the duration/gate computation:

```python
        # 1) real exchanges reset the policy clocks and clear the desire (before wake)
        for sig in ctx.signals:
            if sig.kind == KIND_EXCHANGE:
                actor, _label = read_exchange(sig)
                if actor != "proactive_internal":
                    last_exchange_at = now.isoformat()
                    declined_at = None
                    decline_count = 0
                    agg.on_exchange()
```

Then extend the returned `UpdateState` changes to include the (possibly reset) clocks:
```python
        return [
            UpdateState(
                {
                    "desire_status": agg.status.value,
                    "duration_over_theta": duration,
                    "last_exchange_at": last_exchange_at,
                    "declined_at": declined_at,
                    "decline_count": decline_count,
                }
            )
        ]
```

- [ ] **Step 4: Run the aggregation suite to verify green**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): ContactAggregation — exchange resolution clears desire + resets clocks (spec §12)"
```

---

### Task 4: ContactAggregation — verdict resolution (fulfill / defer / reject)

**Files:**
- Modify: `core/aggregation.py`
- Test: `tests/test_aggregation.py` (extend)

**Interfaces:**
- Consumes: `taxonomy.{KIND_VERDICT, read_verdict}` (Task 1), `sim.aggregation.Verdict`, `sim.drive.Drive`.
- Produces: verdict handling in `ContactAggregation.step`.

**Behavior (spec §12; port of `core/decision.py::apply_verdict`):** a verdict this tick resolves the woken desire, applied **after** exchange and **before** wake:
- `agg.apply_verdict(v)` advances the status (`DEFER`→DEFERRED, else→NONE).
- **FULFILL** — the delivery satiates the drive fully: `u_out = max(0, u_now − beta·1.0)` (via `Drive.satiate`), `duration → 0`, `last_exchange_at = now`, `last_contact_at = now`. (This is the one place aggregation writes `u`.)
- **REJECT** — `declined_at = now`, `decline_count += 1` (feeds the growing backoff on later ticks).
- **DEFER** — status only; pressure is **not** reset (invariant: defer holds the intention).

The `§12` invariants this locks: *reject → growing backoff*, *only a real delivery/exchange satiates*, *defer does not drop pressure*, *own impulse ≠ contact* (Task 3), *wake ≠ send* (the layer only sets status/emits no delivery — sending is Phase D/E).

- [ ] **Step 1: Write the failing tests (append)**

```python
from lifemodel.core.taxonomy import verdict_signal
from lifemodel.sim.aggregation import Verdict


def test_fulfill_satiates_u_and_stamps_contact(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.5, desire_status="active", duration_over_theta=99.0,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["u"] == 0.5  # 1.5 - beta*1.0
    assert changes["duration_over_theta"] == 0.0
    assert changes["last_contact_at"] == now.isoformat()
    assert changes["last_exchange_at"] == now.isoformat()


def test_reject_records_growing_backoff(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.5, desire_status="active", decline_count=1,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"
    assert changes["decline_count"] == 2
    assert changes["declined_at"] == now.isoformat()
    assert "u" not in changes  # reject does not satiate


def test_defer_holds_desire_and_keeps_pressure(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.DEFER, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "deferred"
    assert "u" not in changes  # pressure not dropped


def test_fulfill_resets_duration_even_when_u_stays_high(tmp_path) -> None:
    # FULFILL resets duration_over_theta unconditionally (matching decision.py),
    # NOT merely because the satiated u fell below theta. Here u=5.0 -> satiate
    # to 4.0 (still >= theta) but duration must still reset to 0.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=5.0, desire_status="active", duration_over_theta=500.0,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_signal(origin_id="c1", value=5.0, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.FULFILL, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["u"] == 4.0
    assert changes["duration_over_theta"] == 0.0  # reset regardless of u


def test_reject_then_backoff_blocks_immediate_rewake(tmp_path) -> None:
    # after a REJECT this tick, the fresh declined_at must veto a wake in the same tick
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=5.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_signal(origin_id="c1", value=5.0, delta=0.0, timestamp=None)
    v = verdict_signal(origin_id="v1", verdict=Verdict.REJECT, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, v], tmp_path=tmp_path)))
    assert changes["desire_status"] == "none"  # rejected + backoff vetoes re-wake
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation.py -q`
Expected: FAIL — the four new tests fail (no verdict handling yet).

- [ ] **Step 3: Add verdict handling to `core/aggregation.py`**

Add imports: `from ..sim.aggregation import Aggregator, DesireStatus, Verdict` (extend) and `from ..sim.drive import Drive`; extend the taxonomy import with `KIND_VERDICT, read_verdict`. In `step`, add local accumulators near the top (after `agg = ...`): `last_contact_at = state.last_contact_at`, `u_out: float | None = None`, and `fulfilled = False`. Insert this block **after** the exchange block and **before** the duration computation:

```python
        # 2) a verdict resolves the woken desire (after exchange, before wake)
        for sig in ctx.signals:
            if sig.kind == KIND_VERDICT:
                verdict = read_verdict(sig)
                agg.apply_verdict(verdict)
                if verdict is Verdict.FULFILL:
                    drive = Drive(alpha=0.0, beta=self._beta, u_max=self._u_max, u=u_now)
                    drive.satiate(q=1.0)
                    u_now = drive.u
                    u_out = drive.u
                    fulfilled = True
                    last_exchange_at = now.isoformat()
                    last_contact_at = now.isoformat()
                elif verdict is Verdict.REJECT:
                    declined_at = now.isoformat()
                    decline_count += 1
```

Then change the duration line (from Task 2) so a FULFILL resets it **unconditionally** (matching `core/decision.py::apply_verdict`, which sets `duration_over_theta = 0.0` on FULFILL regardless of the resulting `u`):
```python
        dt = minutes_between(state.last_tick_at, now)
        if fulfilled:
            duration = 0.0
        else:
            duration = state.duration_over_theta + dt if u_now >= self._theta else 0.0
```

Finally extend the returned changes to include `last_contact_at` and, only when set, `u`:

```python
        changes: dict[str, object] = {
            "desire_status": agg.status.value,
            "duration_over_theta": duration,
            "last_exchange_at": last_exchange_at,
            "declined_at": declined_at,
            "decline_count": decline_count,
            "last_contact_at": last_contact_at,
        }
        if u_out is not None:
            changes["u"] = u_out
        return [UpdateState(changes)]
```

(`Drive(alpha=0.0, …)` — alpha is irrelevant to `satiate`; 0.0 makes that explicit.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — all aggregation tests (15) pass and every prior test, including `tests/sim/` scenarios, still passes.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/aggregation.py tests/test_aggregation.py
uv run ruff check core/aggregation.py tests/test_aggregation.py
uv run mypy -p lifemodel
git add core/aggregation.py tests/test_aggregation.py
git commit -m "feat(core): ContactAggregation — verdict resolution (fulfill satiates, reject backs off, defer holds) (spec §12)"
```

---

### Task 5: Wire ContactAggregation into the composition root + pipeline integration

**Files:**
- Modify: `composition.py`
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Consumes: `ContactAggregation` (Tasks 2–4), `GateParams`, `ComponentManifest`/`ComponentRegistry`, `CoreLoop`.
- Produces: a `build_lifemodel(...)` whose registry runs `contact` **then** `contact-aggregation`, so one `coreloop.tick()` rises `u` (neuron) and decides the desire (aggregation) end-to-end.

**Behavior:** register `ContactAggregation` **after** the neuron. Define the gate params once and reuse (`CONTACT_PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)`), passing `theta=CONTACT_PARAMS.theta_u`, `beta=CONTACT_BETA`, `u_max=CONTACT_U_MAX`. Guard against double-registration (same pattern as the neuron). Still no live cutover.

- [ ] **Step 1: Write the failing tests (append to `tests/test_composition.py`)**

```python
from lifemodel.core.aggregation import ContactAggregation


def test_aggregation_registered_after_neuron(tmp_path) -> None:
    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert ids.index("contact") < ids.index("contact-aggregation")
    assert any(isinstance(c, ContactAggregation) for c in lm.registry.enabled())


def test_pipeline_rises_then_wakes_desire(tmp_path) -> None:
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State

    store = JsonStateStore(tmp_path)
    # u already high; 1 min elapsed → neuron keeps it high, aggregation wakes a desire
    store.commit(State(u=3.0, desire_status="none", last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC)))
    lm.coreloop.tick()
    assert store.load().desire_status == "active"


def test_pipeline_exchange_satiates_and_clears(tmp_path) -> None:
    from lifemodel.state.json_store import JsonStateStore
    from lifemodel.state.model import State
    from lifemodel.core.taxonomy import exchange_signal

    store = JsonStateStore(tmp_path)
    store.commit(State(u=3.0, desire_status="active", last_tick_at="2026-07-06T03:59:00+00:00"))
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(datetime(2026, 7, 6, 4, 0, tzinfo=UTC)))
    lm.bus.publish(exchange_signal(origin_id="e1", actor="user", label="two_way", timestamp=None))
    lm.coreloop.tick()
    final = store.load()
    assert final.u == 0.0  # neuron satiated
    assert final.desire_status == "none"  # aggregation cleared the desire
    assert final.last_exchange_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py -q`
Expected: FAIL — `contact-aggregation` not registered.

- [ ] **Step 3: Implement the wiring in `composition.py`**

Add the import `from .core.aggregation import ContactAggregation` and `from .sim.wake import GateParams`. Add a module constant `CONTACT_PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)`. In `build_lifemodel`, after registering the `contact` neuron and before building the `coreloop`, register the aggregation (guarding double-registration exactly like the neuron):
```python
    aggregation = ContactAggregation(
        params=CONTACT_PARAMS,
        theta=CONTACT_PARAMS.theta_u,
        beta=CONTACT_BETA,
        u_max=CONTACT_U_MAX,
    )
    try:
        registry.manifest(aggregation.id)
    except UnknownComponent:
        registry.register(aggregation, ComponentManifest(id=aggregation.id, type="aggregation"))
```
(Import `UnknownComponent` from `.core.registry`; apply the same `try/except` guard to the neuron registration if not already present.)

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — new composition tests pass; every prior test (incl. `tests/sim/`) still passes; live path unchanged.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format composition.py tests/test_composition.py
uv run ruff check composition.py tests/test_composition.py
uv run mypy -p lifemodel
git add composition.py tests/test_composition.py
git commit -m "feat(core): register ContactAggregation after neuron — full autonomic→aggregation pipeline (no cutover)"
```

---

## Phase-B2 Definition of Done

- [ ] `make check` fully green — paste the tail (mypy Success + pytest summary).
- [ ] Five commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`.
- [ ] `tests/sim/` certified scenarios still green (the sim is reused, not changed).
- [ ] The pipeline runs `contact` → `contact-aggregation` under `coreloop.tick()`; the cron/egress path is untouched.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §4 verdict/in_flight taxonomy → Task 1; §12 wake gates + desire birth/dedup → Task 2; §12 exchange clears + resets → Task 3; §12 verdict resolution (fulfill satiates, reject grows backoff, defer holds) → Task 4; DI wiring / full pipeline → Task 5. §12 invariants covered as tests: one-desire-per-crossing + dedup (T2), defer-doesn't-reset (T4), reject→backoff (T4), only-real-contact-satiates + own-impulse≠contact (T3/T4), wake≠send (no delivery emitted anywhere). **Deferred (by design):** salience §5, DEFER→release/availability, WakeCognition intent → Phase D; latent/effective + ActionPending → Phase C; stale-pending, live cutover, delete `decision.py` → Phase E.
- **Type consistency:** `ContactAggregation(*, params, theta, beta, u_max, id="contact-aggregation")` identical in Tasks 2–5. `contact_value(signals, *, default)`, `is_in_flight(signals)`, `read_verdict`, `read_exchange` used with matching signatures. Working-local threading (`last_exchange_at`/`declined_at`/`decline_count`/`last_contact_at`/`u_out`/`agg`) is consistent across the three incremental step edits (Tasks 2→3→4).
- **Order invariant (documented):** within one `step`, exchange → verdict → wake, matching `core/decision.py`'s sequencing so a fresh exchange/reject correctly vetoes a same-tick wake.
- **No placeholders:** every step ships real code + an exact command with expected output.

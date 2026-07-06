# Core Rebuild — Phase E4: /lifemodel debug Rewrite + Delete Monolith Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `/lifemodel debug` view onto the **new model** (physiology E/S/C, latent/effective pressure + ActionPending, desire lifecycle, gates, backstop, timing) so it stops depending on the dead monolith, then **delete `core/decision.py`** and **`impulse.py`** (relocating the one still-needed constant) plus their tests. After this phase nothing imports the monolith and it is gone.

**Architecture:** A pure, Hermes-free `core/introspect.py` computes a `Readings` snapshot **directly from persisted `State` + the new pure helpers** (`circadian`, `inhibition_at`, `effective_pressure`, `evaluate_wake`, `backoff_interval`, `allow_send`) — no `decide_reachout`, no deep-copy. The calibration constants it needs are passed in a small `DebugConfig` (built by `debug.py`, which already imports the composition root's constants) so `introspect` keeps its no-Hermes/no-composition import boundary. `debug.py` renders the sections the owner approved. `IMPULSE_LABEL_PREFIX` moves to `core/wake_packet.py` (the natural home — it prefixes the injected wake-packet); `impulse.py`/`compose_impulse` are deleted.

**Tech Stack:** Python 3.11 stdlib-only. `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; `core/introspect.py` imports no Hermes, no `composition`, no `debug`** (import boundary — it takes a `DebugConfig` instead).
- **Read-only debug (HLA §9):** the debug path never writes/commits/consumes.
- **Do NOT modify** `tick.py`, `heartbeat.py`. Do NOT touch the old ABC cleanup (that is a separate follow-up). Do NOT push/merge/touch `main`. `tests/sim/` must stay green. `mypy -p lifemodel` strict.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Modify `core/wake_packet.py` — add `IMPULSE_LABEL_PREFIX` (Task 1).
- Modify `hooks.py`, `egress_service.py` — import the prefix from `core/wake_packet.py` (Task 1).
- Delete `impulse.py`, `tests/test_impulse.py` (Task 1).
- Rewrite `core/introspect.py` (Task 2), `tests/test_introspect.py` (Task 2).
- Rewrite `debug.py` (Task 3), `tests/test_debug.py` (Task 3).
- Delete `core/decision.py`, `tests/test_decision.py` (Task 4).

---

### Task 1: Relocate `IMPULSE_LABEL_PREFIX`, delete `impulse.py`

**Files:** `core/wake_packet.py`, `hooks.py`, `egress_service.py`, `core/__init__.py`; delete `impulse.py`, `tests/test_impulse.py`; update `tests/test_hooks.py`, `tests/test_egress_service_tick.py`, `tests/test_flat_layout.py`.

- [ ] **Step 1: Add the constant + point the importers at it**

In `core/wake_packet.py`, add near the top (after imports):
```python
#: Marker prefixed to an injected proactive prompt so the being's own hooks
#: recognise their own nudge (correlation + self-exclusion). Was `impulse.py`.
IMPULSE_LABEL_PREFIX = "[lifemodel:impulse]"
```
Re-export `IMPULSE_LABEL_PREFIX` from `core/__init__.py`. Change the imports in `hooks.py` and `egress_service.py` from `from .impulse import IMPULSE_LABEL_PREFIX` to `from .core.wake_packet import IMPULSE_LABEL_PREFIX`. Update `tests/test_hooks.py` and `tests/test_egress_service_tick.py` imports from `lifemodel.impulse` to `lifemodel.core.wake_packet`.

**Confirm the exact prefix string** first: `grep -n IMPULSE_LABEL_PREFIX impulse.py` — copy its literal value verbatim into `core/wake_packet.py` (do not guess; keep it byte-identical so correlation still works).

- [ ] **Step 2: Delete the dead module + its test + fix the layout test**

```bash
git rm impulse.py tests/test_impulse.py
```
`tests/test_flat_layout.py` enumerates modules — remove `impulse` from any expected-module list there (run it to see the exact assertion and update it).

- [ ] **Step 3: Run the suite**

Run: `uv run pytest -q`
Expected: PASS — nothing imports `impulse` anymore; prefix resolves from `core/wake_packet.py`.

- [ ] **Step 4: Format, type-check, commit**

```bash
uv run ruff format core/wake_packet.py core/__init__.py hooks.py egress_service.py tests/
uv run ruff check core/wake_packet.py core/__init__.py hooks.py egress_service.py tests/
uv run mypy -p lifemodel
git add -A
git commit -m "refactor(cutover): relocate IMPULSE_LABEL_PREFIX to core; delete impulse.py"
```

---

### Task 2: Rewrite `core/introspect.py` onto the new model

**Files:** Rewrite `core/introspect.py`; rewrite `tests/test_introspect.py`.

**Behavior:** `compute_readings(state, *, now, cfg)` returns a frozen `Readings` computed directly from `State` + the new helpers. `DebugConfig` carries the calibration. No `decide_reachout`, no deep-copy, no Hermes/composition import.

- [ ] **Step 1: Write the failing tests (replace `tests/test_introspect.py`)**

```python
# tests/test_introspect.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.introspect import DebugConfig, Readings, compute_readings
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State

CFG = DebugConfig(
    params=GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0),
    theta=1.0, i0=1.0, grace_min=45.0, halflife_min=60.0,
    peak_hour_utc=13.0, max_per_day=3, min_interval_min=60.0,
)
NOW = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)


def test_reads_physiology_and_drive() -> None:
    state = State(u=2.0, energy=0.7, fatigue=0.3, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert isinstance(r, Readings)
    assert r.energy == 0.7
    assert r.fatigue == 0.3
    assert 0.0 <= r.circadian <= 1.0
    assert r.u == 2.0
    assert r.inhibition == 0.0  # no ActionPending
    assert abs(r.effective - 2.0) < 1e-9  # u*(1-0)
    assert r.would_wake is True  # effective >= theta, no gates


def test_action_pending_suppresses_effective_and_wake() -> None:
    state = State(u=3.0, action_pending_since="2026-07-06T03:50:00+00:00", last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)  # 10 min ago -> in grace -> inhibition 1
    assert r.inhibition == 1.0
    assert r.action_pending_phase == "grace"
    assert r.effective == 0.0
    assert r.would_wake is False
    assert r.wake_reason == "no_wake_below_threshold"  # effective 0 < theta


def test_backstop_readings() -> None:
    log = ["2026-07-06T03:30:00+00:00", "2026-07-06T02:00:00+00:00"]  # 2 today, last 30m ago
    state = State(u=2.0, proactive_send_log=log, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.sends_today == 2
    assert r.sends_cap == 3
    assert r.send_allowed is False  # last send 30 min ago < 60 min interval


def test_silence_window_and_backoff() -> None:
    state = State(
        u=2.0, last_exchange_at="2026-07-06T03:55:00+00:00",  # 5 min ago, w=15 -> 10 left
        declined_at="2026-07-06T03:40:00+00:00", decline_count=1,  # 20 min ago, r0=30 -> 10 left
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert abs((r.silence_window_remaining_min or 0) - 10.0) < 1e-6
    assert abs((r.backoff_remaining_min or 0) - 10.0) < 1e-6
    assert r.would_wake is False  # silence window blocks
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_introspect.py -q`
Expected: FAIL — old `introspect` has no `DebugConfig`/`Readings`/new fields.

- [ ] **Step 3: Replace `core/introspect.py` entirely**

```python
# core/introspect.py
"""Pure, Hermes-free readings for the /lifemodel debug view (spec §16).

``compute_readings`` turns a persisted :class:`State` into a frozen
:class:`Readings` snapshot for the renderer — computed directly from state and
the new pure helpers (circadian, inhibition, effective pressure, wake gates,
backstop), never from the (deleted) decision monolith. The calibration arrives
in a :class:`DebugConfig` so this module keeps its import boundary (no Hermes, no
composition, no debug).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..sim.wake import GateParams, LaneState, backoff_interval, evaluate_wake
from ..state.model import State
from .backstop import allow_send
from .circadian import circadian
from .pressure import effective_pressure, inhibition_at
from .timeutil import minutes_between


@dataclass(frozen=True)
class DebugConfig:
    """Calibration the readings need (built by ``debug`` from the composition root)."""

    params: GateParams
    theta: float
    i0: float
    grace_min: float
    halflife_min: float
    peak_hour_utc: float
    max_per_day: int
    min_interval_min: float


@dataclass(frozen=True)
class Readings:
    schema_version: int
    tick_count: int
    # physiology
    energy: float
    fatigue: float
    circadian: float
    alertness: float
    # drive
    u: float
    inhibition: float
    action_pending_phase: str  # "none" | "grace" | "decaying"
    action_pending_remaining_min: float | None  # grace remaining, if in grace
    effective: float
    theta: float
    pct_to_wake: float
    duration_over_theta: float
    # desire lifecycle
    desire_status: str
    pending: bool
    pending_since: str | None
    last_contact_at: str | None
    last_exchange_at: str | None
    # gates
    would_wake: bool
    wake_reason: str
    silence_window_remaining_min: float | None
    decline_count: int
    backoff_remaining_min: float | None
    # backstop
    sends_today: int
    sends_cap: int
    send_allowed: bool
    # timing
    last_tick_at: str | None
    last_tick_ago_min: float | None
    egress_service_alive_at: str | None
    egress_service_ago_min: float | None


def _ago(iso: str | None, now: datetime) -> float | None:
    return None if iso is None else minutes_between(iso, now)


def _action_pending(state: State, now: datetime, cfg: DebugConfig) -> tuple[float, str, float | None]:
    if state.action_pending_since is None:
        return 0.0, "none", None
    inh = inhibition_at(
        state.action_pending_since, now, i0=cfg.i0, grace_min=cfg.grace_min, halflife_min=cfg.halflife_min
    )
    elapsed = minutes_between(state.action_pending_since, now)
    if elapsed <= cfg.grace_min:
        return inh, "grace", cfg.grace_min - elapsed
    return inh, "decaying", None


def _silence_remaining(state: State, now: datetime, w: float) -> float | None:
    if state.last_exchange_at is None:
        return None
    left = w - minutes_between(state.last_exchange_at, now)
    return left if left > 0 else None


def _backoff_remaining(state: State, now: datetime, cfg: DebugConfig) -> float | None:
    if state.declined_at is None:
        return None
    r_n = backoff_interval(
        decline_count=state.decline_count, r0=cfg.params.r0, k=cfg.params.k, r_max=cfg.params.r_max
    )
    left = r_n - minutes_between(state.declined_at, now)
    return left if left > 0 else None


def _sends_today(send_log: list[str], now: datetime) -> int:
    from datetime import timedelta

    day_ago = now - timedelta(hours=24)
    count = 0
    for ts in send_log:
        try:
            t = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if t.tzinfo is not None and t >= day_ago:
            count += 1
    return count


def compute_readings(state: State, *, now: datetime, cfg: DebugConfig) -> Readings:
    u = state.u
    inhibition, phase, grace_left = _action_pending(state, now, cfg)
    effective = effective_pressure(u, inhibition)

    exch_min = -minutes_between(state.last_exchange_at, now) if state.last_exchange_at is not None else None
    decl_min = -minutes_between(state.declined_at, now) if state.declined_at is not None else None
    lane = LaneState(
        last_exchange_at=exch_min, in_flight=False, declined_at=decl_min, decline_count=state.decline_count
    )
    outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=cfg.params)

    return Readings(
        schema_version=state.schema_version,
        tick_count=state.tick_count,
        energy=state.energy,
        fatigue=state.fatigue,
        circadian=circadian(now, peak_hour_utc=cfg.peak_hour_utc),
        alertness=max(0.0, min(1.0, circadian(now, peak_hour_utc=cfg.peak_hour_utc) - state.fatigue)),
        u=u,
        inhibition=inhibition,
        action_pending_phase=phase,
        action_pending_remaining_min=grace_left,
        effective=effective,
        theta=cfg.theta,
        pct_to_wake=(effective / cfg.theta) if cfg.theta else 0.0,
        duration_over_theta=state.duration_over_theta,
        desire_status=state.desire_status,
        pending=state.pending_proactive_id is not None,
        pending_since=state.pending_proactive_since,
        last_contact_at=state.last_contact_at,
        last_exchange_at=state.last_exchange_at,
        would_wake=outcome.is_urge,
        wake_reason=outcome.value,
        silence_window_remaining_min=_silence_remaining(state, now, cfg.params.w),
        decline_count=state.decline_count,
        backoff_remaining_min=_backoff_remaining(state, now, cfg),
        sends_today=_sends_today(state.proactive_send_log, now),
        sends_cap=cfg.max_per_day,
        send_allowed=allow_send(
            state.proactive_send_log, now, max_per_day=cfg.max_per_day, min_interval_min=cfg.min_interval_min
        ),
        last_tick_at=state.last_tick_at,
        last_tick_ago_min=_ago(state.last_tick_at, now),
        egress_service_alive_at=state.egress_service_alive_at,
        egress_service_ago_min=_ago(state.egress_service_alive_at, now),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_introspect.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Format, type-check, commit** (note: `debug.py` still imports the OLD introspect names — it is rewritten in Task 3; `mypy`/`pytest` on the whole package will be red until then, so scope this task's gate to the introspect test + `ruff`/`mypy` on the file):

```bash
uv run ruff format core/introspect.py tests/test_introspect.py
uv run ruff check core/introspect.py tests/test_introspect.py
git add core/introspect.py tests/test_introspect.py
git commit -m "refactor(debug): rewrite introspect readings onto the new model (no decision monolith)"
```

---

### Task 3: Rewrite `debug.py` render onto the new `Readings`

**Files:** Rewrite `debug.py`; rewrite `tests/test_debug.py`.

**Behavior:** `render_dump_for_dir(base_dir)` builds the graph via `build_lifemodel`, assembles a `DebugConfig` from the composition constants, computes `Readings`, and renders the sections (PHYSIOLOGY / DRIVE / DESIRE / GATES / BACKSTOP / TIMING) plus a graceful `<unreadable>` banner on a corrupt store. Keep it read-only.

- [ ] **Step 1: Write the failing tests (replace `tests/test_debug.py`)**

```python
# tests/test_debug.py
from __future__ import annotations

from lifemodel.debug import render_dump_for_dir
from lifemodel.state.json_store import JsonStateStore
from lifemodel.state.model import State


def test_dump_renders_the_sections(tmp_path) -> None:
    JsonStateStore(tmp_path).commit(State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00"))
    out = render_dump_for_dir(tmp_path)
    for section in ("PHYSIOLOGY", "DRIVE", "DESIRE", "GATES", "BACKSTOP", "TIMING"):
        assert section in out
    assert "effective" in out.lower()


def test_dump_survives_a_corrupt_store(tmp_path) -> None:
    (tmp_path / "state.json").write_text("{ not json", encoding="utf-8")
    out = render_dump_for_dir(tmp_path)
    assert "unreadable" in out.lower()  # graceful banner, no crash
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_debug.py -q`
Expected: FAIL — old renderer references old `Readings` fields.

- [ ] **Step 3: Replace `debug.py`**

```python
# debug.py
"""The /lifemodel debug dump — the owner's read-only window (spec §16).

Renders the being's new-model state in labelled sections (physiology, drive,
desire lifecycle, gates, backstop, timing), each showing raw + derived values.
Read-only: builds the graph via the single composition root, computes pure
:class:`Readings`, never writes. Stdlib only; imports Hermes-free adapters.
"""

from __future__ import annotations

from pathlib import Path

from . import composition
from .core.introspect import DebugConfig, Readings, compute_readings
from .sim.wake import GateParams
from .state.errors import StateError


def _cfg() -> DebugConfig:
    # max_per_day / min_interval mirror core.backstop.allow_send's defaults —
    # the live egress calls allow_send() without overrides, so those defaults ARE
    # the live limits (spec §14: <=3/day, 60 min).
    return DebugConfig(
        params=composition.CONTACT_PARAMS,
        theta=composition.CONTACT_PARAMS.theta_u,
        i0=composition.CONTACT_I0,
        grace_min=composition.CONTACT_GRACE_MIN,
        halflife_min=composition.CONTACT_INHIBITION_HALFLIFE_MIN,
        peak_hour_utc=composition.CIRCADIAN_PEAK_UTC_HOUR,
        max_per_day=3,
        min_interval_min=60.0,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _n(x: float) -> str:
    return f"{x:.4g}"


def _opt(x: float | None, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:.1f}{unit}"


def render_dump_for_dir(base_dir: Path) -> str:
    lm = composition.build_lifemodel(base_dir=base_dir)
    now = lm.clock.now()
    try:
        state = lm.state.load()
    except StateError as exc:
        return f"lifemodel debug  (read-only)\n{'=' * 30}\n\n<unreadable: {exc}>\n"
    return render_debug_dump(readings=compute_readings(state, now=now, cfg=_cfg()))


def render_debug_dump(*, readings: Readings) -> str:
    r = readings
    lines: list[str] = ["lifemodel debug  (read-only)", "=" * 30, ""]

    lines += [
        "PHYSIOLOGY",
        f"  energy(E) {_pct(r.energy)}   fatigue(S) {_n(r.fatigue)}   circadian(C) {_n(r.circadian)}",
        f"  alertness ~{_n(r.alertness)}   (higher C, lower S = sharper)",
        "",
        "DRIVE (contact)",
        f"  latent u {_n(r.u)}   inhibition {_n(r.inhibition)} [{r.action_pending_phase}"
        + (f", {_opt(r.action_pending_remaining_min, 'm grace left')}" if r.action_pending_remaining_min else "")
        + "]",
        f"  effective = u*(1-inhibition) = {_n(r.effective)}   theta {_n(r.theta)}   -> {_pct(r.pct_to_wake)} to wake",
        f"  interpretation: {'a pull is over threshold' if r.effective >= r.theta else 'below the wake threshold'}",
        "",
        "DESIRE",
        f"  status {r.desire_status}   pending_turn {r.pending}"
        + (f" (since {r.pending_since})" if r.pending else ""),
        f"  last_contact {r.last_contact_at or 'n/a'}   last_exchange {r.last_exchange_at or 'n/a'}",
        "",
        "GATES (why wake / no wake)",
        f"  would_wake {r.would_wake}   reason {r.wake_reason}",
        f"  silence_window_left {_opt(r.silence_window_remaining_min, ' min')}"
        f"   decline_backoff_left {_opt(r.backoff_remaining_min, ' min')} (declines {r.decline_count})",
        "",
        "BACKSTOP (hard send limit)",
        f"  sends_today {r.sends_today}/{r.sends_cap}   send_allowed_now {r.send_allowed}",
        "",
        "TIMING",
        f"  tick {r.tick_count}   last_tick {_opt(r.last_tick_ago_min, ' min ago')}"
        f"   service_alive {_opt(r.egress_service_ago_min, ' min ago')}",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — debug + introspect green; `test_debug`/`test_introspect` pass; `tests/sim/` green.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format debug.py tests/test_debug.py
uv run ruff check debug.py tests/test_debug.py
uv run mypy -p lifemodel
git add debug.py tests/test_debug.py
git commit -m "feat(debug): render /lifemodel debug on the new model (physiology/drive/gates/backstop)"
```

---

### Task 4: Delete the monolith `core/decision.py`

**Files:** delete `core/decision.py`, `tests/test_decision.py`; fix `tests/test_flat_layout.py`.

**Behavior:** nothing imports `core/decision.py` now (verify), so delete it and its unit test. Update the flat-layout module enumeration.

- [ ] **Step 1: Verify no importers remain**

Run: `grep -rnE '^\s*(from|import)\b.*\bdecision\b' --include='*.py' . | grep -v docs/`
Expected: only `tests/test_decision.py` (which is deleted next). If anything else appears, STOP and report.

- [ ] **Step 2: Delete + fix layout test**

```bash
git rm core/decision.py tests/test_decision.py
```
Remove `decision` from the expected-module list in `tests/test_flat_layout.py` (run it to find the exact assertion).

- [ ] **Step 3: Run the full gate**

Run: `make check`
Expected: fully green — the monolith is gone; the being runs entirely on the layered pipeline.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(cutover): delete the core/decision.py monolith — pipeline is the sole brain"
```

---

## Phase-E4 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Four commits on `core/rebuild`, one per task.
- [ ] `core/decision.py`, `impulse.py`, `tests/test_decision.py`, `tests/test_impulse.py` deleted; `grep` for `decision`/`impulse` imports is empty (outside docs).
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review

- **Spec coverage:** §16 observability — the new debug shows latent/effective/inhibition, energy E/S/C, gates, backstop, timing (owner-approved sections). Monolith + dead `impulse.py` deleted; `IMPULSE_LABEL_PREFIX` relocated (correlation preserved). **Deferred (follow-up):** dead ABC cleanup (`SilentAggregator`/`Aggregator`/`Neuron`/`Layer`/`ActGate` + `LifeModel.aggregator`/`neurons`) — a separate mechanical phase; `COGNITION` debug section (projection_id from the event log) — nice-to-have.
- **Import boundary kept:** `introspect` takes `DebugConfig` (no composition/Hermes import); `debug` builds it from the composition constants.
- **Read-only:** debug never writes/commits.
- **No placeholders:** every step ships real code + an exact command with expected output.

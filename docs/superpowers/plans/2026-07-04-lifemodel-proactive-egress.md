# Proactive Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the being's proactive outreach a clean, self-owned assistant turn in the user's live conversation (native reach-in turn), reusable by any plugin, with the tested cron path kept as an independent fallback.

**Architecture:** Two upstream-shaped core primitives are monkey-patched into Hermes at plugin load — `inject_proactive_turn` (runner-owned native-turn injection) and `register_gateway_service` (runner-owned supervised background task). The plugin drives an in-process service loop that accumulates pressure and, on threshold, injects a single internal-labeled user turn so the being composes a native reply (`[user(impulse), assistant(proactive)]`, never assistant-first). A liveness stamp lets the always-registered cron heartbeat defer while the in-process service is alive and take over as fallback when it is not.

**Tech Stack:** Python 3.11, stdlib-only runtime, hexagonal ports/adapters, `uv`, pytest (+pytest-asyncio, asyncio_mode=auto), ruff, mypy --strict, structlog (optional/dev).

## Global Constraints

- Python floor `>=3.11,<3.12`; target-version `py311`.
- **Runtime deps stay empty (stdlib-only).** Do not add a runtime dependency. structlog is dev-only; log via the `EventLogger` seam.
- All Hermes host modules (`gateway.*`, `cron.*`, `agent.*`, `tools.*`, `hermes_constants`) are `ignore_missing_imports` in mypy and are **NOT importable in unit tests**. Every Hermes touchpoint MUST sit behind an injected seam (callable/accessor) with a lazy default import, so unit tests run without Hermes.
- **Fail closed, always.** No code path here may raise into the gateway. On any missing attribute / version drift / error: log and degrade (return an `UNAVAILABLE`/`FAILED` outcome or no-op); never crash.
- `message_id` passed to Hermes MUST be `None` or numeric (Telegram adapter does `int(message_id)`).
- The impulse is a **single internal-labeled `role=user` text** that serves as both the model seed and the honest transcript record (see spec §4). No hidden-input, no dual-text (dual-text is upstream-scope).
- Reach-in only works **inside the gateway process**; resolve **only known lanes** (the home origin), never arbitrary platform/chat strings.
- Quality gate: `make check` (= `uv run ruff format --check .` ; `uv run ruff check .` ; `uv run mypy src` ; `uv run pytest`) must stay green. Coverage must not regress.
- Follow existing repo idioms: `from __future__ import annotations`, `Protocol`/`@runtime_checkable` ports in `ports/`, impls in `adapters/`, pure types in `domain/`, injected seams for I/O, `EventLogger.info(event, **fields)` structured logging, `build_lifemodel(...)` as the composition root.
- Spec: `docs/superpowers/specs/2026-07-04-lifemodel-proactive-egress-design.md`. Bead: `lm-64s`.

---

### Task 1: Reach outcome + proactive egress port

**Files:**
- Create: `src/lifemodel/domain/egress.py`
- Create: `src/lifemodel/ports/proactive.py`
- Modify: `src/lifemodel/ports/__init__.py` (re-export)
- Test: `tests/test_proactive_port.py`

**Interfaces:**
- Produces: `ReachOutcome` (Enum: `DELIVERED`, `SKIPPED_BUSY`, `UNAVAILABLE`, `FAILED`; property `ok -> bool`). `ProactiveEgressPort` (Protocol, `@runtime_checkable`) with `reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome`. `target` is the home-origin dict `{"platform","chat_id","thread_id"}` (as produced by `heartbeat._resolve_home_origin`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proactive_port.py
from __future__ import annotations

from lifemodel.domain.egress import ReachOutcome
from lifemodel.ports import ProactiveEgressPort
from lifemodel.ports.proactive import ProactiveEgressPort as DirectPort


def test_reach_outcome_ok_only_for_delivered() -> None:
    assert ReachOutcome.DELIVERED.ok is True
    assert ReachOutcome.SKIPPED_BUSY.ok is False
    assert ReachOutcome.UNAVAILABLE.ok is False
    assert ReachOutcome.FAILED.ok is False


def test_port_is_runtime_checkable_and_reexported() -> None:
    assert ProactiveEgressPort is DirectPort

    class Impl:
        def reach_out(self, target: dict[str, str | None], impulse: str) -> ReachOutcome:
            return ReachOutcome.DELIVERED

    assert isinstance(Impl(), ProactiveEgressPort)

    class NotImpl:
        pass

    assert not isinstance(NotImpl(), ProactiveEgressPort)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proactive_port.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.domain.egress'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lifemodel/domain/egress.py
from __future__ import annotations

from enum import Enum


class ReachOutcome(Enum):
    """Result of one proactive reach-out attempt (HLA §7, egress)."""

    DELIVERED = "delivered"        # native reach-in turn injected into the live session
    SKIPPED_BUSY = "skipped_busy"  # a turn is active in the session; retry next tick
    UNAVAILABLE = "unavailable"    # reach-in primitive not available (no runner / version drift)
    FAILED = "failed"              # attempted but errored (already logged, fail-closed)

    @property
    def ok(self) -> bool:
        """True only when a native proactive turn was actually delivered."""
        return self is ReachOutcome.DELIVERED
```

```python
# src/lifemodel/ports/proactive.py
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from ..domain.egress import ReachOutcome


@runtime_checkable
class ProactiveEgressPort(Protocol):
    """Reach out to the user first, as a native assistant turn (spec §3.1/§4)."""

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        """Inject *impulse* as an internal user turn on *target* lane so the being
        composes and delivers a native reply. *target* = home-origin dict
        {platform, chat_id, thread_id}. Never raises — returns a ReachOutcome."""
        ...
```

Add the re-export to `src/lifemodel/ports/__init__.py` (append alongside the existing `StatePort, DeliveryPort, ClockPort` exports):

```python
from .proactive import ProactiveEgressPort

__all__ = [*__all__, "ProactiveEgressPort"]  # keep existing __all__ entries
```
(If `ports/__init__.py` builds `__all__` as a literal list, add `"ProactiveEgressPort"` to it and add the `from .proactive import ProactiveEgressPort` import — match the file's existing style rather than the `[*__all__, ...]` shortcut.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proactive_port.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/domain/egress.py src/lifemodel/ports/proactive.py src/lifemodel/ports/__init__.py tests/test_proactive_port.py
git commit -m "feat(egress): ReachOutcome + ProactiveEgressPort"
```

---

### Task 2: Impulse composer (pure)

**Files:**
- Create: `src/lifemodel/impulse.py`
- Test: `tests/test_impulse.py`

**Interfaces:**
- Consumes: `WakePacket` (`domain/wake.py`: fields `reason, pressure_kind, pressure, threshold, energy, budget, last_contact_at, version`).
- Produces: `compose_impulse(packet: WakePacket, *, now: datetime, last_contact_at: datetime | None) -> str` — a single internal-labeled `role=user` text (spec §4). Deterministic; no I/O.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_impulse.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lifemodel.domain.wake import WakePacket
from lifemodel.impulse import IMPULSE_LABEL_PREFIX, compose_impulse

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)


def _packet() -> WakePacket:
    return WakePacket(reason="silence", pressure_kind="idle", pressure=28.0, threshold=10.0)


def test_impulse_is_labeled_internal_and_not_user_authored() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=_T0 - timedelta(hours=5))
    assert text.startswith(IMPULSE_LABEL_PREFIX)
    lowered = text.lower()
    assert "not from the user" in lowered or "не от пользователя" in lowered
    # never starts with a slash (would enter Hermes command routing — spec §5 guard f)
    assert not text.lstrip().startswith("/")


def test_impulse_reports_whole_hours_of_silence() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=_T0 - timedelta(hours=5, minutes=40))
    assert "5" in text  # floor(5h40m) == 5 hours


def test_impulse_handles_unknown_last_contact() -> None:
    text = compose_impulse(_packet(), now=_T0, last_contact_at=None)
    assert text.startswith(IMPULSE_LABEL_PREFIX)
    assert "/" != text.strip()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_impulse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.impulse'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lifemodel/impulse.py
"""Compose the proactive impulse — a single internal-labeled ``role=user`` text.

Spec §4: this text serves as BOTH the model seed AND the honest transcript record.
It must (a) clearly mark itself as not-user-authored, (b) never start with '/'
(would enter Hermes command routing — spec §5 guard f), (c) elicit a warm,
human, self-owned message rather than machinery talk (soul lever, bead lm-pbm).
"""

from __future__ import annotations

from datetime import datetime

from .domain.wake import WakePacket

IMPULSE_LABEL_PREFIX = "[lifemodel · внутренний импульс — не от пользователя]"


def _hours_silent(now: datetime, last_contact_at: datetime | None) -> int | None:
    if last_contact_at is None:
        return None
    delta = now - last_contact_at
    return max(0, int(delta.total_seconds() // 3600))


def compose_impulse(
    packet: WakePacket, *, now: datetime, last_contact_at: datetime | None
) -> str:
    """Return the internal-labeled user text that seeds the proactive turn."""
    hours = _hours_silent(now, last_contact_at)
    if hours is None:
        window = "давно не общались"
    else:
        window = f"около {hours} ч тишины"
    body = (
        "Прошло время без разговора по-человечески "
        f"({window}). Если есть, чем поделиться от себя — "
        "напиши сейчас, коротко и живо, своими словами. Ничего не требуется, "
        "если сказать нечего."
    )
    return f"{IMPULSE_LABEL_PREFIX} {body}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_impulse.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/impulse.py tests/test_impulse.py
git commit -m "feat(egress): pure impulse composer (single internal label)"
```

---

### Task 3: Core primitive `inject_proactive_turn` + `reachin_available`

**Files:**
- Create: `src/lifemodel/gateway_core.py`
- Test: `tests/test_gateway_core_inject.py`

**Interfaces:**
- Produces:
  - `reachin_available(runner: Any | None) -> bool` — version-guard: True only if runner has the exact attributes the primitive needs.
  - `inject_proactive_turn(runner: Any, target: Mapping[str, str | None], prompt: str, *, message_id: int | None = None, make_event: MakeEvent = _default_make_event, schedule: Schedule = _default_schedule, logger: EventLogger | None = None) -> ReachOutcome` — resolves lane→source→adapter and runs a native `internal=True` turn on the gateway loop. Returns `DELIVERED` / `UNAVAILABLE` / `FAILED`. Never raises.
  - Type aliases `MakeEvent = Callable[[str, Any, int | None], Any]`, `Schedule = Callable[[Any, Any], None]`.

**Notes on seams (why):** `make_event` builds a Hermes `MessageEvent` (real default imports `gateway.platforms.base`); `schedule` runs the coroutine on the gateway loop (real default = `asyncio.run_coroutine_threadsafe`). Both are injected so unit tests need no Hermes. Adapter selection: `_profile_adapters` is **secondary-only** (run.py:2585) — use `runner._profile_adapters[source.profile]` when `source.profile` is truthy, else `runner.adapters` (spec §3.1, codex fix).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway_core_inject.py
from __future__ import annotations

from typing import Any

from lifemodel.domain.egress import ReachOutcome
from lifemodel.gateway_core import inject_proactive_turn, reachin_available

_TARGET = {"platform": "telegram", "chat_id": "115679831", "thread_id": None}


class _Source:
    def __init__(self, platform: str = "telegram", chat_id: str = "115679831", profile: str = "") -> None:
        self.platform = platform
        self.chat_id = chat_id
        self.profile = profile


class _FakeRunner:
    """Duck-types only what inject_proactive_turn touches."""

    def __init__(self, *, source: Any = None, running: bool = True) -> None:
        self._gateway_loop = object()
        self._running = running
        self._draining = False
        self._running_agents: set[Any] = set()
        self._source = source if source is not None else _Source()
        self.adapters = {"telegram": object()}
        self._profile_adapters: dict[str, dict[str, Any]] = {}
        self.built_with: Any = None

    def _build_process_event_source(self, evt: dict[str, Any]) -> Any:
        self.built_with = evt
        return self._source


def _make_event(text: str, source: Any, message_id: int | None) -> dict[str, Any]:
    return {"text": text, "source": source, "message_id": message_id, "internal": True}


def test_reachin_available_true_for_complete_runner() -> None:
    assert reachin_available(_FakeRunner()) is True
    assert reachin_available(None) is False

    class Partial:  # missing _gateway_loop etc.
        adapters: dict[str, Any] = {}

    assert reachin_available(Partial()) is False


def test_inject_builds_internal_event_and_schedules_on_loop() -> None:
    runner = _FakeRunner()
    scheduled: list[tuple[Any, Any]] = []

    def schedule(coro: Any, loop: Any) -> None:
        scheduled.append((coro, loop))

    outcome = inject_proactive_turn(
        runner, _TARGET, "hello", make_event=_make_event, schedule=schedule
    )

    assert outcome is ReachOutcome.DELIVERED
    assert runner.built_with == {"platform": "telegram", "chat_id": "115679831", "thread_id": None}
    assert len(scheduled) == 1
    coro, loop = scheduled[0]
    assert loop is runner._gateway_loop
    coro.close()  # we injected a fake adapter with no real handle_message; close the coroutine


def test_inject_unavailable_when_no_source() -> None:
    runner = _FakeRunner(source=None)
    runner._source = None  # _build_process_event_source returns None -> lane unknown
    outcome = inject_proactive_turn(runner, _TARGET, "hi", make_event=_make_event, schedule=lambda c, l: None)
    assert outcome is ReachOutcome.UNAVAILABLE


def test_inject_unavailable_when_draining() -> None:
    runner = _FakeRunner()
    runner._draining = True
    outcome = inject_proactive_turn(runner, _TARGET, "hi", make_event=_make_event, schedule=lambda c, l: None)
    assert outcome is ReachOutcome.UNAVAILABLE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gateway_core_inject.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.gateway_core'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lifemodel/gateway_core.py
"""Interim monkey-patch of two upstream-shaped Hermes core primitives (spec §3).

These functions have the exact signatures we intend to upstream as GatewayRunner
methods (with a PluginContext facade). For now the plugin calls them directly with
an explicitly-resolved runner + injected seams (so they unit-test without Hermes).
Everything is fail-closed: nothing here may raise into the gateway.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .domain.egress import ReachOutcome
from .logging import EventLogger, get_logger

MakeEvent = Callable[[str, Any, "int | None"], Any]
Schedule = Callable[[Any, Any], None]

# Attributes inject_proactive_turn depends on — the version-guard surface.
_REQUIRED_RUNNER_ATTRS = (
    "_gateway_loop",
    "_build_process_event_source",
    "adapters",
    "_running",
    "_draining",
)


def reachin_available(runner: Any | None) -> bool:
    """True only if *runner* exposes every attribute inject_proactive_turn needs."""
    if runner is None:
        return False
    return all(hasattr(runner, attr) for attr in _REQUIRED_RUNNER_ATTRS)


def _default_make_event(text: str, source: Any, message_id: int | None) -> Any:
    from gateway.platforms.base import MessageEvent, MessageType

    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        message_id=message_id,
    )


def _default_schedule(coro: Any, loop: Any) -> None:
    import asyncio

    asyncio.run_coroutine_threadsafe(coro, loop)


def _select_adapter(runner: Any, source: Any) -> Any | None:
    profile = getattr(source, "profile", "") or ""
    if profile:
        adapters = getattr(runner, "_profile_adapters", {}) or {}
        by_profile = adapters.get(profile) or {}
        return by_profile.get(getattr(source, "platform", None))
    for platform, adapter in getattr(runner, "adapters", {}).items():
        if platform == getattr(source, "platform", None):
            return adapter
    return None


def inject_proactive_turn(
    runner: Any,
    target: Mapping[str, str | None],
    prompt: str,
    *,
    message_id: int | None = None,
    make_event: MakeEvent = _default_make_event,
    schedule: Schedule = _default_schedule,
    logger: EventLogger | None = None,
) -> ReachOutcome:
    """Run a native ``internal=True`` turn on *target* lane. Fail-closed."""
    log = logger or get_logger("lifemodel.reachin")
    if not reachin_available(runner):
        log.info("reachin_unavailable", reason="runner_incomplete")
        return ReachOutcome.UNAVAILABLE
    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        log.info("reachin_unavailable", reason="not_running_or_draining")
        return ReachOutcome.UNAVAILABLE
    try:
        evt = {
            "platform": target.get("platform"),
            "chat_id": target.get("chat_id"),
            "thread_id": target.get("thread_id"),
        }
        source = runner._build_process_event_source(evt)
        if source is None or not getattr(source, "chat_id", None):
            log.info("reachin_unavailable", reason="unknown_lane")
            return ReachOutcome.UNAVAILABLE
        adapter = _select_adapter(runner, source)
        if adapter is None:
            log.info("reachin_unavailable", reason="no_adapter")
            return ReachOutcome.UNAVAILABLE
        event = make_event(prompt, source, message_id)  # message_id None (spec constraint)
        schedule(adapter.handle_message(event), runner._gateway_loop)
        log.info("reachin_injected", chat_id=getattr(source, "chat_id", None))
        return ReachOutcome.DELIVERED
    except Exception as exc:  # noqa: BLE001 - fail-closed, never crash the gateway
        log.info("reachin_failed", error=f"{type(exc).__name__}: {exc}")
        return ReachOutcome.FAILED
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gateway_core_inject.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/gateway_core.py tests/test_gateway_core_inject.py
git commit -m "feat(egress): inject_proactive_turn core primitive + version-guard"
```

---

### Task 4: `ReachInEgress` adapter

**Files:**
- Create: `src/lifemodel/adapters/reachin.py`
- Test: `tests/test_reachin_adapter.py`

**Interfaces:**
- Consumes: `inject_proactive_turn`, `reachin_available` (Task 3); `ReachOutcome` (Task 1).
- Produces: `ReachInEgress` implementing `ProactiveEgressPort`. Constructor:
  `ReachInEgress(*, runner_accessor: Callable[[], Any | None], inject: InjectFn = inject_proactive_turn, logger: EventLogger | None = None)`.
  `reach_out(target, impulse) -> ReachOutcome`: resolves runner; if none/incomplete → `UNAVAILABLE`; if a turn is active (`runner._running_agents` truthy) → `SKIPPED_BUSY` (don't interrupt a live conversation; spec §5 defers fine-grained FIFO to upstream); else delegates to `inject`.
  Also a module default accessor `default_runner_accessor() -> Any | None` that lazily reads `gateway.run._gateway_runner_ref()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reachin_adapter.py
from __future__ import annotations

from typing import Any

from lifemodel.adapters.reachin import ReachInEgress
from lifemodel.domain.egress import ReachOutcome
from lifemodel.ports import ProactiveEgressPort

_TARGET = {"platform": "telegram", "chat_id": "115679831", "thread_id": None}


class _Runner:
    def __init__(self, *, busy: bool = False, complete: bool = True) -> None:
        self._running_agents = {"x"} if busy else set()
        if complete:
            self._gateway_loop = object()
            self._build_process_event_source = lambda evt: object()
            self.adapters = {"telegram": object()}
            self._running = True
            self._draining = False


def test_is_proactive_egress_port() -> None:
    egress = ReachInEgress(runner_accessor=lambda: None)
    assert isinstance(egress, ProactiveEgressPort)


def test_unavailable_when_no_runner() -> None:
    egress = ReachInEgress(runner_accessor=lambda: None)
    assert egress.reach_out(_TARGET, "hi") is ReachOutcome.UNAVAILABLE


def test_skips_when_session_busy() -> None:
    calls: list[Any] = []
    egress = ReachInEgress(
        runner_accessor=lambda: _Runner(busy=True),
        inject=lambda *a, **k: calls.append((a, k)) or ReachOutcome.DELIVERED,
    )
    assert egress.reach_out(_TARGET, "hi") is ReachOutcome.SKIPPED_BUSY
    assert calls == []  # inject NOT called while busy


def test_delegates_to_inject_when_idle() -> None:
    captured: dict[str, Any] = {}

    def fake_inject(runner: Any, target: Any, prompt: str, **kwargs: Any) -> ReachOutcome:
        captured.update(runner=runner, target=target, prompt=prompt, kwargs=kwargs)
        return ReachOutcome.DELIVERED

    runner = _Runner()
    egress = ReachInEgress(runner_accessor=lambda: runner, inject=fake_inject)
    assert egress.reach_out(_TARGET, "impulse-text") is ReachOutcome.DELIVERED
    assert captured["runner"] is runner
    assert captured["target"] == _TARGET
    assert captured["prompt"] == "impulse-text"
    assert captured["kwargs"].get("message_id") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reachin_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.adapters.reachin'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lifemodel/adapters/reachin.py
"""Primary proactive-egress adapter: native reach-in turn (spec §3.1/§6)."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..domain.egress import ReachOutcome
from ..gateway_core import inject_proactive_turn, reachin_available
from ..logging import EventLogger, get_logger

RunnerAccessor = Callable[[], "Any | None"]
InjectFn = Callable[..., ReachOutcome]


def default_runner_accessor() -> Any | None:
    """Lazily read the live GatewayRunner (weakref singleton, run.py:2588)."""
    try:
        import gateway.run as grun

        ref = getattr(grun, "_gateway_runner_ref", None)
        return ref() if callable(ref) else None
    except Exception:  # noqa: BLE001 - not in a gateway process / import failure
        return None


class ReachInEgress:
    """Deliver a proactive turn by injecting an internal user turn in the live session."""

    def __init__(
        self,
        *,
        runner_accessor: RunnerAccessor,
        inject: InjectFn = inject_proactive_turn,
        logger: EventLogger | None = None,
    ) -> None:
        self._runner_accessor = runner_accessor
        self._inject = inject
        self._log = logger or get_logger("lifemodel.reachin")

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        runner = self._runner_accessor()
        if not reachin_available(runner):
            return ReachOutcome.UNAVAILABLE
        if getattr(runner, "_running_agents", None):
            self._log.info("reachin_skip_busy")
            return ReachOutcome.SKIPPED_BUSY
        return self._inject(runner, target, impulse, message_id=None, logger=self._log)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reachin_adapter.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/adapters/reachin.py tests/test_reachin_adapter.py
git commit -m "feat(egress): ReachInEgress adapter (busy-skip + delegate)"
```

---

### Task 5: Delivery-aware proactive tick

**Files:**
- Create: `src/lifemodel/egress_service.py`
- Test: `tests/test_egress_service_tick.py`

**Interfaces:**
- Consumes: `LifeModel` (`composition.py`: fields `state, bus, clock, aggregator, neurons`); `ProactiveEgressPort`; `ReachOutcome`; `DEFAULT_WAKE_COOLDOWN` (`tick.py`).
- Produces: `run_proactive_tick(lm: LifeModel, egress: ProactiveEgressPort, target: Mapping[str, str | None], *, logger: EventLogger, cooldown: timedelta = DEFAULT_WAKE_COOLDOWN, busy: bool = False) -> ReachOutcome`. Mirrors `tick.run_tick` (accumulate pressure, decide) but is delivery-aware: it drains pressure + stamps `last_contact_at`/`cooldown_until` **only after** `egress.reach_out(...)` returns `DELIVERED`; on any other outcome it leaves pressure intact (retry next tick). Always bumps `tick_count`/`last_tick_at` and stamps `egress_service_alive_at` (Task 6), commits once.

**Reuse note (DRY):** import the pressure-accumulation and cooldown constants/helpers from `tick.py` rather than re-deriving them: `from .tick import DEFAULT_WAKE_COOLDOWN`. Signal accumulation is `state.pressure += sum(s.salience for s in signals)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_egress_service_tick.py
from __future__ import annotations

from datetime import datetime, timezone

from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import ThresholdAggregator
from lifemodel.domain.egress import ReachOutcome
from lifemodel.egress_service import run_proactive_tick
from lifemodel.logging import get_logger
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)
_TARGET = {"platform": "telegram", "chat_id": "1", "thread_id": None}


class _RecordingEgress:
    def __init__(self, outcome: ReachOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[dict[str, str | None], str]] = []

    def reach_out(self, target: dict[str, str | None], impulse: str) -> ReachOutcome:
        self.calls.append((dict(target), impulse))
        return self.outcome


def _lm(pressure: float) -> object:
    return build_lifemodel(
        base_dir=__import__("pathlib").Path("/unused"),
        state=FakeStateStore(State(pressure=pressure)),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=ThresholdAggregator(threshold=10.0),
        neurons=(),
    )


def test_below_threshold_does_not_reach_out() -> None:
    lm = _lm(pressure=1.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert egress.calls == []
    assert outcome is ReachOutcome.SKIPPED_BUSY or outcome is ReachOutcome.UNAVAILABLE or outcome is ReachOutcome.FAILED or outcome is ReachOutcome.DELIVERED  # not reached-out; sentinel below
    assert lm.state.load().pressure > 0.0  # pressure NOT drained


def test_delivered_drains_pressure_and_stamps_contact() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert outcome is ReachOutcome.DELIVERED
    assert len(egress.calls) == 1
    st = lm.state.load()
    assert st.pressure == 0.0
    assert st.last_contact_at is not None
    assert st.cooldown_until is not None


def test_failed_delivery_keeps_pressure() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.FAILED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"))
    assert outcome is ReachOutcome.FAILED
    st = lm.state.load()
    assert st.pressure == 28.0        # NOT drained — retry next tick
    assert st.last_contact_at is None


def test_busy_skips_delivery_and_keeps_pressure() -> None:
    lm = _lm(pressure=28.0)
    egress = _RecordingEgress(ReachOutcome.DELIVERED)
    outcome = run_proactive_tick(lm, egress, _TARGET, logger=get_logger("t"), busy=True)
    assert outcome is ReachOutcome.SKIPPED_BUSY
    assert egress.calls == []
    assert lm.state.load().pressure == 28.0
```

(Remove the sentinel line in `test_below_threshold_does_not_reach_out` and just assert `egress.calls == []` and `pressure > 0.0` — the outcome for a no-wake tick is defined in Step 3 as `ReachOutcome.SKIPPED_BUSY` only when `busy`; for below-threshold return a dedicated value; see implementation. Simplify the assertion to `assert egress.calls == []`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_egress_service_tick.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.egress_service'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/lifemodel/egress_service.py
"""In-process proactive tick + service loop (spec §3.2/§6).

run_proactive_tick is the delivery-aware analog of tick.run_tick: it accumulates
pressure and decides via the aggregator, but drains pressure / stamps contact ONLY
after a native reach-out is DELIVERED. proactive_service_loop is the supervised
coroutine the gateway runs; it self-guards on _running/_draining.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from .composition import LifeModel
from .domain.egress import ReachOutcome
from .impulse import compose_impulse
from .logging import EventLogger, get_logger
from .ports.proactive import ProactiveEgressPort
from .tick import DEFAULT_WAKE_COOLDOWN


def _iso(dt: datetime | None) -> datetime | None:
    return dt


def run_proactive_tick(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    logger: EventLogger,
    cooldown: timedelta = DEFAULT_WAKE_COOLDOWN,
    busy: bool = False,
) -> ReachOutcome:
    """One in-process proactive tick. Delivery-aware; fail-closed on the caller side."""
    state = lm.state.load()
    now = lm.clock.now()

    # Accumulate pressure from this tick's neurons (identical to tick.run_tick).
    for neuron in lm.neurons:
        for signal in neuron.fire(now):
            lm.bus.publish(signal)
    signals = lm.bus.consume_unprocessed()
    state.pressure += sum((s.salience for s in signals), 0.0)

    decision = lm.aggregator.decide(signals, pressure=state.pressure)
    in_cooldown = (
        state.cooldown_until is not None
        and now < datetime.fromisoformat(state.cooldown_until)
    )

    outcome = ReachOutcome.SKIPPED_BUSY  # default "did not reach out this tick"
    if decision.wake and decision.packet is not None and not in_cooldown and not busy:
        last_contact = (
            datetime.fromisoformat(state.last_contact_at)
            if state.last_contact_at is not None
            else None
        )
        impulse = compose_impulse(decision.packet, now=now, last_contact_at=last_contact)
        outcome = egress.reach_out(target, impulse)
        if outcome is ReachOutcome.DELIVERED:
            state.pressure = 0.0
            state.last_contact_at = now.isoformat()
            state.cooldown_until = (now + cooldown).isoformat()
        else:
            logger.info("proactive_not_delivered", outcome=outcome.value)
    elif busy:
        outcome = ReachOutcome.SKIPPED_BUSY

    state.tick_count += 1
    state.last_tick_at = now.isoformat()
    state.egress_service_alive_at = now.isoformat()  # Task 6 field
    lm.state.commit(state)
    logger.info("proactive_tick", pressure=state.pressure, outcome=outcome.value)
    return outcome
```

> Note for implementer: `neuron.fire(now)` — confirm the exact neuron method name against `core/neuron.py`/`domain/signal.py` (the repo's `StubTimerNeuron`). If the neuron API differs, mirror exactly what `tick.run_tick` calls to produce signals; this loop MUST match `run_tick`'s accumulation so both brains agree.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_egress_service_tick.py -v`
Expected: PASS (4 passed). (Requires Task 6's `State.egress_service_alive_at` field — implement Task 6 first if `commit` rejects the attribute; see ordering note below.)

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/egress_service.py tests/test_egress_service_tick.py
git commit -m "feat(egress): delivery-aware run_proactive_tick (drain-on-delivered)"
```

> **Ordering:** Task 5's test sets/reads `State.egress_service_alive_at`, added in Task 6. Do Task 6 **before** Task 5's Step 4 (or land them together). The plan lists Task 6 next; an implementer may swap their order freely — they share one commit boundary only if done together.

---

### Task 6: Liveness watchdog — state field + cron defer

**Files:**
- Modify: `src/lifemodel/state/model.py` (add `egress_service_alive_at`)
- Modify: `src/lifemodel/tick.py` (defer when the in-process service is alive)
- Test: `tests/test_state_model.py` (extend), `tests/test_tick_defer.py` (new)

**Interfaces:**
- Produces: `State.egress_service_alive_at: str | None` (ISO-8601 UTC, default None). `tick.service_is_alive(state: State, *, now: datetime, max_age: timedelta = SERVICE_LIVENESS_MAX_AGE) -> bool`. `run_tick` returns `WakeDecision.stay_asleep()` **without mutating pressure** when `service_is_alive(...)` is True (the in-process service owns ticking; cron defers — spec §6 fallback).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tick_defer.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lifemodel.composition import build_lifemodel
from lifemodel.core.aggregator import ThresholdAggregator
from lifemodel.logging import get_logger
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeSignalBus, FakeStateStore
from lifemodel.tick import SERVICE_LIVENESS_MAX_AGE, run_tick, service_is_alive

_T0 = datetime(2026, 7, 4, 18, 0, tzinfo=timezone.utc)


def test_service_alive_when_stamp_fresh() -> None:
    st = State(egress_service_alive_at=(_T0 - timedelta(seconds=30)).isoformat())
    assert service_is_alive(st, now=_T0) is True


def test_service_dead_when_stamp_stale_or_absent() -> None:
    assert service_is_alive(State(), now=_T0) is False
    stale = State(egress_service_alive_at=(_T0 - SERVICE_LIVENESS_MAX_AGE - timedelta(seconds=1)).isoformat())
    assert service_is_alive(stale, now=_T0) is False


def test_run_tick_defers_and_does_not_touch_pressure_when_service_alive() -> None:
    fresh = State(pressure=28.0, egress_service_alive_at=(_T0 - timedelta(seconds=10)).isoformat())
    lm = build_lifemodel(
        base_dir=__import__("pathlib").Path("/unused"),
        state=FakeStateStore(fresh),
        bus=FakeSignalBus(),
        clock=FakeClock(_T0),
        aggregator=ThresholdAggregator(threshold=10.0),
        neurons=(),
    )
    decision = run_tick(lm, logger=get_logger("t"))
    assert decision.wake is False
    assert lm.state.load().pressure == 28.0  # NOT accumulated/drained while deferring
```

Extend `tests/test_state_model.py` with:

```python
def test_state_roundtrips_egress_service_alive_at() -> None:
    from lifemodel.state.model import State

    st = State(egress_service_alive_at="2026-07-04T18:00:00+00:00")
    assert State.from_dict(st.to_dict()).egress_service_alive_at == "2026-07-04T18:00:00+00:00"
    assert State().egress_service_alive_at is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tick_defer.py tests/test_state_model.py::test_state_roundtrips_egress_service_alive_at -v`
Expected: FAIL — `AttributeError`/`ImportError` (`egress_service_alive_at`, `service_is_alive`, `SERVICE_LIVENESS_MAX_AGE` undefined).

- [ ] **Step 3: Write minimal implementation**

In `src/lifemodel/state/model.py`, add the field to the `State` dataclass (after `cooldown_until`, keeping serialization order stable). If `from_dict` validates timestamps via `_as_opt_iso`, route the new field through the same tolerant path:

```python
    cooldown_until: str | None = None
    egress_service_alive_at: str | None = None  # in-proc egress-service liveness stamp (ISO-8601 UTC)
```
(In `from_dict`, add `egress_service_alive_at=_as_opt_iso(data.get("egress_service_alive_at"))` — match the existing per-field construction. `to_dict()` via `asdict` includes it automatically.)

In `src/lifemodel/tick.py`, add:

```python
from datetime import timedelta  # if not already imported

SERVICE_LIVENESS_MAX_AGE = timedelta(minutes=3)  # ~3× the 60s service interval


def service_is_alive(state: State, *, now: datetime, max_age: timedelta = SERVICE_LIVENESS_MAX_AGE) -> bool:
    """True if the in-process egress service stamped liveness within *max_age*."""
    stamp = state.egress_service_alive_at
    if stamp is None:
        return False
    try:
        return now - datetime.fromisoformat(stamp) <= max_age
    except ValueError:
        return False
```

In `run_tick(...)`, immediately after loading state (`state = lm.state.load()`) and computing `now`, add the defer guard **before** any pressure accumulation:

```python
    if service_is_alive(state, now=now):
        logger.info(EVENT_TICK, deferred="service_alive", pressure=state.pressure)
        return WakeDecision.stay_asleep()
```
(Use the existing `EVENT_TICK` import already present in tick.py. This returns before neurons run, so pressure is untouched — the in-process service is the sole brain while alive.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tick_defer.py tests/test_state_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/state/model.py src/lifemodel/tick.py tests/test_tick_defer.py tests/test_state_model.py
git commit -m "feat(egress): liveness stamp + cron defer to in-proc service"
```

---

### Task 7: Gateway-service primitive + supervised loop + ctx shim

**Files:**
- Modify: `src/lifemodel/gateway_core.py` (add `register_gateway_service`, `install_core_shim`)
- Modify: `src/lifemodel/egress_service.py` (add `proactive_service_loop`)
- Test: `tests/test_gateway_service.py`

**Interfaces:**
- Produces:
  - `register_gateway_service(runner: Any, key: str, coro_factory: Callable[[], Any], *, logger: EventLogger | None = None) -> bool` — spawns `coro_factory()` as a supervised task on `runner._gateway_loop`, tracks it in `runner._background_tasks` (so the gateway cancels it on shutdown), returns True on success, False fail-closed.
  - `install_core_shim(ctx: Any, *, logger: EventLogger | None = None) -> None` — best-effort monkey-patch of `inject_proactive_turn`/`register_gateway_service` onto `type(ctx)` as methods that resolve the runner via `default_runner_accessor` (so any plugin can call `ctx.inject_proactive_turn(...)`). Never raises.
  - `proactive_service_loop(*, build_lm: Callable[[], LifeModel], egress: ProactiveEgressPort, target, runner_accessor, logger, interval_seconds: float = 60.0, cooldown=DEFAULT_WAKE_COOLDOWN) -> Coroutine` — awaits `_running`, then each `interval_seconds` runs `run_proactive_tick` (busy = truthy `runner._running_agents`); exits when `_running` is False / `_draining` is True.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway_service.py
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lifemodel.gateway_core import install_core_shim, register_gateway_service


class _Runner:
    def __init__(self) -> None:
        self._gateway_loop = asyncio.get_event_loop()
        self._background_tasks: set[Any] = set()


def test_register_gateway_service_tracks_task_in_background_tasks() -> None:
    runner = _Runner()

    async def _svc() -> None:
        await asyncio.sleep(0)

    async def _run() -> None:
        ok = register_gateway_service(runner, "lifemodel-egress", lambda: _svc())
        assert ok is True
        assert len(runner._background_tasks) == 1
        await asyncio.sleep(0.01)  # let it finish

    asyncio.run(_run())


def test_register_gateway_service_fail_closed_without_loop() -> None:
    class Bad:
        pass

    assert register_gateway_service(Bad(), "k", lambda: None) is False


def test_install_core_shim_adds_methods_best_effort() -> None:
    class Ctx:
        pass

    ctx = Ctx()
    install_core_shim(ctx)  # must not raise
    assert hasattr(ctx, "inject_proactive_turn")
    assert hasattr(ctx, "register_gateway_service")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gateway_service.py -v`
Expected: FAIL — `ImportError` (`register_gateway_service`, `install_core_shim` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `src/lifemodel/gateway_core.py`:

```python
def _spawn_on_loop(loop: Any, coro: Any) -> Any:
    import asyncio

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop and running is not None:
        return loop.create_task(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop)


def register_gateway_service(
    runner: Any,
    key: str,
    coro_factory: Callable[[], Any],
    *,
    logger: EventLogger | None = None,
) -> bool:
    """Spawn a gateway-owned supervised task. Fail-closed."""
    log = logger or get_logger("lifemodel.service")
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None:
        log.info("gateway_service_unavailable", key=key, reason="no_loop")
        return False
    try:
        task = _spawn_on_loop(loop, coro_factory())
        bucket = getattr(runner, "_background_tasks", None)
        if isinstance(bucket, set):
            bucket.add(task)
            done = getattr(task, "add_done_callback", None)
            if callable(done):
                done(bucket.discard)
        log.info("gateway_service_started", key=key)
        return True
    except Exception as exc:  # noqa: BLE001 - fail-closed
        log.info("gateway_service_failed", key=key, error=f"{type(exc).__name__}: {exc}")
        return False


def install_core_shim(ctx: Any, *, logger: EventLogger | None = None) -> None:
    """Best-effort: expose the two primitives as PluginContext methods (reusable)."""
    log = logger or get_logger("lifemodel.shim")
    try:
        from .adapters.reachin import default_runner_accessor

        cls = type(ctx)

        def _ctx_inject(self: Any, target: Mapping[str, str | None], prompt: str, **kw: Any) -> ReachOutcome:
            return inject_proactive_turn(default_runner_accessor(), target, prompt, **kw)

        def _ctx_register(self: Any, key: str, coro_factory: Callable[[], Any], **kw: Any) -> bool:
            return register_gateway_service(default_runner_accessor(), key, coro_factory, **kw)

        if not hasattr(cls, "inject_proactive_turn"):
            cls.inject_proactive_turn = _ctx_inject  # type: ignore[attr-defined]
        if not hasattr(cls, "register_gateway_service"):
            cls.register_gateway_service = _ctx_register  # type: ignore[attr-defined]
        log.info("core_shim_installed", cls=cls.__name__)
    except Exception as exc:  # noqa: BLE001 - decorative; never block load
        log.info("core_shim_skipped", error=f"{type(exc).__name__}: {exc}")
```

Append to `src/lifemodel/egress_service.py`:

```python
async def proactive_service_loop(
    *,
    build_lm: Callable[[], LifeModel],
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    runner_accessor: Callable[[], Any | None],
    logger: EventLogger,
    interval_seconds: float = 60.0,
    cooldown: timedelta = DEFAULT_WAKE_COOLDOWN,
) -> None:
    """Supervised in-process brain: tick every interval until shutdown."""
    import asyncio

    # Wait for the gateway to finish starting (adapters wired, _running True).
    for _ in range(600):  # ~5 min cap; then proceed best-effort
        runner = runner_accessor()
        if runner is not None and getattr(runner, "_running", False):
            break
        if runner is not None and getattr(runner, "_draining", False):
            return
        await asyncio.sleep(0.5)

    logger.info("proactive_service_loop_started", interval=interval_seconds)
    while True:
        runner = runner_accessor()
        if runner is None or getattr(runner, "_draining", False) or not getattr(runner, "_running", False):
            logger.info("proactive_service_loop_stop")
            return
        busy = bool(getattr(runner, "_running_agents", None))
        try:
            run_proactive_tick(build_lm(), egress, target, logger=logger, cooldown=cooldown, busy=busy)
        except Exception as exc:  # noqa: BLE001 - a tick error must not kill the loop
            logger.info("proactive_tick_error", error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(interval_seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gateway_service.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/gateway_core.py src/lifemodel/egress_service.py tests/test_gateway_service.py
git commit -m "feat(egress): register_gateway_service + supervised loop + ctx shim"
```

---

### Task 8: Wire register(ctx) — choose brain, start service

**Files:**
- Modify: `src/lifemodel/__init__.py`
- Test: `tests/test_plugin_egress_wiring.py`

**Interfaces:**
- Consumes: everything above + `heartbeat._resolve_home_origin`, `register_heartbeat`, `build_lifemodel`, `state_dir`.
- Produces: updated `register(ctx)` that: installs the core shim; resolves the home origin; if reach-in is available (`reachin_available(default_runner_accessor())`) AND a home origin exists → builds `ReachInEgress` + starts `proactive_service_loop` via `register_gateway_service` (in-process brain); **also always** registers the cron heartbeat (it defers via the liveness stamp while the service is alive, and is the fallback brain otherwise). All best-effort; a failure in either path must not break plugin load. Add a factory `_build_egress_wiring(home, sdir, origin, logger) -> tuple[Callable[[], LifeModel], ProactiveEgressPort, Mapping]` for testability.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plugin_egress_wiring.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

import lifemodel


class FakeCtx:
    profile_name = "test-being"

    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}

    def register_command(self, name: str, handler: Callable[..., Any], description: str = "", args_hint: str = "") -> None:
        self.commands[name] = handler


def test_register_starts_service_when_reachin_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "115679831")

    started: list[str] = []
    monkeypatch.setattr(lifemodel, "reachin_available", lambda runner: True)
    monkeypatch.setattr(lifemodel, "default_runner_accessor", lambda: object())
    monkeypatch.setattr(
        lifemodel, "register_gateway_service",
        lambda runner, key, factory, **kw: started.append(key) or True,
    )
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: None)

    ctx = FakeCtx()
    lifemodel.register(ctx)  # must not raise
    assert "lifemodel" in ctx.commands
    assert started == ["lifemodel-egress"]


def test_register_falls_back_to_cron_when_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(lifemodel, "reachin_available", lambda runner: False)
    monkeypatch.setattr(lifemodel, "default_runner_accessor", lambda: None)

    started: list[str] = []
    heartbeat: list[bool] = []
    monkeypatch.setattr(lifemodel, "register_gateway_service", lambda *a, **k: started.append("x") or True)
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: heartbeat.append(True))

    ctx = FakeCtx()
    lifemodel.register(ctx)
    assert started == []          # service NOT started
    assert heartbeat == [True]    # cron fallback registered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plugin_egress_wiring.py -v`
Expected: FAIL — `AttributeError`/`ImportError` (the new names are not yet imported into `lifemodel.__init__`).

- [ ] **Step 3: Write minimal implementation**

Add imports at the top of `src/lifemodel/__init__.py` (module-level, so tests can monkeypatch them on `lifemodel`):

```python
from .adapters.reachin import ReachInEgress, default_runner_accessor
from .composition import build_lifemodel
from .egress_service import proactive_service_loop
from .gateway_core import install_core_shim, reachin_available, register_gateway_service
from .heartbeat import _resolve_home_origin, register_heartbeat
```

Add a wiring helper and extend `register(ctx)` — after the existing `logger = EventTee(...)` and `register_command(...)` block, replace the heartbeat `try/except` at the bottom with:

```python
    install_core_shim(ctx, logger=logger)

    origin = _resolve_home_origin()
    started = False
    if origin is not None and reachin_available(default_runner_accessor()):
        try:
            egress = ReachInEgress(runner_accessor=default_runner_accessor, logger=logger)

            def _factory() -> Any:
                return proactive_service_loop(
                    build_lm=lambda: build_lifemodel(base_dir=sdir, logger=logger),
                    egress=egress,
                    target=origin,
                    runner_accessor=default_runner_accessor,
                    logger=logger,
                )

            started = register_gateway_service(
                default_runner_accessor(), "lifemodel-egress", _factory, logger=logger
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; never break load
            logger.info("egress_service_wiring_skipped", error=f"{type(exc).__name__}: {exc}")

    # The cron heartbeat is ALWAYS registered: it defers (liveness stamp) while the
    # in-process service is alive, and is the fallback brain when it is not (spec §6).
    src_dir = Path(__file__).resolve().parent.parent
    try:
        register_heartbeat(home, src_dir, logger=logger)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.info("heartbeat_registration_skipped", error=f"{type(exc).__name__}: {exc}")

    logger.info("egress_wiring", service_started=started, has_origin=origin is not None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plugin_egress_wiring.py tests/test_plugin.py -v`
Expected: PASS (existing `test_plugin.py` still green — the `/lifemodel` command path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/lifemodel/__init__.py tests/test_plugin_egress_wiring.py
git commit -m "feat(egress): wire register(ctx) — in-proc service + cron fallback"
```

---

### Task 9: Guarded real-Hermes integration test

**Files:**
- Create: `tests/hermes_reachin_integration.py` (out-of-process driver, real Hermes — mirrors `tests/hermes_wake_delivery_integration.py`)
- Create: `tests/test_reachin_integration.py` (guarded wrapper — skips without Hermes, like `tests/test_wake_delivery_integration.py`)

**Interfaces:**
- Consumes: `tests/_hermes_probe.py::find_hermes_python`; `gateway_core.inject_proactive_turn` with the REAL default seams against a real (headless) `GatewayRunner`-shaped object or a faithful stub built from the installed Hermes types.

**Purpose:** the unit tests above use fakes; this asserts the primitive builds a real `MessageEvent(internal=True, message_id=None)`, resolves a source from a real `session_store` origin, and selects the right adapter — against the actual installed Hermes (`~/.hermes/hermes-agent`). It is SKIPPED when Hermes is absent (CI without Hermes), matching the repo's existing integration pattern.

- [ ] **Step 1: Write the guarded wrapper test**

```python
# tests/test_reachin_integration.py
from __future__ import annotations

import pytest

from tests._hermes_probe import find_hermes_python

pytestmark = pytest.mark.skipif(find_hermes_python() is None, reason="Hermes not installed")


def test_reachin_builds_real_message_event() -> None:
    # Driven out-of-process against real Hermes so our stdlib-only unit env stays clean.
    from tests.hermes_reachin_integration import run_reachin_probe

    result = run_reachin_probe()
    assert result["event_internal"] is True
    assert result["message_id"] is None
    assert result["outcome"] == "delivered"
```

- [ ] **Step 2: Run it (skips locally if Hermes absent)**

Run: `uv run pytest tests/test_reachin_integration.py -v`
Expected: SKIPPED (no Hermes on the unit runner) OR PASS (with Hermes). Either is acceptable — it must not FAIL/ERROR.

- [ ] **Step 3: Write the out-of-process driver**

Model it on `tests/hermes_wake_delivery_integration.py`: locate the Hermes python via `find_hermes_python()`, run a subprocess that imports `gateway.platforms.base`, constructs a fake runner exposing `_gateway_loop` (a real `asyncio` loop), `_build_process_event_source` (returns a real `SessionSource`), `adapters={Platform.TELEGRAM: <recording adapter>}`, `_running=True`, `_draining=False`, `_running_agents=set()`, then calls `inject_proactive_turn(...)` with the REAL default `make_event`/`schedule` and a recording adapter whose `handle_message` captures the event. Return JSON `{"event_internal": ..., "message_id": ..., "outcome": ...}` to stdout; parse it in `run_reachin_probe()`.

Keep the driver self-contained (single subprocess string like the existing driver). Assert the captured event has `internal is True`, `message_id is None`, `message_type == MessageType.TEXT`.

- [ ] **Step 4: Run the full gate**

Run: `make check`
Expected: all green (ruff format+lint, mypy --strict, pytest). The integration test SKIPS without Hermes.

- [ ] **Step 5: Commit**

```bash
git add tests/hermes_reachin_integration.py tests/test_reachin_integration.py
git commit -m "test(egress): guarded real-Hermes reach-in integration probe"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 `inject_proactive_turn` → Task 3 (+ real seams verified in Task 9). Adapter-profile rule, shutdown-gate, message_id=None all in Task 3. Busy-skip (interim substitute for fine-grained FIFO) in Task 4; FIFO itself is explicitly deferred to upstream (documented in spec §8).
- §3.2 `register_gateway_service` (runner-owned, supervised, cancellable, fail-closed) → Task 7. Lifecycle (`_running`/`_draining` guards, `_background_tasks` tracking) covered; start-after-_running via the loop's wait-for-_running preamble.
- §4 impulse-as-user-turn, single internal label → Task 2 + Task 5 (compose_impulse used in run_proactive_tick). Dual-text correctly NOT implemented (upstream).
- §5 guards: slash-command (Task 2 test asserts no leading `/`); memory/counters/hooks/onboarding/goal/sender-prefix are gateway-side effects of a persisted user turn — the interim mitigations are (a) the honest label and (b) `internal=True`; the remaining guards are flagged as upstream `turn_origin` work in the spec. **Gap acknowledged:** the interim does not add plugin-side guards for memory-retain/onboarding/goal-continuation (they require Hermes-side hooks we are not patching in the interim). This is a deliberate scope boundary — see "Deferred" below.
- §6 DeliveryPort/adapters + gateway-service brain + cron fallback → Tasks 4, 5, 7, 8. Liveness-watchdog fallback → Task 6.
- §7 monkey-patch fail-closed → version-guard (Task 3), fail-closed returns everywhere, `install_core_shim` best-effort (Task 7).
- §9 acceptance (clean assistant turn, continuity, labeled seed, fallback, service isolation, `make check`) → Tasks 3–9.

**2. Placeholder scan:** No "TBD/TODO". Two implementer notes flag verification-against-source (neuron API in Task 5; `_as_opt_iso` routing in Task 6) — these are "confirm the exact existing name", not missing content. Task 9 Step 3 describes the driver in prose rather than full code because it is an out-of-process subprocess string modeled on an existing repo file; the implementer copies that file's structure. Acceptable per the "follow established pattern" rule.

**3. Type consistency:** `ReachOutcome` values, `ProactiveEgressPort.reach_out(target, impulse)`, `inject_proactive_turn(runner, target, prompt, *, message_id, make_event, schedule, logger)`, `register_gateway_service(runner, key, coro_factory, *, logger)`, `run_proactive_tick(lm, egress, target, *, logger, cooldown, busy)`, `proactive_service_loop(*, build_lm, egress, target, runner_accessor, logger, interval_seconds, cooldown)` — names/signatures match across tasks 1→8. `target` is consistently the origin dict.

**Deferred (out of this plan's scope — see spec §8):** true dual-text; durable `turn_origin` metadata + the memory/onboarding/goal/sender-prefix guards that need Hermes-side hooks; fine-grained per-session FIFO. These belong to the upstream hermes-agent PR (a separate bead/brainstorm). This plan delivers a working, testable in-process proactive-egress path with a tested cron fallback.

---

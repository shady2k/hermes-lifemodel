# Turn Observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a *turn* (a Hermes LLM exchange) a first-class **traced** unit symmetric to the tick — a per-turn root span with a child span per `pre_llm_call` injector and per tool, into the same `observability.sqlite` — plus one shared per-injector metric and a shell-callable `python -m lifemodel.activity` reader over the durable stores.

**Architecture:** A process-lifetime **`TurnRecorder`** service (built once in `register()`, holding the already-acquired live `TraceWriter` + shared `StdlibTracer` + shared `MetricRegistry` + a bounded turn ledger) is threaded into every turn hook. A dedicated first `pre_llm_call` observer opens the turn root (keyed by the host's `(session_id, turn_id)`); each injector opens/closes its own child inside its fail-soft body; tools ride `pre/post_tool_call`; `post_llm_call` closes the root. Reads are a new `activity.py` module (+ `__main__` CLI), read-only over the durable stores. It is **not** an `ExecutionFrame` — it never takes the state-actor lock, never touches `State`.

**Tech Stack:** Python 3.11 **stdlib only** at runtime, **relative imports** in runtime code; tests use absolute imports (`lifemodel.…`). `uv`/`ruff`/`mypy --strict`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-07-18-turn-observability-design.md`. Task **lm-hg7**.

## Global Constraints

- **Runtime = stdlib only, RELATIVE imports** (`.core.turn_recorder`, `..ports.tracer`); tests absolute (`lifemodel.…`). Every task ends green: `make check` (ruff format --check, ruff check, mypy --strict -p lifemodel, pytest).
- **Fail-soft everywhere** — every tracing call is in its own `try/except`; instrumentation failure NEVER enters or replaces the injector/tool result path and NEVER crashes the host turn. The `TurnRecorder`'s public methods never raise (except `injector_span.__exit__` which re-propagates the BODY's exception after closing `failed`, so the injector's own `except` still runs).
- **Redaction (D10)** — span attrs carry ids / counts / outcomes / latency, **never** `content`, never the full `conversation_history` / prompt.
- **Same store, no new table** — turn spans go into `observability.sqlite` via `TraceSink.submit_span` with `tick=None` and `frame_kind=turn`. No `trace_correlations` row for an ordinary (reactive) turn (unresolved rows are protected from pruning forever).
- **`SpanStatus` is the closed set `{"ok","suppressed","failed"}`** (`ports/tracer.py:59`). A still-open root has `ended_at=None`/`status=None` (the reader renders it *incomplete*). A reconciled dead turn is closed `status="failed"` + attr `outcome="abandoned"` — do NOT invent a new status literal.
- **Metric labels are validated by KEY only, not VALUE** (`core/metrics.py`) — outcome strings come from the closed frozensets in `core/turn_metrics.py`; a typo silently forks a series.
- **Key = the host's `(session_id, turn_id)`** — both are passed to `pre_llm_call`, `post_llm_call`, and (as `tool_call_id`) the tool hooks. The injectors currently discard `turn_id` (`**_ignored`).

## File Structure

- **Create** `core/turn_metrics.py` — `TURN_INJECTOR_TOTAL` spec + closed component/outcome constants + `register_turn_metrics`.
- **Create** `core/turn_recorder.py` — the `TurnRecorder` service + `InjectorSpan` handle + `NULL_TURN_RECORDER`.
- **Create** `activity.py` — `activity_for_dir(base_dir, raw_args)` + `__main__` CLI (the reader).
- **Modify** `core/coreloop.py` — stamp `frame_kind="execution"` + `trigger` on the tick root span.
- **Modify** `core/tick_metrics.py` — **remove** `FELT_DISPLAY_TOTAL` (name + spec).
- **Modify** `hooks.py` — thread `recorder` into the four injector factories (wrap each body in `recorder.injector_span`, set the typed outcome at every return) + `make_post_llm_observer` (call `close_turn`); add `make_open_turn_observer` + the tool-hook handlers `make_tool_span_open`/`make_tool_span_close`.
- **Modify** `__init__.py` — build the `TurnRecorder` once; register the open-turn observer FIRST; thread the recorder into every injector, the post_llm observer, and register `pre_tool_call`/`post_tool_call`.
- **Tests:** create `tests/test_turn_metrics.py`, `tests/test_turn_recorder.py`, `tests/test_activity_view.py`, `tests/test_tool_spans.py`, `tests/test_turn_observability_harness.py`; extend `tests/test_coreloop.py`, each injector test, and the wiring test (`tests/test_plugin.py`).

---

## Task 1: Turn metric surface — `core/turn_metrics.py`

**Files:**
- Create: `core/turn_metrics.py`
- Test: `tests/test_turn_metrics.py`

**Interfaces:**
- Consumes: `MetricRegistry`, `MetricSpec` (`.metrics`).
- Produces: `TURN_INJECTOR_TOTAL: str`; `INJECTOR_{FELT,GENESIS,BELIEF,COMMITMENT}: str`; `{FELT,GENESIS,BELIEF,COMMITMENT}_OUTCOMES: frozenset[str]`; `register_turn_metrics(registry: MetricRegistry) -> None`.

- [ ] **Step 1: Write the failing test** — `tests/test_turn_metrics.py`:

```python
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_metrics import (
    BELIEF_OUTCOMES,
    COMMITMENT_OUTCOMES,
    FELT_OUTCOMES,
    GENESIS_OUTCOMES,
    TURN_INJECTOR_TOTAL,
    register_turn_metrics,
)


def test_register_is_idempotent_and_declares_component_outcome_labels():
    reg = MetricRegistry()
    register_turn_metrics(reg)
    register_turn_metrics(reg)  # a fresh graph / second recorder must not double-declare
    reg.inc(TURN_INJECTOR_TOTAL, component="belief", outcome="surfaced")
    metric = reg.get(TURN_INJECTOR_TOTAL)
    assert metric.value(component="belief", outcome="surfaced") == 1.0


def test_every_injector_outcome_set_carries_error_and_is_closed():
    for outcomes in (FELT_OUTCOMES, GENESIS_OUTCOMES, BELIEF_OUTCOMES, COMMITMENT_OUTCOMES):
        assert "error" in outcomes  # the fail-soft branch always has a home
    assert "light" in FELT_OUTCOMES and "surfaced" in BELIEF_OUTCOMES
    assert "injected" in GENESIS_OUTCOMES and "surfaced" in COMMITMENT_OUTCOMES
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: lifemodel.core.turn_metrics`):

Run: `uv run pytest tests/test_turn_metrics.py -q`

- [ ] **Step 3: Implement** — `core/turn_metrics.py`:

```python
"""Turn-hook metric surface (lm-hg7) — the per-turn injector outcome counter.

Symmetric to :mod:`core.tick_metrics` (the tick surface): the TURN path (the
``pre_llm_call`` injectors) gets ONE shared counter,
``lifemodel_turn_injector_total``, carrying the injector name on ``component`` and
its per-call verdict on ``outcome`` (both allowed keys in ``MetricSpec``'s closed
label set). It REPLACES felt-state's retired ``lifemodel_felt_display_total`` —
that was exactly this counter for one injector. Emitted once per injector
invocation by the :class:`~lifemodel.core.turn_recorder.TurnRecorder`'s
``injector_span`` close, fail-open.

The outcome strings are the SINGLE SOURCE (the registry validates label *keys*,
not *values*, so a typo silently forks a series): keep every emission's outcome in
the matching frozenset here.
"""

from __future__ import annotations

from .metrics import MetricRegistry, MetricSpec

TURN_INJECTOR_TOTAL = "lifemodel_turn_injector_total"

INJECTOR_FELT = "felt_state"
INJECTOR_GENESIS = "genesis"
INJECTOR_BELIEF = "belief"
INJECTOR_COMMITMENT = "commitment"

#: felt-state's per-call gate verdict (the retired FELT_DISPLAY_TOTAL vocabulary) + error.
FELT_OUTCOMES = frozenset(
    {"light", "not_warmed", "not_salient", "task", "cooldown_unchanged", "error"}
)
#: genesis's disjoint no-inject branches + the one inject + error.
GENESIS_OUTCOMES = frozenset(
    {"injected", "born", "carried_by_impulse", "own_impulse", "not_due", "stale_identity", "error"}
)
BELIEF_OUTCOMES = frozenset({"surfaced", "empty", "unavailable", "error"})
COMMITMENT_OUTCOMES = frozenset({"surfaced", "empty", "unavailable", "error"})

TURN_INJECTOR_SPEC = MetricSpec(
    name=TURN_INJECTOR_TOTAL,
    kind="counter",
    help="pre_llm_call injector invocations by injector (component) and per-call verdict (outcome).",
    label_keys=("component", "outcome"),
)


def register_turn_metrics(registry: MetricRegistry) -> None:
    """Declare the turn metric into *registry* (fail-fast on a bad spec, idempotent
    for the identical one — safe for a second recorder / a fresh graph to re-run)."""
    registry.register(TURN_INJECTOR_SPEC)
```

*(If `MetricRegistry` exposes the current value differently than `.get(...).value(...)`, match the accessor the existing `tests/test_tick_metrics.py` / `test_metrics.py` use — read one first.)*

- [ ] **Step 4: Run — expect PASS**. Run: `uv run pytest tests/test_turn_metrics.py -q`
- [ ] **Step 5: Commit**

```bash
git add core/turn_metrics.py tests/test_turn_metrics.py
git commit -m "feat(turn-obs): turn_injector_total metric + closed outcome vocab (lm-hg7)"
```

---

## Task 2: Stamp the tick root span with `frame_kind` + `trigger` — `core/coreloop.py`

The unified timeline must distinguish a tick from a turn. Today the tick root persists neither (`trigger` lives only in `TickReport`).

**Files:**
- Modify: `core/coreloop.py` (the root-span persist in `_run_tick`)
- Test: `tests/test_coreloop.py` (extend)

**Interfaces:**
- Produces (durable): the tick root span now carries attrs `frame_kind="execution"` and `trigger=<FrameTrigger value>`.

- [ ] **Step 1: Write the failing test** — extend `tests/test_coreloop.py` (adapt to the file's existing fake-writer capture helper):

```python
def test_tick_root_span_carries_frame_kind_and_trigger(...):
    # run one EVENT-triggered tick through the loop with a capturing TraceSink
    report = loop.tick(trigger=FrameTrigger.EVENT)
    root = _captured_root_span(writer)  # the span with parent_span_id is None
    assert root.attrs["frame_kind"] == "execution"
    assert root.attrs["trigger"] == "event"
```

- [ ] **Step 2: Run — expect FAIL** (`KeyError: 'frame_kind'`).
- [ ] **Step 3: Implement** — in `_run_tick`, stamp the root span before it is persisted. The root span is `root_span` (from `start_span(root, tick=tick_no, started_at=started)`, `coreloop.py:247`). Add, right after it is created:

```python
        root_span.set(frame_kind="execution", trigger=trigger.value)
```

Confirm the root span IS persisted (it is closed/persisted at end of `_run_tick`; if the current code does not `_persist_span(root_span, …)`, add that close+persist alongside the existing root bookkeeping so the attrs land). `frame_kind`/`trigger` are plain attrs on the existing `attrs_json`.

- [ ] **Step 4: Run — expect PASS.** Also `uv run pytest tests/test_coreloop.py tests/test_trace_store.py tests/test_trace_view.py -q`.
- [ ] **Step 5: Commit**

```bash
git add core/coreloop.py tests/test_coreloop.py
git commit -m "feat(turn-obs): stamp frame_kind+trigger on the tick root span (lm-hg7)"
```

---

## Task 3: `TurnRecorder` — construction + `ensure_turn` (root, ledger, reconcile, bounding)

**Files:**
- Create: `core/turn_recorder.py`
- Test: `tests/test_turn_recorder.py`

**Interfaces:**
- Consumes: `TracerPort`, `TraceContext` (`..ports.tracer`); `TraceSink` (`..state.trace_store`); `MetricRegistry` (`.metrics`); `ClockPort` (`..ports.clock` — match the existing import path); `to_iso` (`.timeutil`); `register_turn_metrics` (`.turn_metrics`).
- Produces:
  - `TurnRecorder(*, tracer, writer, metrics, clock, ledger_ttl_s=900.0, max_entries=256)`.
  - `ensure_turn(session_id: str, turn_id: str, *, model: str = "", platform: str = "", origin: str = "reactive", upstream_traceparent: str | None = None) -> None` — idempotent per key; persists the root; reconciles older-open turns for the session to `failed`/`abandoned`; bounded lazy cleanup.
  - Internal: `_ledger: dict[tuple[str, str], _Entry]` guarded by one `threading.Lock`; `_Entry(ctx: TraceContext, opened_at_iso: str, opened_mono: float, session_id, turn_id)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_turn_recorder.py`:

```python
import threading

from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.turn_recorder import TurnRecorder
from lifemodel.testing.fakes import FakeTracer, FakeClock  # match the repo's fake names


class CapturingSink:
    def __init__(self) -> None:
        self.spans: list[dict] = []

    def submit_span(self, **kw) -> bool:
        self.spans.append(kw)
        return True

    def submit_event(self, **kw) -> bool:
        return True

    def submit_correlation(self, **kw) -> bool:
        return True


def _recorder():
    return TurnRecorder(
        tracer=FakeTracer(), writer=CapturingSink(), metrics=MetricRegistry(), clock=FakeClock()
    )


def test_ensure_turn_persists_open_root_with_frame_kind_and_no_end():
    rec = _recorder()
    rec.ensure_turn("s1", "t1", model="opus", platform="telegram", origin="reactive")
    (root,) = [s for s in rec._writer.spans if s["component"] == "turn"]
    assert root["tick"] is None
    assert root["ended_at"] is None and root["status"] is None
    assert root["attrs"]["frame_kind"] == "turn"
    assert root["attrs"]["turn_id"] == "t1" and root["attrs"]["origin"] == "reactive"


def test_ensure_turn_is_idempotent_per_key():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.ensure_turn("s1", "t1")  # same turn — no second root
    assert len([s for s in rec._writer.spans if s["component"] == "turn"]) == 1


def test_a_new_turn_reconciles_the_prior_open_turn_of_the_same_session():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.ensure_turn("s1", "t2")  # t1 never closed → abandoned
    closed = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "failed"]
    assert len(closed) == 1
    assert closed[0]["attrs"]["turn_id"] == "t1"
    assert closed[0]["attrs"]["outcome"] == "abandoned"


def test_reactive_mints_fresh_trace_proactive_continues_upstream():
    rec = _recorder()
    rec.ensure_turn("s1", "t1", origin="reactive")
    rec.ensure_turn("s2", "t9", origin="proactive", upstream_traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")
    roots = {s["attrs"]["turn_id"]: s for s in rec._writer.spans if s["component"] == "turn" and s["ended_at"] is None}
    assert roots["t9"]["trace_id"] == "a" * 32  # continued the upstream trace id
    assert roots["t1"]["trace_id"] != "a" * 32


def test_ledger_is_bounded_and_never_raises_on_a_broken_sink():
    class BoomSink(CapturingSink):
        def submit_span(self, **kw):
            raise RuntimeError("disk gone")

    rec = TurnRecorder(tracer=FakeTracer(), writer=BoomSink(), metrics=MetricRegistry(), clock=FakeClock(), max_entries=2)
    for i in range(5):
        rec.ensure_turn("s1", f"t{i}")  # must not raise despite the sink blowing up
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`). Run: `uv run pytest tests/test_turn_recorder.py -q`
- [ ] **Step 3: Implement** — `core/turn_recorder.py` (this task: everything up to and including `ensure_turn`; later tasks add `injector_span`, tools, `close_turn`):

```python
"""``TurnRecorder`` — a turn is a first-class traced unit (lm-hg7).

Process-lifetime service (built once in ``register()``) that makes a Hermes turn
observable the way the tick already is: a per-turn ROOT span + a CHILD span per
``pre_llm_call`` injector and per tool, into the SAME ``observability.sqlite`` as
the tick, through the already-acquired live :class:`~lifemodel.state.trace_store.TraceWriter`.

It is deliberately NOT an :class:`~lifemodel.core.frame.ExecutionFrame`: a turn is
an asynchronous observability scope spanning host work across threads, so this
never takes the state-actor lock and never touches ``State``. It only writes to the
trace sink + metric registry + its own small in-memory ledger.

Every public method is fail-soft (a tracing hiccup must never crash the host turn).
The ledger is keyed by the host's ``(session_id, turn_id)`` and bounded (TTL + max
entries, lazy cleanup) — a restart simply loses it, and an open root left by a
crash/interrupt ages out or is reconciled ``failed`` by the next turn of the session.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..ports.clock import ClockPort
from ..ports.tracer import TraceContext, TracerPort
from ..state.trace_store import TraceSink
from .metrics import MetricRegistry
from .timeutil import to_iso
from .turn_metrics import TURN_INJECTOR_TOTAL, register_turn_metrics

_LOG = logging.getLogger("lifemodel.turn_recorder")

_TURN_ROOT_COMPONENT = "turn"


@dataclass
class _Entry:
    ctx: TraceContext
    opened_at_iso: str
    opened_mono: float
    session_id: str
    turn_id: str


class TurnRecorder:
    def __init__(
        self,
        *,
        tracer: TracerPort,
        writer: TraceSink,
        metrics: MetricRegistry,
        clock: ClockPort,
        ledger_ttl_s: float = 900.0,
        max_entries: int = 256,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._tracer = tracer
        self._writer = writer
        self._metrics = metrics
        self._clock = clock
        self._ttl_s = ledger_ttl_s
        self._max = max_entries
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._ledger: dict[tuple[str, str], _Entry] = {}
        # Declare the turn metric into the shared registry (idempotent) so injector
        # closes land on a real metric rather than failing open as unknown.
        try:
            register_turn_metrics(self._metrics)
        except Exception:  # noqa: BLE001 - a metric-declare hiccup must not sink the recorder
            _LOG.exception("turn metric registration failed")

    # --- the open door ------------------------------------------------------ #
    def ensure_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        model: str = "",
        platform: str = "",
        origin: str = "reactive",
        upstream_traceparent: str | None = None,
    ) -> None:
        try:
            key = (session_id, turn_id)
            now_iso = to_iso(self._clock.now())
            with self._lock:
                if key in self._ledger:  # idempotent — the first injector already opened it
                    return
                self._reconcile_session_locked(session_id, now_iso)
                self._evict_locked(now_iso)
                ctx = self._tracer.start_root(upstream_traceparent=upstream_traceparent)
                self._ledger[key] = _Entry(ctx, now_iso, self._monotonic(), session_id, turn_id)
            # Persist OUTSIDE the lock (the sink is thread-safe + async): eager, with
            # ended_at/status NULL, so a crash still leaves a discoverable parent.
            self._submit(
                ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=now_iso,
                ended_at=None,
                status=None,
                attrs={
                    "frame_kind": "turn",
                    "turn_id": turn_id,
                    "session_id": session_id,
                    "origin": origin,
                    "model": model,
                    "platform": platform,
                },
            )
        except Exception:  # noqa: BLE001 - opening a turn trace must never crash the host turn
            _LOG.exception("ensure_turn failed session=%s turn=%s", session_id, turn_id)

    # --- internals ---------------------------------------------------------- #
    def _reconcile_session_locked(self, session_id: str, now_iso: str) -> None:
        """Close any OTHER still-open turn of this session as failed/abandoned — the
        next turn is the only reliable signal a prior one died (post_llm_call is not
        guaranteed). Caller holds the lock; the submit is best-effort after."""
        dead = [e for (sid, _), e in self._ledger.items() if sid == session_id]
        for entry in dead:
            self._ledger.pop((entry.session_id, entry.turn_id), None)
            self._submit(
                entry.ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=entry.opened_at_iso,
                ended_at=now_iso,
                status="failed",
                attrs={"frame_kind": "turn", "turn_id": entry.turn_id, "outcome": "abandoned"},
            )

    def _evict_locked(self, now_iso: str) -> None:
        # TTL first (age out silent leaks), then cap the map. Lazy — no sweeper thread.
        cutoff = self._monotonic() - self._ttl_s
        for key in [k for k, e in self._ledger.items() if e.opened_mono < cutoff]:
            self._ledger.pop(key, None)
        while len(self._ledger) > self._max:
            oldest = min(self._ledger, key=lambda k: self._ledger[k].opened_mono)
            self._ledger.pop(oldest, None)

    def _submit(self, ctx: TraceContext, **kw: Any) -> None:
        try:
            self._writer.submit_span(
                trace_id=ctx.trace_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                tick=None,
                **kw,
            )
        except Exception:  # noqa: BLE001 - fail-open, exactly like the tick's _persist_span
            _LOG.exception("submit_span failed")

    def _child_span_id(self, parent: TraceContext) -> TraceContext:
        return self._tracer.child_of(parent)
```

*(Note: `FakeTracer.start_root` must honour `upstream_traceparent` — continue the parsed trace_id. If the repo's `FakeTracer` ignores it, extend it minimally in this task; check `testing/fakes.py` first.)*

- [ ] **Step 4: Run — expect PASS.** Run: `uv run pytest tests/test_turn_recorder.py -q`
- [ ] **Step 5: Commit**

```bash
git add core/turn_recorder.py tests/test_turn_recorder.py testing/fakes.py
git commit -m "feat(turn-obs): TurnRecorder.ensure_turn — open root, reconcile, bound (lm-hg7)"
```

---

## Task 4: `TurnRecorder.injector_span` — child span + metric

**Files:**
- Modify: `core/turn_recorder.py`
- Test: `tests/test_turn_recorder.py` (extend)

**Interfaces:**
- Produces:
  - `injector_span(session_id, turn_id, component) -> ContextManager[InjectorSpan]`.
  - `InjectorSpan.set(*, outcome: str, **attrs: Any) -> None` — records the verdict; the last set wins.
  - On clean `__exit__`: persist a `turn.injector.<component>` child (`status="ok"`) + `metrics.inc(TURN_INJECTOR_TOTAL, component=<component>, outcome=<outcome or "unknown">)`.
  - On `__exit__` WITH a body exception: persist the child `status="failed"` + `metrics.inc(..., outcome="error")`, then **return False** (re-propagate — the injector's own `except` runs). If there is no open turn for the key, the span degrades to a bare parentless child (best-effort), never a crash.

- [ ] **Step 1: Write the failing tests** — extend `tests/test_turn_recorder.py`:

```python
import pytest

from lifemodel.core.turn_metrics import TURN_INJECTOR_TOTAL


def test_injector_span_success_persists_ok_child_and_increments_outcome():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    with rec.injector_span("s1", "t1", "belief") as span:
        span.set(outcome="surfaced", count=2, ids=["belief:ab", "belief:cd"])
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"][0]
    assert child["status"] == "ok" and child["attrs"]["outcome"] == "surfaced"
    assert child["attrs"]["count"] == 2
    assert rec._metrics.get(TURN_INJECTOR_TOTAL).value(component="belief", outcome="surfaced") == 1.0


def test_injector_span_reraises_and_marks_failed_with_error_outcome():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    with pytest.raises(RuntimeError):
        with rec.injector_span("s1", "t1", "belief") as span:
            span.set(outcome="surfaced")  # then the body blows up before completing
            raise RuntimeError("boom")
    child = [s for s in rec._writer.spans if s["component"] == "turn.injector.belief"][0]
    assert child["status"] == "failed" and child["attrs"]["outcome"] == "error"
    assert rec._metrics.get(TURN_INJECTOR_TOTAL).value(component="belief", outcome="error") == 1.0
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: injector_span`).
- [ ] **Step 3: Implement** — add to `core/turn_recorder.py`:

```python
from collections.abc import Iterator
from contextlib import contextmanager


class InjectorSpan:
    """The mutable handle an injector stamps its verdict onto (set-then-close)."""

    __slots__ = ("_outcome", "_attrs")

    def __init__(self) -> None:
        self._outcome: str = "unknown"
        self._attrs: dict[str, Any] = {}

    def set(self, *, outcome: str, **attrs: Any) -> None:
        self._outcome = outcome
        self._attrs.update(attrs)
```

and the method on `TurnRecorder`:

```python
    @contextmanager
    def injector_span(self, session_id: str, turn_id: str, component: str) -> Iterator[InjectorSpan]:
        parent = self._ledger_ctx(session_id, turn_id)  # None if no open turn
        child = self._child_span_id(parent) if parent is not None else self._tracer.start_root()
        span = InjectorSpan()
        started = to_iso(self._clock.now())
        comp = f"turn.injector.{component}"
        try:
            yield span
        except Exception:
            self._close_injector(child, comp, started, status="failed", outcome="error", span=span)
            raise  # re-propagate so the injector's own fail-soft except runs
        else:
            self._close_injector(child, comp, started, status="ok", outcome=span._outcome, span=span)

    def _ledger_ctx(self, session_id: str, turn_id: str) -> TraceContext | None:
        with self._lock:
            entry = self._ledger.get((session_id, turn_id))
            return entry.ctx if entry is not None else None

    def _close_injector(
        self, ctx: TraceContext, comp: str, started: str, *, status: str, outcome: str, span: InjectorSpan
    ) -> None:
        try:
            self._submit(
                ctx,
                component=comp,
                started_at=started,
                ended_at=to_iso(self._clock.now()),
                status=status,
                attrs={"outcome": outcome, **span._attrs},
            )
            self._metrics.inc(TURN_INJECTOR_TOTAL, component=comp.rsplit(".", 1)[-1], outcome=outcome)
        except Exception:  # noqa: BLE001 - never let the tracing close mask/replace the body
            _LOG.exception("injector span close failed component=%s", comp)
```

- [ ] **Step 4: Run — expect PASS.** Run: `uv run pytest tests/test_turn_recorder.py -q`
- [ ] **Step 5: Commit**

```bash
git add core/turn_recorder.py tests/test_turn_recorder.py
git commit -m "feat(turn-obs): injector_span child span + turn_injector_total emit (lm-hg7)"
```

---

## Task 5: `TurnRecorder` tool spans — `tool_open` / `tool_close` (keyed by `tool_call_id`)

**Files:**
- Modify: `core/turn_recorder.py`
- Test: `tests/test_turn_recorder.py` (extend)

**Interfaces:**
- Produces:
  - `tool_open(session_id, turn_id, *, tool: str, tool_call_id: str) -> None` — stash an open child keyed by `tool_call_id` (separate small map, lock-guarded).
  - `tool_close(tool_call_id, *, status: str = "ok", **attrs: Any) -> None` — persist `turn.tool.<tool>` child (`started_at` from open, `ended_at` now); drop the entry. Unknown `tool_call_id` → best-effort no-op.

- [ ] **Step 1: Write the failing test** — extend `tests/test_turn_recorder.py`:

```python
def test_tool_open_close_persists_child_keyed_by_call_id():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.tool_open("s1", "t1", tool="commitment", tool_call_id="call_7")
    rec.tool_open("s1", "t1", tool="check_in", tool_call_id="call_8")  # concurrent, distinct id
    rec.tool_close("call_7", status="ok", action="discharge")
    child = [s for s in rec._writer.spans if s["component"] == "turn.tool.commitment"][0]
    assert child["status"] == "ok" and child["attrs"]["action"] == "discharge"
    rec.tool_close("nope")  # unknown id — best-effort no-op, no raise
```

- [ ] **Step 2: Run — expect FAIL**.
- [ ] **Step 3: Implement** — add a `self._tools: dict[str, tuple[TraceContext, str, str]] = {}` (call_id → (child_ctx, tool, started_iso)) in `__init__` under the same lock, plus:

```python
    def tool_open(self, session_id: str, turn_id: str, *, tool: str, tool_call_id: str) -> None:
        try:
            parent = self._ledger_ctx(session_id, turn_id)
            child = self._child_span_id(parent) if parent is not None else self._tracer.start_root()
            with self._lock:
                self._tools[tool_call_id] = (child, tool, to_iso(self._clock.now()))
        except Exception:  # noqa: BLE001
            _LOG.exception("tool_open failed tool=%s", tool)

    def tool_close(self, tool_call_id: str, *, status: str = "ok", **attrs: Any) -> None:
        try:
            with self._lock:
                found = self._tools.pop(tool_call_id, None)
            if found is None:
                return
            child, tool, started = found
            self._submit(
                child,
                component=f"turn.tool.{tool}",
                started_at=started,
                ended_at=to_iso(self._clock.now()),
                status=status if status in ("ok", "suppressed", "failed") else "ok",
                attrs=dict(attrs),
            )
        except Exception:  # noqa: BLE001
            _LOG.exception("tool_close failed id=%s", tool_call_id)
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit**

```bash
git add core/turn_recorder.py tests/test_turn_recorder.py
git commit -m "feat(turn-obs): tool spans keyed by tool_call_id (lm-hg7)"
```

---

## Task 6: `TurnRecorder.close_turn` — completion child + close the root

**Files:**
- Modify: `core/turn_recorder.py`
- Test: `tests/test_turn_recorder.py` (extend)

**Interfaces:**
- Produces: `close_turn(session_id, turn_id, *, final_output: str = "", reasoning: str = "", status: str = "ok") -> None` — write a `turn.completion` child (bounded `final_output`/`reasoning`, truncated) + close the root span (`ended_at` now, `status`); drop the ledger entry. Idempotent / unknown key → best-effort no-op.

- [ ] **Step 1: Write the failing test** — extend `tests/test_turn_recorder.py`:

```python
def test_close_turn_writes_completion_and_closes_root():
    rec = _recorder()
    rec.ensure_turn("s1", "t1")
    rec.close_turn("s1", "t1", final_output="ok, talk soon", reasoning="short and warm")
    completion = [s for s in rec._writer.spans if s["component"] == "turn.completion"][0]
    assert "talk soon" in completion["attrs"]["final_output"]
    closed_root = [s for s in rec._writer.spans if s["component"] == "turn" and s["status"] == "ok"]
    assert closed_root and closed_root[-1]["ended_at"] is not None
    assert ("s1", "t1") not in rec._ledger  # entry removed
    rec.close_turn("s1", "t1")  # second close — no raise, no duplicate root close
```

- [ ] **Step 2: Run — expect FAIL**.
- [ ] **Step 3: Implement** — add (with a module const `_MAX_TEXT = 4000`):

```python
    def close_turn(
        self, session_id: str, turn_id: str, *, final_output: str = "", reasoning: str = "", status: str = "ok"
    ) -> None:
        try:
            key = (session_id, turn_id)
            with self._lock:
                entry = self._ledger.pop(key, None)
            if entry is None:
                return
            child = self._child_span_id(entry.ctx)
            now_iso = to_iso(self._clock.now())
            self._submit(
                child,
                component="turn.completion",
                started_at=now_iso,
                ended_at=now_iso,
                status="ok",
                attrs={"final_output": final_output[:_MAX_TEXT], "reasoning": reasoning[:_MAX_TEXT]},
            )
            self._submit(
                entry.ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=entry.opened_at_iso,
                ended_at=now_iso,
                status=status if status in ("ok", "suppressed", "failed") else "ok",
                attrs={"frame_kind": "turn", "turn_id": turn_id, "session_id": session_id},
            )
        except Exception:  # noqa: BLE001
            _LOG.exception("close_turn failed session=%s turn=%s", session_id, turn_id)
```

Add `NULL_TURN_RECORDER = TurnRecorder(tracer=StdlibTracer(), writer=NULL_TRACE_SINK, metrics=MetricRegistry(), clock=SystemClock())` at module end **only if** a null default is needed by callers — otherwise the hooks accept `TurnRecorder | None`. (Decide in Task 7; prefer `| None` + a nullcontext helper to avoid importing adapters into core.)

- [ ] **Step 4: Run — expect PASS.** Run: `uv run pytest tests/test_turn_recorder.py -q && make check`
- [ ] **Step 5: Commit**

```bash
git add core/turn_recorder.py tests/test_turn_recorder.py
git commit -m "feat(turn-obs): close_turn — completion child + root close (lm-hg7)"
```

---

## Task 7: Instrument the felt-state injector + retire `FELT_DISPLAY_TOTAL`

**Files:**
- Modify: `hooks.py` (`make_felt_state_injector`), `core/tick_metrics.py` (remove `FELT_DISPLAY_TOTAL`)
- Test: extend the felt injector test (find it: `tests/test_felt*`/`tests/test_reactive_felt*`); grep for every `FELT_DISPLAY_TOTAL` reference and update.

**Interfaces:**
- Consumes: `TurnRecorder` (`.core.turn_recorder`), `INJECTOR_FELT` (`.core.turn_metrics`).
- Produces: `make_felt_state_injector(..., recorder: TurnRecorder | None = None)`; each invocation opens a `turn.injector.felt_state` child and emits `turn_injector_total{component=felt_state, outcome=<decision.value>}`. `FELT_DISPLAY_TOTAL` no longer exists.

- [ ] **Step 1: Write/adjust the failing test** — the felt injector must now increment the unified metric (not `FELT_DISPLAY_TOTAL`), threading a real `TurnRecorder` with a capturing sink + shared registry:

```python
def test_felt_injector_emits_turn_injector_total_and_opens_a_child(...):
    reg = MetricRegistry()
    rec = TurnRecorder(tracer=FakeTracer(), writer=sink, metrics=reg, clock=FakeClock())
    rec.ensure_turn("s1", "t1")
    injector = make_felt_state_injector(build_lm, recorder=rec, metrics=reg)
    injector(session_id="s1", turn_id="t1", user_message="hi", conversation_history=[])
    assert reg.get(TURN_INJECTOR_TOTAL).value(component="felt_state", outcome="not_warmed") >= 1.0
    assert any(s["component"] == "turn.injector.felt_state" for s in sink.spans)
```

- [ ] **Step 2: Run — expect FAIL**.
- [ ] **Step 3: Implement**
  1. In `core/tick_metrics.py`: delete the `FELT_DISPLAY_TOTAL = "..."` constant (lines ~49-55) AND its `MetricSpec(name=FELT_DISPLAY_TOTAL, …)` entry in `UNIVERSAL_SPECS`.
  2. In `hooks.py` `make_felt_state_injector`: add `recorder: TurnRecorder | None = None` to the signature; add `turn_id: str = ""` to `_injector`'s kwargs; wrap the body in the injector span and replace the metric line. The single verdict is `decision.value`:

```python
    def _injector(self, *, session_id="", turn_id="", user_message="", conversation_history=None, **_ignored):
        try:
            with _injector_span(recorder, session_id, turn_id, INJECTOR_FELT) as span:
                lm = build_lm()
                ...
                decision = decide(state, turn, params, now)
                span.set(outcome=decision.value, shows=decision.shows, notice=notice is not None)
                # (delete: metrics.inc(FELT_DISPLAY_TOTAL, outcome=decision.value))
                if decision.shows:
                    ...
                return {"context": "\n\n".join(blocks)} if blocks else None
        except Exception as exc:
            _record_observer_failure(observer_name=PRE_LLM_OBSERVER, exc=exc, health=health, metrics=metrics)
            return None
```

  3. Add ONE shared helper at the top of `hooks.py` so every injector wraps uniformly and stays green when `recorder is None`:

```python
from contextlib import contextmanager, nullcontext

@contextmanager
def _injector_span(recorder, session_id, turn_id, component):
    if recorder is None:
        yield _NullInjectorSpan()
        return
    with recorder.injector_span(session_id, turn_id, component) as span:
        yield span

class _NullInjectorSpan:
    def set(self, **_): ...
```

- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/ -k "felt or metric" -q && make check`
- [ ] **Step 5: Commit**

```bash
git add hooks.py core/tick_metrics.py tests/
git commit -m "feat(turn-obs): felt injector -> turn_injector_total; retire FELT_DISPLAY_TOTAL (lm-hg7)"
```

---

## Task 8: Instrument the genesis injector

**Files:** Modify `hooks.py` (`make_genesis_injector`); Test: extend `tests/test_genesis*`.

**Interface:** `recorder: TurnRecorder | None = None`; outcome per branch:

| branch (in `_injector`) | `span.set(outcome=…)` |
|---|---|
| `state.genesis_completed_at is not None` → return None | `"born"` |
| `GENESIS_TAG in user_message` (stamp shown) → return None | `"carried_by_impulse"` |
| `_is_own_impulse(...)` → return None | `"own_impulse"` |
| `not should_launch(...)` → return None | `"not_due"` |
| `identity_stale()` → return None | `"stale_identity"` |
| else (block, stamp shown) → return {"context": …} | `"injected"` |

- [ ] **Step 1: Failing test** — e.g. an unborn, should-launch case asserts `turn_injector_total{component=genesis, outcome=injected}` and a `turn.injector.genesis` child; a born case asserts `outcome=born`.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — add `recorder`/`turn_id`; wrap the body in `_injector_span(recorder, session_id, turn_id, INJECTOR_GENESIS)`; call `span.set(outcome=…)` immediately before EACH `return`.
- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/ -k genesis -q`
- [ ] **Step 5: Commit** `feat(turn-obs): genesis injector child span + outcome (lm-hg7)`

---

## Task 9: Instrument the belief injector

**Files:** Modify `hooks.py` (`make_belief_injector`); Test: extend `tests/test_belief_injector.py`.

**Interface:** `recorder: TurnRecorder | None = None`; outcome per branch:

| branch | `span.set(outcome=…)` |
|---|---|
| `memory is None` → return None | `"unavailable"` |
| `not beliefs` → return None | `"empty"` |
| else (block, stamp) → return {"context": …} | `"surfaced"` (+ `count`, `ids`) |

- [ ] **Step 1: Failing test** — a store with surfaceable beliefs asserts `turn_injector_total{component=belief, outcome=surfaced}` + a `turn.injector.belief` child carrying `count`/`ids` (never content); an empty store asserts `outcome=empty`.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — add `recorder`/`turn_id`; wrap the body; set the outcome before each return (fold the existing `ids`/`count` into `span.set(outcome="surfaced", count=len(ids), ids=ids)`).
- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/ -k belief -q`
- [ ] **Step 5: Commit** `feat(turn-obs): belief injector child span + outcome (lm-hg7)`

---

## Task 10: Instrument the commitment injector (keep overflow)

**Files:** Modify `hooks.py` (`make_commitment_injector`); Test: extend `tests/test_commitment_injector.py`.

**Interface:** `recorder: TurnRecorder | None = None`; outcome per branch (`COMMITMENT_INJECTOR_OVERFLOW` stays — overflow is orthogonal to outcome):

| branch | `span.set(outcome=…)` |
|---|---|
| `memory is None` → return None | `"unavailable"` |
| `not fetched` → return None | `"empty"` |
| else (block; overflow metric unchanged) → return {"context": …} | `"surfaced"` (+ `count`, `ids`, `overflow`) |

- [ ] **Step 1: Failing test** — active commitments assert `turn_injector_total{component=commitment, outcome=surfaced}` + `turn.injector.commitment` child; over-cap still increments `COMMITMENT_INJECTOR_OVERFLOW` AND sets `overflow=True` on the child.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — as Task 9, keeping the existing `metrics.inc(COMMITMENT_INJECTOR_OVERFLOW)` line untouched.
- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/ -k commitment -q`
- [ ] **Step 5: Commit** `feat(turn-obs): commitment injector child span + outcome (lm-hg7)`

---

## Task 11: Wire it live — `register()` builds the recorder + all turn hooks

**Files:**
- Modify: `hooks.py` (add `make_open_turn_observer`, `make_tool_span_open`, `make_tool_span_close`; thread `recorder` into `make_post_llm_observer` → `close_turn`)
- Modify: `__init__.py` (`register`)
- Test: `tests/test_tool_spans.py` (new) + extend `tests/test_plugin.py` (wiring)

**Interfaces:**
- Consumes: `_outcome_writer` (already acquired, `__init__.py:574`), the shared `tracer`/`metrics` from `build_lifemodel`, `SystemClock`.
- Produces: one `TurnRecorder` instance threaded into: the open-turn observer (registered FIRST on `pre_llm_call`), all four injectors, the `post_llm_call` observer (close), and `pre_tool_call`/`post_tool_call`.

- [ ] **Step 1: Write the failing tests**
  - `tests/test_tool_spans.py`: `make_tool_span_open`/`_close` handlers, given a recorder, open+close a `turn.tool.<tool>` child on the host's `pre/post_tool_call` kwargs (`session_id`, `turn_id`, `tool_call_id`, `tool_name`, `status`, `duration_ms`).
  - `tests/test_plugin.py` (extend): after `register(ctx)`, a `pre_tool_call` and a `post_tool_call` hook are registered, the FIRST `pre_llm_call` callback is the open-turn observer, and a full fake turn (open → felt injector → tool → post_llm close) leaves a `turn` root + `turn.injector.felt_state` + `turn.tool.*` + `turn.completion` in the capturing sink, all under ONE `trace_id`. A raise inside the recorder does not crash the injector (still returns its context/None).

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**
  1. `hooks.py`:

```python
def make_open_turn_observer(recorder: TurnRecorder) -> Callable[..., None]:
    """The FIRST pre_llm_call callback: open the turn root before the injectors run.
    Returns None (injects no context). Distinguishes proactive from reactive by the
    presence of our own impulse marker / a proactive origin traceparent kwarg."""
    def _open(*, session_id="", turn_id="", user_message="", **kw):
        origin = "proactive" if _is_own_impulse(user_message) else "reactive"
        recorder.ensure_turn(
            session_id, turn_id,
            model=str(kw.get("model", "")), platform=str(kw.get("platform", "")),
            origin=origin,
            upstream_traceparent=kw.get("proactive_origin_traceparent"),  # None for reactive
        )
        return None
    return _open


def make_tool_span_open(recorder: TurnRecorder) -> Callable[..., None]:
    def _pre(*, session_id="", turn_id="", tool_name="", tool_call_id="", **_kw):
        recorder.tool_open(session_id, turn_id, tool=tool_name, tool_call_id=tool_call_id)
        return None
    return _pre


def make_tool_span_close(recorder: TurnRecorder) -> Callable[..., None]:
    def _post(*, tool_call_id="", status="ok", duration_ms=None, **_kw):
        recorder.tool_close(tool_call_id, status=str(status), duration_ms=duration_ms)
        return None
    return _post
```

Verify the exact `pre/post_tool_call` kwarg names against `~/.hermes/hermes-agent/model_tools.py:974`/`:1176` and adjust (`tool_name` vs `tool`, the status field). In `make_post_llm_observer`, add `recorder: TurnRecorder | None = None` and, after the existing appraisal/proactive work, call `recorder.close_turn(session_id, turn_id, final_output=<final text>, reasoning=<reasoning>)` — reuse whatever `_log_proactive_reasoning` already reads for the text/reasoning (spec open-Q #3); if the reactive reasoning isn't exposed, pass `final_output` only.

  2. `__init__.py` `register()`: right after `_outcome_writer` is acquired (`:574`), build the recorder and register the open observer BEFORE the felt injector (`:626`). Thread `recorder=_turn_recorder` into every `make_*_injector(...)` call and into `make_post_llm_observer(...)`:

```python
    _turn_lm = build_lifemodel(base_dir=sdir, trace_writer=_outcome_writer)
    _turn_recorder = TurnRecorder(
        tracer=_turn_lm.tracer, writer=_outcome_writer, metrics=_turn_lm.metrics, clock=SystemClock()
    )
    with wire("open_turn_observer", required=True, health=health, logger=_LOG):
        ctx.register_hook("pre_llm_call", make_open_turn_observer(_turn_recorder))
    with wire("tool_span_hooks", required=True, health=health, logger=_LOG):
        ctx.register_hook("pre_tool_call", make_tool_span_open(_turn_recorder))
        ctx.register_hook("post_tool_call", make_tool_span_close(_turn_recorder))
```

(`_turn_lm.tracer`/`.metrics` are the shared instances; the writer is the acquired one.)

- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/test_tool_spans.py tests/test_plugin.py -q && make check`
- [ ] **Step 5: Commit** `feat(turn-obs): wire TurnRecorder — open observer, tool hooks, close on post_llm (lm-hg7)`

---

## Task 12: The reader — `activity.py` + `python -m lifemodel.activity`

**Files:**
- Create: `activity.py`
- Test: `tests/test_activity_view.py`

**Interfaces:**
- Consumes: `observability_db_path` (`.state.trace_store`), read-only sqlite; `render_dump_for_dir` (`.debug`) for the state header; `local_time` (`.debug`); `build_why_graph`/`display_id` reuse optional.
- Produces:
  - `activity_for_dir(base_dir: Path, raw_args: str) -> str` — `""`/`"last [N]"` → the timeline (state header + interleaved tick/turn units newest-first, filtered by `frame_kind`); `"turn <trace_id>"` → that turn's child tree with ids enriched from `lifemodel.sqlite`; anything else → a usage line.
  - `__main__`: `python3 -m lifemodel.activity [last N | turn <trace_id>]` resolving `base_dir` to `~/.hermes/workspace/lifemodel` by default (overridable via `LIFEMODEL_BASE_DIR`).

- [ ] **Step 1: Write the failing tests** — `tests/test_activity_view.py` seeds a temp `observability.sqlite` (reuse the `test_trace_view`/`test_trace_store` fixtures) with: one execution root (`frame_kind=execution`), one completed turn root (`frame_kind=turn`, `ended_at` set) + its injector/tool/completion children, and one OPEN turn root (`ended_at=None`).

```python
def test_timeline_interleaves_and_labels_frame_kind_newest_first(tmp_path):
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "turn" in out and "execution" in out
    # a turn line is present with its outcome summary, not drowned/omitted

def test_open_turn_renders_incomplete_not_success(tmp_path):
    _seed_open_turn(tmp_path)
    out = activity_for_dir(tmp_path, "last 10")
    assert "incomplete" in out.lower()

def test_turn_detail_shows_child_tree(tmp_path):
    _seed(tmp_path)
    out = activity_for_dir(tmp_path, f"turn {SEEDED_TURN_TRACE_ID}")
    assert "turn.injector.belief" in out and "turn.completion" in out

def test_reader_tolerates_old_span_without_frame_kind(tmp_path):
    _seed_legacy_tick_without_frame_kind(tmp_path)
    activity_for_dir(tmp_path, "last 10")  # no crash, renders it as execution/unknown
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — `activity.py`: open the db `?mode=ro`; a timeline query that selects ROOT rows (`parent_span_id IS NULL`) ordered by `started_at DESC LIMIT N`, labels each by `attrs_json->>'frame_kind'` (default `execution` when absent, so old rows don't crash), renders one line each (turn: `origin`, injector outcomes summary, `incomplete` when `ended_at IS NULL`; tick: `trigger`, ran/suppressed counts); a `turn <trace_id>` query that selects all rows for the trace and renders the parent→child tree (reuse `trace_view`'s tree renderer if cleanly importable), enriching `belief:`/`commitment:` ids by a read against `lifemodel.sqlite`. Prepend `render_dump_for_dir(base_dir)`'s state block. Everything fail-soft (a missing/locked store → a friendly line, never a crash), mirroring `stats_view._safe_now`/`_safe_window`.

- [ ] **Step 4: Run — expect PASS.** `uv run pytest tests/test_activity_view.py -q && make check`. Then a live smoke: `cd <repo> && python3 -m lifemodel.activity last 10` against the live being prints the state header + tick lines (turns appear only post-deploy).
- [ ] **Step 5: Commit** `feat(turn-obs): activity reader + python -m lifemodel.activity CLI (lm-hg7)`

---

## Task 13: Real-code observability harness

**Files:**
- Create: `tests/test_turn_observability_harness.py`

Drive a reactive turn through the REAL injectors + a real tool via a live-shaped `TurnRecorder` (real `StdlibTracer`, a real `TraceWriter` on a temp `observability.sqlite`, a shared `MetricRegistry`), then READ the durable store back exactly the way the agent will — assert the turn root + `turn.injector.*` children + a `turn.tool.*` child + `turn.completion` all landed under one `trace_id`, `activity_for_dir(tmp, "turn <id>")` renders them, and `turn_injector_total` carries the expected `{component,outcome}` series. This is the end-to-end proof that "read it back from the durable store" works.

- [ ] **Step 1: Write the harness test** (per above). Flush the writer before reading (`writer.flush(...)`), mirroring `trace_view`.
- [ ] **Step 2: Run — expect FAIL** (until the pieces align), then **PASS**.
- [ ] **Step 3: `make check`** — full green.
- [ ] **Step 4: Commit** `test(turn-obs): real-code end-to-end harness — write turn, read it back (lm-hg7)`

---

## Self-Review

- **Spec coverage:** §4 TurnRecorder → T3–T6, T11; §5 correlation/lifecycle → T3 (reconcile), T4 (injector children), T5 (tools), T6 (close), T11 (open observer + proactive traceparent); §6 storage/frame_kind → T2 + T3; §7 metric → T1 + T4 + T7 (retire FELT_DISPLAY_TOTAL); §8 attrs/completion → T4/T6; §9 reader → T12; §10 fail-soft → every task's `try/except` + T13. All spec sections map to a task.
- **Type/name consistency:** `TURN_INJECTOR_TOTAL`, `INJECTOR_{FELT,GENESIS,BELIEF,COMMITMENT}`, `injector_span`/`InjectorSpan.set(outcome=…)`, `ensure_turn`/`tool_open`/`tool_close`/`close_turn`, component strings `turn` / `turn.injector.<c>` / `turn.tool.<t>` / `turn.completion` — used identically across tasks.
- **Open items surfaced for the worker (not blockers):** exact `MetricRegistry` value accessor (T1 note); `FakeTracer.start_root(upstream_traceparent=…)` honouring continuation (T3 note); exact `pre/post_tool_call` + `post_llm_call` kwarg names in the host (T11 note, verify against `~/.hermes/hermes-agent/`); whether reactive reasoning is exposed to `post_llm_call` (spec open-Q #3, degrade to final_output only).

## Execution notes (this repo's SDD workflow)

Per the owner's established preference: implementers (sonnet) run TDD + `make check` + self-review per task and **commit directly** — NO per-task review checkpoints. Review concentrates in ONE final whole-branch codex review at the end, adjudicated by the orchestrator. One task at a time, in order (later tasks consume earlier interfaces). Merge to `main` + `make deploy` stay owner-gated.

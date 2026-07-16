# Waking Mind — Slice 1: Event-Seeded Thought Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When an ordinary owner↔being conversation turn completes, the being *appraises* what came up and durably **captures a thought** ("worth returning to") — without any hook writing the store directly, and without touching the 0-LLM tick's cost profile.

**Architecture:** The appraisal (judgment) runs **out-of-band in the `post_llm_call` hook** (which already holds the completed `user_message` + `assistant_response`) via an injected `Appraiser` port. On a positive appraisal the hook seeds a one-shot `EVENT` frame carrying a `thought_seed` **signal** (the bounded appraisal *result*, not raw text). A new 0-LLM **`ThoughtCapture`** aggregation component consumes that signal and emits a `PutRecord(thought)` intent through the existing atomic committer. Idempotent by a content-digest id, so a host retry upserts one row.

**Tech Stack:** Python 3.11 stdlib only (runtime), `uv`/`ruff`/`mypy --strict`/`pytest` (dev). The being's typed BDI store (`memory_records` via `MemoryPort`), the intent bus (`PutRecord`/`PutOp`), the signal taxonomy (`core/taxonomy.py`), the `Thought` write-door (`core/thought_view.py`), and the real-code test harness (`testing/`).

## Global Constraints

- **bd:** this is slice 1 of epic **lm-705** — task **lm-705.1**. Spec: `docs/superpowers/specs/2026-07-16-waking-mind-attention-economy-design.md` (§3.1, §4.1).
- **Create ≠ process.** This slice *creates* thoughts only. It NEVER processes/ruminates (slice 2), never mints a desire (slice 3), never runs the arbiter (slice 4). A captured thought is born `active` and just sits in the backlog.
- **A hook never writes the store directly** (spec §4.1). The hook seeds a *signal*; a core component emits the `PutRecord`. Mutation only via the intent bus + end-of-tick committer.
- **0-LLM core, S5 held.** `ThoughtCapture.step` makes no LLM call. The appraisal in slice 1 is a **deterministic heuristic** (no LLM) — the LLM-vs-rides-the-tail appraiser is a deferred bead (spec §8 open question). Idle heartbeat ticks stay byte-identical.
- **Idempotent capture.** The thought id is the content digest (`seed_thought_id`), so a duplicate appraisal (a host `post_llm` retry of the same turn) upserts ONE row, never a pile.
- **Fail-loud/fail-soft as the surrounding hooks.** The appraisal seam lives inside the existing `post_llm` observer's plugin-owned try/except (`hooks.py:428`): a throw is logged + swallowed, never crashes the host turn.
- **Every step ends green:** `make check` (ruff format --check · ruff check · mypy -p lifemodel · pytest) must pass before each commit.

---

## File Structure

- **Create** `lifemodel/core/appraisal.py` — the `Appraiser` Protocol, the `ThoughtSeed` result dataclass, and the deterministic `HeuristicAppraiser`. One responsibility: *decide whether a completed exchange is worth a thought, and what that thought says.*
- **Create** `lifemodel/core/thought_capture.py` — the `ThoughtCapture` 0-LLM component: `thought_seed` signal → `PutRecord(thought)`.
- **Modify** `lifemodel/core/taxonomy.py` — add the `thought_seed` signal kind + builder + reader (mirrors `proactive_outcome_signal` at `core/taxonomy.py:128`).
- **Modify** `lifemodel/hooks.py` — extend `make_post_llm_observer` (`hooks.py:362`) to take an `appraiser` and run an `EVENT` frame with a `thought_seed` signal on a genuine reactive exchange.
- **Modify** `lifemodel/composition.py` — register `ThoughtCapture` in the component registry inside `build_lifemodel`.
- **Modify** `lifemodel/__init__.py` — pass a `HeuristicAppraiser` into `make_post_llm_observer` at the `post_llm_observer` wiring (`__init__.py:469`).
- **Create** `lifemodel/testing/appraisal.py` — `FakeAppraiser` (returns a preset `ThoughtSeed | None`).
- **Create** tests: `tests/test_appraisal.py`, `tests/test_thought_capture.py`, `tests/test_thought_capture_seam.py` (hook), `tests/test_thought_capture_harness.py` (real-code sim).

---

## Task 1: The `thought_seed` signal (taxonomy)

**Files:**
- Modify: `lifemodel/core/taxonomy.py`
- Test: `tests/test_taxonomy_thought_seed.py`

**Interfaces:**
- Produces: `KIND_THOUGHT_SEED: str`; `thought_seed_signal(*, origin_id: str, content: str, salience: float, actionability: float = 0.0, other_regarding_value: float = 0.0, timestamp: str | None) -> Signal`; `read_thought_seed(signal: Signal) -> ThoughtSeedRead` where `ThoughtSeedRead` is a frozen dataclass `(content: str, salience: float, actionability: float, other_regarding_value: float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_taxonomy_thought_seed.py
import pytest
from lifemodel.core.taxonomy import (
    KIND_THOUGHT_SEED,
    thought_seed_signal,
    read_thought_seed,
)


def test_thought_seed_roundtrip():
    sig = thought_seed_signal(
        origin_id="seed-1",
        content="the owner mentioned a dentist appointment on Friday",
        salience=0.6,
        actionability=0.3,
        other_regarding_value=0.5,
        timestamp="2026-07-16T00:00:00+00:00",
    )
    assert sig.kind == KIND_THOUGHT_SEED
    read = read_thought_seed(sig)
    assert read.content == "the owner mentioned a dentist appointment on Friday"
    assert read.salience == 0.6
    assert read.actionability == 0.3
    assert read.other_regarding_value == 0.5


def test_read_thought_seed_rejects_wrong_kind():
    from lifemodel.domain.signal import Signal

    with pytest.raises(ValueError):
        read_thought_seed(Signal(origin_id="x", kind="not_a_seed", payload={}, timestamp=None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy_thought_seed.py -v`
Expected: FAIL with `ImportError: cannot import name 'KIND_THOUGHT_SEED'`.

- [ ] **Step 3: Add the signal to `core/taxonomy.py`**

Add near the other `KIND_*` constants and builders (mirroring `proactive_outcome_signal`, `core/taxonomy.py:128`):

```python
from dataclasses import dataclass

KIND_THOUGHT_SEED = "thought_seed"


@dataclass(frozen=True)
class ThoughtSeedRead:
    """The validated payload of a ``thought_seed`` signal — the appraisal result a
    completed exchange produced, on its way to a captured ``Thought`` (slice 1)."""

    content: str
    salience: float
    actionability: float
    other_regarding_value: float


def thought_seed_signal(
    *,
    origin_id: str,
    content: str,
    salience: float,
    actionability: float = 0.0,
    other_regarding_value: float = 0.0,
    timestamp: str | None,
) -> Signal:
    """Build a ``thought_seed`` signal — a bounded appraisal result seeded by the
    ``post_llm`` appraisal seam, consumed by ``ThoughtCapture`` (spec §4.1)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_THOUGHT_SEED,
        payload={
            "content": content,
            "salience": float(salience),
            "actionability": float(actionability),
            "other_regarding_value": float(other_regarding_value),
        },
        timestamp=timestamp,
    )


def read_thought_seed(signal: Signal) -> ThoughtSeedRead:
    """Validate and extract a :class:`ThoughtSeedRead` from a ``thought_seed`` signal."""
    if signal.kind != KIND_THOUGHT_SEED:
        raise ValueError(f"not a thought_seed signal: kind={signal.kind!r}")
    content = signal.payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"invalid thought_seed payload: {signal.payload!r}")
    return ThoughtSeedRead(
        content=content,
        salience=float(signal.payload.get("salience", 0.0)),
        actionability=float(signal.payload.get("actionability", 0.0)),
        other_regarding_value=float(signal.payload.get("other_regarding_value", 0.0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy_thought_seed.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lifemodel/core/taxonomy.py tests/test_taxonomy_thought_seed.py
git commit -m "feat(thought-capture): thought_seed signal — the appraisal result on the bus (lm-705.1)"
```

---

## Task 2: The `Appraiser` port + deterministic `HeuristicAppraiser`

**Files:**
- Create: `lifemodel/core/appraisal.py`
- Create: `lifemodel/testing/appraisal.py`
- Test: `tests/test_appraisal.py`

**Interfaces:**
- Produces: `ThoughtSeed` (frozen dataclass: `content: str`, `salience: float`, `actionability: float = 0.0`, `other_regarding_value: float = 0.0`); `Appraiser` Protocol with `appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None`; `HeuristicAppraiser` implementing it; `FakeAppraiser(seed: ThoughtSeed | None)` in testing.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_appraisal.py
from lifemodel.core.appraisal import HeuristicAppraiser, ThoughtSeed


def test_heuristic_seeds_on_a_forward_reference():
    seed = HeuristicAppraiser().appraise(
        user_message="I've got a dentist appointment on Friday, dreading it",
        assistant_response="Ah, hope it goes smoothly — tell me how it went?",
    )
    assert seed is not None
    assert "friday" in seed.content.lower() or "dentist" in seed.content.lower()
    assert 0.0 < seed.salience <= 1.0


def test_heuristic_declines_on_small_talk():
    seed = HeuristicAppraiser().appraise(
        user_message="ok thanks",
        assistant_response="anytime!",
    )
    assert seed is None


def test_heuristic_declines_on_empty():
    assert HeuristicAppraiser().appraise(user_message="", assistant_response="") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_appraisal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lifemodel.core.appraisal'`.

- [ ] **Step 3: Create `core/appraisal.py`**

```python
"""The appraisal seam — decide whether a completed exchange is worth a thought.

Runs OUT-OF-BAND (in the ``post_llm`` hook, which holds the finished turn), never
inside the 0-LLM tick. Slice 1 ships a deterministic, no-LLM :class:`HeuristicAppraiser`
so the whole capture pipeline is testable and cost-free; the richer LLM/rides-the-tail
appraiser is a deferred bead (spec §8). The being's REPLY is its thinking; this only
drops a mental bookmark to return to later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Cheap, language-agnostic-ish markers that a turn left something open worth
#: returning to — a forward reference (a plan, a future event) or an unresolved
#: question. Intentionally crude: slice 1 proves the PLUMBING; appraisal QUALITY is
#: the next refinement, tuned against live traces ("нужно пробовать").
_FORWARD_MARKERS = (
    "tomorrow", "next week", "friday", "monday", "tuesday", "wednesday",
    "thursday", "saturday", "sunday", "later", "soon", "appointment",
    "deadline", "interview", "meeting", "trip", "plan to", "going to",
)
_MIN_CONTENT_CHARS = 24


@dataclass(frozen=True)
class ThoughtSeed:
    """An appraisal result: the content + salience of a thought worth capturing."""

    content: str
    salience: float
    actionability: float = 0.0
    other_regarding_value: float = 0.0


@runtime_checkable
class Appraiser(Protocol):
    """Judge a completed owner↔being exchange; return a seed, or ``None`` to decline."""

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None: ...


class HeuristicAppraiser:
    """A deterministic, no-LLM appraiser (slice 1). Seeds a thought when the user's
    message is substantive AND carries a forward-reference / open-loop marker."""

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None:
        text = user_message.strip()
        if len(text) < _MIN_CONTENT_CHARS:
            return None
        low = text.lower()
        if not any(marker in low for marker in _FORWARD_MARKERS):
            return None
        # First-person content: this is the being's own note to itself.
        content = f"the owner said: {text}" if len(text) <= 200 else f"the owner said: {text[:200]}…"
        return ThoughtSeed(
            content=content,
            salience=0.5,
            actionability=0.3,
            other_regarding_value=0.5,
        )
```

- [ ] **Step 4: Create `testing/appraisal.py`**

```python
"""Test double for the appraisal seam (slice 1)."""

from __future__ import annotations

from lifemodel.core.appraisal import ThoughtSeed


class FakeAppraiser:
    """Returns a fixed *seed* (or ``None`` to decline) regardless of input, and
    records the last call so a seam test can assert it was invoked."""

    def __init__(self, seed: ThoughtSeed | None) -> None:
        self._seed = seed
        self.calls: list[tuple[str, str]] = []

    def appraise(self, *, user_message: str, assistant_response: str) -> ThoughtSeed | None:
        self.calls.append((user_message, assistant_response))
        return self._seed
```

- [ ] **Step 5: Run tests to verify they pass, then commit**

Run: `uv run pytest tests/test_appraisal.py -v`
Expected: PASS (3 passed).

```bash
git add lifemodel/core/appraisal.py lifemodel/testing/appraisal.py tests/test_appraisal.py
git commit -m "feat(thought-capture): Appraiser port + deterministic HeuristicAppraiser (lm-705.1)"
```

---

## Task 3: The `ThoughtCapture` component (seed signal → `PutRecord(thought)`)

**Files:**
- Create: `lifemodel/core/thought_capture.py`
- Test: `tests/test_thought_capture.py`

**Interfaces:**
- Consumes: `read_thought_seed` (Task 1); `build_thought`/`encode_thought`/`seed_thought_id` (`core/thought_view.py`); `PutRecord`/`PutOp`; `creation_provenance` (`core/trace.py`).
- Produces: `ThoughtCapture` class with `id = "thought-capture"` and `step(self, ctx: TickContext) -> Sequence[Intent]`; `THOUGHT_CAPTURE_ID = "thought-capture"`.

- [ ] **Step 1: Write the failing test** (uses the same in-tick primitives as `tests/test_aggregation.py`)

```python
# tests/test_thought_capture.py
from lifemodel.core.intents import PutRecord
from lifemodel.core.taxonomy import thought_seed_signal
from lifemodel.core.thought_capture import ThoughtCapture
from lifemodel.core.thought_view import seed_thought_id
from lifemodel.domain.objects import ThoughtState
from lifemodel.testing.tick import make_tick_context  # existing helper (see test_aggregation.py)


def test_capture_emits_one_put_for_a_seed():
    content = "the owner said: dentist on Friday"
    ctx = make_tick_context(
        signals=[thought_seed_signal(
            origin_id="s1", content=content, salience=0.5,
            timestamp="2026-07-16T00:00:00+00:00",
        )],
    )
    intents = list(ThoughtCapture().step(ctx))
    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(puts) == 1
    draft = puts[0].op.draft
    assert draft.kind == "thought"
    assert draft.id == seed_thought_id(content)          # deterministic / idempotent
    assert draft.state == ThoughtState.ACTIVE.value
    assert draft.payload["content"] == content
    assert draft.payload["trigger"] == "event"


def test_capture_is_idempotent_on_identical_content():
    content = "the owner said: interview next week"
    sig = thought_seed_signal(origin_id="s2", content=content, salience=0.5,
                              timestamp="2026-07-16T00:00:00+00:00")
    ctx = make_tick_context(signals=[sig, sig])          # same content twice this frame
    puts = [i for i in ThoughtCapture().step(ctx) if isinstance(i, PutRecord)]
    assert {p.op.draft.id for p in puts} == {seed_thought_id(content)}  # one id, not two


def test_no_seed_no_put():
    ctx = make_tick_context(signals=[])
    assert list(ThoughtCapture().step(ctx)) == []
```

> **If `lifemodel/testing/tick.py::make_tick_context` does not exist**, add it as a thin helper mirroring how `tests/test_aggregation.py` constructs a `TickContext` (a `SignalFrame` snapshot + a `NoopTracer` child span + fake clock); fold that creation into this task's first commit.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_thought_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lifemodel.core.thought_capture'`.

- [ ] **Step 3: Create `core/thought_capture.py`**

```python
"""``ThoughtCapture`` — the 0-LLM component that persists appraised thoughts (§4.1).

Slice 1 of the waking mind (lm-705.1). Consumes ``thought_seed`` signals (seeded by
the ``post_llm`` appraisal seam) and emits a ``PutRecord(thought)`` per DISTINCT seed,
through the intent bus + end-of-tick committer — never a direct store write. Born
``active``; capture only (no processing / no desire / no arbiter — those are later
slices). Idempotent: the thought id is the content digest, so a duplicate appraisal
(a host retry) upserts ONE row.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import PutOp
from .component import TickContext
from .intents import Intent, PutRecord
from .taxonomy import KIND_THOUGHT_SEED, read_thought_seed
from .thought_view import build_thought, encode_thought, seed_thought_id
from .trace import creation_provenance

THOUGHT_CAPTURE_ID = "thought-capture"


class ThoughtCapture:
    """Persist each appraised ``thought_seed`` as an ``active`` thought (0-LLM)."""

    id: str = THOUGHT_CAPTURE_ID

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        seen: set[str] = set()
        intents: list[Intent] = []
        for signal in ctx.signals:
            if signal.kind != KIND_THOUGHT_SEED:
                continue
            seed = read_thought_seed(signal)
            thought_id = seed_thought_id(seed.content)
            if thought_id in seen:
                continue  # collapse duplicate seeds this frame → one upsert
            seen.add(thought_id)
            provenance = creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="aggregation",
                reason="event-seeded thought capture",
                source_signal_ids=(signal.origin_id,),
            )
            thought = build_thought(
                id=thought_id,
                content=seed.content,
                trigger="event",
                salience=seed.salience,
                actionability=seed.actionability,
                other_regarding_value=seed.other_regarding_value,
                source="thought-capture",
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_thought(thought))))
        if intents and ctx.logger is not None:
            ctx.logger.span.set(captured=len(intents))
        return intents
```

> **Confirm** the exact `creation_provenance` keyword args against `core/trace.py:25` (it is called with `source_object_ids=` in `core/cognition.py:157`; use `source_signal_ids=` if that is the signal-lineage kwarg, else drop it — the causal stamp is a nicety, not load-bearing for capture).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_thought_capture.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lifemodel/core/thought_capture.py lifemodel/testing/tick.py tests/test_thought_capture.py
git commit -m "feat(thought-capture): ThoughtCapture component — seed signal to PutRecord(thought), idempotent (lm-705.1)"
```

---

## Task 4: The `post_llm` appraisal seam

**Files:**
- Modify: `lifemodel/hooks.py` (`make_post_llm_observer`, `hooks.py:362`)
- Test: `tests/test_thought_capture_seam.py`

**Interfaces:**
- Consumes: `Appraiser` (Task 2); `thought_seed_signal` (Task 1); `run_frame` / `FrameTrigger.EVENT` (already imported in `hooks.py`).
- Produces: `make_post_llm_observer(build_lm, *, appraiser: Appraiser | None = None, health=..., metrics=...)` — an added keyword; when a **genuine reactive exchange** completes (NOT a pending-proactive turn, non-empty user + assistant text, not a control command / own impulse) and `appraiser` returns a seed, it runs one `EVENT` frame carrying a `thought_seed` signal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_thought_capture_seam.py
from lifemodel.core.appraisal import ThoughtSeed
from lifemodel.core.thought_view import read_live_thoughts, seed_thought_id
from lifemodel.hooks import make_post_llm_observer
from lifemodel.testing.appraisal import FakeAppraiser
from lifemodel.testing.harness import build_capture_lifemodel  # thin harness builder (Task 6)


def test_reactive_exchange_captures_a_thought():
    lm = build_capture_lifemodel()                      # fakes + ThoughtCapture registered
    content = "the owner said: trip on Friday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.5))
    )
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    thoughts = read_live_thoughts(lm.state)             # FakeStateStore is also a MemoryPort
    assert [t.id for t in thoughts] == [seed_thought_id(content)]


def test_declining_appraiser_captures_nothing():
    lm = build_capture_lifemodel()
    observer = make_post_llm_observer(lambda: lm, appraiser=FakeAppraiser(None))
    observer(user_message="ok thanks", assistant_response="anytime")
    assert read_live_thoughts(lm.state) == ()


def test_no_appraiser_is_a_noop():                      # back-compat: existing wiring unaffected
    lm = build_capture_lifemodel()
    observer = make_post_llm_observer(lambda: lm)       # appraiser omitted
    observer(user_message="I have a trip on Friday", assistant_response="Sounds lovely!")
    assert read_live_thoughts(lm.state) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_thought_capture_seam.py -v`
Expected: FAIL — `make_post_llm_observer() got an unexpected keyword argument 'appraiser'`.

- [ ] **Step 3: Extend `make_post_llm_observer`**

Add the `appraiser` keyword and an appraisal branch for non-proactive turns. Inside the existing `try:` (so a throw stays swallowed, `hooks.py:428`), after the early-return that handles the pending-proactive path:

```python
def make_post_llm_observer(
    build_lm: Callable[[], LifeModel],
    *,
    appraiser: "Appraiser | None" = None,   # NEW — from lifemodel.core.appraisal
    health: BrainHealth | None = None,
    metrics: MetricRegistry | None = None,
) -> Callable[..., None]:
    def _observer(
        *,
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: Any = None,
        **_ignored: Any,
    ) -> None:
        try:
            lm = build_lm()
            state = lm.state.load()
            if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
                # NOT a proactive read-back → an ordinary owner↔being exchange.
                # Appraise it (out-of-band) and, on a seed, capture a thought via
                # a core component in its own EVENT frame (spec §4.1). The hook
                # never writes the store; it only seeds a signal.
                _maybe_capture_thought(lm, appraiser, user_message, assistant_response)
                return
            # ... existing pending-proactive outcome path unchanged ...
```

Add the helper (module-level, near the observer):

```python
def _maybe_capture_thought(
    lm: LifeModel,
    appraiser: "Appraiser | None",
    user_message: str,
    assistant_response: str,
) -> None:
    """Appraise a completed reactive exchange; on a seed, run an EVENT frame that
    carries a ``thought_seed`` signal for ``ThoughtCapture`` (spec §4.1)."""
    if appraiser is None:
        return
    text = user_message.strip()
    # Only genuine dialogue: skip empties, our own impulse, and control commands
    # (same sensor band-pass as the inbound observer, hooks.py:436/441).
    if not text or _is_own_impulse(text) or _is_control_command(text):
        return
    if _is_no_reply(assistant_response):
        return
    seed = appraiser.appraise(user_message=user_message, assistant_response=assistant_response)
    if seed is None:
        return
    assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
    now = lm.clock.now()
    run_frame(
        lm.coreloop,
        [
            thought_seed_signal(
                origin_id=f"thought-seed-{seed_thought_id(seed.content)}",
                content=seed.content,
                salience=seed.salience,
                actionability=seed.actionability,
                other_regarding_value=seed.other_regarding_value,
                timestamp=to_iso(now),
            )
        ],
        trigger=FrameTrigger.EVENT,
    )
```

Add imports at the top of `hooks.py`:

```python
from .core.appraisal import Appraiser
from .core.taxonomy import thought_seed_signal   # extend the existing taxonomy import
from .core.thought_view import seed_thought_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_thought_capture_seam.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lifemodel/hooks.py tests/test_thought_capture_seam.py
git commit -m "feat(thought-capture): post_llm appraisal seam — a completed exchange seeds a thought (lm-705.1)"
```

---

## Task 5: Composition wiring

**Files:**
- Modify: `lifemodel/composition.py` (`build_lifemodel`)
- Modify: `lifemodel/__init__.py` (`post_llm_observer` wiring, `__init__.py:469`)
- Test: `tests/test_composition.py` (extend)

**Interfaces:**
- Consumes: `ThoughtCapture` (Task 3); `HeuristicAppraiser` (Task 2); `ComponentManifest`/`ComponentLayer` (`core/registry.py`, `core/component.py`).
- Produces: `ThoughtCapture` present in the built registry; the live `post_llm` hook carrying a `HeuristicAppraiser`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition.py  (add)
def test_build_lifemodel_registers_thought_capture():
    from lifemodel.composition import build_lifemodel
    from lifemodel.core.thought_capture import THOUGHT_CAPTURE_ID

    lm = build_lifemodel(base_dir=_tmp_base_dir())   # reuse this module's existing tmp helper
    ids = {m.id for m in lm.coreloop._registry.manifests()}
    assert THOUGHT_CAPTURE_ID in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_composition.py::test_build_lifemodel_registers_thought_capture -v`
Expected: FAIL — `THOUGHT_CAPTURE_ID` not in the registry.

- [ ] **Step 3: Register `ThoughtCapture` in `build_lifemodel`**

Where the other components register (mirror the existing aggregation component's `registry.register(...)` call):

```python
from .core.thought_capture import ThoughtCapture, THOUGHT_CAPTURE_ID
from .core.registry import ComponentManifest
from .core.component import ComponentLayer   # confirm the AGGREGATION member name

registry.register(
    ThoughtCapture(),
    ComponentManifest(
        id=THOUGHT_CAPTURE_ID,
        type="thought-capture",
        layer=ComponentLayer.AGGREGATION,   # 0-LLM, consumes signals, writes memory
        metric_surface=(),                  # declares no domain metrics (slice 1)
        accepts_signals=True,
    ),
)
```

- [ ] **Step 4: Pass a `HeuristicAppraiser` into the live `post_llm` hook (`__init__.py:469`)**

```python
from .core.appraisal import HeuristicAppraiser

# inside the wire("post_llm_observer", ...) block:
ctx.register_hook(
    "post_llm_call",
    make_post_llm_observer(
        lambda: build_lifemodel(
            base_dir=sdir, trace_writer=_outcome_writer, event_ring=_outcome_ring
        ),
        appraiser=HeuristicAppraiser(),
        health=health,
        metrics=metrics,
    ),
)
```

- [ ] **Step 5: Run tests + full gate, then commit**

Run: `uv run pytest tests/test_composition.py -v && make check`
Expected: PASS; `make check` green.

```bash
git add lifemodel/composition.py lifemodel/__init__.py tests/test_composition.py
git commit -m "feat(thought-capture): wire ThoughtCapture + HeuristicAppraiser into composition (lm-705.1)"
```

---

## Task 6: Real-code sim — capture end-to-end through the live pipeline

**Files:**
- Modify: `lifemodel/testing/harness.py` (add `build_capture_lifemodel`)
- Test: `tests/test_thought_capture_harness.py`

**Interfaces:**
- Consumes: the real `CoreLoop`, `FakeStateStore(memory=FakeMemoryStore(...))`, `run_frame`, `ThoughtCapture`.
- Produces: `build_capture_lifemodel(*, clock=None) -> LifeModel` — a real-code `LifeModel` whose registry contains `ThoughtCapture`, over fake ports (mirrors the existing harness builder used by `tests/test_frame_acceptance.py`).

- [ ] **Step 1: Write the failing test** (drives the REAL frame, asserts the store)

```python
# tests/test_thought_capture_harness.py
from lifemodel.core.appraisal import ThoughtSeed
from lifemodel.core.thought_view import read_live_thoughts, seed_thought_id
from lifemodel.hooks import make_post_llm_observer
from lifemodel.testing.appraisal import FakeAppraiser
from lifemodel.testing.harness import build_capture_lifemodel


def test_exchange_to_thought_row_end_to_end():
    lm = build_capture_lifemodel()
    content = "the owner said: interview on Monday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.7))
    )
    observer(user_message="big interview on Monday, nervous", assistant_response="You'll do great.")
    live = read_live_thoughts(lm.state)
    assert [t.content for t in live] == [content]
    assert live[0].salience == 0.7


def test_retry_of_same_exchange_upserts_one_row():
    lm = build_capture_lifemodel()
    content = "the owner said: interview on Monday"
    observer = make_post_llm_observer(
        lambda: lm, appraiser=FakeAppraiser(ThoughtSeed(content=content, salience=0.7))
    )
    observer(user_message="big interview on Monday", assistant_response="You'll do great.")
    observer(user_message="big interview on Monday", assistant_response="You'll do great.")  # retry
    live = read_live_thoughts(lm.state)
    assert len(live) == 1
    assert live[0].id == seed_thought_id(content)


def test_idle_heartbeat_is_still_zero_capture():
    from lifemodel.core.frame import FrameTrigger, run_frame

    lm = build_capture_lifemodel()
    run_frame(lm.coreloop, [], trigger=FrameTrigger.HEARTBEAT)   # empty world
    assert read_live_thoughts(lm.state) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_thought_capture_harness.py -v`
Expected: FAIL — `build_capture_lifemodel` not defined.

- [ ] **Step 3: Add `build_capture_lifemodel` to `testing/harness.py`**

Mirror the harness builder the frame-acceptance tests already use (a `CoreLoop` over `FakeStateStore(memory=FakeMemoryStore(clock=clock))`, a `NoopTracer`, a `ComponentRegistry`), and register `ThoughtCapture`:

```python
def build_capture_lifemodel(*, clock: ClockPort | None = None) -> LifeModel:
    """A real-code LifeModel over fake ports with ThoughtCapture registered — the
    slice-1 sim seam (spec §6). Reuses the existing fake-port CoreLoop builder in
    this module; adds only the ThoughtCapture registration."""
    lm = _build_fake_lifemodel(clock=clock)              # existing private builder
    lm.coreloop._registry.register(
        ThoughtCapture(),
        ComponentManifest(
            id=THOUGHT_CAPTURE_ID, type="thought-capture",
            layer=ComponentLayer.AGGREGATION, metric_surface=(), accepts_signals=True,
        ),
    )
    return lm
```

> If `testing/harness.py` has no reusable private builder, factor the `CoreLoop`+fakes construction the existing harness tests use into `_build_fake_lifemodel` as part of this task (a pure refactor, no behavior change), then add `build_capture_lifemodel` on top.

- [ ] **Step 4: Run tests + full gate, then commit**

Run: `uv run pytest tests/test_thought_capture_harness.py -v && make check`
Expected: PASS (3 passed); `make check` green.

```bash
git add lifemodel/testing/harness.py tests/test_thought_capture_harness.py
git commit -m "feat(thought-capture): real-code sim — exchange to thought row, idempotent, idle stays zero (lm-705.1)"
```

---

## Self-Review (run after the plan is executed)

- **Spec coverage (§3.1/§4.1):** appraisal seam (Task 4) ✓ · hook never writes the store, a component emits PutRecord (Task 3) ✓ · event-seeded only (`trigger="event"`, Task 3) ✓ · idempotent capture (content-digest id, Task 3/6) ✓ · 0-LLM core + idle-zero (Task 6) ✓ · durable Thought via the typed registry door (Task 3) ✓.
- **Explicitly deferred to later slices (do NOT build here):** processing/rumination + the lifecycle *bounds* (no_progress increment, park-backoff, max attempts, terminal) → slice 2 (lm-705.2); thought→contact-desire → slice 3 (lm-705.3); the arbiter → slice 4 (lm-705.4). Backlog decay/expiry of un-processed thoughts rides slice 2 (it is a processing-tier concern).
- **Type consistency:** `ThoughtSeed` (core/appraisal) vs `ThoughtSeedRead` (taxonomy payload) are intentionally distinct — the appraiser's output type vs the signal's validated payload; the hook maps one to the other. `seed_thought_id(content)` is the single id source used by Task 3 (write) and Tasks 4/6 (assert).
- **Confirm-before-code flags (framework constants only, logic is fully specified):** the `ComponentLayer.AGGREGATION` member name (`core/component.py`) and the `creation_provenance` signal-lineage kwarg (`core/trace.py:25`).

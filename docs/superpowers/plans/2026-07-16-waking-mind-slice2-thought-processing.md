# Waking Mind — Slice 2: Private Thought Processing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On an idle tick, the being selects **one** live thought (K=1, most-salient), ruminates on it via the non-delivering internal-cognition seam under a hard FR20 quota, and applies a **typed outcome** (`resolve`/`park`/`drop`) that advances the thought's bounded lifecycle — discharging the nag, all cost-capped, never delivered.

**Architecture:** Two new 0-LLM cognition components ride the seam built in lm-705.6. `ThoughtProcessingSelector` (heartbeat) re-arms expired-parked thoughts (`parked→active`) and, when the gates pass, emits a `LaunchInternalCognition` for the top-salience **active** thought. The adapter-owned `InternalCognitionRunner` awaits the aux call off-lock; on completion `ThoughtProcessingApply` (the runner's injected `apply`, completion-frame-only) reads the in-flight subject, validates the parsed outcome, and emits the `TransitionRecord`. This slice is also the **first live emitter** of `LaunchInternalCognition`, so it lands the two latent prereqs that a live emitter must own: **birth-voice** threading and a **single-flight** gate.

**Tech Stack:** Python 3.11 stdlib only (runtime), `uv`/`ruff`/`mypy --strict`/`pytest` (dev). The being's typed BDI store (`memory_records` via `MemoryPort`), the intent bus (`TransitionRecord`/`TransitionOp` + `MemoryPatch.payload_merge`), the signal taxonomy (`internal_result` / `read_internal_result`), the thought view (`core/thought_view.py`), the internal-cognition seam (`core/internal_cognition.py`, `adapters/internal_runner.py`, `core/budget.py`), and the real-code fake-port harness (`testing/harness.py`).

## Global Constraints

- **bd:** slice 2 of epic **lm-705** — task **lm-705.2**. Spec: `docs/superpowers/specs/2026-07-16-waking-mind-attention-economy-design.md` (§3.2, §4.1, §4.5). Rides the seam from lm-705.6 (`docs/superpowers/plans/2026-07-16-internal-cognition-seam-lm-705-6.md`).
- **Create ≠ process (spec §2).** This slice never *creates* a thought (slice 1) and never mints a desire (slice 3). It only *processes* existing thoughts and advances their lifecycle. The reply is the thinking during a live turn — no rumination inside a dialogue turn.
- **Non-delivery is structural (lm-705.6).** The internal path calls the `LlmPort`, never `egress.reach_out`/`inject_proactive_turn`; there is no `post_llm` outcome for it. The ONLY egress touch on a completion frame is `dispatch_launches` for a *proactive* launch some OTHER component incidentally returned — and that path MUST carry the birth `voice` (prereq #1).
- **Idle stays 0-LLM (S5, spec §4.5).** The selector emits nothing when the active backlog is empty, the budget is spent, the interval hasn't elapsed, or a pass is in flight. A heartbeat over an empty world is byte-identical to before this slice.
- **Cost is a hard FR20 ceiling independent of energy (spec §4.5).** Reuse the durable daily call quota (`internal_calls_today`/`internal_calls_day`, `reserve_internal_call`) already built. Add a durable **min inter-processing interval** and enforce **single-flight**. A day with a live backlog spends **≤ the FR20 ceiling**.
- **Bounded lifecycle is a required contract, not a hope (spec §4.1).** Every non-transient outcome is a valid state transition **from `active`**; a thought never loops `active→active` or `parked→parked` (the transition table forbids both — `domain/objects/thought.py:38`). Attempts are bounded by `no_progress_count` (cap → `drop`) and `park_count` (cap → `expire`). A **transient** call failure never penalizes the thought.
- **No residue/opinion is written here (spec §4.1).** `resolve` is plain. The `reflection` text rides the observability span (FR24 storage/debug), it is **not** persisted on the thought — the residue field belongs to lm-adz.
- **Observability is forced with a closed reason set (spec §5).** Positive processing decisions are **not** suppressions: they log as span fields (`processing_reason=…`, `thought_id=…`), never through `SuppressionReason`. The thought id is always a field, never inside the reason string.
- **State fields are additive** — `to_dict` is `asdict(self)`; `from_dict` gets one explicit `_as_opt_str`/`_as_str` line per new field. No migration (`state/sqlite_store.py` forward-compat load).
- **Deferred to follow-up beads (Task 9), not built here:** shared FR20 with proactive (spec §4.5 — amended: internal rumination is the newly-unbounded spend and is capped; proactive stays bounded by its own drive/backstop dynamics), launch→completion **trace-weave** (they already correlate via `correlation_id`; span-parenting touches the trace root), **cheap-model routing** (blocked on host — `ctx.llm.acomplete_structured` hard-codes `task=None`; see `adapters/plugin_llm_adapter.py` docstring), and **refund-on-transient-failure** (Minor; mitigated by single-flight + interval + the transient/malformed split).
- **Every step ends green:** `make check` (ruff format --check · ruff check · mypy -p lifemodel · pytest) before each commit.

## File Structure

- **Modify** `state/model.py` — add `pending_internal_subject_id: str | None`, `last_internal_call_at: str | None` to `State` (+ `from_dict`).
- **Modify** `core/budget.py` — add `DEFAULT_DAILY_INTERNAL_CALL_CEILING`, `DEFAULT_MIN_INTERPROCESSING_INTERVAL`, pure gates `internal_budget_available(...)`, `internal_interval_elapsed(...)`.
- **Modify** `core/intents.py` — add `subject_id`, `instructions`, `json_schema` (all optional) to `LaunchInternalCognition` so an emitter fully specifies its own call.
- **Create** `core/thought_processing.py` — the whole processing brain: constants + JSON schema + prompt, the pure `decide_processing_transition(...)`, `ThoughtProcessingSelector` (heartbeat emitter), `ThoughtProcessingApply` (completion consumer), the `ProcessingReason` closed enum.
- **Modify** `core/internal_cognition.py` — `run_internal_completion(..., voice=None)`: thread `voice` into `dispatch_launches` (prereq #1) and clear `pending_internal_subject_id` alongside `pending_internal_id`.
- **Modify** `adapters/internal_runner.py` — constructor takes `voice`; `launch(request, correlation_id, *, subject_id=None)` adds the single-flight gate (prereq #2), stamps `pending_internal_subject_id` + `last_internal_call_at` in the reserve frame, passes `voice` to the completion; `recover_stale` also clears the subject.
- **Modify** `composition.py` — register `ThoughtProcessingSelector` in `build_lifemodel`.
- **Modify** `adapters/being_platform.py` — pass `voice=self._voice` to the runner, use `ThoughtProcessingApply()` as its `apply`, map `subject_id`/`instructions`/`json_schema` from the launch into the request + `launch(...)` call; import the FR20 ceiling from `core/budget.py` (DRY).
- **Modify** `testing/harness.py` — add `build_processing_lifemodel(...)` (real-code `CoreLoop` with the selector registered).
- **Tests:** `tests/test_budget_processing.py`, `tests/test_internal_intent_subject.py`, `tests/test_thought_processing_decide.py`, `tests/test_thought_processing_selector.py`, `tests/test_thought_processing_apply.py`, `tests/test_internal_runner_single_flight.py`, `tests/test_internal_completion_voice.py`, extend `tests/test_composition.py`, `tests/test_thought_processing_harness.py`, extend `tests/hermes_internal_cognition_integration.py`.

---

## Task 1: Durable state fields + pure FR20 budget/interval gates

**Files:** Modify `state/model.py`, `core/budget.py`; Create `tests/test_budget_processing.py`.

**Interfaces:**
- Produces: `State.pending_internal_subject_id: str | None = None`, `State.last_internal_call_at: str | None = None`; `DEFAULT_DAILY_INTERNAL_CALL_CEILING: int = 50`, `DEFAULT_MIN_INTERPROCESSING_INTERVAL: timedelta`; `internal_budget_available(state: State, *, now: datetime, daily_ceiling: int) -> bool`; `internal_interval_elapsed(state: State, *, now: datetime, min_interval: timedelta) -> bool`.
- Consumes: `State`, `reserve_internal_call` (already in `core/budget.py`), `from_iso` (`core/timeutil.py`).

- [ ] **Step 1: Failing test** (`tests/test_budget_processing.py`)

```python
from datetime import datetime, timedelta, timezone

from lifemodel.core.budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from lifemodel.state.model import State


def _now(day="2026-07-16", hh=12, mm=0):
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00+00:00")


def test_default_ceiling_is_positive():
    assert DEFAULT_DAILY_INTERNAL_CALL_CEILING == 50
    assert DEFAULT_MIN_INTERPROCESSING_INTERVAL > timedelta(0)


def test_budget_available_below_ceiling_and_rolls_day():
    s = State(internal_calls_today=2, internal_calls_day="2026-07-16")
    assert internal_budget_available(s, now=_now(), daily_ceiling=3) is True
    at_cap = State(internal_calls_today=3, internal_calls_day="2026-07-16")
    assert internal_budget_available(at_cap, now=_now(), daily_ceiling=3) is False
    # a new day resets the count → available again
    assert internal_budget_available(at_cap, now=_now(day="2026-07-17"), daily_ceiling=3) is True


def test_interval_elapsed_when_never_run_or_past_window():
    fresh = State(last_internal_call_at=None)
    assert internal_interval_elapsed(fresh, now=_now(), min_interval=timedelta(minutes=30)) is True
    recent = State(last_internal_call_at="2026-07-16T11:45:00+00:00")
    assert internal_interval_elapsed(recent, now=_now(hh=12, mm=0), min_interval=timedelta(minutes=30)) is False
    assert internal_interval_elapsed(recent, now=_now(hh=12, mm=20), min_interval=timedelta(minutes=30)) is True


def test_new_state_defaults_are_neutral():
    s = State()
    assert s.pending_internal_subject_id is None
    assert s.last_internal_call_at is None
    # additive round-trip through from_dict
    assert State.from_dict(s.to_dict()).pending_internal_subject_id is None
    assert State.from_dict(s.to_dict()).last_internal_call_at is None
```

- [ ] **Step 2: Run → fail** (`ImportError` on the new names).

Run: `uv run pytest tests/test_budget_processing.py -v`

- [ ] **Step 3: Add the two `State` fields** (`state/model.py`, right after `internal_calls_day` at line 268):

```python
    #: The durable object a *currently in-flight* internal-cognition pass concerns
    #: (lm-705.2): for a processing pass, the ``kind='thought'`` id being ruminated
    #: on, so the completion frame's :class:`~lifemodel.core.thought_processing.ThoughtProcessingApply`
    #: knows WHICH thought the typed outcome applies to. ``None`` when no pass is in
    #: flight, or the pass has no single durable subject (e.g. noticing, lm-705.5).
    #: Set atomically with :attr:`pending_internal_id` in the runner's reserve frame,
    #: cleared with it on completion. Additive: ``from_dict`` defaults it to ``None``.
    pending_internal_subject_id: str | None = None
    #: When the being last LAUNCHED an internal-cognition pass (lm-705.2) — the
    #: min-inter-processing-interval key (spec §4.5). Server-local ISO-8601 (like every
    #: lifemodel timestamp). Stamped in the runner's reserve frame on a successful
    #: launch (never on a denied/gated one, so the interval clock only advances when a
    #: pass really started). ``None`` before the first pass ever. Additive: ``from_dict``
    #: defaults it to ``None``.
    last_internal_call_at: str | None = None
```

Add the `from_dict` lines (after the `internal_calls_day=` line, ~line 395):

```python
            pending_internal_subject_id=_as_opt_str(
                data.get("pending_internal_subject_id"), "pending_internal_subject_id"
            ),
            last_internal_call_at=_as_opt_str(
                data.get("last_internal_call_at"), "last_internal_call_at"
            ),
```

- [ ] **Step 4: Add the gates** (`core/budget.py`). At the top, extend imports and add constants + two pure functions:

```python
from datetime import datetime, timedelta

from ..core.timeutil import from_iso

#: FR20 v1 default daily ceiling (spec §4.5) — the ONE source of truth, imported by
#: both the runner (the atomic reserve) and the selector (the pre-check), so the two
#: can never drift. A generous-but-real cap: idle ticks stay 0-LLM, this only bounds
#: the spike once a live emitter (lm-705.2) exists. Tune against live traces.
DEFAULT_DAILY_INTERNAL_CALL_CEILING = 50

#: Min wall-clock gap between two internal-cognition launches (spec §4.5) — paces
#: rumination so a live backlog is chewed a little at a time, not all at once.
DEFAULT_MIN_INTERPROCESSING_INTERVAL = timedelta(minutes=30)


def internal_budget_available(state: State, *, now: datetime, daily_ceiling: int) -> bool:
    """True iff another internal-cognition call fits under today's FR20 ceiling.

    The read-only pre-check the selector runs so it never emits a launch the runner's
    atomic :func:`reserve_internal_call` would just deny (and to log the honest
    ``skipped_no_budget`` reason). Shares the day-rollover convention with
    ``reserve_internal_call``: a call on a new day counts against 0."""
    used = state.internal_calls_today if state.internal_calls_day == _day(now) else 0
    return used < daily_ceiling


def internal_interval_elapsed(
    state: State, *, now: datetime, min_interval: timedelta
) -> bool:
    """True iff at least *min_interval* has passed since the last launch (spec §4.5).

    ``True`` when no pass has ever run (``last_internal_call_at is None``). Fail-open:
    an unparseable stored timestamp reads as elapsed rather than wedging processing."""
    if state.last_internal_call_at is None:
        return True
    try:
        last = from_iso(state.last_internal_call_at)
    except (ValueError, TypeError):
        return True
    return now - last >= min_interval
```

> **Confirm:** `_day(now)` already exists in `core/budget.py` (used by `reserve_internal_call`). `from_iso` is in `core/timeutil.py` (used across the codebase, e.g. `domain/memory.py`).

- [ ] **Step 5: Run → pass.** `make check`.

- [ ] **Step 6: Commit** `feat(thought-processing): durable subject/interval state + pure FR20 gates (lm-705.2)`.

---

## Task 2: `LaunchInternalCognition` carries its own call spec (subject/instructions/schema)

**Files:** Modify `core/intents.py`; Create `tests/test_internal_intent_subject.py`.

**Interfaces:**
- Produces: `LaunchInternalCognition` gains `subject_id: str | None = None`, `instructions: str = ""`, `json_schema: dict[str, Any] | None = None` (all optional, back-compatible). The emitter fully specifies its own aux call, so the adapter maps intent→`InternalCognitionRequest` generically without knowing the pass type.
- Consumes: nothing new.

- [ ] **Step 1: Failing test** (`tests/test_internal_intent_subject.py`)

```python
from lifemodel.core.intents import LaunchInternalCognition

_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def test_defaults_are_back_compatible():
    # the lm-705.6 construction shape still works unchanged
    intent = LaunchInternalCognition(prompt="p", correlation_id="c", origin_traceparent=_TP)
    assert intent.subject_id is None
    assert intent.instructions == ""
    assert intent.json_schema is None


def test_emitter_can_fully_specify_its_call():
    schema = {"type": "object", "properties": {"outcome": {"type": "string"}}}
    intent = LaunchInternalCognition(
        prompt="the thought", correlation_id="process-t1@x", origin_traceparent=_TP,
        subject_id="thought:seed:abc", instructions="ruminate", json_schema=schema,
    )
    assert intent.subject_id == "thought:seed:abc"
    assert intent.instructions == "ruminate"
    assert intent.json_schema == schema
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3:** Add the three fields to `LaunchInternalCognition` (`core/intents.py`, after `origin_traceparent`). Add `from typing import Any` if not already imported (it is — line 18):

```python
    prompt: str
    correlation_id: str
    origin_traceparent: str
    #: The durable object this pass concerns (lm-705.2), threaded to
    #: :attr:`~lifemodel.state.model.State.pending_internal_subject_id` by the runner
    #: so the completion frame's apply knows its subject. ``None`` for a subjectless
    #: pass (noticing, lm-705.5).
    subject_id: str | None = None
    #: The aux call's system framing. ``""`` → the adapter's generic default (the
    #: seam's content-free non-delivery notice). An emitter that needs specific framing
    #: (processing) supplies its own, so the adapter never has to know the pass type.
    instructions: str = ""
    #: The typed-result JSON Schema for a structured pass (processing's outcome), or
    #: ``None`` for a plain-text pass. Carried onto the ``InternalCognitionRequest``.
    json_schema: dict[str, Any] | None = None
```

> **Note:** a frozen dataclass with a `dict` field is unhashable — that is fine, intents are never hashed (they go into lists, `coreloop.py:375` matches by `isinstance`).

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `feat(thought-processing): LaunchInternalCognition carries subject/instructions/schema (lm-705.2)`.

---

## Task 3: The pure processing lifecycle — `decide_processing_transition`

**Files:** Create `core/thought_processing.py` (constants + schema + prompt + `ProcessingReason` + the pure decision fn); Create `tests/test_thought_processing_decide.py`.

**Interfaces:**
- Produces: `ProcessingReason(StrEnum)`; `MAX_NO_PROGRESS_COUNT=3`, `MAX_PARK_CYCLES=3`, `PARK_BACKOFFS=(timedelta(hours=6), timedelta(hours=24), timedelta(hours=72))`; `PROCESSING_JSON_SCHEMA: dict`; `PROCESSING_INSTRUCTIONS: str`; `build_processing_prompt(thought: Thought) -> str`; `ProcessingDecision` frozen dataclass `(transition: TransitionOp | None, reason: ProcessingReason)`; `decide_processing_transition(thought: Thought, *, parsed: dict | None, raw: str, now: datetime) -> ProcessingDecision`.
- Consumes: `Thought`/`ThoughtState` (`domain/objects`), `TransitionOp`/`MemoryPatch` (`domain/memory`), `to_iso` (`core/timeutil`).

- [ ] **Step 1: Failing test** (`tests/test_thought_processing_decide.py`)

```python
from datetime import datetime, timezone

from lifemodel.core.thought_processing import (
    MAX_NO_PROGRESS_COUNT,
    MAX_PARK_CYCLES,
    ProcessingReason,
    decide_processing_transition,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.objects import ThoughtState

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _t(*, no_progress=0, park=0):
    return build_thought(
        id="thought:seed:abc", content="dentist on Friday",
        no_progress_count=no_progress, park_count=park,
    )


def test_resolve_is_terminal():
    d = decide_processing_transition(_t(), parsed={"outcome": "resolve"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.RESOLVED
    assert d.transition.from_state == ThoughtState.ACTIVE.value
    assert d.transition.to_state == ThoughtState.RESOLVED.value


def test_drop_is_terminal():
    d = decide_processing_transition(_t(), parsed={"outcome": "drop"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.DROPPED
    assert d.transition.to_state == ThoughtState.DROPPED.value


def test_park_sets_backoff_and_bumps_park_count():
    d = decide_processing_transition(_t(park=0), parsed={"outcome": "park"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.PARKED
    assert d.transition.to_state == ThoughtState.PARKED.value
    assert d.transition.patch.payload_merge["park_count"] == 1
    assert d.transition.patch.payload_merge["parked_until"] == "2026-07-16T18:00:00+00:00"  # +6h


def test_park_at_cap_expires_instead():
    d = decide_processing_transition(_t(park=MAX_PARK_CYCLES), parsed={"outcome": "park"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.EXPIRED_PARK_CAP
    assert d.transition.to_state == ThoughtState.EXPIRED.value


def test_malformed_parks_and_bumps_no_progress():
    d = decide_processing_transition(_t(no_progress=0), parsed=None, raw="not json at all", now=NOW)
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS
    assert d.transition.to_state == ThoughtState.PARKED.value
    assert d.transition.patch.payload_merge["no_progress_count"] == 1


def test_malformed_at_no_progress_cap_drops():
    d = decide_processing_transition(
        _t(no_progress=MAX_NO_PROGRESS_COUNT - 1), parsed=None, raw="junk", now=NOW
    )
    assert d.reason == ProcessingReason.DROPPED_NO_PROGRESS
    assert d.transition.to_state == ThoughtState.DROPPED.value
    assert d.transition.patch.payload_merge["no_progress_count"] == MAX_NO_PROGRESS_COUNT


def test_transient_failure_does_not_touch_the_thought():
    d = decide_processing_transition(_t(), parsed=None, raw="   ", now=NOW)
    assert d.reason == ProcessingReason.TRANSIENT_FAILURE
    assert d.transition is None  # thought stays active, retried next interval


def test_unknown_outcome_string_is_malformed_not_a_crash():
    d = decide_processing_transition(_t(), parsed={"outcome": "banana"}, raw="{...}", now=NOW)
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`).

- [ ] **Step 3: Create `core/thought_processing.py`** (this step's portion — constants + pure fn; the two components come in Tasks 4/5):

```python
"""The waking mind's rumination brain (lm-705.2, spec §3.2/§4.1/§4.5).

Two 0-LLM cognition components ride the non-delivering internal-cognition seam
(lm-705.6): :class:`ThoughtProcessingSelector` (heartbeat) picks ONE live thought
and emits a :class:`~lifemodel.core.intents.LaunchInternalCognition`;
:class:`ThoughtProcessingApply` (completion-frame) turns the typed aux result into
the thought's next state. The lifecycle rules — attempt/park bounds — live here
(spec §4.1: "a required contract, not a hope"), so a thought is chewed a bounded
number of times and then terminates (``resolve``/``drop``/``expire``), never spirals.

Non-delivery is structural (the seam calls the ``LlmPort``, never egress). No
residue/opinion is written (spec §4.1) — the ``reflection`` rides the span for FR24
debug, never the thought. Every from-state is ``active`` (the selector re-arms
expired-parked thoughts to ``active`` first), so no transition is a forbidden
``active→active``/``parked→parked`` self-loop (``domain/objects/thought.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ..domain.memory import MemoryPatch, TransitionOp
from ..domain.objects import Thought, ThoughtState
from .timeutil import to_iso

#: How many non-progress (malformed/no-parse) attempts a thought tolerates before it
#: is ``drop``ped (spec §4.1 "max total processing attempts"). Distinct from park.
MAX_NO_PROGRESS_COUNT = 3
#: How many park cycles before a thought ``expire``s rather than re-arming (spec §4.1).
MAX_PARK_CYCLES = 3
#: The widening park backoff (spec §4.1) — the 6h/24h/72h ladder the Thought schema's
#: ``park_count`` docstring already names (``domain/objects/thought.py``). Indexed by
#: the pre-increment ``park_count``, clamped to the last rung.
PARK_BACKOFFS: tuple[timedelta, ...] = (
    timedelta(hours=6),
    timedelta(hours=24),
    timedelta(hours=72),
)

THOUGHT_KIND = "thought"

#: The processing pass's typed result contract (spec §4.1 "deterministic schema +
#: validation"). ``outcome`` is the disposition; ``reflection`` is a short first-person
#: note that rides the span (FR24 debug), never persisted.
PROCESSING_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["resolve", "park", "drop"]},
        "reflection": {"type": "string"},
    },
    "required": ["outcome"],
    "additionalProperties": False,
}

#: The pass's system framing — first-person, private, non-delivered.
PROCESSING_INSTRUCTIONS = (
    "You are the being's own private mind, quietly turning over one of your thoughts. "
    "Nothing you write here is shown to anyone — this is rumination, not a message. "
    "Reflect briefly, in the first person, then decide the thought's disposition: "
    "'resolve' if you have thought it through and it needs nothing more; "
    "'park' if it is worth returning to later but not now; "
    "'drop' if it no longer matters. "
    "Answer as JSON: an 'outcome' of resolve/park/drop and a short 'reflection'."
)


class ProcessingReason(StrEnum):
    """The closed set of processing-decision reasons (spec §5) — positive choices, NOT
    suppressions. Logged as a span field alongside ``thought_id``, never in a string."""

    # selector
    CHOSE_PROCESS = "chose_process"
    UNPARKED = "unparked"
    SKIPPED_EMPTY_BACKLOG = "skipped_empty_backlog"
    SKIPPED_IN_FLIGHT = "skipped_in_flight"
    SKIPPED_NO_BUDGET = "skipped_no_budget"
    SKIPPED_INTERVAL = "skipped_interval"
    # apply
    RESOLVED = "processed_resolve"
    PARKED = "processed_park"
    DROPPED = "processed_drop"
    EXPIRED_PARK_CAP = "processed_expired_park_cap"
    PARKED_NO_PROGRESS = "processed_park_no_progress"
    DROPPED_NO_PROGRESS = "processed_drop_no_progress"
    TRANSIENT_FAILURE = "processed_transient_failure"
    NO_SUBJECT = "processed_no_subject"


@dataclass(frozen=True)
class ProcessingDecision:
    """A pure decision: the guarded transition to apply (or ``None`` for a transient
    failure that leaves the thought untouched) plus the closed reason for the span."""

    transition: TransitionOp | None
    reason: ProcessingReason


def build_processing_prompt(thought: Thought) -> str:
    """The bounded input_text handed to the aux call — the thought and its history."""
    revisited = (
        f"\n\n(You have revisited this {thought.no_progress_count} time(s) without resolving it.)"
        if thought.no_progress_count
        else ""
    )
    return f"The thought you are turning over:\n\n{thought.content}{revisited}"


def _transition(thought: Thought, to: ThoughtState, merge: dict[str, object]) -> TransitionOp:
    return TransitionOp(
        kind=THOUGHT_KIND,
        id=thought.id,
        from_state=ThoughtState.ACTIVE.value,
        to_state=to.value,
        patch=MemoryPatch(payload_merge=merge),
    )


def _park_or_terminate(
    thought: Thought, *, now: datetime, no_progress: bool
) -> ProcessingDecision:
    """Park with a widening backoff, or terminate at a bound. Bumps ``no_progress_count``
    when *no_progress* (a malformed attempt), always bumps ``park_count``."""
    new_np = thought.no_progress_count + (1 if no_progress else 0)
    new_park = thought.park_count + 1
    if no_progress and new_np >= MAX_NO_PROGRESS_COUNT:
        return ProcessingDecision(
            _transition(thought, ThoughtState.DROPPED, {"no_progress_count": new_np}),
            ProcessingReason.DROPPED_NO_PROGRESS,
        )
    if new_park > MAX_PARK_CYCLES:
        return ProcessingDecision(
            _transition(
                thought, ThoughtState.EXPIRED,
                {"no_progress_count": new_np, "park_count": new_park},
            ),
            ProcessingReason.EXPIRED_PARK_CAP,
        )
    backoff = PARK_BACKOFFS[min(thought.park_count, len(PARK_BACKOFFS) - 1)]
    merge = {
        "no_progress_count": new_np,
        "park_count": new_park,
        "parked_until": to_iso(now + backoff),
    }
    reason = ProcessingReason.PARKED_NO_PROGRESS if no_progress else ProcessingReason.PARKED
    return ProcessingDecision(_transition(thought, ThoughtState.PARKED, merge), reason)


def decide_processing_transition(
    thought: Thought, *, parsed: dict | None, raw: str, now: datetime
) -> ProcessingDecision:
    """Map an aux result to the thought's next state (pure; spec §4.1).

    ``resolve``/``drop`` are terminal. ``park`` backs off (or expires at the park cap).
    A malformed result (no valid ``outcome``, but the model DID respond) is a
    no-progress attempt → park+bump (or drop at the no-progress cap). A TRANSIENT
    failure (empty ``raw`` — the call itself failed/timed out) leaves the thought
    untouched, so provider flakiness never drops a good thought (refund-of-attempt)."""
    outcome = parsed.get("outcome") if isinstance(parsed, dict) else None
    if outcome == "resolve":
        return ProcessingDecision(
            _transition(thought, ThoughtState.RESOLVED, {}), ProcessingReason.RESOLVED
        )
    if outcome == "drop":
        return ProcessingDecision(
            _transition(thought, ThoughtState.DROPPED, {}), ProcessingReason.DROPPED
        )
    if outcome == "park":
        return _park_or_terminate(thought, now=now, no_progress=False)
    if not raw.strip():
        return ProcessingDecision(None, ProcessingReason.TRANSIENT_FAILURE)
    return _park_or_terminate(thought, now=now, no_progress=True)
```

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `feat(thought-processing): pure lifecycle — resolve/park/drop + bounded attempts (lm-705.2)`.

---

## Task 4: `ThoughtProcessingSelector` — the heartbeat emitter

**Files:** Modify `core/thought_processing.py` (add the selector); Create `tests/test_thought_processing_selector.py`.

**Interfaces:**
- Produces: `THOUGHT_PROCESSING_SELECTOR_ID = "thought-processing-selector"`; `ThoughtProcessingSelector(*, daily_ceiling=DEFAULT_DAILY_INTERNAL_CALL_CEILING, min_interval=DEFAULT_MIN_INTERPROCESSING_INTERVAL)` with `id` and `step(ctx) -> Sequence[Intent]`.
- Consumes: `live_thoughts` (`core/thought_view`), `internal_budget_available`/`internal_interval_elapsed` (`core/budget`), `format_traceparent` (`ports/tracer`), `LaunchInternalCognition`/`TransitionRecord` (`core/intents`), `TransitionOp` (`domain/memory`).

- [ ] **Step 1: Failing test** (`tests/test_thought_processing_selector.py`) — uses the bare-`TickContext` helper the other component tests use (`lifemodel.testing.tick.make_tick_context`, added in slice 1):

```python
from datetime import datetime, timezone

from lifemodel.core.budget import DEFAULT_MIN_INTERPROCESSING_INTERVAL
from lifemodel.core.intents import LaunchInternalCognition, TransitionRecord
from lifemodel.core.thought_processing import (
    ProcessingReason,
    ThoughtProcessingSelector,
)
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _rec(thought):
    # a MemoryRecord as the start-of-tick snapshot would carry it
    from lifemodel.testing.harness import draft_to_record  # helper added in THIS task's Step 3
    return draft_to_record(encode_thought(thought), now=NOW)


def _active(id_, salience):
    return build_thought(id=id_, content=f"c{id_}", state=ThoughtState.ACTIVE, salience=salience)


def test_selects_top_salience_active_thought():
    ctx = make_tick_context(
        state=State(),
        now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.3)), _rec(_active("thought:seed:b", 0.9))],
    )
    intents = list(ThoughtProcessingSelector().step(ctx))
    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(launches) == 1
    assert launches[0].subject_id == "thought:seed:b"          # top salience
    assert launches[0].json_schema is not None                 # a structured pass
    assert launches[0].instructions                            # processing framing


def test_empty_backlog_emits_nothing():
    ctx = make_tick_context(state=State(), now=NOW, objects=[])
    assert list(ThoughtProcessingSelector().step(ctx)) == []


def test_single_flight_blocks_when_a_pass_is_in_flight():
    ctx = make_tick_context(
        state=State(pending_internal_id="process-x"), now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)] == []


def test_interval_gate_blocks_a_recent_pass():
    ctx = make_tick_context(
        state=State(last_internal_call_at="2026-07-16T11:50:00+00:00"), now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)] == []


def test_budget_gate_blocks_at_ceiling():
    ctx = make_tick_context(
        state=State(internal_calls_today=50, internal_calls_day="2026-07-16"), now=NOW,
        objects=[_rec(_active("thought:seed:a", 0.9))],
    )
    assert [i for i in ThoughtProcessingSelector().step(ctx) if isinstance(i, LaunchInternalCognition)] == []


def test_rearms_expired_parked_thought_and_does_not_launch_it():
    parked = build_thought(
        id="thought:seed:p", content="cp", state=ThoughtState.PARKED, salience=0.9,
        park_count=1, parked_until="2026-07-16T06:00:00+00:00",  # past → re-eligible
    )
    ctx = make_tick_context(state=State(), now=NOW, objects=[_rec(parked)])
    intents = list(ThoughtProcessingSelector().step(ctx))
    transitions = [i for i in intents if isinstance(i, TransitionRecord)]
    launches = [i for i in intents if isinstance(i, LaunchInternalCognition)]
    assert len(transitions) == 1
    assert transitions[0].op.from_state == ThoughtState.PARKED.value
    assert transitions[0].op.to_state == ThoughtState.ACTIVE.value
    assert launches == []            # re-armed this tick, processed a later tick


def test_still_parked_thought_is_not_rearmed():
    parked = build_thought(
        id="thought:seed:p", content="cp", state=ThoughtState.PARKED, salience=0.9,
        parked_until="2026-07-17T00:00:00+00:00",  # future → still parked
    )
    ctx = make_tick_context(state=State(), now=NOW, objects=[_rec(parked)])
    assert list(ThoughtProcessingSelector().step(ctx)) == []
```

> **Add the `draft_to_record(draft: MemoryDraft, *, now: datetime) -> MemoryRecord` helper to `testing/harness.py` as this task's Step 3a** (encode a `MemoryDraft` into a `MemoryRecord` the start-of-tick snapshot would carry — stamp `created_at`/`updated_at=to_iso(now)`, `revision=0`, `schema_version=draft.schema_version`, copy the rest from the draft). Tasks 5 and 8 reuse it. The selector only reads `kind`/`id`/`state`/`payload`/`salience`, so the stamped fields are cosmetic but keep the record well-formed.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Add the selector** to `core/thought_processing.py`. New imports at the top of the module:

```python
from ..ports.tracer import format_traceparent
from .budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from .component import TickContext
from .intents import Intent, LaunchInternalCognition, TransitionRecord
from .thought_view import build_thought, live_thoughts  # build_thought only if needed
from .timeutil import from_iso  # for parked_until comparison
```

Then:

```python
THOUGHT_PROCESSING_SELECTOR_ID = "thought-processing-selector"


class ThoughtProcessingSelector:
    """Pick ONE live thought to ruminate on this tick, and re-arm expired parks (§4.1).

    0-LLM: it only emits intents. Re-arms every parked thought past its ``parked_until``
    (``parked→active``) so parking means "return later", not "shelve till expiry". Then,
    if the gates pass (single-flight, FR20 budget, min interval), emits ONE
    ``LaunchInternalCognition`` for the top-salience ACTIVE thought — the being's private,
    non-delivered pass. Emits no launch (idle 0-LLM, S5) when the active backlog is empty
    or any gate holds; the reason is a span field either way (spec §5)."""

    id: str = THOUGHT_PROCESSING_SELECTOR_ID

    def __init__(
        self,
        *,
        daily_ceiling: int = DEFAULT_DAILY_INTERNAL_CALL_CEILING,
        min_interval: timedelta = DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    ) -> None:
        self._daily_ceiling = daily_ceiling
        self._min_interval = min_interval

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        thoughts = live_thoughts(ctx.objects)
        intents: list[Intent] = []
        actives = []
        for t in thoughts:
            if t.state == ThoughtState.PARKED.value:
                if self._parked_is_due(t, ctx.now):
                    intents.append(
                        TransitionRecord(
                            op=TransitionOp(
                                kind=THOUGHT_KIND,
                                id=t.id,
                                from_state=ThoughtState.PARKED.value,
                                to_state=ThoughtState.ACTIVE.value,
                            )
                        )
                    )
            elif t.state == ThoughtState.ACTIVE.value:
                actives.append(t)

        reason, subject = self._pick(ctx, actives)
        if subject is not None:
            intents.append(
                LaunchInternalCognition(
                    prompt=build_processing_prompt(subject),
                    correlation_id=f"process-{subject.id}@{to_iso(ctx.now)}",
                    origin_traceparent=format_traceparent(ctx.trace),
                    subject_id=subject.id,
                    instructions=PROCESSING_INSTRUCTIONS,
                    json_schema=PROCESSING_JSON_SCHEMA,
                )
            )
        if ctx.logger is not None:
            ctx.logger.span.set(processing_reason=reason.value)
            if subject is not None:
                ctx.logger.span.set(thought_id=subject.id)
        return intents

    def _parked_is_due(self, thought: Thought, now: datetime) -> bool:
        if not thought.parked_until:
            return True  # parked with no window set → treat as due (defensive)
        try:
            return from_iso(thought.parked_until) <= now
        except (ValueError, TypeError):
            return True

    def _pick(
        self, ctx: TickContext, actives: list[Thought]
    ) -> tuple[ProcessingReason, Thought | None]:
        if not actives:
            return ProcessingReason.SKIPPED_EMPTY_BACKLOG, None
        if ctx.state.pending_internal_id is not None:
            return ProcessingReason.SKIPPED_IN_FLIGHT, None
        if not internal_interval_elapsed(
            ctx.state, now=ctx.now, min_interval=self._min_interval
        ):
            return ProcessingReason.SKIPPED_INTERVAL, None
        if not internal_budget_available(
            ctx.state, now=ctx.now, daily_ceiling=self._daily_ceiling
        ):
            return ProcessingReason.SKIPPED_NO_BUDGET, None
        return ProcessingReason.CHOSE_PROCESS, actives[0]  # live_thoughts is salience-desc
```

> `live_thoughts` returns most-salient-first (`core/thought_view.py:139`), so `actives[0]` is the top active thought. `build_thought` import is not needed by the selector itself — drop it if unused (mypy/ruff will flag).

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `feat(thought-processing): ThoughtProcessingSelector — re-arm parks + emit one launch, gated (lm-705.2)`.

---

## Task 5: `ThoughtProcessingApply` — the completion consumer

**Files:** Modify `core/thought_processing.py` (add the apply); Create `tests/test_thought_processing_apply.py`.

**Interfaces:**
- Produces: `THOUGHT_PROCESSING_APPLY_ID = "thought-processing-apply"`; `ThoughtProcessingApply()` with `id` and `step(ctx) -> Sequence[Intent]`.
- Consumes: `read_internal_result`/`KIND_INTERNAL_RESULT` (`core/taxonomy`), `decide_processing_transition` (Task 3), `_decode_live` equivalence via reading `ctx.objects` for the subject thought.

- [ ] **Step 1: Failing test** (`tests/test_thought_processing_apply.py`)

```python
from datetime import datetime, timezone

from lifemodel.core.intents import TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _ctx(subject_id, *, parsed, raw, thought_state=ThoughtState.ACTIVE):
    thought = build_thought(id="thought:seed:a", content="ca", state=thought_state)
    sig = internal_result_signal(
        origin_id="r1", correlation_id="process-x", raw=raw, parsed=parsed,
        timestamp="2026-07-16T12:00:00+00:00",
    )
    return make_tick_context(
        state=State(pending_internal_id="process-x", pending_internal_subject_id=subject_id),
        now=NOW, objects=[draft_to_record(encode_thought(thought), now=NOW)], signals=[sig],
    )


def test_resolve_emits_terminal_transition():
    ctx = _ctx("thought:seed:a", parsed={"outcome": "resolve"}, raw="{...}")
    trs = [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)]
    assert len(trs) == 1
    assert trs[0].op.id == "thought:seed:a"
    assert trs[0].op.to_state == ThoughtState.RESOLVED.value


def test_malformed_bumps_no_progress():
    ctx = _ctx("thought:seed:a", parsed=None, raw="junk")
    trs = [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)]
    assert trs[0].op.to_state == ThoughtState.PARKED.value
    assert trs[0].op.patch.payload_merge["no_progress_count"] == 1


def test_transient_failure_emits_no_transition():
    ctx = _ctx("thought:seed:a", parsed=None, raw="   ")
    assert [i for i in ThoughtProcessingApply().step(ctx) if isinstance(i, TransitionRecord)] == []


def test_no_subject_is_a_noop():          # a subjectless (noticing) pass, or cleared subject
    ctx = _ctx(None, parsed={"outcome": "resolve"}, raw="{...}")
    assert list(ThoughtProcessingApply().step(ctx)) == []


def test_no_internal_result_signal_is_a_noop():   # runs on every completion frame; guards
    ctx = make_tick_context(state=State(), now=NOW, objects=[], signals=[])
    assert list(ThoughtProcessingApply().step(ctx)) == []


def test_subject_no_longer_live_is_a_noop():      # thought already terminal — nothing to do
    ctx = _ctx("thought:seed:gone", parsed={"outcome": "resolve"}, raw="{...}")
    assert list(ThoughtProcessingApply().step(ctx)) == []
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Add the apply** to `core/thought_processing.py`. Extend imports:

```python
from .taxonomy import KIND_INTERNAL_RESULT, read_internal_result
```

Then:

```python
THOUGHT_PROCESSING_APPLY_ID = "thought-processing-apply"


class ThoughtProcessingApply:
    """Turn a completed processing pass's typed result into the thought's next state.

    The runner's injected ``apply`` (lm-705.6): it runs only inside the
    ``ASYNC_COMPLETION`` frame :func:`~lifemodel.core.internal_cognition.run_internal_completion`
    seeds, so it guards on an ``internal_result`` signal + a matching in-flight subject
    and no-ops otherwise (a subjectless noticing pass, a cleared/terminal subject, or a
    non-completion frame all fall through to ``[]``). Emits at most one
    ``TransitionRecord`` (the atomic committer applies it under the lock)."""

    id: str = THOUGHT_PROCESSING_APPLY_ID

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        subject_id = ctx.state.pending_internal_subject_id
        if subject_id is None:
            return []
        result = next(
            (
                read_internal_result(s)
                for s in ctx.signals
                if s.kind == KIND_INTERNAL_RESULT
                and s.payload.get("correlation_id") == ctx.state.pending_internal_id
            ),
            None,
        )
        if result is None:
            return []
        thought = self._live_subject(ctx, subject_id)
        if thought is None:
            self._log(ctx, ProcessingReason.NO_SUBJECT, subject_id)
            return []
        decision = decide_processing_transition(
            thought, parsed=result.parsed, raw=result.raw, now=ctx.now
        )
        self._log(ctx, decision.reason, subject_id)
        return [TransitionRecord(op=decision.transition)] if decision.transition is not None else []

    def _live_subject(self, ctx: TickContext, subject_id: str) -> Thought | None:
        for t in live_thoughts(ctx.objects):
            if t.id == subject_id and t.state == ThoughtState.ACTIVE.value:
                return t
        return None

    def _log(self, ctx: TickContext, reason: ProcessingReason, thought_id: str) -> None:
        if ctx.logger is not None:
            ctx.logger.span.set(processing_reason=reason.value, thought_id=thought_id)
```

> **Note:** the subject is required to be `active` — the selector only launches passes for active thoughts, and `run_internal_completion` runs BEFORE its own `pending_*` clear, so the subject is still the active thought that was launched. A subject found `parked`/terminal (a race with an admin mutation) reads as `NO_SUBJECT` → no-op, never a stale transition.

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `feat(thought-processing): ThoughtProcessingApply — typed outcome to guarded transition (lm-705.2)`.

---

## Task 6: First-emitter prereqs — birth-voice (#1) + single-flight (#2) + subject wiring

**Files:** Modify `core/internal_cognition.py`, `adapters/internal_runner.py`; Create `tests/test_internal_runner_single_flight.py`, `tests/test_internal_completion_voice.py`.

**Interfaces:**
- Produces: `run_internal_completion(lm, egress, target, *, correlation_id, result, apply, voice=None)`; `InternalCognitionRunner(..., voice=None)`; `runner.launch(request, correlation_id, *, subject_id=None) -> bool` (single-flight: returns `False` when `pending_internal_id` is already set).
- Consumes: `dispatch_launches(lm, report, egress, target, *, voice=None)` (already accepts `voice`, `core/proactive.py:134`).

- [ ] **Step 1: Failing tests.**

`tests/test_internal_runner_single_flight.py` — single-flight + subject/interval stamping (mirror `tests/test_internal_runner.py`'s fake-loop + `FakeLlmPort` setup):

```python
# ... reuse the harness/fixtures from tests/test_internal_runner.py ...

def test_single_flight_denies_a_second_launch(runner_fixture):
    runner, lm = runner_fixture  # pending_internal_id starts None
    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True
    # first launch set pending_internal_id → the second is denied WITHOUT reserving budget
    calls_before = lm.state_actor.state.internal_calls_today
    assert runner.launch(REQ, "c-2", subject_id="thought:seed:b") is False
    assert lm.state_actor.state.internal_calls_today == calls_before   # no second reserve

def test_launch_stamps_subject_and_interval(runner_fixture):
    runner, lm = runner_fixture
    assert runner.launch(REQ, "c-1", subject_id="thought:seed:a") is True
    assert lm.state_actor.state.pending_internal_subject_id == "thought:seed:a"
    assert lm.state_actor.state.last_internal_call_at is not None

def test_recover_stale_clears_subject_too(runner_fixture):
    runner, lm = runner_fixture
    lm.state_actor.apply([UpdateState({"pending_internal_id": "x", "pending_internal_subject_id": "y"})])
    runner.recover_stale(lm)
    assert lm.state_actor.state.pending_internal_id is None
    assert lm.state_actor.state.pending_internal_subject_id is None
```

`tests/test_internal_completion_voice.py` — the birth-voice thread (a completion frame carrying an incidental `LaunchProactive` goes through the voice pre-flight):

```python
def test_completion_dispatches_incidental_proactive_through_voice():
    # build a completion lm whose graph WILL surface a LaunchProactive on the frame,
    # a fake egress, and a spy `voice`; run run_internal_completion(..., voice=spy)
    # assert the spy voice was consulted before egress delivered (mirrors the birth
    # pre-flight assertion in tests/test_being_platform_genesis*.py / dispatch_launches tests)
    ...
    assert spy_voice.consulted is True
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3a: Thread `voice` through the completion** (`core/internal_cognition.py`). Add the param + import and pass it to `dispatch_launches`:

```python
from ..adapters.session_end import GatewayBirthVoice  # TYPE hint only — keep core Hermes-free:
```

> **Correction (core stays Hermes-free):** do NOT import a Hermes/adapter type. Type `voice` structurally. `dispatch_launches` already accepts `voice` as an opaque object (`core/proactive.py`), so `run_internal_completion` just forwards it untyped (`voice: object | None = None`), exactly as the adapter passes `self._voice` today. Update the signature and the one call:

```python
def run_internal_completion(
    lm: LifeModel,
    egress: ProactiveEgressPort,
    target: Mapping[str, str | None],
    *,
    correlation_id: str,
    result: InternalCognitionResult,
    apply: Component,
    voice: object | None = None,
) -> ReachOutcome | None:
    ...
    report = run_frame(lm.coreloop, [signal], trigger=FrameTrigger.ASYNC_COMPLETION)
    outcome = dispatch_launches(lm, report, egress, target, voice=voice)  # ← birth pre-flight (prereq #1)
    lm.state_actor.apply(
        [UpdateState({"pending_internal_id": None, "pending_internal_subject_id": None})]  # ← clear subject too
    )
    return outcome
```

> Confirm `dispatch_launches`'s `voice` parameter type in `core/proactive.py`; forward with the same annotation it uses (likely a `BirthVoice`-ish `Protocol` or `object | None`). Match it so mypy stays green.

- [ ] **Step 3b: Runner** (`adapters/internal_runner.py`): add `voice` to `__init__`, the single-flight gate + subject/interval stamp to `launch`, forward `voice` in `_run`, clear subject in `recover_stale`/`_clear_pending_fail_loud`:

```python
def __init__(self, ..., voice: object | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    ...
    self._voice = voice

def launch(self, request, correlation_id, *, subject_id: str | None = None) -> bool:
    lm = self._build_lm()
    assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
    now = lm.clock.now()
    with state_actor_lock():
        state = lm.state_actor.state
        if state.pending_internal_id is not None:
            return False                      # single-flight (prereq #2)
        reserved = reserve_internal_call(state, now=now, daily_ceiling=self._daily_ceiling)
        if reserved is None:
            return False
        lm.state_actor.apply([UpdateState({
            "internal_calls_today": reserved.internal_calls_today,
            "internal_calls_day": reserved.internal_calls_day,
            "pending_internal_id": correlation_id,
            "pending_internal_subject_id": subject_id,
            "last_internal_call_at": to_iso(now),
        })])
    task = self._gateway_loop.create_task(self._run(request, correlation_id))
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)
    return True
```

In `_run`, pass `voice=self._voice` to `run_internal_completion(...)`. In `recover_stale` and `_clear_pending_fail_loud`, extend the `UpdateState` to also null `pending_internal_subject_id`. Add `from ..core.timeutil import to_iso`.

- [ ] **Step 4: Run → pass** (plus existing `tests/test_internal_runner.py` — update its `launch(...)` calls to the new keyword-only `subject_id`; the old positional 2-arg form still works since `subject_id` is keyword-only with a default). `make check`.

- [ ] **Step 5: Commit** `feat(internal-cognition): birth-voice thread + single-flight gate + durable subject (lm-705.2, prereqs #1/#2)`.

---

## Task 7: Compose the selector + wire the apply/voice/subject through the adapter

**Files:** Modify `composition.py`, `adapters/being_platform.py`; extend `tests/test_composition.py`.

**Interfaces:**
- Produces: `ThoughtProcessingSelector` present in `build_lifemodel`'s registry (COGNITION layer); the adapter's runner built with `apply=ThoughtProcessingApply()`, `voice=self._voice`, and driving `launch(..., subject_id=..., instructions=..., json_schema=...)`.
- Consumes: `ThoughtProcessingSelector`/`ThoughtProcessingApply` (Tasks 4/5), `DEFAULT_DAILY_INTERNAL_CALL_CEILING` (from `core/budget`).

- [ ] **Step 1: Failing test** (`tests/test_composition.py`, add):

```python
def test_build_lifemodel_registers_processing_selector():
    from lifemodel.composition import build_lifemodel
    from lifemodel.core.thought_processing import THOUGHT_PROCESSING_SELECTOR_ID

    lm = build_lifemodel(base_dir=_tmp_base_dir())
    ids = {m.id for m in lm.coreloop._registry.manifests()}
    assert THOUGHT_PROCESSING_SELECTOR_ID in ids
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3a: Register the selector** in `build_lifemodel` (`composition.py`, mirror the `ThoughtCapture` registration from slice 1):

```python
from .core.thought_processing import ThoughtProcessingSelector, THOUGHT_PROCESSING_SELECTOR_ID

registry.register(
    ThoughtProcessingSelector(),
    ComponentManifest(
        id=THOUGHT_PROCESSING_SELECTOR_ID,
        type="thought-processing-selector",
        layer=ComponentLayer.COGNITION,   # a cognitive decision to ruminate
        metric_surface=(),
        accepts_signals=True,
    ),
)
```

> `ThoughtProcessingApply` is NOT registered here — it is the runner's injected `apply`, run ONLY inside the completion frame (`run_internal_completion` registers it into that frame's ephemeral registry). Registering it in `build_lifemodel` would run it on every heartbeat (harmless — it guards on the signal — but the seam's design is that `apply` is completion-only).

- [ ] **Step 3b: Wire the adapter** (`adapters/being_platform.py`):
  1. Replace the module constant import: `from .core.budget import DEFAULT_DAILY_INTERNAL_CALL_CEILING` (remove the local `DEFAULT_DAILY_INTERNAL_CALL_CEILING = 50` at line 88 — DRY, one source of truth).
  2. In `connect()`'s runner construction (line ~422), pass `apply=ThoughtProcessingApply()` (replacing `NullInternalApply`) and `voice=self._voice`.
  3. In `_tick()` (line ~224), map the launch's own call spec through:

```python
        if self._internal_runner is not None:
            for launch in report.internal_launches:
                self._internal_runner.launch(
                    InternalCognitionRequest(
                        instructions=launch.instructions or _INTERNAL_COGNITION_INSTRUCTIONS,
                        input_text=launch.prompt,
                        json_schema=launch.json_schema,
                    ),
                    launch.correlation_id,
                    subject_id=launch.subject_id,
                )
```

Add `from ..core.thought_processing import ThoughtProcessingApply` and keep `NullInternalApply` imported only if still used elsewhere (it is not after this change — drop the import).

- [ ] **Step 4: Run → pass** (plus `tests/test_being_platform_internal_cognition.py` — update any `runner.launch(...)` positional calls to the new keyword `subject_id`; the tick-drive test should now also carry `subject_id`). `make check`.

- [ ] **Step 5: Commit** `feat(thought-processing): register selector + wire apply/voice/subject through the adapter (lm-705.2)`.

---

## Task 8: Real-code sim — backlog health, bounds, idle-0-LLM, cost cap (spec §6)

**Files:** Modify `testing/harness.py` (add `build_processing_lifemodel` + `draft_to_record`); Create `tests/test_thought_processing_harness.py`.

**Interfaces:**
- Produces: `build_processing_lifemodel(*, clock=None) -> LifeModel` (real `CoreLoop` over fake ports with `ThoughtProcessingSelector` registered). (`draft_to_record` was already added to `testing/harness.py` in Task 4.)
- Consumes: the existing fake-port builder (`_build_fake_lifemodel` / `build_capture_lifemodel` from slice 1), `run_frame`, `run_internal_completion`.

- [ ] **Step 1: Failing tests** (drive the REAL frame + the REAL completion; assert the store):

```python
def test_seeded_thought_is_processed_to_resolved():
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:a", content="dentist Friday", salience=0.8)
    # heartbeat → selector emits a launch (captured from the report)
    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert len(report.internal_launches) == 1
    launch = report.internal_launches[0]
    # simulate the runner's reserve (set pending_* the way launch() would) ...
    _set_pending(lm, launch)
    # ... then the completion with a "resolve" result via the REAL apply
    run_internal_completion(
        lm, _fake_egress(), {}, correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw='{"outcome":"resolve"}', parsed={"outcome": "resolve"}),
        apply=ThoughtProcessingApply(),
    )
    assert read_thought(lm.state, "thought:seed:a") is None       # resolved → no longer live

def test_idle_empty_backlog_is_zero_launches():
    lm = build_processing_lifemodel()
    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    assert report.internal_launches == ()

def test_repeated_malformed_drops_at_no_progress_cap():
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:a", content="c", salience=0.8)
    for _ in range(MAX_NO_PROGRESS_COUNT):
        _reset_pacing(lm)                                # clear interval/pending between rounds
        report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
        if not report.internal_launches:                 # thought parked → re-arm it
            run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
            continue
        launch = report.internal_launches[0]
        _set_pending(lm, launch)
        run_internal_completion(
            lm, _fake_egress(), {}, correlation_id=launch.correlation_id,
            result=InternalCognitionResult(raw="junk", parsed=None),
            apply=ThoughtProcessingApply(),
        )
    assert read_thought(lm.state, "thought:seed:a") is None       # dropped (bounded, no spiral)

def test_daily_ceiling_caps_launches():
    lm = build_processing_lifemodel()   # ceiling forced low via a small daily_ceiling selector
    ...                                  # after `ceiling` launches, further heartbeats emit none
```

> These sim tests exercise the pieces the adapter glues in Task 7 (`launch`'s reserve + `run_internal_completion`) directly, because the harness has no gateway asyncio loop. `_set_pending`/`_reset_pacing`/`_seed_active_thought` are small local helpers over `lm.state_actor`/the memory store — write them at the top of the test module. Assert the invariants the spec §6 names: **backlog health** (processed, no starve/spiral), **bounds terminate** (drop/expire), **idle 0-LLM**, **cost ≤ ceiling**.

- [ ] **Step 2: Run → fail** (`build_processing_lifemodel` not defined).

- [ ] **Step 3:** Add `build_processing_lifemodel` (mirror slice 1's `build_capture_lifemodel`, registering `ThoughtProcessingSelector` instead) to `testing/harness.py`. (`draft_to_record` is already there from Task 4.)

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `test(thought-processing): real-code sim — backlog health, bounds terminate, idle-0-LLM, cost cap (lm-705.2)`.

---

## Task 9: Host-integration extension + follow-up beads + spec amendments

**Files:** Modify `tests/hermes_internal_cognition_integration.py`; docs `docs/superpowers/specs/2026-07-16-waking-mind-attention-economy-design.md`; `bd` beads.

- [ ] **Step 1: Extend the isolated-`HERMES_HOME` host-integration test** (never the live being): seed one active thought, drive a heartbeat → a real `LaunchInternalCognition` → the runner calls the aux model via the sanctioned `ctx.llm` lane, **delivers nothing** (no gateway turn / no message on any lane), the completion applies a **real thought transition** (e.g. `resolve` → the thought is no longer live), `pending_internal_id`/`pending_internal_subject_id` clear; single-flight denies a concurrent launch; the FR20 ceiling denies the N+1 that day. Reuse the existing test's `HERMES_HOME` scaffolding + slow/integration marker.

- [ ] **Step 2: File the deferred-prereq beads** (children of lm-705 or lm-1w2 hardening epic, as fits):

```bash
bd create --title "Shared FR20 ceiling: rumination + proactive draw from one durable budget (spec §4.5)" \
  --type task -p 2 --parent lm-705 \
  --description "Amended in lm-705.2: internal rumination is capped; proactive stays bounded by its own drive/backstop dynamics. Unify into one durable expensive-cognition-call ceiling that BOTH paths reserve from; amend spec §4.5 to match. Touches the load-bearing proactive path — do after live traces."
bd create --title "Trace-weave: parent the internal-cognition completion span on its launch (origin_traceparent)" \
  --type task -p 3 --parent lm-705 \
  --description "lm-705.2 carries LaunchInternalCognition.origin_traceparent but run_frame/coreloop.tick roots its own trace, so completion is not child_of the launch. They already correlate via correlation_id (a span field). Thread origin_traceparent into the ASYNC_COMPLETION frame's root so the launch→call→completion weave lands under one trace_id (spec §4.4)."
bd create --title "Refund-on-transient-failure for the FR20 internal call quota (Minor)" \
  --type task -p 3 --parent lm-705 \
  --description "A provider outage / timeout still consumes an FR20 call reservation. lm-705.2 mitigates (single-flight + min interval + the transient/malformed split leaves the thought un-penalized), but the CALL quota is not refunded. Refund a reservation on a transient (empty-raw) failure so an outage cannot burn the day's ceiling."
bd create --title "Cheap-model routing for internal cognition (blocked on host aux-slot lane)" \
  --type task -p 3 --parent lm-705 \
  --description "ctx.llm.acomplete_structured hard-codes task=None (agent/plugin_llm.py), so a third-party plugin cannot route lifemodel_internal to a cheap aux model via the sanctioned lane. lm-705.2 routes to the main model + bounds cost by the FR20 call quota. Wire model=/provider= override via plugins.entries.lifemodel.llm.allow_model_override when the host exposes it; see adapters/plugin_llm_adapter.py + lm-fgs."
```

> Check `bd ready`/`bd list` first for an existing cheap-model / upstream bead (e.g. **lm-fgs**) and link rather than duplicate if one already covers it.

- [ ] **Step 3: Amend the spec** (`...waking-mind-attention-economy-design.md`): in §4.5 note the shared-FR20 amendment (internal capped; proactive bounded by its own dynamics; unified ceiling → follow-up bead) and the cheap-model host-block; add a line to §10 review log referencing lm-705.2's decisions. Keep it to a few honest sentences — do not rewrite the spec.

- [ ] **Step 4:** `make check` (the integration test may be gated behind its marker). Commit `test(thought-processing): host-integration processing pass + deferred-prereq beads + spec amendments (lm-705.2)`.

---

## Self-Review (run after execution)

- **Spec coverage (§3.2/§4.1/§4.5):** non-delivering processing path reused from the seam ✓ (Tasks 6/7) · typed outcome schema + validation ✓ (Task 3) · atomic transition commit ✓ (`TransitionRecord`, Task 5) · bounded lifecycle: max attempts→drop, no_progress increment, park backoff, max park cycles→expire, terminal-on-malformed ✓ (Task 3) · hard FR20 quota + min inter-processing interval ✓ (Tasks 1/6) · idle 0-LLM / S5 ✓ (Task 4 gates + Task 8 sim) · forced observability with a closed positive-reason set, thought id as a field ✓ (`ProcessingReason`, Tasks 4/5) · sim through the real-code harness ✓ (Task 8).
- **First-emitter prereqs:** birth-voice threaded (#1, Task 6) ✓ · single-flight gate (#2, Task 6) ✓ · shared-FR20 amended + bead (#3, Task 9) ✓ · trace-weave bead (#4, Task 9) ✓ · cheap-model host-block bead (#5, Task 9) ✓ · refund-on-transient bead (#6, Task 9) ✓.
- **Not built here (slice boundaries):** minting a contact desire → slice 3 (lm-705.3); residue/opinion field → lm-adz; the arbiter → slice 4 (lm-705.4); spontaneous thoughts → Phase 6.
- **Type consistency:** `decide_processing_transition(thought, *, parsed, raw, now)` and `ProcessingDecision(transition, reason)` are used identically in Tasks 3/5; `launch(request, correlation_id, *, subject_id=None)` matches between Task 6 (runner) and Task 7 (adapter); `ThoughtProcessingSelector`/`ThoughtProcessingApply` ids match between Tasks 4/5/7. `internal_budget_available`/`internal_interval_elapsed` signatures match between Task 1 (def) and Task 4 (use).
- **Confirm-before-code (framework details, logic fully specified):** the `dispatch_launches` `voice` parameter's exact type in `core/proactive.py` (forward it verbatim); `make_tick_context`'s exact kwargs (`state=`, `now=`, `objects=`, `signals=`) in `testing/tick.py`; `ComponentLayer.COGNITION` member name; the fake-port builder name in `testing/harness.py` slice 1 added (`build_capture_lifemodel`/`_build_fake_lifemodel`).

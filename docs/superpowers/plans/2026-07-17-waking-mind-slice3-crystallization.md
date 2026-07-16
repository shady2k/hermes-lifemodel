# Waking Mind — Slice 3: Thought Crystallization + `Commitment` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A processed thought **crystallizes** into a durable BDI object. Rumination's typed outcome gains a **closed, discriminated** variant `crystallize_commitment`; on it, `ThoughtProcessingApply` builds a new **`Commitment`** (a follow-up the being owes — "ask how their interview went"), emits its `PutRecord`, and resolves the source thought — **in one atomic commit, producing the object and stopping there (no contact, no send)**.

**Architecture:** Add the first catalog **extension type** `Commitment` (frozen dataclass + state machine + registry registration), a `commitment_view` write/read door (mirroring `thought_view`/`desire_view`), and extend the slice-2 processing seam: the outcome schema/instructions gain `crystallize_commitment`; the pure `decide_processing_transition` routes it (carrying the validated commitment fields, or routing a malformed crystallize through the existing `no_progress_count` bound); `ThoughtProcessingApply` builds+encodes the `Commitment` (registry-validated), emits `PutRecord(commitment)` **plus** the source-thought `TransitionRecord(active→resolved)`, and logs `processed_crystallize` + `crystallized_kind`/`crystallized_id`. A builder/registry failure falls back to the no-progress bound.

**Tech Stack:** Python 3.11 stdlib only (runtime), `uv`/`ruff`/`mypy --strict`/`pytest` (dev). The typed BDI object core (`domain/objects/`), the closed `KindRegistry` (`domain/objects/registry.py`), the intent bus (`PutRecord`/`TransitionRecord` + `PutOp`/`TransitionOp`), the internal-cognition completion seam (`core/internal_cognition.py`), the real-code fake-port harness (`testing/harness.py`), and the real-Hermes integration driver (`tests/hermes_internal_cognition_integration.py`).

## Global Constraints

- **bd:** slice 3 of epic **lm-705** — task **lm-705.3** (re-scoped v3). Spec: `docs/superpowers/specs/2026-07-16-waking-mind-attention-economy-design.md` (§3 item 3, §4.1, **§4.2**, §5, §6, §7), status **v3.1.1** (codex-reviewed twice: `019f6c40`). Rides slice 2 (`core/thought_processing.py`, shipped).
- **Closed, discriminated crystallization — NOT a runtime-arbitrary kind (spec §4.1/§4.2).** The outcome enum gains exactly `crystallize_commitment`; a new target later is a new registered variant + builder, never an accepted arbitrary `kind` string. `PutRecord` does **not** validate a kind — only a trusted builder + `default_registry().encode()` does; a free kind+payload could clobber a singleton.
- **Crystallization failure is bounded (spec §4.1, codex I2).** A `crystallize_commitment` whose payload fails validation (pure field check *or* `registry.encode`) is a **no-progress** outcome, routed through the existing `no_progress_count` bound (→ park+bump, drop at the cap) — never an uncaught exception that strands the thought without incrementing its cap. Only an empty-`raw` transport failure stays transient/unpenalized.
- **Atomic commit (spec §4.2).** The source-thought `TransitionRecord(active→resolved)` and the `PutRecord(commitment)` are emitted **together** — the `StateActor` batches them under one `BEGIN IMMEDIATE`, so a rollback leaves neither a resolved thought nor a stray commitment.
- **`Commitment` model (spec §4.2, finalized by codex I3):** **non-singleton** (many coexist); a **deterministic** id from the source thought + a content fingerprint (never random, never a bare global content hash); typed trigger (`trigger_kind ∈ time|event|condition`, `trigger_value`, optional `due_at`); `basis ∈ promised|follow_up|self_assumed`; `salience`/`expires_at` on the **base envelope**, not the semantic payload; full state machine `active → honoured|dropped|expired|deferred`, `deferred → active|honoured|dropped|expired`; registry-guarded. **Boundary vs `Intention`:** a `Commitment` is an enduring *owed follow-up / source object*, not an executable send-gating plan — it becomes a `Desire`→`Intention` only in the **deferred** contact work.
- **"No send" is structural but narrow (spec §4.2, codex I4).** This slice emits **no** `LaunchProactive`, and **no** current component consumes a `Commitment` as a contact source (aggregation reads only the contact `Desire`/`Intention`; `CognitionLauncher` launches only from the singleton active contact `Desire`; snapshot-per-frame means a freshly-inserted commitment is not read in the same frame). *(A completion frame can still dispatch an **incidental** proactive launch from an unrelated pre-existing contact `Desire` — that is the seam's strand-fix, not this slice sending.)* Acceptance test: a completion frame with **no** pre-existing contact desire → **zero** `LaunchProactive`.
- **Provenance direction target→source (spec §4.2, codex M2).** The `Commitment` carries `source_thought_ids` + qualified `provenance.source_object_ids`; the source Thought gets **no** new `crystallized_object_id` field (the registry rejects unknown payload keys).
- **Observability (spec §5, codex I6).** Add the closed reason `processed_crystallize` + span fields `thought_id`, `crystallized_kind`, `crystallized_id` (never a reason-per-id), stamped when the build+encode succeed. `processed_crystallize` records the *decision* — the component span persists **before** the commit, so durable success is the committed row, not the span alone.
- **Registry consistency (spec §4.2, codex I5).** Registering `commitment` must keep the catalog **terminal-consistent**: no state string terminal for one kind and live for another. `active`/`deferred` are live for `Commitment` (consistent with `desire`); `honoured` is new + terminal; `dropped`/`expired` are terminal across kinds. Extend the object-kinds consistency test.
- **State fields:** no new `State` (`state/model.py`) fields — the crystallization is a memory-object write, not a `runtime_state` scalar. **No DB migration** — a new `kind` needs none (`domain/objects/registry.py`).
- **Creation is not this slice.** The appraiser that *creates* thoughts is dormant live (lm-705.11); slice 3 is exercised by **seeded** thoughts. Slice 3 does not touch thought creation.
- **Every step ends green:** `make check` (ruff format --check · ruff check · mypy -p lifemodel · pytest).

## File Structure

- **Create** `domain/objects/commitment.py` — `Commitment(BaseObject)` + `CommitmentState`/`CommitmentBasis`/`CommitmentTriggerKind` enums + `COMMITMENT_TRANSITIONS`.
- **Modify** `domain/objects/registry.py` — add `Commitment`/`COMMITMENT_TRANSITIONS` to `_CATALOG`.
- **Modify** `domain/objects/__init__.py` — export the new symbols.
- **Create** `core/commitment_view.py` — `COMMITMENT_KIND`, deterministic `crystallized_commitment_id`, `build_commitment`, `encode_commitment`, `read_live_commitments`, `LIVE_COMMITMENT_STATES` (mirror `core/thought_view.py`).
- **Modify** `core/thought_processing.py` — extend `PROCESSING_JSON_SCHEMA` + `PROCESSING_INSTRUCTIONS` + `ProcessingReason` + `ProcessingDecision` + `decide_processing_transition` (Task 3), and `ThoughtProcessingApply.step` (Task 4).
- **Modify** `testing/harness.py` — a `seed_active_thought` helper if not already present (Task 5 may reuse slice-2 helpers).
- **Modify** `tests/hermes_internal_cognition_integration.py` — a crystallize scenario (Task 6).
- **Tests:** `tests/test_commitment_object.py`, `tests/test_commitment_view.py`, `tests/test_thought_processing_crystallize_decide.py`, `tests/test_thought_processing_crystallize_apply.py`, `tests/test_thought_processing_crystallize_harness.py`, and the existing object-kinds consistency test (extended).

---

## Task 1: The `Commitment` domain type + catalog registration + consistency

**Files:** Create `domain/objects/commitment.py`; Modify `domain/objects/registry.py`, `domain/objects/__init__.py`; Create `tests/test_commitment_object.py`; extend the existing object-kinds terminal-consistency test.

**Interfaces:**
- Produces: `Commitment(BaseObject)` with semantic fields `content: str`, `basis: CommitmentBasis`, `trigger_kind: CommitmentTriggerKind`, `trigger_value: str`, `due_at: str | None`, `source_thought_ids: tuple[str, ...]`, `other_regarding_value: float`; `CommitmentState` (`ACTIVE/DEFERRED/HONOURED/DROPPED/EXPIRED`), `CommitmentBasis` (`PROMISED/FOLLOW_UP/SELF_ASSUMED`), `CommitmentTriggerKind` (`TIME/EVENT/CONDITION`); `COMMITMENT_TRANSITIONS`; `KIND="commitment"`, `SCHEMA_VERSION=1`.
- Consumes: `domain/objects/base.py` (`BaseObject`, `BaseFields`, `req_str`, `opt_str`, `req_float`, `req_enum`, `req_str_tuple`, `state_set`); `domain/objects/registry.py` (`KindSpec`, `_CATALOG`).

- [ ] **Step 1: Failing test** (`tests/test_commitment_object.py`)

```python
import pytest

from lifemodel.domain.objects import (
    Commitment,
    CommitmentBasis,
    CommitmentState,
    CommitmentTriggerKind,
    default_registry,
)
from lifemodel.domain.objects.commitment import COMMITMENT_TRANSITIONS


def _commitment(**over):
    base = dict(
        id="commitment:seed:abc",
        state=CommitmentState.ACTIVE.value,
        source="thought-processing-apply",
        content="ask how their interview on Friday went",
        basis=CommitmentBasis.FOLLOW_UP,
        trigger_kind=CommitmentTriggerKind.EVENT,
        trigger_value="next time we talk",
        due_at=None,
        source_thought_ids=("thought:seed:xyz",),
        other_regarding_value=0.8,
        salience=0.6,
    )
    base.update(over)
    return Commitment(**base)


def test_commitment_roundtrips_through_the_registry():
    reg = default_registry()
    c = _commitment()
    record = reg.encode(c)
    assert record.kind == "commitment"
    assert record.salience == 0.6  # base-envelope field, not the payload
    back = reg.decode(_as_record(record))
    assert isinstance(back, Commitment)
    assert back.content == c.content
    assert back.basis == CommitmentBasis.FOLLOW_UP
    assert back.trigger_kind == CommitmentTriggerKind.EVENT
    assert back.source_thought_ids == ("thought:seed:xyz",)


def test_commitment_is_a_known_kind():
    assert "commitment" in default_registry().kinds()


def test_commitment_transition_table_is_complete():
    reg = default_registry()
    reg.validate_transition("commitment", "active", "honoured")
    reg.validate_transition("commitment", "active", "deferred")
    reg.validate_transition("commitment", "deferred", "active")
    with pytest.raises(Exception):
        reg.validate_transition("commitment", "honoured", "active")  # terminal


def test_catalog_is_terminal_consistent_including_commitment():
    # no state string may be terminal for one kind and live (non-terminal) for another
    reg = default_registry()
    live = reg.live_states()
    for kind in reg.kinds():
        terminal = reg.terminal_states_of(kind)
        assert not (terminal & live), f"{kind}: {terminal & live} both terminal and live"


def _as_record(draft):
    from lifemodel.domain.memory import MemoryRecord

    return MemoryRecord(
        kind=draft.kind, id=draft.id, state=draft.state, payload=draft.payload,
        source=draft.source, recipient_id=draft.recipient_id, salience=draft.salience,
        confidence=draft.confidence, expires_at=draft.expires_at,
        created_at="2026-07-17T00:00:00+00:00", updated_at="2026-07-17T00:00:00+00:00",
        revision=0, schema_version=draft.schema_version,
    )
```

- [ ] **Step 2: Run → fail** (`ImportError: cannot import name 'Commitment'`).

Run: `uv run pytest tests/test_commitment_object.py -v`

- [ ] **Step 3: Create `domain/objects/commitment.py`** (mirror `domain/objects/desire.py` exactly — same imports, `@dataclass(frozen=True, kw_only=True)`, `_semantic_payload`/`_rebuild`):

```python
"""``Commitment`` — a follow-up the being *owes*, having thought it over (§4.2, v3).

The first catalog EXTENSION type (D8): what the being decided to do after processing a
thought — "ask how their interview went", "come back to the moving-house topic". HLA §4.1:
"the strongest non-intrusive reason, serving the other". A crystallization target
(``core/thought_processing.py``): a processed thought becomes a ``Commitment`` via ``PutRecord``,
and the source thought resolves. NON-singleton (many coexist), unlike the contact ``Desire``.

Distinct from ``Intention``: a ``Commitment`` is an enduring *owed follow-up / source object*;
an ``Intention`` (Bratman) is an executable, send-gating plan. Turning a commitment into an
outreach (commitment → contact ``Desire`` → ``Intention``) is the deferred contact work, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import (
    BaseFields,
    BaseObject,
    opt_str,
    req_enum,
    req_float,
    req_str,
    req_str_tuple,
    state_set,
)


class CommitmentState(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    HONOURED = "honoured"
    DROPPED = "dropped"
    EXPIRED = "expired"


class CommitmentBasis(StrEnum):
    """WHY the being holds this — so an ordinary interesting thought cannot masquerade as a debt."""

    PROMISED = "promised"
    FOLLOW_UP = "follow_up"
    SELF_ASSUMED = "self_assumed"


class CommitmentTriggerKind(StrEnum):
    """WHEN to honour it (Gollwitzer if-then): a wall-clock time, an event, or a condition."""

    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"


#: The explicit transition table (held by the registry). Terminal states are keys with
#: empty out-sets. ``active``/``deferred`` are live; ``honoured``/``dropped``/``expired`` terminal.
COMMITMENT_TRANSITIONS: dict[str, frozenset[str]] = {
    CommitmentState.ACTIVE: state_set(
        CommitmentState.DEFERRED,
        CommitmentState.HONOURED,
        CommitmentState.DROPPED,
        CommitmentState.EXPIRED,
    ),
    CommitmentState.DEFERRED: state_set(
        CommitmentState.ACTIVE,
        CommitmentState.HONOURED,
        CommitmentState.DROPPED,
        CommitmentState.EXPIRED,
    ),
    CommitmentState.HONOURED: state_set(),
    CommitmentState.DROPPED: state_set(),
    CommitmentState.EXPIRED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Commitment(BaseObject):
    content: str
    basis: CommitmentBasis
    trigger_kind: CommitmentTriggerKind
    trigger_value: str
    due_at: str | None
    source_thought_ids: tuple[str, ...]
    other_regarding_value: float

    KIND: ClassVar[str] = "commitment"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "content": self.content,
            "basis": str(self.basis),
            "trigger_kind": str(self.trigger_kind),
            "trigger_value": self.trigger_value,
            "due_at": self.due_at,
            "source_thought_ids": list(self.source_thought_ids),
            "other_regarding_value": self.other_regarding_value,
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            content=req_str(payload, "content"),
            basis=req_enum(payload, "basis", CommitmentBasis),
            trigger_kind=req_enum(payload, "trigger_kind", CommitmentTriggerKind),
            trigger_value=req_str(payload, "trigger_value"),
            due_at=opt_str(payload, "due_at"),
            source_thought_ids=req_str_tuple(payload, "source_thought_ids"),
            other_regarding_value=req_float(payload, "other_regarding_value"),
        )
```

- [ ] **Step 4: Register in `domain/objects/registry.py`** — extend the imports and `_CATALOG`:

```python
from .commitment import COMMITMENT_TRANSITIONS, Commitment
# ...
_CATALOG: tuple[KindSpec, ...] = (
    KindSpec(cls=Desire, transitions=DESIRE_TRANSITIONS),
    KindSpec(cls=Intention, transitions=INTENTION_TRANSITIONS),
    KindSpec(cls=UserModel, transitions=USER_MODEL_TRANSITIONS),
    KindSpec(cls=Thought, transitions=THOUGHT_TRANSITIONS),
    KindSpec(cls=Commitment, transitions=COMMITMENT_TRANSITIONS),
)
```

- [ ] **Step 5: Export from `domain/objects/__init__.py`** — add to the imports and `__all__`: `Commitment`, `CommitmentBasis`, `CommitmentState`, `CommitmentTriggerKind` (import from `.commitment`; keep `__all__` sorted).

- [ ] **Step 6:** Locate the existing terminal-consistency test (grep `terminal.*live|live_states` under `tests/` — e.g. `tests/test_object_kinds.py` / `tests/test_registry*.py`). If it enumerates kinds explicitly, add `commitment`; if it iterates `reg.kinds()` it already covers it — confirm it passes. (Task 1 Step 1's `test_catalog_is_terminal_consistent_including_commitment` is the safety net regardless.)

- [ ] **Step 7: Run → pass.** `make check`.

- [ ] **Step 8: Commit** `feat(commitment): Commitment BDI type + catalog registration (lm-705.3)`.

---

## Task 2: `commitment_view` — the write/read door + deterministic id

**Files:** Create `core/commitment_view.py`; Create `tests/test_commitment_view.py`.

**Interfaces:**
- Produces: `COMMITMENT_KIND="commitment"`; `crystallized_commitment_id(source_thought_id: str, content: str) -> str` (deterministic: source thought id + a content fingerprint, never random, never a bare global hash); `build_commitment(*, id, content, basis, trigger_kind, trigger_value, due_at=None, source_thought_ids, other_regarding_value=0.0, salience=0.0, source="thought-processing-apply", provenance=None) -> Commitment`; `encode_commitment(c: Commitment) -> MemoryDraft`; `read_live_commitments(memory: MemoryPort, *, limit=None) -> tuple[Commitment, ...]`; `LIVE_COMMITMENT_STATES = frozenset({"active", "deferred"})`.
- Consumes: `domain/objects` (`Commitment`, `CommitmentBasis`, `CommitmentState`, `CommitmentTriggerKind`, `Provenance`, `default_registry`, `derive_id`), `ports/memory` (`MemoryPort`). Mirror `core/thought_view.py`.

- [ ] **Step 1: Failing test** (`tests/test_commitment_view.py`)

```python
from lifemodel.core.commitment_view import (
    build_commitment,
    crystallized_commitment_id,
    encode_commitment,
)
from lifemodel.domain.objects import CommitmentBasis, CommitmentState, CommitmentTriggerKind


def test_id_is_deterministic_and_scoped_to_the_source_thought():
    a = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    b = crystallized_commitment_id("thought:seed:x", "ask about the interview")
    c = crystallized_commitment_id("thought:seed:y", "ask about the interview")  # other source
    d = crystallized_commitment_id("thought:seed:x", "different content")
    assert a == b            # reproducible → idempotent
    assert a != c and a != d  # distinct episode / distinct content ≠ conflated
    assert a.startswith("commitment:")


def test_build_and_encode_roundtrip():
    c = build_commitment(
        id=crystallized_commitment_id("thought:seed:x", "ask about the interview"),
        content="ask how their interview went",
        basis=CommitmentBasis.FOLLOW_UP,
        trigger_kind=CommitmentTriggerKind.EVENT,
        trigger_value="next time we talk",
        source_thought_ids=("thought:seed:x",),
        other_regarding_value=0.8,
        salience=0.6,
    )
    assert c.state == CommitmentState.ACTIVE.value
    draft = encode_commitment(c)  # goes through registry.encode → validates
    assert draft.kind == "commitment"
    assert draft.salience == 0.6
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Create `core/commitment_view.py`** (mirror `core/thought_view.py`'s `_REGISTRY`, `build_*`, `encode_*`, `seed_*_id`, `read_live_*` shape):

```python
"""The commitment view — the registry door onto ``kind='commitment'`` rows (§4.2, v3).

The ONE place a ``Commitment`` is constructed/encoded/read, mirroring
:mod:`lifemodel.core.thought_view`. Ids are DETERMINISTIC (never random, HLA §4.1):
:func:`crystallized_commitment_id` scopes a stable content fingerprint to the *source
thought*, so re-crystallizing the same thought upserts ONE row, and distinct episodes
(different source thought) or distinct content never conflate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    Commitment,
    CommitmentBasis,
    CommitmentState,
    CommitmentTriggerKind,
    Provenance,
    default_registry,
    derive_id,
)
from ..ports.memory import MemoryPort

COMMITMENT_KIND = "commitment"

LIVE_COMMITMENT_STATES: frozenset[str] = frozenset(
    {CommitmentState.ACTIVE.value, CommitmentState.DEFERRED.value}
)

_REGISTRY = default_registry()


def crystallized_commitment_id(source_thought_id: str, content: str) -> str:
    """A deterministic id scoping a content fingerprint to its source thought (never
    random; never a bare global content hash — distinct episodes must not conflate)."""
    digest = hashlib.sha256(f"{source_thought_id}\x00{content.strip()}".encode()).hexdigest()[:16]
    return derive_id(COMMITMENT_KIND, "seed", digest)


def build_commitment(
    *,
    id: str,
    content: str,
    basis: CommitmentBasis,
    trigger_kind: CommitmentTriggerKind,
    trigger_value: str,
    due_at: str | None = None,
    source_thought_ids: tuple[str, ...],
    other_regarding_value: float = 0.0,
    salience: float = 0.0,
    source: str = "thought-processing-apply",
    provenance: Provenance | None = None,
) -> Commitment:
    """Construct a typed :class:`Commitment` (the one constructor). Born ``active``."""
    return Commitment(
        id=id,
        state=str(CommitmentState.ACTIVE),
        source=source,
        salience=salience,
        provenance=provenance,
        content=content,
        basis=basis,
        trigger_kind=trigger_kind,
        trigger_value=trigger_value,
        due_at=due_at,
        source_thought_ids=source_thought_ids,
        other_regarding_value=other_regarding_value,
    )


def encode_commitment(commitment: Commitment) -> MemoryDraft:
    """Encode through the registry (the single write door; validates on write)."""
    return _REGISTRY.encode(commitment)


def _decode_live(record: MemoryRecord | None) -> Commitment | None:
    if record is None or record.kind != COMMITMENT_KIND:
        return None
    if record.state not in LIVE_COMMITMENT_STATES:
        return None
    obj = _REGISTRY.decode(record)
    return obj if isinstance(obj, Commitment) else None


def read_live_commitments(memory: MemoryPort, *, limit: int | None = None) -> tuple[Commitment, ...]:
    """The live (``active``/``deferred``) commitments, most-salient first."""
    records = memory.find(kind=COMMITMENT_KIND, order_by="salience_desc")
    live = tuple(
        sorted(
            (c for record in records if (c := _decode_live(record)) is not None),
            key=lambda c: (-c.salience, c.id),
        )
    )
    return live if limit is None else live[:limit]
```

> **Confirm** `memory.find(kind=..., order_by="salience_desc")` matches `core/thought_view.py:169`'s call exactly (same `order_by` string).

- [ ] **Step 4: Run → pass.** `make check`.

- [ ] **Step 5: Commit** `feat(commitment): commitment_view — deterministic id + build/encode/read door (lm-705.3)`.

---

## Task 3: Crystallization outcome schema + pure decision

**Files:** Modify `core/thought_processing.py`; Create `tests/test_thought_processing_crystallize_decide.py`.

**Interfaces:**
- Produces: `PROCESSING_JSON_SCHEMA` gains outcome `"crystallize_commitment"` + an optional `"commitment"` object; `ProcessingReason.CRYSTALLIZED_COMMITMENT = "processed_crystallize"`; `ProcessingDecision` gains `crystallize: JsonObject | None = None` (the validated commitment sub-object, or `None`); `decide_processing_transition` routes `crystallize_commitment` — returns a decision carrying the commitment fields (transition left to the apply), or routes a malformed/absent `commitment` through `_park_or_terminate(no_progress=True)`.
- Consumes: the existing `decide_processing_transition`/`_park_or_terminate`/`ProcessingReason`.

- [ ] **Step 1: Failing test** (`tests/test_thought_processing_crystallize_decide.py`)

```python
from datetime import datetime, timezone

from lifemodel.core.thought_processing import (
    ProcessingReason,
    decide_processing_transition,
)
from lifemodel.core.thought_view import build_thought
from lifemodel.domain.objects import ThoughtState

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _t(no_progress=0):
    return build_thought(id="thought:seed:x", content="interview Friday", no_progress_count=no_progress)


_GOOD = {
    "content": "ask how their interview went",
    "basis": "follow_up",
    "trigger_kind": "event",
    "trigger_value": "next time we talk",
}


def test_crystallize_carries_the_fields_and_leaves_the_transition_to_apply():
    d = decide_processing_transition(
        _t(), parsed={"outcome": "crystallize_commitment", "commitment": _GOOD}, raw="{...}", now=NOW
    )
    assert d.reason == ProcessingReason.CRYSTALLIZED_COMMITMENT
    assert d.crystallize == _GOOD          # the validated sub-object rides the decision
    assert d.transition is None            # apply computes active→resolved after a successful build


def test_crystallize_without_a_commitment_object_is_no_progress():
    d = decide_processing_transition(
        _t(), parsed={"outcome": "crystallize_commitment"}, raw="{...}", now=NOW
    )
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS
    assert d.crystallize is None
    assert d.transition.to_state == ThoughtState.PARKED.value


def test_crystallize_with_a_non_object_commitment_is_no_progress():
    d = decide_processing_transition(
        _t(), parsed={"outcome": "crystallize_commitment", "commitment": "oops"}, raw="{...}", now=NOW
    )
    assert d.reason == ProcessingReason.PARKED_NO_PROGRESS


def test_existing_outcomes_unchanged():
    assert decide_processing_transition(_t(), parsed={"outcome": "resolve"}, raw="x", now=NOW).reason == ProcessingReason.RESOLVED
    assert decide_processing_transition(_t(), parsed=None, raw="   ", now=NOW).reason == ProcessingReason.TRANSIENT_FAILURE
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Edit `core/thought_processing.py`:**

  1. **`PROCESSING_JSON_SCHEMA`** — add the outcome + the commitment sub-object:

```python
PROCESSING_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["resolve", "park", "drop", "crystallize_commitment"],
        },
        "reflection": {"type": "string"},
        "commitment": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "basis": {"type": "string", "enum": ["promised", "follow_up", "self_assumed"]},
                "trigger_kind": {"type": "string", "enum": ["time", "event", "condition"]},
                "trigger_value": {"type": "string"},
                "due_at": {"type": "string"},
                "other_regarding_value": {"type": "number"},
            },
            "required": ["content", "basis", "trigger_kind", "trigger_value"],
            "additionalProperties": False,
        },
    },
    "required": ["outcome"],
    "additionalProperties": False,
}
```

  2. **`PROCESSING_INSTRUCTIONS`** — append the crystallize option:

```python
PROCESSING_INSTRUCTIONS = (
    "You are the being's own private mind, quietly turning over one of your thoughts. "
    "Nothing you write here is shown to anyone — this is rumination, not a message. "
    "Reflect briefly, in the first person, then decide the thought's disposition: "
    "'resolve' if you have thought it through and it needs nothing more; "
    "'park' if it is worth returning to later but not now; "
    "'drop' if it no longer matters; "
    "'crystallize_commitment' if thinking it over left you with a follow-up you OWE them — "
    "something to come back to for their sake — and fill 'commitment' with what you will do "
    "('content'), why you hold it ('basis': promised/follow_up/self_assumed), and when to honour "
    "it ('trigger_kind': time/event/condition + 'trigger_value'). "
    "Answer as JSON: an 'outcome', a short 'reflection', and 'commitment' only when crystallizing."
)
```

  3. **`ProcessingReason`** — add one member (after `NO_SUBJECT`):

```python
    CRYSTALLIZED_COMMITMENT = "processed_crystallize"
```

  4. **`ProcessingDecision`** — add the field:

```python
@dataclass(frozen=True)
class ProcessingDecision:
    transition: TransitionOp | None
    reason: ProcessingReason
    crystallize: JsonObject | None = None  # the validated commitment sub-object (apply builds it)
```

  5. **`decide_processing_transition`** — add the `crystallize_commitment` branch **before** the `park` branch (after `drop`):

```python
    if outcome == "crystallize_commitment":
        commitment = parsed.get("commitment") if isinstance(parsed, dict) else None
        if not isinstance(commitment, dict):
            # schema said crystallize but no valid commitment object → no progress (codex I2)
            return _park_or_terminate(thought, now=now, no_progress=True)
        return ProcessingDecision(
            transition=None,
            reason=ProcessingReason.CRYSTALLIZED_COMMITMENT,
            crystallize=commitment,
        )
```

- [ ] **Step 4: Run → pass.** `make check` (existing `tests/test_thought_processing_decide.py` stays green — the new field defaults to `None`).

- [ ] **Step 5: Commit** `feat(crystallize): processing outcome crystallize_commitment + pure decision (lm-705.3)`.

---

## Task 4: `ThoughtProcessingApply` builds the `Commitment` + emits `PutRecord` (+ failure→no-progress)

**Files:** Modify `core/thought_processing.py`; Create `tests/test_thought_processing_crystallize_apply.py`.

**Interfaces:**
- Consumes: `ProcessingDecision.crystallize` (Task 3), `commitment_view` (`build_commitment`/`encode_commitment`/`crystallized_commitment_id`, Task 2), `PutRecord`/`PutOp` (`core/intents`/`domain/memory`), `InvalidPayload` (`domain/objects`), `creation_provenance` (`core/trace.py` — same helper `ThoughtCapture` uses; confirm its kwargs).
- Produces: on a `crystallize` decision — `[TransitionRecord(active→resolved), PutRecord(commitment)]` + span fields `processed_crystallize`/`crystallized_kind`/`crystallized_id`; on a build/encode failure — the no-progress transition instead (no `PutRecord`).

- [ ] **Step 1: Failing test** (`tests/test_thought_processing_crystallize_apply.py`) — reuses `make_tick_context`, `draft_to_record`, `internal_result_signal` like `tests/test_thought_processing_apply.py`:

```python
from datetime import datetime, timezone

from lifemodel.core.intents import PutRecord, TransitionRecord
from lifemodel.core.taxonomy import internal_result_signal
from lifemodel.core.thought_processing import ThoughtProcessingApply
from lifemodel.core.thought_view import build_thought, encode_thought
from lifemodel.domain.objects import ThoughtState
from lifemodel.state.model import State
from lifemodel.testing.harness import draft_to_record
from lifemodel.testing.tick import make_tick_context

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)
_GOOD = {"content": "ask how their interview went", "basis": "follow_up",
         "trigger_kind": "event", "trigger_value": "next time we talk"}


def _ctx(parsed):
    thought = build_thought(id="thought:seed:x", content="interview Friday", state=ThoughtState.ACTIVE)
    sig = internal_result_signal(origin_id="r1", correlation_id="c1", raw="{...}", parsed=parsed,
                                 timestamp="2026-07-17T12:00:00+00:00")
    return make_tick_context(
        state=State(pending_internal_id="c1", pending_internal_subject_id="thought:seed:x"),
        now=NOW, objects=[draft_to_record(encode_thought(thought), now=NOW)], signals=[sig],
    )


def test_crystallize_emits_resolve_transition_and_a_commitment_put():
    ctx = _ctx({"outcome": "crystallize_commitment", "commitment": _GOOD})
    intents = list(ThoughtProcessingApply().step(ctx))
    trs = [i for i in intents if isinstance(i, TransitionRecord)]
    puts = [i for i in intents if isinstance(i, PutRecord)]
    assert len(trs) == 1 and trs[0].op.to_state == ThoughtState.RESOLVED.value
    assert len(puts) == 1 and puts[0].op.draft.kind == "commitment"
    assert puts[0].op.draft.payload["content"] == "ask how their interview went"
    # provenance target→source: the commitment points back at the thought
    assert "thought:seed:x" in puts[0].op.draft.payload["source_thought_ids"]


def test_crystallize_with_bad_enum_falls_back_to_no_progress_no_put():
    # schema-shaped but domain-invalid (bad basis) → registry.encode rejects → no-progress
    bad = {**_GOOD, "basis": "not_a_basis"}
    ctx = _ctx({"outcome": "crystallize_commitment", "commitment": bad})
    intents = list(ThoughtProcessingApply().step(ctx))
    assert [i for i in intents if isinstance(i, PutRecord)] == []          # no commitment persisted
    trs = [i for i in intents if isinstance(i, TransitionRecord)]
    assert trs and trs[0].op.to_state == ThoughtState.PARKED.value          # bounded no-progress
    assert trs[0].op.patch.payload_merge["no_progress_count"] == 1
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Edit `ThoughtProcessingApply.step`** — add a crystallize branch. New imports at the top of `core/thought_processing.py`:

```python
from ..domain.objects import InvalidPayload
from .commitment_view import build_commitment, crystallized_commitment_id, encode_commitment
from .intents import Intent, LaunchInternalCognition, PutRecord, TransitionRecord  # + PutRecord
from ..domain.memory import PutOp  # add to the existing domain.memory import
from ..domain.objects import CommitmentBasis, CommitmentTriggerKind  # for typed build
from .trace import creation_provenance
```

Replace the tail of `step` (from the `decision = decide_processing_transition(...)` line) with:

```python
        decision = decide_processing_transition(
            thought, parsed=result.parsed, raw=result.raw, now=ctx.now
        )
        self._maybe_log_reflection(ctx, result.parsed)
        if decision.crystallize is not None:
            return self._crystallize(ctx, thought, decision.crystallize)
        self._log(ctx, decision.reason, subject_id)
        return [TransitionRecord(op=decision.transition)] if decision.transition is not None else []

    def _crystallize(
        self, ctx: TickContext, thought: Thought, fields: JsonObject
    ) -> Sequence[Intent]:
        """Build the Commitment + emit its PutRecord alongside the thought's resolve; a
        builder/registry failure falls back to the bounded no-progress path (codex I2)."""
        try:
            commitment = build_commitment(
                id=crystallized_commitment_id(thought.id, str(fields["content"])),
                content=str(fields["content"]),
                basis=CommitmentBasis(str(fields["basis"])),
                trigger_kind=CommitmentTriggerKind(str(fields["trigger_kind"])),
                trigger_value=str(fields["trigger_value"]),
                due_at=(str(fields["due_at"]) if fields.get("due_at") is not None else None),
                source_thought_ids=(thought.id,),
                other_regarding_value=float(fields.get("other_regarding_value") or 0.0),
                salience=thought.salience,
                provenance=creation_provenance(
                    ctx.trace,
                    created_by=self.id,
                    component="cognition",
                    reason="thought crystallized into a commitment",
                    source_object_ids=(thought.id,),
                ),
            )
            draft = encode_commitment(commitment)  # registry-validates → InvalidPayload on bad data
        except (InvalidPayload, ValueError, KeyError, TypeError):
            fallback = _park_or_terminate(thought, now=ctx.now, no_progress=True)
            self._log(ctx, fallback.reason, thought.id)
            return [TransitionRecord(op=fallback.transition)] if fallback.transition else []
        if ctx.logger is not None:
            ctx.logger.span.set(
                processing_reason=ProcessingReason.CRYSTALLIZED_COMMITMENT.value,
                thought_id=thought.id,
                crystallized_kind=commitment.KIND,
                crystallized_id=commitment.id,
            )
        return [
            TransitionRecord(op=_transition(thought, ThoughtState.RESOLVED, {})),
            PutRecord(op=PutOp(draft=draft)),
        ]

    def _maybe_log_reflection(self, ctx: TickContext, parsed: JsonObject | None) -> None:
        if ctx.logger is not None and isinstance(parsed, dict) and "reflection" in parsed:
            ctx.logger.span.set(reflection=str(parsed.get("reflection", ""))[:500])
```

> **Note:** the broad `except (InvalidPayload, ValueError, KeyError, TypeError)` is the **single bounded failure gate** — a missing/non-str/bad-enum field (which the JSON schema *should* have caught, but the port is fail-soft on malformed model output) routes to no-progress, never an uncaught strand (codex I2). Also **remove** the old inline reflection-logging block in `step` (lines ~327-334 of the current file) — it moves verbatim into `_maybe_log_reflection`, called once for every outcome.

- [ ] **Step 4: Run → pass.** `make check` (existing `tests/test_thought_processing_apply.py` stays green — non-crystallize paths unchanged).

- [ ] **Step 5: Commit** `feat(crystallize): apply builds Commitment + PutRecord, failure→no-progress (lm-705.3)`.

---

## Task 5: Real-code sim — crystallization end-to-end + zero-launch + bounded failure

**Files:** Create `tests/test_thought_processing_crystallize_harness.py` (reuse `build_processing_lifemodel` from `testing/harness.py`, slice 2).

**Interfaces:** Consumes `build_processing_lifemodel`, `run_frame`, `run_internal_completion`, `read_live_commitments`, `read_thought`, a fake egress (copy the tiny one from `tests/test_internal_runner.py`).

- [ ] **Step 1: Failing tests** (drive the REAL frame + completion + store):

```python
def test_seeded_thought_crystallizes_into_a_real_commitment():
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="interview Friday", salience=0.8)
    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = report.internal_launches[0]
    _set_pending(lm, launch)
    run_internal_completion(
        lm, _fake_egress(), {}, correlation_id=launch.correlation_id,
        result=InternalCognitionResult(
            raw='{"outcome":"crystallize_commitment",...}',
            parsed={"outcome": "crystallize_commitment", "commitment": {
                "content": "ask how their interview went", "basis": "follow_up",
                "trigger_kind": "event", "trigger_value": "next time we talk"}},
        ),
        apply=ThoughtProcessingApply(),
    )
    commitments = read_live_commitments(lm.state)
    assert len(commitments) == 1 and commitments[0].content == "ask how their interview went"
    assert commitments[0].source_thought_ids == ("thought:seed:x",)
    assert read_thought(lm.state, "thought:seed:x") is None          # source thought resolved

def test_crystallization_emits_no_proactive_launch(monkeypatch):
    # codex I4 no-send acceptance: with NO pre-existing contact desire, a crystallization
    # completion frame produces ZERO LaunchProactive (fake egress reach_out never called).
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="interview Friday", salience=0.8)
    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = report.internal_launches[0]; _set_pending(lm, launch)
    egress = _fake_egress()
    run_internal_completion(lm, egress, {}, correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw="{...}", parsed={"outcome": "crystallize_commitment",
            "commitment": {"content": "c", "basis": "follow_up", "trigger_kind": "event",
                           "trigger_value": "later"}}),
        apply=ThoughtProcessingApply())
    assert egress.calls == []                                        # non-delivery, structural

def test_atomicity_bad_commitment_leaves_no_row_and_no_resolve():
    lm = build_processing_lifemodel()
    _seed_active_thought(lm, id="thought:seed:x", content="c", salience=0.8)
    report = run_frame(lm.coreloop, trigger=FrameTrigger.HEARTBEAT)
    launch = report.internal_launches[0]; _set_pending(lm, launch)
    run_internal_completion(lm, _fake_egress(), {}, correlation_id=launch.correlation_id,
        result=InternalCognitionResult(raw="{...}", parsed={"outcome": "crystallize_commitment",
            "commitment": {"content": "c", "basis": "BAD", "trigger_kind": "event",
                           "trigger_value": "later"}}),
        apply=ThoughtProcessingApply())
    assert read_live_commitments(lm.state) == ()                     # no stray commitment
    t = read_thought(lm.state, "thought:seed:x")
    assert t is not None and t.no_progress_count == 1                # bounded no-progress, still live
```

> `_seed_active_thought`/`_set_pending`/`_fake_egress` are the same local helpers `tests/test_thought_processing_harness.py` (slice 2) defines — copy them (or import if slice 2 exposed them). Assert the spec §6 crystallization-slice invariants: **crystallization** (real commitment row + resolved thought), **no-send** (zero launch), **bounded failure** (bad payload → no row, no-progress).

- [ ] **Step 2: Run → fail** (until the crystallize path is complete). **Step 3:** no new production code — this is the end-to-end proof over Tasks 1–4. **Step 4:** `make check`.

- [ ] **Step 5: Commit** `test(crystallize): real-code sim — commitment created, no send, atomic failure (lm-705.3)`.

---

## Task 6: Host-integration scenario + spec pointer + bead close

**Files:** Modify `tests/hermes_internal_cognition_integration.py`; docs; `bd`.

- [ ] **Step 1: Extend the real-Hermes driver** with a Part-B "crystallize" scenario (mirror the existing Part-B pattern): seed a real active thought, scripted aux result `{"outcome":"crystallize_commitment","commitment":{...}}`, `apply=ThoughtProcessingApply()`, `subject_id=<thought id>`; await; assert new `_REQUIRED_TRUE_KEYS`: `bc_commitment_created` (a real `kind=commitment` row exists with the content), `bc_thought_resolved` (source thought no longer live), `bc_egress_never_called` (non-delivery), `bc_pending_cleared`. Add the keys to the driver's `_REQUIRED_TRUE_KEYS`.

- [ ] **Step 2:** `make check` (the integration wrapper `tests/test_internal_cognition_integration.py` runs the driver against real Hermes if present, else skips). Commit `test(crystallize): host-integration — a real Commitment crystallized, non-delivered (lm-705.3)`.

- [ ] **Step 3:** `bd close lm-705.3` with a summary (implemented + reviewed + green; note the deferred contact/arbiter work reads these commitments). Add a one-line §10 note to the spec that lm-705.3 shipped (mirroring the lm-705.2 build note), and commit the doc.

---

## Self-Review (run after execution)

- **Spec coverage (§4.1/§4.2/§5):** closed discriminated `crystallize_commitment` (not arbitrary kind) ✓ (Task 3) · trusted builder + `registry.encode` validation ✓ (Tasks 2/4) · failure routes through `no_progress_count` bound ✓ (Tasks 3/4) · atomic thought-resolve + commitment-`PutRecord` ✓ (Task 4, Task 5 atomicity test) · `Commitment` model: non-singleton, deterministic id, typed trigger, basis, full transitions, base-envelope salience, Intention boundary ✓ (Tasks 1/2) · target→source provenance ✓ (Task 4) · no-send guarantee + zero-launch test ✓ (Task 5) · `processed_crystallize` + `crystallized_kind`/`crystallized_id` span fields ✓ (Task 4) · registry terminal-consistency ✓ (Task 1) · host-integration vs real Hermes ✓ (Task 6).
- **Not built here (slice boundaries):** turning a `Commitment` into an outreach (commitment→`Desire`→`Intention`→send), collision/discharge/liveness — the deferred **contact/arbiter** work; `Opinion`/`Prediction` types (same mechanism, later); thought **creation**/appraiser (lm-705.11); the arbiter (slice 4).
- **Type consistency:** `ProcessingDecision(transition, reason, crystallize=None)` used identically in Tasks 3/4; `build_commitment(...)`/`crystallized_commitment_id(source_thought_id, content)` signatures match between Task 2 (def) and Task 4 (call); `CommitmentBasis`/`CommitmentTriggerKind` enum values match the JSON schema enums (`promised|follow_up|self_assumed`, `time|event|condition`) between Task 1 and Task 3.
- **Confirm-before-code (framework details):** `creation_provenance(...)` kwargs in `core/trace.py` (mirror the `ThoughtCapture` call — `source_object_ids=` vs `source_signal_ids=`); the existing object-kinds terminal-consistency test file name; `memory.find(order_by="salience_desc")` string.

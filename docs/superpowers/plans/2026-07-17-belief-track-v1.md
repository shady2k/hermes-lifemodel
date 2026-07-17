# Belief-track v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the being a first-class **`belief`** (a *defeasible* inferred proposition with confidence + evidence) and one channel that makes it matter now: born in the noticing pass, surfaced into the live turn by a gated, sensitivity-aware `pre_llm_call` injector so the being's understanding shapes its reply.

**Architecture:** A new `belief` kind in the closed BDI catalog, mirroring `Commitment` exactly (domain dataclass + one `_CATALOG` line + a `core/belief_view.py`). Beliefs are produced where the evidence lives — the **noticing** pass classifies each carried seed as `thought` or `belief` — not by a context-poor processing crystallise. A third `pre_llm_call` injector reads a **bounded** slice of active beliefs (`MemoryPort.find(kind, state, limit)` — not the decode-all `read_live_thoughts` shape), gates them (confidence threshold, non-private, a `surfaced_belief_ids` cooldown ring in `AgentState`), and splices a fallible-framed block onto the outgoing message. Cache-prefix-safe and fail-soft, exactly like `make_felt_state_injector`.

**Tech Stack:** Python 3.11 stdlib only (runtime), `uv`/`ruff`/`mypy --strict`/`pytest`. Existing: the closed `KindRegistry` (`domain/objects/registry.py`), the `Commitment` template (`domain/objects/commitment.py`, `core/commitment_view.py`), the noticing seam (`core/noticing.py`), the `pre_llm_call` injector pattern (`hooks.py`, `__init__.py`), the `AgentState` ring + atomic-stamp pattern (`state/model.py`, `state/sqlite_store.py`).

**Spec:** `docs/superpowers/specs/2026-07-17-fact-track-design.md` (v2). bd **lm-705.19** (this plan) / **lm-705.20** (v1.5 export, out of scope).

## Global Constraints

- **Runtime = stdlib only, RELATIVE imports** (`..domain.objects`, `.belief_view`); tests use absolute imports (`lifemodel.…`). Every step ends green: `make check` (ruff format --check, ruff check, mypy --strict -p lifemodel, pytest).
- **Defeasible by construction:** a belief carries **mandatory `confidence` validated to `[0,1]`** (the registry does NOT range-check — the view builder must) and **evidence** (`source_message_ids`, the cited-in-segment ids). Stored ≠ authoritative.
- **Born in noticing, not processing.** No `crystallize_belief`; processing is untouched. A belief seed with ungrounded source ids is dropped (the existing anti-hallucination contract).
- **Sensitivity floor (FR26):** a belief defaults to `Sensitivity.SENSITIVE` (a proposition about a person is sensitive); the model may raise it to `PRIVATE`. `PRIVATE` is NEVER surfaced (this plan) and never exported (v1.5).
- **Injector is bounded + gated + fail-soft:** a `find(..., limit=BOUND)` query (never decode-all), small `N` surfaced, confidence≥θ, non-private, per-belief cooldown; any raise → record + return `None`; the `{"context": …}` splices onto a copy of the outgoing message only (never the cached prompt / rolling history).
- **Observability redaction (D10, tightened for this kind):** spans/logs carry the belief `id`, `subject`, `confidence`, `sensitivity`, evidence ids, and the pass reflection — **never the full `content` verbatim** in a reason field.
- **No store migration:** a belief is a `memory_records` row (`kind="belief"`) via `payload_json`; adding a kind needs no schema migration. It DOES change the catalog's kind set — update any test asserting the exact set.

## File Structure

- **Create** `domain/objects/belief.py` — `BeliefState`, `BELIEF_TRANSITIONS`, `Belief(BaseObject)`.
- **Modify** `domain/objects/registry.py` — one `KindSpec(cls=Belief, transitions=BELIEF_TRANSITIONS)` in `_CATALOG`.
- **Modify** `domain/objects/__init__.py` — export `Belief`, `BeliefState`, `BELIEF_TRANSITIONS`.
- **Create** `core/belief_view.py` — `build_belief`, `belief_id`, `belief_from_seed_fields`, `encode_belief`, `live_beliefs`, `read_active_beliefs`.
- **Modify** `state/model.py` — `surfaced_belief_ids` ring + `SURFACED_BELIEF_IDS_CAP` + `from_dict`/`to_dict`.
- **Modify** `state_commands.py` — add `surfaced_belief_ids` to `_SET_PROTECTED`.
- **Modify** `state/sqlite_store.py` — `stamp_surfaced_beliefs` atomic-merge method.
- **Modify** `core/noticing.py` — seed schema (`kind`/`confidence`/`sensitivity`), instructions, `NoticedSeed`/`validate_noticed_seeds`, `NoticingApply` belief branch + redacted logging.
- **Modify** `hooks.py` — `make_belief_injector`.
- **Modify** `__init__.py` — register the belief injector.
- **Tests:** `tests/test_belief_object.py`, `tests/test_belief_view.py`, `tests/test_state_surfaced_beliefs.py`, extend `tests/test_noticing_apply.py`/`test_noticing_trigger.py`, `tests/test_belief_injector.py`, `tests/test_belief_harness.py`.

---

## Task 1: The `belief` kind (domain object + registry + view)

**Files:**
- Create: `domain/objects/belief.py`, `core/belief_view.py`, `tests/test_belief_object.py`, `tests/test_belief_view.py`
- Modify: `domain/objects/registry.py`, `domain/objects/__init__.py`

**Interfaces:**
- Produces: `Belief(BaseObject)` with `KIND="belief"`; `BeliefState`; `BELIEF_TRANSITIONS`; `build_belief(...)`, `belief_id(source_thought_id, content)`, `belief_from_seed_fields(...)`, `encode_belief(belief)->MemoryDraft`, `live_beliefs(objects)->tuple[Belief,...]`, `read_active_beliefs(memory, *, min_confidence, exclude_private, limit)->list[Belief]`.
- Consumes: `BaseObject`/`BaseFields`/decode helpers (`domain/objects/base.py`), `derive_id`, `_REGISTRY`/`default_registry`, `MemoryPort`, `Sensitivity` (`domain/objects/provenance.py`).

- [ ] **Step 1: Write the failing object test** (`tests/test_belief_object.py`)

```python
import pytest
from lifemodel.domain.objects import Belief, BeliefState
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.domain.objects.registry import default_registry
from lifemodel.domain.objects.errors import InvalidTransition


def _belief(**over):
    kw = dict(id="belief:seed:abcd", state=BeliefState.ACTIVE.value, source="noticing",
              content="They tend to get anxious before a loss of status.", subject="owner",
              source_message_ids=("t1",), source_thought_ids=("thought:seed:x",),
              confidence=0.7, sensitivity=Sensitivity.SENSITIVE)
    kw.update(over)
    return Belief(**kw)


def test_belief_round_trips_through_the_registry():
    reg = default_registry()
    b = _belief()
    draft = reg.encode(b)
    record = draft_to_record(draft)  # helper mirrors tests/test_commitment_object.py
    assert reg.decode(record) == b


def test_active_reaches_every_terminal_but_terminals_are_sealed():
    reg = default_registry()
    for term in (BeliefState.SUPERSEDED, BeliefState.DROPPED, BeliefState.EXPIRED):
        reg.check_transition("belief", BeliefState.ACTIVE.value, term.value)  # no raise
        with pytest.raises(InvalidTransition):
            reg.check_transition("belief", term.value, BeliefState.ACTIVE.value)
```
(Mirror `tests/test_commitment_object.py` for the exact `draft_to_record`/`check_transition` helpers and assertion style.)

- [ ] **Step 2: Run → fail** (`Belief` undefined).
Run: `uv run pytest tests/test_belief_object.py -x -q` — Expected: ImportError / fail.

- [ ] **Step 3: Create `domain/objects/belief.py`** (mirror `commitment.py`)

```python
"""The ``belief`` kind — a defeasible proposition the being has inferred about the
person or world (spec 2026-07-17). Fallible by construction: it carries a mandatory
``confidence`` and its ``source_message_ids`` evidence, and is never authoritative
merely because it is stored. Distinct from Opinion (an evaluative stance), Prediction
(future), UserModel (closed receptivity facets), and Thought (reasoning stream)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import BaseFields, BaseObject, req_float, req_str, req_str_tuple


class BeliefState(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DROPPED = "dropped"
    EXPIRED = "expired"


BELIEF_TRANSITIONS: dict[str, frozenset[str]] = {
    BeliefState.ACTIVE: frozenset(
        {BeliefState.SUPERSEDED, BeliefState.DROPPED, BeliefState.EXPIRED}
    ),
    BeliefState.SUPERSEDED: frozenset(),
    BeliefState.DROPPED: frozenset(),
    BeliefState.EXPIRED: frozenset(),
}


@dataclass(frozen=True, kw_only=True)
class Belief(BaseObject):
    content: str
    subject: str
    source_message_ids: tuple[str, ...]
    source_thought_ids: tuple[str, ...]

    KIND: ClassVar[str] = "belief"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "content": self.content,
            "subject": self.subject,
            "source_message_ids": list(self.source_message_ids),
            "source_thought_ids": list(self.source_thought_ids),
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base.as_kwargs(),
            content=req_str(payload, "content"),
            subject=req_str(payload, "subject"),
            source_message_ids=req_str_tuple(payload, "source_message_ids"),
            source_thought_ids=req_str_tuple(payload, "source_thought_ids"),
        )
```
Confirm `req_str_tuple`/`req_str`/`BaseFields.as_kwargs` names against `commitment.py`'s `_rebuild` (§ Explore report) and adjust if the helper names differ. `confidence`/`sensitivity`/`salience` are inherited envelope fields — NOT semantic payload here (they ride `BaseObject`).

- [ ] **Step 4: Register** — in `domain/objects/registry.py` `_CATALOG`, add after the `Commitment` line:
```python
    KindSpec(cls=Belief, transitions=BELIEF_TRANSITIONS),
```
Import `Belief, BELIEF_TRANSITIONS` at the top. Export in `domain/objects/__init__.py`'s `__all__`: `Belief`, `BeliefState`, `BELIEF_TRANSITIONS`. **Grep for any test asserting the exact kind set** (`grep -rn "user_model.*thought.*commitment\|{'desire'" tests/`) and add `"belief"` there.

- [ ] **Step 5: Write the failing view test** (`tests/test_belief_view.py`)

```python
import pytest
from lifemodel.core.belief_view import (
    build_belief, belief_id, belief_from_seed_fields, encode_belief,
    live_beliefs, read_active_beliefs,
)
from lifemodel.domain.objects import BeliefState
from lifemodel.domain.objects.provenance import Sensitivity
from lifemodel.domain.objects.errors import InvalidPayload


def test_id_is_deterministic_content_digest_scoped_to_thought():
    a = belief_id("thought:seed:x", "They get anxious before status loss.")
    assert a == belief_id("thought:seed:x", "  They get anxious before status loss. ")
    assert a != belief_id("thought:seed:y", "They get anxious before status loss.")
    assert a.startswith("belief:seed:")


def test_from_seed_rejects_out_of_range_confidence():
    with pytest.raises(InvalidPayload):
        belief_from_seed_fields(source_thought_id="thought:seed:x",
                                fields={"content": "c", "confidence": 1.5},
                                source_message_ids=("t1",), provenance=None)


def test_from_seed_floors_sensitivity_to_sensitive_by_default():
    b = belief_from_seed_fields(source_thought_id="thought:seed:x",
                                fields={"content": "c", "confidence": 0.7},
                                source_message_ids=("t1",), provenance=None)
    assert b.sensitivity == Sensitivity.SENSITIVE
    assert b.state == BeliefState.ACTIVE.value
    assert b.source_message_ids == ("t1",)
```

- [ ] **Step 6: Run → fail.** `uv run pytest tests/test_belief_view.py -x -q`.

- [ ] **Step 7: Create `core/belief_view.py`** (mirror `core/commitment_view.py`)

```python
"""View helpers for the ``belief`` kind — the one construction/encode/read door
(mirrors ``core/commitment_view.py``)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..domain.memory import JsonObject, MemoryDraft, MemoryRecord
from ..domain.objects import Belief, BeliefState
from ..domain.objects.base import derive_id
from ..domain.objects.errors import InvalidPayload
from ..domain.objects.provenance import Provenance, Sensitivity
from ..domain.objects.registry import default_registry
from ..ports.memory import MemoryPort

_REGISTRY = default_registry()
BELIEF_KIND = Belief.KIND
LIVE_BELIEF_STATES = frozenset({BeliefState.ACTIVE})


def belief_id(source_thought_id: str, content: str) -> str:
    digest = hashlib.sha256(f"{source_thought_id}\x00{content.strip()}".encode()).hexdigest()[:16]
    return derive_id(BELIEF_KIND, "seed", digest)


def build_belief(*, id: str, content: str, subject: str = "owner",
                 source_message_ids: Sequence[str] = (), source_thought_ids: Sequence[str] = (),
                 confidence: float, salience: float = 0.0,
                 sensitivity: Sensitivity = Sensitivity.SENSITIVE,
                 source: str = "noticing", provenance: Provenance | None = None) -> Belief:
    return Belief(
        id=id, state=BeliefState.ACTIVE.value, source=source, salience=salience,
        confidence=_validated_confidence(confidence), sensitivity=sensitivity,
        provenance=provenance, content=content, subject=subject,
        source_message_ids=tuple(source_message_ids), source_thought_ids=tuple(source_thought_ids),
    )


def _validated_confidence(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidPayload(f"belief confidence must be a number, got {value!r}")
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise InvalidPayload(f"belief confidence must be in [0,1], got {f}")
    return f


def _floor_sensitivity(raw: object) -> Sensitivity:
    # Conservative floor: a proposition about a person is at least SENSITIVE; the
    # model may escalate to PRIVATE. Anything else (incl. a "normal" request) floors up.
    if raw == Sensitivity.PRIVATE.value:
        return Sensitivity.PRIVATE
    return Sensitivity.SENSITIVE


def belief_from_seed_fields(*, source_thought_id: str, fields: JsonObject,
                            source_message_ids: Sequence[str], salience: float = 0.0,
                            provenance: Provenance | None = None) -> Belief:
    if not isinstance(fields, dict):
        raise InvalidPayload("belief fields must be an object")
    content = fields.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InvalidPayload("belief content must be a non-empty string")
    try:
        bid = belief_id(source_thought_id, content)
    except UnicodeEncodeError as exc:  # lone surrogate in content
        raise InvalidPayload(f"belief content not encodable: {exc}") from exc
    return build_belief(
        id=bid, content=content.strip(), subject=str(fields.get("subject", "owner")),
        source_message_ids=source_message_ids, source_thought_ids=(source_thought_id,),
        confidence=_validated_confidence(fields.get("confidence")), salience=salience,
        sensitivity=_floor_sensitivity(fields.get("sensitivity")), provenance=provenance,
    )


def encode_belief(belief: Belief) -> MemoryDraft:
    return _REGISTRY.encode(belief)


def _decode_live(record: MemoryRecord) -> Belief | None:
    if record.kind != BELIEF_KIND or record.state not in LIVE_BELIEF_STATES:
        return None
    decoded = _REGISTRY.decode(record)
    assert isinstance(decoded, Belief)
    return decoded


def live_beliefs(objects: Sequence[MemoryRecord]) -> tuple[Belief, ...]:
    live = [b for r in objects if (b := _decode_live(r)) is not None]
    return tuple(sorted(live, key=lambda b: (-b.salience, b.id)))


def read_active_beliefs(memory: MemoryPort, *, min_confidence: float = 0.0,
                        exclude_private: bool = True, limit: int) -> list[Belief]:
    # BOUNDED store query (never decode-all): fetch a small superset ordered by
    # recency, then apply the payload-level filters (confidence/sensitivity) the SQL
    # columns can't express, and cap at ``limit``.
    fetch = max(limit * 6, 12)
    records = memory.find(kind=BELIEF_KIND, state=BeliefState.ACTIVE.value,
                          order_by="created_desc", limit=fetch)
    out: list[Belief] = []
    for r in records:
        b = _decode_live(r)
        if b is None or (b.confidence or 0.0) < min_confidence:
            continue
        if exclude_private and b.sensitivity == Sensitivity.PRIVATE:
            continue
        out.append(b)
        if len(out) >= limit:
            break
    return out
```
Confirm `derive_id`, `MemoryDraft`, `OrderBy` value `"created_desc"`, and `memory.find` signature against the codebase before finalising.

- [ ] **Step 8: Run → pass.** `make check`.

- [ ] **Step 9: Commit** `feat(belief): defeasible belief kind + view (registry, confidence/evidence, bounded read) (lm-705.19)`.

---

## Task 2: `surfaced_belief_ids` cooldown ring in AgentState

**Files:** Modify `state/model.py`, `state_commands.py`, `state/sqlite_store.py`; Create `tests/test_state_surfaced_beliefs.py`.

**Interfaces:**
- Produces: `State.surfaced_belief_ids: tuple[str, ...] = ()`, `SURFACED_BELIEF_IDS_CAP = 64`; `SQLiteRuntimeStore.stamp_surfaced_beliefs(ids: Sequence[str]) -> None` (atomic bounded append, mirrors `stamp_affect_display`).
- Consumes: the `noticed_source_ids` ring pattern (`state/model.py`), the `_SET_PROTECTED` map (`state_commands.py`), the atomic field-merge pattern (`state/sqlite_store.py:stamp_affect_display`).

- [ ] **Step 1: Failing test** (`tests/test_state_surfaced_beliefs.py`): `State(surfaced_belief_ids=("belief:seed:a",))` round-trips through `to_dict`/`from_dict` (tuple → list on persist, back to tuple); a `stamp_surfaced_beliefs(["b1","b2"])` appends deduped, bounded to `SURFACED_BELIEF_IDS_CAP`, and survives a fresh store (mirror `tests/test_sqlite_store.py`'s affect-display stamp test); `state_commands.settable_fields()` does NOT include it and `_SET_PROTECTED` DOES (so `test_every_state_field_is_settable_or_protected` passes).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Add the field + `SURFACED_BELIEF_IDS_CAP` to `state/model.py` (mirror `noticed_source_ids`, incl. the `from_dict` `_as_str_tuple(...)` line and `to_dict` `list(...)`). Add `"surfaced_belief_ids": "internal injector cooldown ring; not hand-written"` to `_SET_PROTECTED` in `state_commands.py`. Add `stamp_surfaced_beliefs` to `sqlite_store.py` (copy `stamp_affect_display`'s `BEGIN IMMEDIATE` field-merge; append+dedup+cap the ring). **Landmine:** `test_every_state_field_is_settable_or_protected` will fail until the `_SET_PROTECTED` entry lands; the round-trip identity test needs the `from_dict`/`to_dict` lines.
- [ ] **Step 4: Run → pass.** `make check`.
- [ ] **Step 5: Commit** `feat(belief): surfaced_belief_ids cooldown ring + atomic stamp in AgentState (lm-705.19)`.

---

## Task 3: Noticing produces beliefs (schema + instructions + apply branch)

**Files:** Modify `core/noticing.py`; extend `tests/test_noticing_apply.py`, `tests/test_noticing_trigger.py` (schema only).

**Interfaces:**
- Produces: `NOTICING_JSON_SCHEMA` seeds gain `"kind": "thought"|"belief"` (default `"thought"`), `"confidence": number`, `"sensitivity": "normal"|"sensitive"|"private"`; `NoticedSeed` gains `kind`/`confidence`/`sensitivity`; `validate_noticed_seeds` threads + validates them; `NoticingApply._seed_intents` branches: a `belief` seed → `belief_from_seed_fields(...)` → `PutRecord(encode_belief(...))`; a `thought` seed → `Thought` unchanged.
- Consumes: `belief_from_seed_fields`/`encode_belief` (Task 1), `creation_provenance`, the existing `validate_noticed_seeds` anti-hallucination + dedup.

- [ ] **Step 1: Failing test** (`tests/test_noticing_apply.py`): a completion whose seed is `{"kind":"belief","gist":"they get anxious before status loss","content":"They get anxious before a loss of status.","source_message_ids":["t1"],"confidence":0.75}` over a segment containing `t1` → the apply emits a `PutRecord` for a `Belief` whose `source_message_ids==("t1",)`, `confidence==0.75`, `sensitivity==SENSITIVE`, plus resolves the surveyed segment as today; a `belief` seed citing an id NOT in the segment → dropped (anti-hallucination); a `belief` seed with `confidence` missing/out-of-range → dropped (not a crash); a plain `{"kind":"thought",...}` (or no kind) → a `Thought` exactly as today; a `belief` with `sensitivity:"private"` → a `Belief` with `PRIVATE` sensitivity.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Extend `NOTICING_JSON_SCHEMA` (add the three seed properties; `content` required only when `kind=="belief"` — enforce in `validate_noticed_seeds`, since JSON-schema conditional-required is avoided here). Add the belief clause to `NOTICING_INSTRUCTIONS` in the being's judgment-first voice (a belief = a fallible understanding you'd act on across conversations, stated with how sure you are; do not inflate a one-off; cite the exact turn_ids). Extend `NoticedSeed` + `_validate_one_seed`/`validate_noticed_seeds` to carry `kind`/`content`/`confidence`/`sensitivity` and drop a malformed belief seed (missing content, ungrounded ids, bad confidence). In `NoticingApply._seed_intents`, branch on `seed.kind`: build a `Belief` via `belief_from_seed_fields(source_thought_id=<the seed's anchor thought id? — beliefs are not thoughts>, ...)`. **Design note for the implementer:** a belief seed has no parent thought; use a deterministic surrogate source for the id + provenance — derive `belief_id(source_thought_id=f"notice:{survey_id}", content=…)` (stable per survey+content) and set provenance `created_by=NOTICING_APPLY_ID, reason="believed", source_object_ids=source_message_ids`. Keep the consumed-ring dedup for belief source ids too. Redacted logging: log the belief `id`/`subject`/`confidence`/`sensitivity` on the span, NOT `content`.
- [ ] **Step 4: Run → pass.** `make check`.
- [ ] **Step 5: Commit** `feat(noticing): a seed may be a grounded belief (confidence + evidence), applied as a Belief (lm-705.19)`.

---

## Task 4: The `make_belief_injector` `pre_llm_call` hook

**Files:** Modify `hooks.py`, `__init__.py`; Create `tests/test_belief_injector.py`.

**Interfaces:**
- Produces: `make_belief_injector(build_lm, *, params=DEFAULT_BELIEF_INJECT_PARAMS, health=None, metrics=None) -> Callable[..., dict[str,str]|None]` returning `_injector(*, session_id="", user_message="", **_ignored) -> dict[str,str]|None`; registered as a third `pre_llm_call` hook in `__init__.py`.
- Consumes: `read_active_beliefs` (Task 1), `State.surfaced_belief_ids` + `stamp_surfaced_beliefs` (Task 2), the `make_felt_state_injector` structure (fresh `build_lm()`, `{"context":…}`, fail-soft, `_record_observer_failure`).

- [ ] **Step 1: Failing test** (`tests/test_belief_injector.py`): with two active beliefs (confidence 0.8 and 0.4) and `min_confidence=0.6`, `N=2`, the injector returns a `{"context": …}` containing the 0.8 belief's content, framed with a fallibility marker (assert a substring like "I could be wrong"), and NOT the 0.4 one; a `PRIVATE` belief is never in the output; after surfacing, `stamp_surfaced_beliefs` recorded the surfaced id and a second immediate call does NOT re-surface it (cooldown), returning `None` when nothing else qualifies; no active beliefs → `None`; a `read_active_beliefs` that raises → `None` + a recorded observer failure (never raises); the returned block is not persisted (pure per-call).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Implement `make_belief_injector` (copy `make_felt_state_injector`'s skeleton: `try/except`→`_record_observer_failure(observer_name=PRE_LLM_OBSERVER, …)`→`None`; fresh `lm=build_lm()`). Body: `state=lm.state.load()`; `cooldown=set(state.surfaced_belief_ids)`; `beliefs=[b for b in read_active_beliefs(lm.memory, min_confidence=params.min_confidence, exclude_private=True, limit=params.top_n + len(cooldown)) if b.id not in cooldown][:params.top_n]`; if none → `None`; compose the block: a first-person, fallible-framed list (`"Some things I think I've come to understand about them (my own read — I could be wrong): …"` + one line per belief content); `stamp_surfaced_beliefs([b.id for b in beliefs])` via the duck-typed `getattr(lm.state, "stamp_surfaced_beliefs", None)` pattern (atomic, never a stale full-State commit); log count/ids/latency (NOT content); return `{"context": block}`. Add `DEFAULT_BELIEF_INJECT_PARAMS` (`min_confidence=0.6`, `top_n=2`). Confirm `lm.memory` is the right accessor for a `MemoryPort` on `LifeModel`; if the injector context lacks a memory port, thread one the way `buffer=` is threaded to felt-state. Register in `__init__.py` (a third `with wire("belief_injector", required=True, …): ctx.register_hook("pre_llm_call", make_belief_injector(lambda: build_lifemodel(base_dir=sdir), health=health, metrics=metrics))`).
- [ ] **Step 4: Run → pass.** `make check`.
- [ ] **Step 5: Commit** `feat(belief): gated sensitivity-aware pre_llm injector surfaces held beliefs (lm-705.19)`.

---

## Task 5: Real-code sim + observability + close-out

**Files:** Create `tests/test_belief_harness.py`; verify D10; docs; `bd`.

- [ ] **Step 1: Real-code sim** (`tests/test_belief_harness.py`, mirror `tests/test_noticing_harness.py`): over a real on-disk store, a noticing pass whose completion carries a grounded `belief` seed → a `Belief` row exists (confidence + evidence + SENSITIVE); then a `pre_llm_call` via `make_belief_injector` surfaces it ONCE, fallible-framed; a second turn does not re-surface it (cooldown); a below-θ belief is never surfaced; a `PRIVATE` belief is never surfaced.
- [ ] **Step 2: Observability check** — assert the noticing/apply span carries the belief `id`/`subject`/`confidence`/`sensitivity` + reflection but NOT the full `content`; the injector span carries count/ids/latency. `make check`.
- [ ] **Step 3: Commit** `test(belief): real-code sim — noticing forms a belief, a turn surfaces it once (lm-705.19)`.
- [ ] **Step 4:** Update the noticing spec §10 review-log with a one-line note that noticing now also produces beliefs; commit `docs`. Do NOT close lm-705.19 here — the whole-branch review + merge + deploy close it (owner-gated).

---

## Self-Review (run after execution)

- **Defeasible:** confidence mandatory + `[0,1]`-validated in the view (Task 1); evidence = cited-in-segment `source_message_ids` (Task 3); stored ≠ authoritative (fallible framing, Task 4).
- **Born in noticing** (Task 3), processing untouched; a belief with ungrounded ids or bad confidence is dropped, not crashed.
- **Injector bounded + gated + fail-soft** (Task 4): `find(limit=…)` not decode-all; confidence/non-private/cooldown gates; raise → `None`; ephemeral splice.
- **Sensitivity floor** (Task 1) default SENSITIVE, PRIVATE never surfaced (Tasks 1/4).
- **Observability redaction** (Tasks 3/5): id/subject/confidence/sensitivity, never full content.
- **No store migration**; the catalog kind-set test updated (Task 1).
- **Deferred (honest):** reconciliation/supersession-op, expiry/decay, reinforcement, the thought→desire track, the v1.5 provider export (lm-705.20).

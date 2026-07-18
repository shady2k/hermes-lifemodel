# Waking-mind thought capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `create_thought` tool so the being, mid-reply, drops a thought (with its own rough sense of how much it matters) that a **restricted capture path** dedups and persists as a durable thought — lighting up the inert waking-mind vector, additively, without waking the rest of the brain.

**Architecture:** The tool builds `thought_seed` impulses and hands them to a NEW restricted entrypoint `CoreLoop.capture_thoughts` that runs ONLY `ThoughtCapture` under the one state-actor lock, commits mutation-only (no tick advance, no launches, no other component), and persists its own `thought-capture` span. Dedup is atomic at the store via a new `PutOp(create_only=True)` (`ON CONFLICT DO NOTHING`) — no resurrection of terminal rows, race-safe. The dead post-hoc `Appraiser` seam is retired; the `commitment` tool and `noticing` are untouched.

**Tech Stack:** Python 3.11 **stdlib only** at runtime, **relative imports** in runtime code; tests use absolute imports (`lifemodel.…`). `uv`/`ruff`/`mypy --strict`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-07-18-waking-mind-thought-capture-design.md`. bd **lm-705.11**.

## Global Constraints

- **Runtime = stdlib only, RELATIVE imports** (`.core.thought_capture`, `..domain.memory`); tests absolute (`lifemodel.…`). Every task ends green: `make check` (ruff format --check, ruff check, mypy --strict -p lifemodel, pytest).
- **Observability-first is NON-NEGOTIABLE (D10, owner):** the capture path persists a `thought-capture` **span** with decision attrs; the tool emits a per-call **metric** and a structured **log** — logs/spans carry counts / producer / thought ids, **NEVER `content`** (D10 redaction, mirror the `commitment` tool's `id/basis/action` logging).
- **Restricted capture — the load-bearing safety invariant:** capture runs ONLY `ThoughtCapture`, under `state_actor_lock()`, committing mutation-only (`commit_tick(None, mutations)`); it MUST NOT run `run_frame`/the full registry, advance `tick_count`/`last_tick_at`, produce or strand any `LaunchProactive`/`LaunchInternalCognition`, or mutate any non-thought `State`. Fail-closed: assert every emitted intent is a thought `PutRecord`.
- **Atomic create-if-absent, never resurrect:** the capture `PutOp` is `create_only=True` → `INSERT … ON CONFLICT(kind,id) DO NOTHING`; a conflict on ANY existing state (incl. terminal `resolved`/`dropped`/`expired`) is a no-op — never an upsert, never a provenance overwrite.
- **Additive only:** no change to the `commitment` tool, the injectors, `noticing`, `PROCESSING_INSTRUCTIONS`, or any envelope/`salience` schema. No store migration, no new kind (`thought` already exists).
- **Tool honours the Hermes contract:** handler returns a `json.dumps` **string**, `{"error": …}` on failure, and **NEVER raises** (mirror `make_commitment_tool`/`make_check_in_tool`).
- **Being tool surface after this change:** `check_in`, `write_soul`, `commitment`, `create_thought`.

## File Structure

- **Modify** `domain/memory.py` — add `PutOp.create_only: bool = False`.
- **Modify** `state/sqlite_store.py` — `_put_on(..., *, create_only=False)` → `DO NOTHING` branch; `commit_tick` PutOp case passes `create_only=mutation.create_only`.
- **Modify** `testing/fakes.py` — `FakeMemoryStore.put(..., *, create_only=False)` no-op on existing; `FakeStateStore.commit_tick` PutOp case threads `create_only`.
- **Modify** `core/taxonomy.py` — add `producer` to `ThoughtSeedRead`, `thought_seed_signal`, `read_thought_seed`.
- **Modify** `core/thought_capture.py` — emit `PutOp(create_only=True)`; take `source` from the seed's `producer` (stop hardcoding `"thought-capture"`).
- **Modify** `core/coreloop.py` — add `CaptureResult` + `CoreLoop.capture_thoughts(seeds)`.
- **Modify** `core/tick_metrics.py` — add `THOUGHT_TOOL_TOTAL` spec.
- **Modify** `hooks.py` — add `THOUGHT_PRODUCER_TOOL`, `DEFAULT_THOUGHT_SALIENCE`, `make_create_thought_tool` (+ private `_thought_seeds_from_args`); remove `_maybe_capture_thought` + the `appraiser=` param from `make_post_llm_observer`.
- **Modify** `core/appraisal.py` — remove the `Appraiser` protocol; keep `ThoughtSeed`.
- **Modify** `__init__.py` — `_CREATE_THOUGHT_SCHEMA` / `_CREATE_THOUGHT_DESCRIPTION`; `register_tool("create_thought", …)`; drop the `appraiser=` wiring intent in the `post_llm` registration.
- **Tests:** create `tests/test_create_only_put.py`, `tests/test_capture_thoughts.py`, `tests/test_create_thought_tool.py`; extend `tests/test_thought_capture.py`, `tests/test_taxonomy_thought_seed.py`; extend `tests/test_plugin.py` (wiring); adjust the post_llm observer tests for the dropped `appraiser=`.

---

## Task 1: `create_only` store primitive — atomic create-if-absent

**Files:**
- Modify: `domain/memory.py` (PutOp)
- Modify: `state/sqlite_store.py` (`_put_on`, `commit_tick` PutOp case)
- Modify: `testing/fakes.py` (`FakeMemoryStore.put`, `FakeStateStore.commit_tick`)
- Test: `tests/test_create_only_put.py` (create)

**Interfaces:**
- Produces: `PutOp(draft: MemoryDraft, create_only: bool = False)`. When `create_only`, committing the op inserts iff no `(kind, id)` row exists in ANY state, else no-op (never upsert). Real + fake stores apply identical semantics.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_create_only_put.py
from datetime import UTC, datetime

from lifemodel.domain.memory import MemoryDraft, PutOp, TransitionOp
from lifemodel.testing.fakes import FakeMemoryStore, FakeStateStore, FixedClock
from lifemodel.state.model import State


def _draft(id: str, state: str, *, source: str) -> MemoryDraft:
    return MemoryDraft(kind="thought", id=id, state=state, payload={"content": id}, source=source)


def _store() -> FakeStateStore:
    clock = FixedClock(datetime(2026, 7, 18, tzinfo=UTC))
    return FakeStateStore(state=State(), memory=FakeMemoryStore(clock=clock), clock=clock)


def test_create_only_inserts_when_absent() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="a"), create_only=True)])
    assert store._memory.get("thought", "t1").source == "a"


def test_create_only_is_noop_on_existing_terminal_row() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "resolved", source="orig"))])  # normal put
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="new"), create_only=True)])
    row = store._memory.get("thought", "t1")
    assert row.state == "resolved" and row.source == "orig"  # NOT resurrected, NOT overwritten


def test_normal_put_still_upserts() -> None:
    store = _store()
    store.commit_tick(None, [PutOp(draft=_draft("t1", "active", source="a"))])
    store.commit_tick(None, [PutOp(draft=_draft("t1", "parked", source="b"))])  # create_only=False
    assert store._memory.get("thought", "t1").state == "parked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_create_only_put.py -v`
Expected: FAIL — `PutOp` has no `create_only` field (TypeError), and the fake ignores it.

- [ ] **Step 3: Add the `create_only` field to `PutOp`** (`domain/memory.py`, the `PutOp` dataclass ~154)

```python
@dataclass(frozen=True)
class PutOp:
    """A queued ``MemoryPort.put`` … (existing docstring)."""

    draft: MemoryDraft
    #: Create-if-absent: when True the committer inserts only if no ``(kind, id)``
    #: row exists in ANY state, and is a no-op on conflict (never an upsert). The
    #: atomic dedup primitive for the capture path (thought-capture spec §3.1):
    #: it never resurrects a terminal row and never overwrites provenance.
    create_only: bool = False
```

- [ ] **Step 4: Thread `create_only` through the real store** — in `state/sqlite_store.py`, change the `commit_tick` PutOp case (~734) to `self._put_on(conn, mutation.draft, now, create_only=mutation.create_only)`, and update `_put_on` (~360) to branch the conflict clause:

```python
def _put_on(self, conn: Connection, draft: MemoryDraft, now: datetime, *, create_only: bool = False) -> None:
    payload_json = json.dumps(draft.payload, allow_nan=False)
    expires_at = normalize_expires_at(draft.expires_at)
    now_iso = stamp_iso_utc(now)
    conflict = (
        "ON CONFLICT(kind, id) DO NOTHING"
        if create_only
        else (
            "ON CONFLICT(kind, id) DO UPDATE SET "
            "state=excluded.state, recipient_id=excluded.recipient_id, "
            "payload_json=excluded.payload_json, salience=excluded.salience, "
            "confidence=excluded.confidence, expires_at=excluded.expires_at, "
            "source=excluded.source, updated_at=excluded.updated_at, "
            "revision=memory_records.revision + 1"
        )
    )
    conn.execute(
        "INSERT INTO memory_records ("
        "kind, id, state, recipient_id, payload_json, salience, confidence, "
        "expires_at, source, created_at, updated_at, revision, schema_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?) " + conflict,
        (
            draft.kind, draft.id, draft.state, draft.recipient_id, payload_json,
            draft.salience, draft.confidence, expires_at, draft.source,
            now_iso, now_iso, draft.schema_version,
        ),
    )
```

- [ ] **Step 5: Thread `create_only` through the fake store** — in `testing/fakes.py`:

`FakeStateStore.commit_tick` PutOp case (~310): `self._memory.put(mutation.draft, create_only=mutation.create_only)`.

`FakeMemoryStore.put` (~352) — accept `create_only` and no-op on an existing row:

```python
def put(self, draft: MemoryDraft, *, create_only: bool = False) -> str:
    ensure_json_serializable(draft.payload)
    expires_at = normalize_expires_at(draft.expires_at)
    now = stamp_iso_utc(self._clock.now())
    key = (draft.kind, draft.id)
    existing = self._rows.get(key)
    if create_only and existing is not None:
        return draft.id  # create-if-absent: any-state conflict is a no-op
    created_at = existing.created_at if existing is not None else now
    revision = existing.revision + 1 if existing is not None else 0
    # … rest unchanged …
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_create_only_put.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: `make check` + commit**

```bash
make check
git add domain/memory.py state/sqlite_store.py testing/fakes.py tests/test_create_only_put.py
git commit -m "feat(thought-capture): PutOp create_only — atomic create-if-absent (lm-705.11)"
```

---

## Task 2: Producer on the thought-seed + `ThoughtCapture` uses create-only

**Files:**
- Modify: `core/taxonomy.py` (`ThoughtSeedRead`, `thought_seed_signal`, `read_thought_seed`)
- Modify: `core/thought_capture.py` (emit create-only PutOp; source from producer)
- Test: `tests/test_taxonomy_thought_seed.py` (extend), `tests/test_thought_capture.py` (extend)

**Interfaces:**
- Consumes: `PutOp(create_only=…)` from Task 1.
- Produces: `thought_seed_signal(..., producer: str = "unknown")` (payload carries `producer`); `ThoughtSeedRead.producer: str`; `ThoughtCapture` now emits `PutRecord(op=PutOp(draft=…, create_only=True))` with `source=<producer>` (defaulting `"thought-capture"` when `producer == "unknown"`, so the retired seam's/tests' back-compat holds).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_taxonomy_thought_seed.py (append)
from lifemodel.core.taxonomy import read_thought_seed, thought_seed_signal


def test_thought_seed_carries_producer() -> None:
    sig = thought_seed_signal(origin_id="o", content="c", salience=0.5,
                              producer="create-thought-tool", timestamp=None)
    assert read_thought_seed(sig).producer == "create-thought-tool"


def test_thought_seed_producer_defaults_unknown() -> None:
    sig = thought_seed_signal(origin_id="o", content="c", salience=0.5, timestamp=None)
    assert read_thought_seed(sig).producer == "unknown"
```

```python
# tests/test_thought_capture.py (append)
def test_capture_emits_create_only_and_records_producer() -> None:
    from lifemodel.core.thought_capture import ThoughtCapture
    from lifemodel.core.taxonomy import thought_seed_signal
    ctx = _ctx_with_signals([  # existing helper in this test module (mint a TickContext)
        thought_seed_signal(origin_id="o", content="hello", salience=0.5,
                            producer="create-thought-tool", timestamp="2026-07-18T00:00:00+00:00"),
    ])
    intents = list(ThoughtCapture().step(ctx))
    assert len(intents) == 1
    op = intents[0].op
    assert op.create_only is True
    assert op.draft.source == "create-thought-tool"
```

*(If `tests/test_thought_capture.py` has no `_ctx_with_signals` helper, add a small one that builds a `TickContext(state=State(), now=…, trace=FakeTracer().start_root(), signals=(…,), objects=())`.)*

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_taxonomy_thought_seed.py tests/test_thought_capture.py -v`
Expected: FAIL — no `producer` kwarg / attr; op is not create-only; source is hardcoded.

- [ ] **Step 3: Add `producer` to the taxonomy** — in `core/taxonomy.py`:

`ThoughtSeedRead` (~184): add field `producer: str`.

`thought_seed_signal` (~192): add param `producer: str = "unknown"` and `"producer": producer` to the payload dict.

`read_thought_seed` (~224): read it defensively (mirror the `turn_id` reader):
```python
    raw_producer = signal.payload.get("producer", "unknown")
    producer = raw_producer if isinstance(raw_producer, str) and raw_producer else "unknown"
```
and pass `producer=producer` into the returned `ThoughtSeedRead(...)`.

- [ ] **Step 4: `ThoughtCapture` emits create-only + producer source** — in `core/thought_capture.py`, in `step` where it builds the thought (~63) and the `PutRecord` (~73):

```python
            thought = build_thought(
                id=thought_id,
                content=seed.content,
                trigger="event",
                salience=seed.salience,
                actionability=seed.actionability,
                other_regarding_value=seed.other_regarding_value,
                source=seed.producer if seed.producer != "unknown" else "thought-capture",
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_thought(thought), create_only=True)))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_taxonomy_thought_seed.py tests/test_thought_capture.py -v`
Expected: PASS.

- [ ] **Step 6: `make check` + commit**

```bash
make check
git add core/taxonomy.py core/thought_capture.py tests/test_taxonomy_thought_seed.py tests/test_thought_capture.py
git commit -m "feat(thought-capture): thought-seed producer + create-only ThoughtCapture (lm-705.11)"
```

---

## Task 3: Restricted capture entrypoint `CoreLoop.capture_thoughts` (the safety-critical task)

**Files:**
- Modify: `core/coreloop.py` (add `CaptureResult` + `capture_thoughts`)
- Test: `tests/test_capture_thoughts.py` (create)

**Interfaces:**
- Consumes: `ThoughtCapture` (Task 2), `state_actor_lock`, `PutRecord`.
- Produces: `CaptureResult(accepted: int, deduped: int, thought_ids: tuple[str, ...])`; `CoreLoop.capture_thoughts(seeds: Sequence[Signal]) -> CaptureResult` — runs ONLY `ThoughtCapture` under the lock, persists a `thought-capture` span, commits mutation-only, and returns the counts. Used by the tool (Task 4).

- [ ] **Step 1: Write the failing safety test** (the load-bearing one)

```python
# tests/test_capture_thoughts.py
from lifemodel.core.taxonomy import thought_seed_signal
from lifemodel.testing.harness import build_capture_harness  # real-code CoreLoop (see note)


def _seed(content: str, producer: str = "create-thought-tool"):
    return thought_seed_signal(origin_id=f"o-{content}", content=content, salience=0.5,
                              producer=producer, timestamp="2026-07-18T00:00:00+00:00")


def test_capture_creates_thoughts_and_touches_nothing_else() -> None:
    h = build_capture_harness()  # CoreLoop + fake stores, with an ALREADY-ACTIVE desire
    before = h.state_store.load()
    result = h.coreloop.capture_thoughts([_seed("alpha"), _seed("beta")])
    after = h.state_store.load()
    # thoughts created …
    assert result.accepted == 2 and result.deduped == 0
    assert {r.id for r in h.memory.find(state="active", limit=50) if r.kind == "thought"} == set(result.thought_ids)
    # … and NOTHING else moved: no tick advance, no state field changed, no launch produced.
    assert after.tick_count == before.tick_count
    assert after == before  # full State frozen-dataclass equality
    assert h.egress.sent == []  # no LaunchProactive dispatched/stranded


def test_capture_dedups_and_does_not_resurrect_terminal() -> None:
    h = build_capture_harness()
    h.coreloop.capture_thoughts([_seed("alpha")])
    (tid,) = [r.id for r in h.memory.find(state="active", limit=50) if r.kind == "thought"]
    h.memory.transition("thought", tid, "active", "resolved")  # terminate it
    result = h.coreloop.capture_thoughts([_seed("alpha")])  # same content
    assert result.deduped == 1 and result.accepted == 1
    assert h.memory.get("thought", tid).state == "resolved"  # NOT resurrected


def test_capture_fails_closed_on_non_put_intent() -> None:
    h = build_capture_harness()
    h.coreloop._capture_component = _EmitsTransition()  # test double emitting a TransitionRecord
    import pytest
    with pytest.raises(AssertionError):
        h.coreloop.capture_thoughts([_seed("alpha")])
```

*(Add `build_capture_harness` to `testing/harness.py` mirroring the existing `build_capture_thought_harness`/`ThoughtCapture` harnesses — a real `CoreLoop` over `FakeStateStore(memory=FakeMemoryStore)` + `FakeTracer` + a fake egress, seeded with one active `desire` row so the "no LaunchProactive stranded" assertion is meaningful.)*

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_capture_thoughts.py -v`
Expected: FAIL — `CoreLoop` has no `capture_thoughts`.

- [ ] **Step 3: Add `CaptureResult` + `capture_thoughts`** to `core/coreloop.py`:

```python
@dataclass(frozen=True)
class CaptureResult:
    """The outcome of one restricted capture (thought-capture spec §3)."""
    accepted: int      # valid seeds persisted-or-deduped this call
    deduped: int       # of those, already present (create-if-absent no-op)
    thought_ids: tuple[str, ...]


class CoreLoop:
    # … existing …

    def capture_thoughts(self, seeds: Sequence[Signal]) -> CaptureResult:
        """Persist thought-seed impulses via ONLY ``ThoughtCapture``, under the one
        state-actor lock, mutation-only — no other component, no tick advance, no
        launches (spec §3). Opens + persists its own ``thought-capture`` span so
        provenance points at a real span (D10). Fail-closed: a non-thought
        ``PutRecord`` intent aborts before any commit."""
        if not seeds:
            return CaptureResult(accepted=0, deduped=0, thought_ids=())
        with state_actor_lock():
            now = self._clock.now()
            state = self._state_actor.state
            root = self._tracer.start_root()
            span = start_span(root, component=THOUGHT_CAPTURE_ID, tick=None, started_at=to_iso(now))
            logger = self._span_logger(span)
            objects = self._live_objects_snapshot(now, logger)  # reuse _run_tick's snapshot read
            ctx = TickContext(
                state=state, now=now, trace=root, signals=tuple(seeds), objects=objects,
                logger=logger, tracer=self._tracer, trace_writer=self._trace_writer,
                metrics=self._metrics,
            )
            intents = list(self._capture_component.step(ctx))
            for intent in intents:  # fail-closed (spec §3)
                if not isinstance(intent, PutRecord):
                    raise AssertionError(f"capture emitted a non-PutRecord intent: {intent!r}")
            # created-vs-deduped for observability — under the lock (consistent vs frames)
            deduped = sum(
                1 for i in intents
                if self._memory is not None and self._memory.get(i.op.draft.kind, i.op.draft.id) is not None
            )
            ids = tuple(i.op.draft.id for i in intents)
            self._state_actor.apply(intents)  # PutRecord-only → commit_tick(None, mutations)
            span.set(captured=len(ids) - deduped, deduped=deduped, thoughts=len(ids))
            self._persist_span(span, now)  # SAME per-component span persist _run_tick uses
        return CaptureResult(accepted=len(ids), deduped=deduped, thought_ids=ids)
```

Wire the two small reuses in `CoreLoop.__init__`: `self._capture_component = ThoughtCapture()` (a stateless component; a field so the fail-closed test can swap a double). Extract the start-of-tick snapshot read from `_run_tick` (~274-282) into `self._live_objects_snapshot(now, logger)` and call it from both places (DRY). Extract the per-component span persist from `_run_tick` into `self._persist_span(span, now)` and call it from both. Import `state_actor_lock`, `THOUGHT_CAPTURE_ID`, `ThoughtCapture`, `PutRecord`, `to_iso` at module top.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_capture_thoughts.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite (the extraction touches `_run_tick`) + commit**

```bash
make check   # confirms the _run_tick snapshot/span extraction didn't regress the tick path
git add core/coreloop.py testing/harness.py tests/test_capture_thoughts.py
git commit -m "feat(thought-capture): restricted CoreLoop.capture_thoughts entrypoint + span (lm-705.11)"
```

---

## Task 4: The `create_thought` tool — handler, schema, metric, wiring

**Files:**
- Modify: `core/tick_metrics.py` (`THOUGHT_TOOL_TOTAL` + its `MetricSpec`)
- Modify: `hooks.py` (`THOUGHT_PRODUCER_TOOL`, `DEFAULT_THOUGHT_SALIENCE`, `make_create_thought_tool`, `_thought_seeds_from_args`)
- Modify: `__init__.py` (`_CREATE_THOUGHT_SCHEMA`, `_CREATE_THOUGHT_DESCRIPTION`, `register_tool`)
- Test: `tests/test_create_thought_tool.py` (create), `tests/test_plugin.py` (extend wiring)

**Interfaces:**
- Consumes: `CoreLoop.capture_thoughts` (Task 3), `thought_seed_signal` (Task 2), `seed_thought_id` (`core/thought_view.py`).
- Produces: `make_create_thought_tool(build_lm, *, metrics=None) -> Callable[..., str]` — the Hermes tool handler; returns `json.dumps({"accepted": N, "deduped": M})`, `{"error": …}` on failure, never raises.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_create_thought_tool.py
import json
from lifemodel.hooks import make_create_thought_tool
from lifemodel.testing.harness import build_capture_harness


def _tool(h):
    return make_create_thought_tool(lambda: h.lm, metrics=h.metrics)


def test_create_thought_captures_and_reports() -> None:
    h = build_capture_harness()
    out = json.loads(_tool(h)({"thoughts": [{"content": "ask about the trip", "salience": 0.7}]}))
    assert out == {"accepted": 1, "deduped": 0}
    rows = [r for r in h.memory.find(state="active", limit=50) if r.kind == "thought"]
    assert rows[0].source == "create-thought-tool" and rows[0].salience == 0.7


def test_create_thought_array_and_intra_call_dedup() -> None:
    h = build_capture_harness()
    out = json.loads(_tool(h)({"thoughts": [{"content": "a"}, {"content": "a"}, {"content": "b"}]}))
    assert out == {"accepted": 2, "deduped": 0}  # "a","a" collapse in-call to one; default salience used


def test_create_thought_empty_is_handled() -> None:
    h = build_capture_harness()
    assert json.loads(_tool(h)({"thoughts": []})) == {"accepted": 0, "deduped": 0}


def test_create_thought_never_raises_returns_error() -> None:
    def boom():
        raise RuntimeError("nope")
    out = json.loads(make_create_thought_tool(boom)({"thoughts": [{"content": "x"}]}))
    assert "error" in out


def test_create_thought_bad_args_returns_error() -> None:
    h = build_capture_harness()
    assert "error" in json.loads(_tool(h)("not a dict"))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_create_thought_tool.py -v`
Expected: FAIL — `make_create_thought_tool` undefined.

- [ ] **Step 3: Add the metric** (`core/tick_metrics.py`, beside `COMMITMENT_TOOL_TOTAL` ~62 and its spec registration ~161):

```python
#: The create_thought tool (lm-705.11): capture calls by ``outcome`` (captured /
#: empty / error). Emitted from the tool handler, fail-open. NEVER carries content.
THOUGHT_TOOL_TOTAL = "lifemodel_thought_tool_total"
```
Register its `MetricSpec` in the same list the other tool metrics use (mirror `COMMITMENT_TOOL_TOTAL`'s spec: `name=THOUGHT_TOOL_TOTAL`, label `outcome`, help text).

- [ ] **Step 4: Add the handler** (`hooks.py`, beside `make_commitment_tool`):

```python
THOUGHT_PRODUCER_TOOL = "create-thought-tool"
DEFAULT_THOUGHT_SALIENCE = 0.5  # neutral non-zero — the being's rough estimate, better than 0


def _thought_seeds_from_args(args: dict[str, Any], *, now_iso: str | None) -> list[Signal]:
    """Turn tool args into thought_seed signals (in-call dedup by content digest)."""
    raw = args.get("thoughts")
    seeds: list[Signal] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        tid = seed_thought_id(content)
        if tid in seen:
            continue
        seen.add(tid)
        salience = item.get("salience")
        salience = float(salience) if isinstance(salience, (int, float)) else DEFAULT_THOUGHT_SALIENCE
        salience = max(0.0, min(1.0, salience))
        seeds.append(thought_seed_signal(
            origin_id=f"thought-seed-{tid}", content=content, salience=salience,
            producer=THOUGHT_PRODUCER_TOOL, timestamp=now_iso,
        ))
    return seeds


def make_create_thought_tool(
    build_lm: Callable[[], LifeModel], *, metrics: MetricRegistry | None = None
) -> Callable[..., str]:
    """The ``create_thought`` tool: the being drops thought(s) it wants to return to,
    captured via the restricted path (spec §3). Honours the Hermes contract (a
    ``json.dumps`` STRING, ``{"error": …}`` on failure, NEVER raises). Logs
    counts/producer, NEVER ``content`` (D10)."""

    def _handler(args: Any = None, **_ignored: Any) -> str:
        try:
            if not isinstance(args, dict):
                _record_thought_tool(metrics, "error")
                return json.dumps({"error": "expected an arguments object"}, ensure_ascii=False)
            lm = build_lm()
            assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
            seeds = _thought_seeds_from_args(args, now_iso=to_iso(lm.clock.now()))
            result = lm.coreloop.capture_thoughts(seeds)
            outcome = "captured" if result.accepted else "empty"
            _record_thought_tool(metrics, outcome)
            _LOG.info(  # counts + ids only — NEVER content (D10)
                "create_thought accepted=%d deduped=%d producer=%s",
                result.accepted, result.deduped, THOUGHT_PRODUCER_TOOL,
            )
            return json.dumps({"accepted": result.accepted, "deduped": result.deduped}, ensure_ascii=False)
        except Exception as exc:  # Hermes tool contract: return {"error": …}, never raise
            _LOG.error("create_thought_failed error=%s", f"{type(exc).__name__}: {exc}", exc_info=True)
            _record_thought_tool(metrics, "error")
            return json.dumps({"error": "the create_thought tool is unavailable right now"}, ensure_ascii=False)

    return _handler


def _record_thought_tool(metrics: MetricRegistry | None, outcome: str) -> None:
    if metrics is not None:
        metrics.inc(THOUGHT_TOOL_TOTAL, outcome=outcome)
```

Add imports to `hooks.py` as needed: `seed_thought_id` (`from .core.thought_view import seed_thought_id`), `thought_seed_signal`, `THOUGHT_TOOL_TOTAL`, `Signal`.

- [ ] **Step 5: Run handler tests to verify pass**

Run: `uv run pytest tests/test_create_thought_tool.py -v`
Expected: PASS.

- [ ] **Step 6: Wire the tool + description/schema** (`__init__.py`, beside the commitment tool `register_tool` ~867):

```python
_CREATE_THOUGHT_DESCRIPTION = (
    "Capture a thought you want to return to later. When something in this exchange "
    "leaves a thread worth revisiting — a question you want to sit with, something you "
    "noticed about them or about yourself, an idea not yet finished — write it here in a "
    "sentence (in whatever language is natural), with a rough sense of how much it tugs "
    "at you (salience 0..1). Your reply is your thinking; this only drops a bookmark your "
    "quieter, later mind will pick up. Not every turn — only when something genuinely "
    "tugs. You may capture more than one at once."
)
_CREATE_THOUGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "the thought to return to"},
                    "salience": {"type": "number", "description": "0..1, how much it tugs (optional)"},
                },
                "required": ["content"],
            },
        }
    },
    "required": ["thoughts"],
}
```
```python
    with wire("create_thought_tool", required=True, health=health, logger=_LOG):
        ctx.register_tool(
            "create_thought",
            toolset="lifemodel",
            schema=_CREATE_THOUGHT_SCHEMA,
            handler=make_create_thought_tool(lambda: build_lifemodel(base_dir=sdir), metrics=metrics),
            description=_CREATE_THOUGHT_DESCRIPTION,
        )
```

- [ ] **Step 7: Wiring smoke test** — extend `tests/test_plugin.py` to assert `create_thought` is registered in the `lifemodel` toolset (mirror the existing `commitment`/`check_in` registration assertions).

- [ ] **Step 8: `make check` + commit**

```bash
make check
git add core/tick_metrics.py hooks.py __init__.py tests/test_create_thought_tool.py tests/test_plugin.py
git commit -m "feat(thought-capture): create_thought tool — handler, schema, metric, wiring (lm-705.11)"
```

---

## Task 5: Retire the dead post-hoc `Appraiser` seam

**Files:**
- Modify: `hooks.py` (drop `_maybe_capture_thought` + the `appraiser=` param on `make_post_llm_observer`; drop the `_is_genuine_reactive_exchange` call site if now unused by the appraiser path only — keep it if the buffer close still uses it)
- Modify: `core/appraisal.py` (remove the `Appraiser` protocol; keep `ThoughtSeed`)
- Modify: `__init__.py` (drop the `appraiser=`-related comment/wiring in the `post_llm` registration — it passes none today, so this is comment/dead-arg cleanup)
- Test: adjust `tests/` for the post_llm observer (any test that passed `appraiser=`), keep the proactive-read-back / buffer-close / turn-close tests green.

**Interfaces:**
- Consumes: nothing new.
- Produces: `make_post_llm_observer` with no `appraiser=` parameter; `core/appraisal.py` exporting only `ThoughtSeed`.

- [ ] **Step 1: Find the call sites**

Run: `grep -rn "appraiser\|_maybe_capture_thought\|Appraiser" --include=*.py . | grep -v __pycache__`
Expected: only `hooks.py`, `core/appraisal.py`, `__init__.py`, `testing/appraisal.py` (a `FakeAppraiser`), and their tests.

- [ ] **Step 2: Write/adjust the failing test** — assert `make_post_llm_observer` no longer accepts `appraiser` and still resolves a proactive read-back + closes the buffer:

```python
# tests/test_post_llm_observer.py (adjust existing)
import inspect
from lifemodel.hooks import make_post_llm_observer


def test_post_llm_observer_has_no_appraiser_param() -> None:
    assert "appraiser" not in inspect.signature(make_post_llm_observer).parameters
```
Keep (do not delete) the existing proactive-read-back / buffer-close / turn-close observer tests — they must stay green after the removal.

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_post_llm_observer.py -v`
Expected: FAIL — `appraiser` still in the signature.

- [ ] **Step 4: Remove the seam**
- `hooks.py`: delete `_maybe_capture_thought`; remove `appraiser: Appraiser | None = None` from `make_post_llm_observer` and the `_maybe_capture_thought(lm, appraiser, …)` call in `_observer`; remove the `from .core.appraisal import Appraiser` import. Leave `_maybe_complete_buffer_entry` and everything else in `_observer` untouched.
- `core/appraisal.py`: delete the `Appraiser` `Protocol` (and its `runtime_checkable`/`Protocol` imports if now unused); keep `ThoughtSeed`.
- `__init__.py`: in the `post_llm` `register_hook`, remove the now-stale `appraiser`-not-wired comment; it already passes no `appraiser=`, so no call-arg change.
- `testing/appraisal.py`: delete `FakeAppraiser` if only the removed protocol used it (grep first; if a test still imports it, delete that test's appraiser assertion too).

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_post_llm_observer.py -v`
Expected: PASS. Then the appraisal + capture suites:
Run: `uv run pytest tests/test_thought_capture.py tests/test_capture_thoughts.py -v`
Expected: PASS.

- [ ] **Step 6: `make check` + commit**

```bash
make check
git add hooks.py core/appraisal.py __init__.py testing/appraisal.py tests/
git commit -m "refactor(thought-capture): retire the dead post-hoc Appraiser seam (lm-705.11)"
```

---

## Self-Review

**Spec coverage:** §3 restricted path → Task 3; §3.1 atomic create-if-absent + resurrection → Task 1 (+ Task 2 emits create-only, Task 3 asserts terminal no-op); §4 tool contract + salience + honest result + metric → Task 4; §5 tool-provided salience, no envelope refactor → Task 4 (`DEFAULT_THOUGHT_SALIENCE`, envelope column unchanged); §6 retire appraiser → Task 5; §7 commitment/noticing untouched → not modified (asserted by the additive-only constraint + Task 4 wiring smoke); §8 producer tagging → Task 2 (+ Task 4 sets `create-thought-tool`); §9 bus-discipline / mutation-only → Task 3; observability (D10) → span in Task 3, metric+log in Task 4. All covered.

**Placeholder scan:** the two "reuse `_run_tick`'s snapshot/span-persist" extractions (Task 3, Step 3) name the exact source lines (`~274-282`, per-component persist) to lift into `_live_objects_snapshot`/`_persist_span` — a concrete refactor, not a TODO. The `build_capture_harness` helper (Task 3) names the existing harnesses to mirror. No "TBD"/"handle edge cases".

**Type consistency:** `PutOp.create_only` (Task 1) is read in `commit_tick`/`_put_on` (Task 1) and emitted by `ThoughtCapture` (Task 2); `thought_seed_signal(producer=…)`/`ThoughtSeedRead.producer` (Task 2) is set by `_thought_seeds_from_args` (Task 4) and read by `ThoughtCapture` (Task 2); `CaptureResult.{accepted,deduped,thought_ids}` (Task 3) is consumed by the tool handler (Task 4). Names consistent across tasks.

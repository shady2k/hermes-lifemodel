# Durable NoticingBuffer + claim/finalize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The `NoticingBuffer` (conversation capture that feeds noticing) becomes **durable** — persisted in a dedicated `conversation_buffer` table in `lifemodel.sqlite` — so a plugin/gateway restart (deploy, crash, reboot) no longer wipes captured-but-not-yet-noticed conversation. Folds in the **claim/finalize** transactional lifecycle: the surveyed segment is *claimed* (an immutable snapshot immune to ring eviction), and *finalized* (cursor advanced + consumed) **atomically with the thought-creation commit**, so a mid-gap eviction or a rolled-back frame can never lose or double-process a segment.

**Architecture:** A `BufferStore` port (core, Hermes-free) with two impls: `InMemoryBufferStore` (the current in-memory behavior, the test fake) and `SqliteBufferStore` (durable, over the `conversation_buffer` table, owned by `SQLiteRuntimeStore`). `NoticingBuffer` keeps its API (`open_pending`/`complete`/`closed_segment`/`clear_through`/`abandon_pending`/`session_ids`) and delegates persistence to the injected store. A **claim** marks the surveyed rows `claimed` under a `survey_id` (carried on the launch correlation) — `segment_through(survey_id)` returns exactly those rows regardless of ring pressure (**codex I2**). **Finalize** is a new `FinalizeBuffer(survey_id)` intent the `StateActor` threads into the single `commit_tick` transaction, so the buffer cursor advances in the same `BEGIN IMMEDIATE` as the `PutRecord(thought)` + consumed-ring `UpdateState` (**codex I3**). On `connect()`, stale `claimed` rows (a pass that died with the process) are released back to `complete`.

**Tech Stack:** Python 3.11 stdlib only (runtime, incl. `sqlite3`), `uv`/`ruff`/`mypy --strict`/`pytest`. The single-store discipline (D7: ONE `lifemodel.sqlite`, atomic across tables), the `schema_migrations` framework (`state/sqlite_store.py:_MIGRATIONS`), the `commit_tick` one-transaction committer, the intent bus (`StateActor`), the noticing seam (`core/noticing_buffer.py`, `core/noticing.py`).

## Global Constraints

- **bd:** closes **lm-705.14** (durable buffer) **and lm-705.13** (claim/finalize lifecycle). Spec: `docs/superpowers/specs/2026-07-16-waking-mind-noticing-internal-cognition-design.md` §4.1 (the buffer was "process-owned" there — this amends it to durable, per the owner's live decision 2026-07-17). Rides lm-705.5 (noticing, shipped).
- **ONE physical store (D7).** The buffer lives in `lifemodel.sqlite` as a **dedicated `conversation_buffer` table** (NOT a separate file — that would break the atomic finalize and reintroduce the split-brain gap D7 removed; NOT a `memory_records` kind — that pollutes the closed BDI catalog with a non-BDI transient; NOT a `runtime_state` field — the buffer would bloat the per-tick State blob).
- **Atomic finalize (codex I3).** Advancing the buffer cursor (clearing the surveyed rows + marking sources consumed) MUST commit in the **same `BEGIN IMMEDIATE`** as the frame's `PutRecord(thought)`/`UpdateState`. A rollback leaves neither the thoughts nor the cursor-advance — the segment is simply re-surveyed later (idempotent via the consumed ring).
- **Claim immunity (codex I2).** The apply acts on the segment that was actually *surveyed* — the `claimed` rows under the launch's `survey_id` — never a recompute against the current (possibly-evicted) ring.
- **Window↔cursor alignment (codex F2b).** The claimed window is a **prefix** aligned with the cursor: claim + finalize the same set, so no un-surveyed older turn is ever cleared.
- **Buffer API unchanged.** `core/noticing.py` (`NoticingTrigger`/`NoticingApply`) and the hooks keep calling the same `NoticingBuffer` methods — only the backend + the claim/finalize seam change. `NoticingBuffer` and `BufferStore` stay Hermes-free (the SQLite impl lives at the store layer).
- **Turn-path safety.** `open_pending`/`complete` still run inside the hooks' fail-loud try; a durable-write hiccup must never crash a host turn (guard + fail-soft, same as today).
- **No behavior change to WHAT is noticed** — only durability + transactional correctness. Idle/size/gates/validation/dedup unchanged.
- **Additive migration, no data loss.** A new `_migrate_v4` creates `conversation_buffer` (`CREATE TABLE IF NOT EXISTS`); existing rows/tables untouched; the migration is idempotent (the `schema_migrations` framework skips applied versions).
- **Every step ends green:** `make check`.

## File Structure

- **Create** `core/buffer_store.py` — the `BufferStore` Protocol + `BufferEntry`/`PendingTurn`/`ClaimedSegment` value types + `InMemoryBufferStore` (the current in-memory logic, extracted).
- **Modify** `core/noticing_buffer.py` — `NoticingBuffer(store: BufferStore | None = None, ...)` delegates to the store (default `InMemoryBufferStore` → back-compat); add `claim(session_id, *, now) -> survey_id | None`, `segment_through(survey_id)`, `release(survey_id)`, `recover_stale_claims()`.
- **Modify** `state/sqlite_store.py` — `_migrate_v4` + `conversation_buffer` table; a `SqliteBufferStore` (the durable `BufferStore` impl over the store's connection); thread a buffer-finalize into `commit_tick`.
- **Modify** `core/intents.py` — `FinalizeBuffer(survey_id: str)` intent.
- **Modify** `core/state_actor.py` — collect `FinalizeBuffer` → pass to `commit_tick(..., finalize_survey_id=...)`.
- **Modify** `core/noticing.py` — the trigger claims (`survey_id` in the correlation); the apply emits `FinalizeBuffer` on success / `release` on transient failure; `segment_through(survey_id)` for the surveyed set.
- **Modify** `adapters/being_platform.py`, `__init__.py` — build the `NoticingBuffer` over a `SqliteBufferStore` (same `base_dir`); `recover_stale_claims()` in `connect()`.
- **Tests:** `tests/test_buffer_store_sqlite.py`, `tests/test_noticing_buffer_durable.py`, `tests/test_finalize_buffer_intent.py`, extend `tests/test_noticing_trigger.py`/`test_noticing_apply.py`/`test_noticing_harness.py`, `tests/test_being_platform_*` recovery.

---

## Task 1: The `conversation_buffer` table (migration) + `SqliteBufferStore`

**Files:** Modify `state/sqlite_store.py`; Create `core/buffer_store.py` (port + value types); Create `tests/test_buffer_store_sqlite.py`.

**Interfaces:**
- Produces (`core/buffer_store.py`): `@dataclass(frozen=True) BufferEntry(session_id, turn_id, source_ids: tuple[str,...], user_text, assistant_text, ts)`; a `BufferStore` `Protocol` with `open_pending(session_id, *, user_text, now)`, `stamp_source(session_id, message_id)`, `complete(session_id, turn_id, *, assistant_text, now)`, `abandon_pending(session_id)`, `completed(session_id, *, now, ttl) -> list[BufferEntry]` (the ordered `complete` rows iff no live pending, TTL-abandoning a stale pending first), `claim(session_id, turn_ids: tuple[str,...], survey_id) -> None`, `claimed(survey_id) -> list[BufferEntry]`, `finalize(survey_id) -> None` (drop the claimed rows), `release(survey_id) -> None` (claimed→complete), `recover_stale_claims() -> None` (all claimed→complete at boot), `session_ids() -> list[str]`.
- Produces (`state/sqlite_store.py`): `_migrate_v4` creating `conversation_buffer(session_id TEXT, turn_id TEXT, state TEXT, source_ids TEXT, user_text TEXT, assistant_text TEXT, opened_at TEXT, ts TEXT, survey_id TEXT, PRIMARY KEY(session_id, turn_id))` + `(4, _migrate_v4)` in `_MIGRATIONS`; `SqliteBufferStore(base_dir, *, clock)` implementing `BufferStore` over that table (its own connection, WAL — same file as the runtime store).

- [ ] **Step 1: Failing test** (`tests/test_buffer_store_sqlite.py`) — over a temp `base_dir`: open→stamp→complete → `completed()` returns the entry; a live pending → `completed() == []`; a TTL-stale pending is abandoned; `claim` marks rows so `claimed(survey_id)` returns exactly them AND they leave `completed()`; `finalize` drops them; `release` returns claimed→complete; `recover_stale_claims` releases a leftover claim; **durability:** a second `SqliteBufferStore` on the SAME `base_dir` sees the rows (survives "restart").
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Add `_migrate_v4` (mirror `_migrate_v2`'s `_create_*_table` shape; register in `_MIGRATIONS`). Create `core/buffer_store.py` (port + `BufferEntry`). Implement `SqliteBufferStore` (each method one small SQL under its own transaction; `completed`/`claimed` are ordered `SELECT`s; `claim`/`finalize`/`release` are `UPDATE`/`DELETE` by key/`survey_id`). **Confirm** the store's connection helper (`_connect`) + `STRICT`/WAL conventions in `sqlite_store.py` and reuse them.
- [ ] **Step 4: Run → pass.** `make check`.
- [ ] **Step 5: Commit** `feat(buffer): conversation_buffer table (migration v4) + SqliteBufferStore (lm-705.14)`.

---

## Task 2: `NoticingBuffer` delegates to a `BufferStore` (in-memory extracted as the fake)

**Files:** Modify `core/noticing_buffer.py`; Create `tests/test_noticing_buffer_durable.py` (+ keep `tests/test_noticing_buffer.py` green).

**Interfaces:**
- Produces: `InMemoryBufferStore` (in `core/buffer_store.py`) — the current `_pending`/`_complete`/`deque` logic, implementing `BufferStore`; `NoticingBuffer(*, store: BufferStore | None = None, max_entries=256, pending_ttl=...)` delegating every method to `store` (default `InMemoryBufferStore(max_entries=...)`), adding `claim`/`segment_through`/`release`/`recover_stale_claims` that forward to the store. `segment_through(survey_id)` returns `store.claimed(survey_id)`.
- Consumes: `BufferStore` (Task 1).

- [ ] **Step 1: Failing test** — a `NoticingBuffer` over an `InMemoryBufferStore` behaves exactly as today (the existing `tests/test_noticing_buffer.py` scenarios), AND a `NoticingBuffer` over a `SqliteBufferStore` (temp dir) does too + survives a fresh `NoticingBuffer` on the same dir (durability through the buffer API).
- [ ] **Step 2: Run → fail** (`claim`/`store=` not defined).
- [ ] **Step 3:** Extract the current in-memory logic into `InMemoryBufferStore` (move the `_pending`/`_complete`/`_lock` bodies; keep the TTL/ring/closed-prefix semantics identical). Make `NoticingBuffer` a thin delegator (its own lock is no longer needed if the store is internally locked — keep the store impls lock-guarded). Add `claim`/`segment_through`/`release`/`recover_stale_claims`.
- [ ] **Step 4: Run → pass** (existing buffer tests + the new durable ones). `make check`.
- [ ] **Step 5: Commit** `feat(buffer): NoticingBuffer delegates to BufferStore; InMemoryBufferStore fake (lm-705.14)`.

---

## Task 3: Claim in the trigger + survey_id on the correlation (codex I2)

**Files:** Modify `core/noticing.py`; extend `tests/test_noticing_trigger.py`.

**Interfaces:**
- Produces: `NoticingTrigger.step` — when it decides to survey a lane, it **claims** the surveyed **prefix window** (the oldest `size_cap` complete entries, aligned with the cursor — codex F2b) via `buffer.claim(session_id, turn_ids, survey_id)`, mints a `survey_id`, and emits `LaunchInternalCognition(correlation_id=f"notice-{session_id}#{survey_id}", ...)` carrying the survey_id. The prompt is built from the claimed window.
- Consumes: `NoticingBuffer.claim`/`segment_through` (Task 2).

- [ ] **Step 1: Failing test** — a due lane → the surveyed entries are `claim`ed (leave `completed()`, appear in `segment_through(survey_id)`); the correlation encodes the survey_id; a second tick does NOT re-survey the claimed window (single-flight + claimed rows gone from `completed`). Ring eviction of newer turns does NOT change `segment_through(survey_id)` (the I2 regression: seed a small `max_entries`, claim, add turns past the cap, assert `segment_through` is unchanged).
- [ ] **Step 2–4:** Implement the claim + survey_id. `survey_id` derivation must be deterministic-per-launch (mirror the existing correlation `to_iso(ctx.now)` + the anchor turn — no `Math.random`). The window is the oldest `size_cap` prefix (so `finalize` clears exactly the surveyed set). `make check`.
- [ ] **Step 5: Commit** `feat(noticing): trigger claims the surveyed prefix + survey_id correlation (lm-705.13, codex I2/F2b)`.

---

## Task 4: Atomic finalize — `FinalizeBuffer` intent through `commit_tick` (codex I3)

**Files:** Modify `core/intents.py`, `core/state_actor.py`, `state/sqlite_store.py`, `core/noticing.py`; Create `tests/test_finalize_buffer_intent.py`.

**Interfaces:**
- Produces: `FinalizeBuffer(survey_id: str)` (`core/intents.py`); `StateActor.apply` collects it and calls `commit_tick(state, mutations, finalize_survey_id=survey_id)`; `commit_tick(..., finalize_survey_id: str | None = None)` — inside its `BEGIN IMMEDIATE`, after the state+mutations, `DELETE FROM conversation_buffer WHERE survey_id=? AND state='claimed'` (the finalize) so it is atomic with the `PutRecord(thought)` + consumed-ring `UpdateState`. `NoticingApply` on a genuine result emits `FinalizeBuffer(survey_id)` alongside the thought `PutRecord`s + the consumed-ring `UpdateState`; on a **transient** failure it emits nothing (leaves the claim) and the runner's completion / recover releases it later.
- Consumes: `commit_tick` (`state/sqlite_store.py`), `SqliteBufferStore` finalize SQL (Task 1).

- [ ] **Step 1: Failing test** — a `StateActor.apply([PutRecord(thought), UpdateState(ring), FinalizeBuffer(sid)])` over a real store commits ALL in one transaction: the thought row exists, the ring advanced, AND the claimed buffer rows are gone; **atomicity:** a batch whose `PutRecord` is a stale/invalid transition rolls back everything — the buffer rows stay `claimed` (not finalized). (Mirror the existing `commit_tick` atomicity test.)
- [ ] **Step 2–4:** Add the intent; thread it through `StateActor` (collect like `UpdateState`); add the `finalize_survey_id` param + the one `DELETE` inside `commit_tick`'s transaction (the committer owns `conversation_buffer`). In `NoticingApply`, replace the direct `buffer.clear_through(...)` with emitting `FinalizeBuffer(survey_id)` (success) / doing nothing (transient → the claim persists for a retry). **Read `NoticingApply` from lm-705.5 first** — it currently clears the buffer directly during `step`; this moves the clear into the atomic commit. `make check`.
- [ ] **Step 5: Commit** `feat(buffer): atomic FinalizeBuffer through commit_tick — cursor advances with the thoughts (lm-705.13, codex I3)`.

---

## Task 5: Wire the durable store + connect-recovery + the release-on-failure path

**Files:** Modify `adapters/being_platform.py`, `__init__.py`; extend `tests/test_being_platform_internal_cognition.py`.

**Interfaces:**
- Produces: `__init__.py`/`being_platform` build the ONE `NoticingBuffer(store=SqliteBufferStore(base_dir, clock=...))` (same `base_dir` as the runtime store) — shared by the hooks AND the graph (as lm-705.5 already threads it). `connect()` calls `buffer.recover_stale_claims()` (release a claim whose pass died with the process) — mirror the existing `runner.recover_stale(...)` call site.
- Consumes: `SqliteBufferStore` (Task 1), `NoticingBuffer` (Task 2), the recovery pattern in `connect()`.

- [ ] **Step 1: Failing test** — after `connect()` on a store with a leftover `claimed` survey, the claim is released (`completed()` returns it again). The buffer built in `register()` is a `SqliteBufferStore`-backed one over the being's `base_dir` (durable across a rebuilt buffer). A transient noticing failure leaves the claim; the next eligible tick re-surveys it (no loss).
- [ ] **Step 2–4:** Wire it. Confirm the `base_dir`/`clock` reach `__init__.py`'s buffer construction (grep the lm-705.5 wiring). `make check`.
- [ ] **Step 5: Commit** `feat(buffer): durable buffer wired through the adapter + connect recovery (lm-705.14)`.

---

## Task 6: Real-code sim + host-integration + close-out

**Files:** extend `tests/test_noticing_harness.py`, `tests/hermes_internal_cognition_integration.py`; docs; `bd`.

- [ ] **Step 1: Real-code sim** — over a `SqliteBufferStore`-backed buffer: a captured segment survives a **simulated restart** (rebuild the buffer on the same dir → the segment is still there → noticing runs on it); claim→finalize advances the cursor **atomically** (a forced commit failure leaves the claim AND no thoughts — nothing half-applied); ring eviction does not corrupt a claimed survey; the window↔cursor alignment clears exactly the surveyed prefix (no older turn lost).
- [ ] **Step 2: Host-integration** — extend the driver: a noticing pass over a durable buffer against real Hermes seeds thoughts AND the `conversation_buffer` rows are finalized (gone) in the same commit; a restart mid-window preserves the segment. `make check`.
- [ ] **Step 3:** `bd close lm-705.14` and `bd close lm-705.13` (summaries); a §10 note in the noticing spec amending "process-owned" → durable + the claim/finalize lifecycle; commit the doc.

---

## Self-Review (run after execution)

- **Durability (lm-705.14):** the buffer lives in `conversation_buffer` in `lifemodel.sqlite` (Task 1) and survives a rebuild/restart (Tasks 2/5/6) — a deploy no longer wipes captured conversation.
- **Claim/finalize (lm-705.13):** the surveyed set is a claimed snapshot immune to ring eviction (Task 3, codex I2); the cursor advances atomically with the thought commit (Task 4, codex I3); the claimed window is the cursor-aligned prefix (Task 3, codex F2b); a transient failure leaves the claim for a clean retry (Task 4/5).
- **One store (D7):** a table, not a file/kind/State-field; the finalize shares `commit_tick`'s one `BEGIN IMMEDIATE` (Task 4).
- **No noticing-behavior change:** idle/size/gates/validation/dedup untouched — only durability + transactionality.
- **Confirm-before-code:** the `_connect`/`STRICT`/WAL conventions in `sqlite_store.py`; the exact `commit_tick` signature + where `StateActor` collects intents; the lm-705.5 buffer-construction site in `__init__.py`; whether `claim` needs rollback if the runner denies the launch (single-flight/budget) — if so, release on the denied-launch path (or rely on `recover_stale_claims`/the next-tick release), flag it.

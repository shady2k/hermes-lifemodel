# Waking Mind — Slice 5: Noticing (involuntary idle-batch thought creation) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The being **notices** what a real conversation left worth carrying — involuntarily, not as a tool it calls. A process-owned **conversation buffer** captures each completed turn; when a lane goes quiet (or the buffer size-caps) over a **closed prefix**, an aux **noticing pass** (non-delivered, on the lm-705.6 seam) judges the recent segment + the being's own live thought backlog and returns **top-K first-person gists with source ids**, which become `active` thoughts. This is the **thought-creation** path that turns the whole waking-mind vector (capture → process → crystallize) live.

**Architecture:** Reuse the shipped seam (lm-705.6) + patterns (lm-705.2/705.3). Add: a **`NoticingBuffer`** process-owned service wired into the `pre_llm`/inbound/`post_llm` hooks (a per-session `pending` slot completed at `post_llm` into a `(session_id, turn_id)` ring entry); a **`NoticingTrigger`** heartbeat component (idle∨size over a closed prefix → `LaunchInternalCognition` with a noticing prompt+schema, `subject_id=None`); a **`NoticingApply`** completion component (subjectless noticing result → validate source ids → seed thoughts via the slice-1 capture door). Noticing and processing (705.2) **share the one seam + single-flight slot**, disambiguated at completion by `pending_internal_subject_id` (processing sets it; noticing leaves it `None`). The `thought_seed`/`ThoughtCapture` contract is extended for **source lineage** + a durable **consumed-id ring** dedup.

**Tech Stack:** Python 3.11 stdlib only (runtime), `uv`/`ruff`/`mypy --strict`/`pytest`. The internal-cognition seam (`core/internal_cognition.py`, `adapters/internal_runner.py`, `core/llm_port.py`), the capture pipeline (`core/taxonomy.py` `thought_seed`, `core/thought_capture.py`, `core/thought_view.py`), the hooks (`hooks.py`), `State` (`state/model.py`), the real-code harness (`testing/harness.py`), the real-Hermes integration driver.

## Global Constraints

- **bd:** slice 5 of epic **lm-705** — task **lm-705.5** (Part II of the noticing spec). Spec: `docs/superpowers/specs/2026-07-16-waking-mind-noticing-internal-cognition-design.md` (§4, §5, §6). Rides lm-705.6 (seam) + lm-705.2 (birth-voice/single-flight/generic launch call-spec, shipped) + lm-705.3 (apply pattern).
- **Appraisal is JUDGMENT, not a heuristic (spec §2).** The aux model judges what is noticed; **no keyword/pattern matching** anywhere. (This is the concrete appraiser the whole vector was dormant for.)
- **Noticing is INVOLUNTARY (spec §2).** A curator-shaped idle batch, never a `note_to_self` tool the being calls mid-reply.
- **NO privacy gate (owner decision, spec §2).** The being is the owner's 1:1 companion — what the human writes to it, it may remember. **Do NOT build any sensitivity/no-store classifier.**
- **Unit of reading = the recent coherent SEGMENT** (bounded by a lull ∨ size-cap), never a lone message, never the whole dialogue. Cross-segment continuity comes from the being's **own live thought backlog** (read as context), not from re-reading old conversation.
- **Non-delivered, async, off-lock (spec §2, reused from 705.6).** The noticing call reaches no human; it runs off the 0-LLM tick; the model call is OFF the state-actor lock (only the FR20 reservation + the typed-result application are serialized). Correlation is `pending_internal_id`, never `pending_proactive_id`; never read back as `[SILENT]`/`SENT`.
- **The being holds pointers, not the transcript (spec §2).** A thought = gist + **source ids**; the full conversation lives in Hermes/hindsight. Footprint is bounded by thought count, not dialogue size.
- **Closed prefix only (spec §4.2).** The trigger surveys only completed turns; **never launches while an unmatched `pending` turn exists for that lane** (a long tool-heavy reply must not be surveyed before its `post_llm`). Honors HLA "not during a turn".
- **Shared seam + single-flight (reused).** Noticing and processing (705.2) share the runner + FR20 quota + the one in-flight slot; the completion disambiguates by `pending_internal_subject_id` (processing: set; noticing: `None`). Idle ticks stay 0-LLM (S5).
- **Source-lineage contract (spec §4.3):** every returned seed carries `source_message_ids`/`turn_id`; **validate** each returned id is actually in the surveyed segment (reject hallucinated sources); preserve immutable creation provenance on an idempotent re-seed; a durable bounded **consumed-source-id ring** so a message that already produced a thought isn't re-seeded even after that thought terminalizes.
- **Buffer is process-owned (spec §4.1), NOT a `LifeModel` field** (graphs are rebuilt per call) — a single lock-protected service injected into the hooks *and* the adapter.
- **Observability forced (spec §5):** closed `reason` enum (`idle_launch`, `size_cap_launch`, `nothing_lingered`, `budget_denied`, `noticed`, …); counts/ids ride as fields (`noticed_count`, `thought_ids`, `source_ids`), never in the reason string.
- **Confirmed host kwargs:** `post_llm_call` receives `session_id`, `turn_id`, `user_message`, `assistant_response`, `conversation_history` (+ more, ignored); `pre_llm_call` receives `session_id`, `user_message`, `conversation_history`, `is_first_turn`, `model`, `platform` (no `turn_id`); the inbound observer receives `event` (`event.id` = platform message id, `hooks.py:541`).
- **Every step ends green:** `make check`.

## File Structure

- **Modify** `core/taxonomy.py` — extend `thought_seed_signal`/`read_thought_seed`/`ThoughtSeedRead` with `source_message_ids: tuple[str,...]` + `turn_id: str | None`; add the `internal_result`→noticing read helper if needed.
- **Modify** `core/thought_capture.py` — `ThoughtCapture` threads the source ids into the built thought's provenance/payload; consult + update the durable consumed-id ring for dedup.
- **Modify** `state/model.py` — add a durable bounded `noticed_source_ids: tuple[str,...]` ring (the consumed-source dedup) + `last_noticing_at: str | None` (pacing, optional).
- **Create** `core/noticing_buffer.py` — `NoticingBuffer` (process-owned, lock-protected: `open_pending`, `stamp_source`, `complete`, `closed_segment`, `clear_through`, TTL→`abandoned`, bounded ring). Pure/stdlib, no Hermes.
- **Create** `core/noticing.py` — `NoticingTrigger` (heartbeat emitter), `NoticingApply` (completion consumer), `NOTICING_JSON_SCHEMA`, `NOTICING_INSTRUCTIONS`, `NoticingReason` enum, the pure seed-validation (`validate_noticed_seeds`).
- **Modify** `hooks.py` — wire the buffer into `make_pre_llm_*`/`make_inbound_observer`/`make_post_llm_observer` (open/stamp/complete); these are additive, buffer optional.
- **Modify** `adapters/being_platform.py` — own the `NoticingBuffer` singleton (or receive it), inject it into the graph so `NoticingTrigger` reads it; the tick already drives `report.internal_launches`.
- **Modify** `composition.py` — register `NoticingTrigger`; the `NoticingApply` is the completion apply (like `ThoughtProcessingApply`, guarded).
- **Modify** `__init__.py` — build the one process-owned `NoticingBuffer`, pass it into the hooks + the platform.
- **Tests:** `tests/test_noticing_buffer.py`, `tests/test_taxonomy_thought_seed_source.py`, `tests/test_thought_capture_source.py`, `tests/test_noticing_trigger.py`, `tests/test_noticing_apply.py`, `tests/test_noticing_seam.py` (hooks), `tests/test_noticing_harness.py` (real-code sim), extend the host-integration driver.

## Global reconciliation (how noticing rides the shipped seam)

- The **generic launch call-spec** (lm-705.2): `LaunchInternalCognition` already carries `prompt`, `instructions`, `json_schema`, `subject_id`. `NoticingTrigger` emits it with `NOTICING_INSTRUCTIONS`/`NOTICING_JSON_SCHEMA` and **`subject_id=None`** (subjectless — no single thought). `being_platform._tick` already maps these to the runner (lm-705.2 Task 7).
- **Completion disambiguation:** `ThoughtProcessingApply.step` guards on `pending_internal_subject_id is not None` → it no-ops for a noticing pass. `NoticingApply.step` guards on `pending_internal_subject_id is None` + a matching `internal_result` with the noticing shape → it seeds thoughts. Both run in the same completion frame; each takes only its own. (No dispatcher.)
- **Write door:** reuse the slice-1 `thought_seed`→`ThoughtCapture` path (one write door), extended for source ids — `NoticingApply` emits `thought_seed` signals (via the seam's completion frame) that `ThoughtCapture` turns into `PutRecord(thought)`; OR `NoticingApply` runs `ThoughtCapture`'s build directly. (Decide in Task 5; leaning: `NoticingApply` builds thoughts via `thought_view.build_thought` + the capture id/provenance rules, emitting `PutRecord` directly — one component, reusing `thought_view`, avoiding a second in-frame signal hop.)

---

## Task 1: Extend the `thought_seed` contract + `State` for source lineage & dedup

**Files:** Modify `core/taxonomy.py`, `state/model.py`; Create `tests/test_taxonomy_thought_seed_source.py`, extend `tests/test_budget_processing.py`-style state test.

**Interfaces:**
- Produces: `thought_seed_signal(..., source_message_ids: tuple[str,...] = (), turn_id: str | None = None)`; `ThoughtSeedRead` gains `source_message_ids: tuple[str,...]`, `turn_id: str | None`; `read_thought_seed` decodes them (back-compatible defaults). `State.noticed_source_ids: tuple[str,...] = ()` (a bounded consumed-source ring, additive) + `State.last_noticing_at: str | None = None`.
- Consumes: the existing `thought_seed` taxonomy + `State` (mirror lm-705.1 / lm-705.2 field additions).

- [ ] **Step 1: Failing test** — a `thought_seed_signal` with `source_message_ids=("m1","m2")`, `turn_id="t1"` round-trips through `read_thought_seed`; an old seed with no source ids still reads (defaults `()`/`None`); `State.noticed_source_ids`/`last_noticing_at` default + round-trip through `to_dict`/`from_dict`.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Add the two payload keys to `thought_seed_signal` + `ThoughtSeedRead` + `read_thought_seed` (mirror the existing salience/actionability handling, `core/taxonomy.py`). Add the two `State` fields (mirror lm-705.2's `pending_internal_subject_id`/`last_internal_call_at` additive pattern: field + default + `from_dict` line). Keep `noticed_source_ids` a bounded ring (a module const cap, e.g. 512 — enforced when the ring is appended in Task 5, not here).
- [ ] **Step 4: Run → pass.** `make check`.
- [ ] **Step 5: Commit** `feat(noticing): thought_seed source lineage + State consumed-id ring (lm-705.5)`.

---

## Task 2: The `NoticingBuffer` process-owned service

**Files:** Create `core/noticing_buffer.py`, `tests/test_noticing_buffer.py`.

**Interfaces:**
- Produces: `NoticingBuffer(*, max_entries=256, pending_ttl=timedelta(minutes=30))` with, all under one `threading.Lock`:
  - `open_pending(session_id: str, *, user_text: str, now: datetime) -> None` — opens/refreshes the session's single `pending` slot.
  - `stamp_source(session_id: str, message_id: str) -> None` — records a platform message id on the open pending slot (from the inbound observer).
  - `complete(session_id: str, turn_id: str, *, assistant_text: str, now: datetime) -> None` — moves the pending slot to a `complete` ring entry keyed `(session_id, turn_id)` with `source_ids`, texts, ts.
  - `closed_segment(session_id: str, *, now: datetime) -> list[BufferEntry]` — the ordered `complete` entries for the lane **iff** no `pending` slot is open for it (else `[]` — closed-prefix rule); ages a stale `pending` (`now - opened_at > pending_ttl`) to `abandoned` first so a dropped turn never wedges the lane.
  - `clear_through(session_id: str, turn_id: str) -> None` — cursor: drop `complete` entries up to and including `turn_id` (called after a successful pass).
  - `BufferEntry` = frozen `(session_id, turn_id, source_ids: tuple[str,...], user_text, assistant_text, ts)`.
- Consumes: stdlib only (`threading`, `datetime`, `collections.deque`). No Hermes, no `LifeModel`.

- [ ] **Step 1: Failing test** — open_pending→stamp_source→complete yields a `closed_segment` with the source id; a still-open pending → `closed_segment == []` (closed-prefix); a pending older than TTL is abandoned so the lane re-opens; `clear_through` drops the surveyed prefix; the ring bounds length; two sessions are independent; thread-safety smoke (concurrent open/complete under the lock).
- [ ] **Step 2–4:** Implement (one `Lock`; a per-session pending slot + a bounded `deque` of complete entries; `closed_segment` returns `[]` when a live pending exists). `make check`.
- [ ] **Step 5: Commit** `feat(noticing): NoticingBuffer — per-session pending→complete ring, closed-prefix + TTL (lm-705.5)`.

---

## Task 3: Wire the buffer into the hooks + the composition singleton

**Files:** Modify `hooks.py`, `__init__.py`; Create `tests/test_noticing_seam.py`.

**Interfaces:**
- Produces: the `pre_llm` hook (or the inbound path) calls `buffer.open_pending(session_id, user_text=user_message, now=...)`; `make_inbound_observer` calls `buffer.stamp_source(session_id, event.id)`; `make_post_llm_observer` calls `buffer.complete(session_id, turn_id, assistant_text=assistant_response, now=...)` on a genuine reactive turn (NOT a pending-proactive read-back, NOT our own impulse/control command — the same band-pass the appraisal seam uses). All buffer args are **optional** (`buffer: NoticingBuffer | None = None`) so existing callers/tests are unaffected.
- Consumes: `NoticingBuffer` (Task 2); the confirmed hook kwargs (`session_id`/`turn_id` at post_llm, `session_id` at pre_llm, `event.id` inbound).

- [ ] **Step 1: Failing test** — drive the observers with a buffer: pre_llm opens a pending; inbound stamps a source; post_llm completes → `closed_segment` returns the entry with the source id; a pending-proactive read-back does NOT complete a reactive entry; a control command / own impulse is skipped.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3:** Thread `buffer` through the three hook factories (add the `session_id`/`turn_id` kwargs to the `post_llm` `_observer` signature — they arrive in `**_ignored` today; name them). Reuse the existing reactive-turn guard (`_is_pending_proactive_turn`, `_is_own_impulse`, `_is_control_command`). In `__init__.py`, build ONE `NoticingBuffer` and pass the **same instance** into `make_pre_llm_*`, `make_inbound_observer`, `make_post_llm_observer`, and the platform (Task 6). **Confirm** the exact pre_llm hook factory name/signature in `hooks.py` (there is a `pre_llm_call` registration in `__init__.py` — thread the buffer there).
- [ ] **Step 4: Run → pass.** `make check` (existing hook tests green — the buffer is optional).
- [ ] **Step 5: Commit** `feat(noticing): buffer wired into pre_llm/inbound/post_llm + composition singleton (lm-705.5)`.

---

## Task 4: `NoticingTrigger` — the idle/size closed-prefix emitter

**Files:** Modify `core/noticing.py` (create; trigger portion), `composition.py`, `adapters/being_platform.py`; Create `tests/test_noticing_trigger.py`.

**Interfaces:**
- Produces: `NOTICING_JSON_SCHEMA` (top-K seeds: `[{gist, source_message_ids, turn_id, salience}]`), `NOTICING_INSTRUCTIONS` (judgment framing — "what did this leave worth carrying?", non-delivered, source ids must be from the segment), `NoticingReason` enum; `NoticingTrigger(buffer, *, idle=timedelta(minutes=N), size_cap=K, daily_ceiling, min_interval)` with `id`/`step(ctx)`: for each lane, if a `closed_segment` exists AND (idle elapsed ∨ size ≥ cap) AND gates pass (single-flight/budget/interval), emit ONE `LaunchInternalCognition(prompt=<segment + top-M backlog>, instructions=NOTICING_INSTRUCTIONS, json_schema=NOTICING_JSON_SCHEMA, subject_id=None, correlation_id, origin_traceparent)`. Emits nothing (0-LLM) otherwise; logs the reason.
- Consumes: `NoticingBuffer` (Task 2), `live_thoughts` (backlog context), the budget gates (`core/budget.py`), `format_traceparent`, the generic `LaunchInternalCognition` (mirror `ThoughtProcessingSelector`).

- [ ] **Step 1: Failing test** — a closed segment past the idle window → one `LaunchInternalCognition` with `subject_id is None`, the noticing schema, and a prompt containing the segment; an open pending → none; below size + within idle → none; single-flight/budget gates each block. (Use `make_tick_context` + a `NoticingBuffer` seeded with entries.)
- [ ] **Step 2–4:** Implement the trigger (mirror `ThoughtProcessingSelector`'s gate/emit shape). The `buffer` is injected via the graph (`being_platform` owns it, threads it into `build_lifemodel` → the trigger). Register `NoticingTrigger` in `composition.py` (COGNITION layer). `make check`.
- [ ] **Step 5: Commit** `feat(noticing): NoticingTrigger — idle/size closed-prefix launch (lm-705.5)`.

---

## Task 5: `NoticingApply` — validate seeds → seed thoughts + dedup + cursor

**Files:** Modify `core/noticing.py` (apply portion); Create `tests/test_noticing_apply.py`.

**Interfaces:**
- Produces: `validate_noticed_seeds(parsed, *, segment_ids: frozenset[str], consumed: frozenset[str]) -> list[NoticedSeed]` (pure — drops seeds whose `source_message_ids` are not all in the segment, or already consumed); `NoticingApply(buffer)` completion component: guards on `pending_internal_subject_id is None` + a noticing-shaped `internal_result`; validates seeds; emits `PutRecord(thought)` per fresh seed (born `active`, via `thought_view.build_thought`, source ids in provenance, content-digest id — reuse slice-1 rules); appends the consumed source ids to `State.noticed_source_ids` (bounded ring) via `UpdateState`; `clear_through` the surveyed cursor on the buffer; logs `noticed`/`nothing_lingered` + `noticed_count`/`thought_ids`/`source_ids`.
- Consumes: `read_internal_result` (`core/taxonomy`), `build_thought`/`encode_thought`/`seed_thought_id` (`core/thought_view`), `PutRecord`/`UpdateState`, the consumed-id ring (Task 1), the buffer cursor (Task 2).

- [ ] **Step 1: Failing test** — a noticing result with two valid seeds (source ids in the segment) → two `PutRecord(thought)` with source ids in provenance + the ids appended to `noticed_source_ids`; a seed with a hallucinated source id (not in the segment) → dropped; a seed whose source id is already in `noticed_source_ids` → dropped (dedup); a subject-set (processing) result → no-op; the surveyed prefix is cleared.
- [ ] **Step 2–4:** Implement `validate_noticed_seeds` (pure) + `NoticingApply`. **Idempotent immutable provenance:** a re-seed of the same content upserts one row with its original provenance (slice-1 invariant — reuse `seed_thought_id` + the capture provenance rule). `make check`.
- [ ] **Step 5: Commit** `feat(noticing): NoticingApply — validate source ids, seed thoughts, dedup ring, clear cursor (lm-705.5)`.

---

## Task 6: Wire the runner apply + buffer through the adapter; noticing pass end-to-end

**Files:** Modify `adapters/being_platform.py`, `__init__.py`; extend `tests/test_being_platform_internal_cognition.py`.

**Interfaces:**
- Produces: `being_platform` owns the `NoticingBuffer` (same instance as the hooks), threads it into `build_lifemodel` (so `NoticingTrigger`/`NoticingApply` receive it); the runner's completion frame runs `NoticingApply` (a standing component) alongside `ThoughtProcessingApply` — each guards on `pending_internal_subject_id`. The tick maps a noticing `LaunchInternalCognition` (subjectless) into the runner exactly as processing (lm-705.2 Task 7, unchanged — the runner already takes `subject_id=launch.subject_id`, `None` here).
- Consumes: the runner's `apply` (lm-705.6) — register `NoticingApply` in `build_lifemodel` (standing, guarded) so it runs in every completion frame; `ThoughtProcessingApply` stays the runner's injected apply.

- [ ] **Step 1: Failing test** — a full-graph tick with a seeded closed buffer segment drives a subjectless `LaunchInternalCognition` into the runner; on completion the noticing result seeds thoughts (no proactive launch, non-delivery). Confirm processing + noticing coexist: a subject-set completion runs `ThoughtProcessingApply`, a subjectless one runs `NoticingApply`.
- [ ] **Step 2–4:** Register `NoticingApply` in `composition.py` (standing completion component, guarded on subjectless). Thread the buffer through `being_platform` → `build_lifemodel` → the trigger/apply. `make check`.
- [ ] **Step 5: Commit** `feat(noticing): wire buffer + NoticingApply through the adapter; processing+noticing coexist (lm-705.5)`.

---

## Task 7: Real-code sim (spec §6)

**Files:** Create `tests/test_noticing_harness.py` (reuse `build_processing_lifemodel` shape + a `FakeLlmPort` with scripted noticing seeds).

- [ ] **Step 1: Failing tests** — a buffered sitting → thoughts with the right source ids; continuity uses the backlog (a scripted seed references a prior thought, not raw old text); idle ∨ size-cap both fire; a `pending` turn blocks the launch; the cursor clears; top-K holds; **idle-with-empty-buffer stays 0-LLM**; the internal correlation never collides with `pending_proactive_id`; **a completion frame that also returns a proactive launch dispatches it** (the §3.2 strand regression); the consumed-id ring dedups across a re-survey.
- [ ] **Step 2–4:** Drive the real frame + completion (like `tests/test_thought_processing_crystallize_harness.py`) with a scripted noticing `FakeLlmPort`/result. `make check`.
- [ ] **Step 5: Commit** `test(noticing): real-code sim — buffer→thoughts, source ids, closed-prefix, dedup, 0-LLM idle (lm-705.5)`.

---

## Task 8: Host-integration + bead close + spec note

**Files:** Modify `tests/hermes_internal_cognition_integration.py`; docs; `bd`.

- [ ] **Step 1:** Extend the real-Hermes driver with a "Part B-noticing" scenario (mirror the existing Part-B): seed a buffer segment, drive a noticing `LaunchInternalCognition`, scripted seeds → real `kind=thought` rows with source ids, **non-delivered**; add `bn_*` keys to `_REQUIRED_TRUE_KEYS`.
- [ ] **Step 2:** `make check`. Commit `test(noticing): host-integration — a real noticing pass seeds thoughts, non-delivered (lm-705.5)`.
- [ ] **Step 3:** `bd close lm-705.5` (summary); §10 spec note (mirroring the lm-705.2/705.3 build notes); commit the doc. **Note in the close:** noticing is the concrete appraiser — with it live, the whole vector (notice → process → crystallize) runs end-to-end; lm-705.11 (turn-tail appraiser) is now optional/complementary.

---

## Self-Review (run after execution)

- **Spec coverage (§4/§5/§6):** buffer service (per-session pending→complete ring, closed-prefix, TTL, cursor) ✓ (Task 2) · wired into pre_llm/inbound/post_llm ✓ (Task 3) · idle/size trigger over a closed prefix ✓ (Task 4) · aux noticing pass, subjectless, on the shared seam ✓ (Tasks 4/6) · source-lineage contract: source ids on the seed, segment-membership validation, consumed-id ring dedup, immutable provenance ✓ (Tasks 1/5) · judgment not heuristic (the model decides) ✓ · no privacy classifier ✓ · non-delivery + no proactive collision + strand-dispatch ✓ (Tasks 6/7) · idle 0-LLM ✓ · closed obs enum ✓ (Tasks 4/5) · host-integration ✓ (Task 8).
- **Reconciliation:** noticing + processing share the seam + single-flight; disambiguated by `pending_internal_subject_id` (Task 6) — assert a subjectless completion runs `NoticingApply` and a subject-set one runs `ThoughtProcessingApply`, never both act.
- **Turn-path care:** every buffer hook call is inside the existing plugin-owned fail-loud try (a buffer hiccup never crashes a host turn); the buffer is optional so existing tests/callers are unaffected.
- **Not built here (boundaries):** deep cross-segment reference (running summary / pull-by-id), a model-facing thought READ tool (FR24 conversational — deferred), processing (705.2, done), the arbiter (705.4).
- **Confirm-before-code:** the exact `pre_llm_call` hook factory name/signature in `hooks.py`/`__init__.py` (thread the buffer there); that `post_llm_call` `turn_id` is non-empty on a real reactive turn (host `agent/turn_context.py`); `event.id` shape at the inbound observer (`hooks.py:541`); the `creation_provenance` source-id kwargs (mirror `ThoughtCapture`).

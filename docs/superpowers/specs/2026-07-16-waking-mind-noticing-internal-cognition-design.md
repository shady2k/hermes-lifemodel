# Waking mind — the internal-cognition seam + noticing (design)

**Beads:** **Part I — lm-705.6** (the shared, non-delivered internal-cognition seam) is a
**foundation built and host-integration-tested FIRST**; **Part II — lm-705.5** (noticing —
the being notices what a conversation left worth carrying) is its first consumer.
lm-705.2 (processing) reuses the same seam. Under epic lm-705 (Phase 5a); builds on
lm-705.1 (capture pipeline, done).
**Date:** 2026-07-16
**Status:** design under review — **v2, reshaped after codex review `019f6a75`** (§10)
**Product source:** BRD FR4 (inner life), FR5 (becoming), FR20 (hard cost ceiling), FR24
(the being explains itself), S5 (idle → 0-LLM), NFR1. HLA §4.1 (Thought + provenance), D10.
Owner principles: **appraisal is judgment, not a keyword heuristic**
(`appraisal-is-judgment-not-heuristic`); **no privacy gate — what the owner writes to the
being, it may remember** (`no-privacy-gate-owner-owns-what-they-write`).

## 1. Context and the load-bearing finding

Slice 1 built the capture *pipeline* (Thought object, `thought_seed` signal, 0-LLM
`ThoughtCapture`, intent-bus write, sim) — on `main`, **dormant** (the placeholder keyword
appraiser was removed; noticing is what fills it). The being must genuinely notice.

**The finding (verified in source):** lifemodel has **no internal LLM seam.** The
`LifeModel` graph (`composition.py:100`) has `state/clock/delivery/registry/coreloop/tracer`
— **no `llm`**. Its only cognition is `LaunchProactive` (`core/cognition.py:198`): a
**delivered** Hermes turn read back via `post_llm`. Noticing needs a **private, non-delivered
judgment pass**; so does slice 2 (processing). So the foundation is a **shared internal,
non-delivered async cognition seam** — its own bead, built first. Hermes provides the
primitives (verified): `ctx.llm` (`hermes_cli/plugins.py:351`),
`agent.auxiliary_client.async_call_llm` (used by `plugins/teams_pipeline/pipeline.py`),
plugin aux tasks via `ctx.register_auxiliary_task` (`plugins.py:1047`) routing
`auxiliary.<task>` — **model routing, not a cost ceiling** (§3.4).

## 2. Invariants (do not reopen)

- **Appraisal is JUDGMENT, not a heuristic.** A model judges what the being notices; no
  keyword/pattern matching.
- **Noticing is INVOLUNTARY** — a curator-shaped idle batch, not a `note_to_self` tool the
  being calls mid-reply. Agency belongs to what it does with what stuck (slices 2–4).
- **No privacy gate (owner decision).** The being is the person's own 1:1 companion, not a
  third party: what the human writes to it, it may remember. **Do NOT build an FR26
  sensitivity/no-store classifier into this pass.** (An explicit owner "forget this" / mute
  is separate CONTROL, FR29 — a later lever, not this slice.)
- **The being holds pointers, not the transcript.** Thought = gist + **source ids**; the full
  conversation lives in Hermes/hindsight. Footprint is bounded by thought count, not dialogue
  size.
- **Unit of reading is the recent coherent SEGMENT** (bounded by lull ∨ size-cap), never a
  lone message, never the whole dialogue. Cross-segment continuity comes from the being's
  **own live thought backlog**, read as context — not from re-reading old conversation.
- **The internal call is NON-DELIVERED, ASYNC, off-lock.** It never reaches the human; it
  runs off the 0-LLM tick; **the model call happens OUTSIDE the state-actor lock** — only the
  budget reservation and the typed-result application are serialized (a frame).
- **The completion frame must dispatch every returned launch.** A frame runs *every* enabled
  component regardless of trigger (`core/coreloop.py:301`; `FrameTrigger` isn't in
  `TickContext`), and `CoreLoop` returns `LaunchProactive` *separately* rather than applying
  it (`coreloop.py:360`). So a naive `run_frame(ASYNC_COMPLETION)` that ignores a returned
  proactive launch would set `pending_proactive_id` with nothing injected → real outreach
  blocked. The executor **must** dispatch all returned launches after releasing the lock.
- **Cost is a durable FR20 quota WE build** (§3.4) — not the aux slot. Idle ticks stay 0-LLM
  (S5). Energy is physiology, not the ceiling.
- **Correlation is separate from `pending_proactive_id`** — an internal pass is never read
  back as `[SILENT]`/`SENT` and never occupies the proactive in-flight gate.
- **Thoughts carry source lineage** (§4.3), reuse the capture core, are born `active` and
  just sit (no processing here).

## 3. Part I — the shared internal-cognition seam (lm-705.6, built + host-tested first)

### 3.1 `InternalCognitionRunner` (adapter-owned, on the gateway loop)

`LaunchProactive` is **not** a general async-job mechanism — it schedules a *gateway agent
turn* (`gateway_core.py`), and Hermes owns the background turn + finalizer→`post_llm`. A
direct aux call bypasses all that, so **we** own the orchestration. An adapter-owned runner
(built in `being_platform.connect()`, on the gateway asyncio loop):
- creates and **retains** an asyncio task per launch (a tracked task set);
- **awaits the aux call off the state-actor lock**;
- handles **timeout / cancel / disconnect / task exceptions** → a typed *failure* outcome;
- on completion: rebuilds a fresh `LifeModel`, runs the completion frame (§3.2), clears the
  internal-pending state;
- on `disconnect()`: cancels tracked tasks; on boot: **recovers stale internal-pending** (a
  launch whose task died) so the buffer never strands.

### 3.2 A generic launch-dispatching frame executor

Replace the bare `run_frame(ASYNC_COMPLETION)` completion path with **one executor that
always dispatches every `LaunchProactive` the frame returns** (after releasing the lock),
exactly as the live tick loop does — so an internal completion frame that happens to also
wake `CognitionLauncher` cannot strand a proactive launch. (Alternative considered:
trigger-aware component eligibility — rejected as more fragile than one honest dispatcher.)

### 3.3 The `LlmPort` + a distinct launch intent

- `LlmPort` wired into the graph over `async_call_llm` / `ctx.llm` (thin adapter; core stays
  Hermes-free; tests inject a fake).
- A `LaunchInternalCognition` intent distinct from `LaunchProactive`: its own **correlation id
  + pending field** (`pending_internal_id`), never the proactive ones. **Delivery is
  structurally impossible** — the runner calls the `LlmPort`, never the gateway egress; there
  is no `post_llm` outcome path for it.
- The typed result re-enters via the executor (§3.2) seeded with a result signal; a core
  component applies it via the intent bus (atomic commit). No hook/runner writes the store.

### 3.4 FR20 — a durable quota WE enforce

The aux slot is model *routing*, not a ceiling (`ctx.llm.acomplete_structured` even calls
`async_call_llm(task=None)`). So: a **durable, atomically-reserved call/token quota in the
runtime store**, **shared with proactive contact** (the phase's shared-budget invariant),
**reserved BEFORE the task is created**. v1: a per-day call-count ceiling (simplest reliable
form); define whether a failed call consumes quota. The aux slot picks the cheap model; the
quota bounds the spend.

### 3.5 Host-integration test (isolated `HERMES_HOME`)

Prove, against a real host (never the live being): plugin aux-task registration + custom-slot
routing; **non-delivery**; gateway-loop execution; task lifecycle + shutdown cancellation;
timeout/failure/stale-pending recovery; typed re-entry under the lock; **dispatch of any other
launch the completion frame returns**; hard-budget reservation.

## 4. Part II — noticing (lm-705.5, on the seam)

### 4.1 The conversation buffer (a process-owned service)

A **single process-owned, lock-protected buffer service** injected into the hooks *and* the
adapter — NOT a field on a freshly-built `LifeModel` (graphs are rebuilt per call).
- **Keyed by `session_id` + `turn_id`** (both available at `post_llm`) — the reliable shared
  key (the inbound message id is *not* handed to `post_llm`). The platform message id, if
  needed as a transcript pointer, rides from the inbound observer (`event.id`,
  `hooks.py:541`) into the same-keyed entry; else `turn_id` is the pointer.
- Entry = `(session_id, turn_id, source_ids, user_text, assistant_text, state, ts)` with an
  explicit **`pending | complete | abandoned`** state: inbound/`pre_llm` opens `pending`,
  `post_llm` (successful turns only) completes it. A turn that never completes ages to
  `abandoned` (TTL).
- **Bounded ring**, cursor semantics (cleared after a successful pass), so length is bounded
  by inter-pass interval, not session length.

### 4.2 The idle/size trigger — only a closed prefix

A loop component fires the noticing pass when the conversation is quiet for `N` min **or** the
buffer crosses a size cap — but **only over a closed coherent prefix**: never while an
unmatched `pending` turn exists for that lane (a long tool-heavy reply must not be surveyed
before its `post_llm`). Honors the HLA "not during a turn" contract (`docs/hla.md:290`).

### 4.3 The noticing pass + the source-lineage contract

The aux call reads (recent segment + a bounded top-M of the live thought backlog) → returns
**top-K** typed seeds, each a first-person *gist* + **`source_message_ids`/`turn_id`** +
salience. Extend the contract (slice-1's `thought_seed`/`ThoughtCapture` carry content only):
- the typed seed carries `source_message_ids` + `turn_id`;
- **validate** every returned id is actually in the surveyed segment (no hallucinated source);
- **preserve immutable creation provenance** on an idempotent re-seed (slice-1 invariant);
- **content-vs-source identity:** content-digest id (as slice 1) *and* a **durable bounded
  "consumed source ids" ring** so a message that already produced a thought isn't re-seeded
  even after that thought terminalizes.
Committed via the §3.2 executor, born `active`; then the surveyed prefix is cleared.

## 5. Observability (forced, D10)

Closed `reason` enum (`idle_launch`, `size_cap_launch`, `nothing_lingered`,
`budget_denied`, `noticed`, …); counts + ids ride as **fields** (`noticed_count`,
`thought_ids`, `source_ids`) — never embedded in the reason string.

## 6. Simulation (real code)

Vivify the seam + noticing through the fake-port harness with a **fake `LlmPort`** (scripted
seeds): a buffered sitting → thoughts with the right source ids; continuity uses the backlog
(a thought references a prior one, not raw old text); idle ∨ size-cap both fire; a `pending`
turn blocks the launch; cursor clears; top-K holds; idle-with-empty-buffer stays 0-LLM; the
internal correlation never collides with `pending_proactive_id`; **a completion frame that
also returns a proactive launch dispatches it** (regression for the §3.2 strand). Plus the
Part I host-integration test (§3.5).

## 7. Boundaries

| Part I — seam (lm-705.6, first) | Part II — noticing (lm-705.5) | Deferred |
|---|---|---|
| InternalCognitionRunner (off-lock, lifecycle) | buffer service + entry states | privacy classifier — **removed by owner** |
| generic launch-dispatch executor | idle/size trigger over a closed prefix | deep cross-segment ref (running summary / pull-by-id) |
| LlmPort + distinct correlation | aux noticing pass → thoughts (gist + source ids) | a model-facing thought read tool (FR24 conversational) |
| durable FR20 quota | source-lineage contract + consumed-id ring | processing (lm-705.2), arbiter (lm-705.4) |
| host-integration test | closed obs enum | map-reduce for a single huge sitting |

## 8. Open questions

- Buffer transcript-pointer: is `turn_id` enough, or do we thread the inbound platform message
  id through? (Leaning `turn_id`; add the platform id only if slice 3 needs a deep-link.)
- FR20 default budget + whether a failed call consumes quota; aux-slot name.
- Segment/window size, top-K, backlog top-M (context vs cost) — tune in sim/live.
- Result-apply: reuse `thought_seed`→`ThoughtCapture` (one write door) vs a direct result
  component (leaning reuse, extended for source ids).

## 9. Acceptance

- **Felt (owner):** the being carries threads from real conversation; never nags a mid-reply
  bookmark.
- **Structural:** a non-delivered aux pass exists, never delivers, never collides with the
  proactive path, and its completion frame cannot strand a proactive launch; thoughts carry
  source ids; nothing seeds from pattern-matching; no privacy classifier.
- **Measured:** idle default 0-LLM; cost ≤ the durable FR20 quota; footprint independent of
  dialogue length; cursor + source-dedup hold.
- **FR24 scope for this slice:** thoughts are inspectable via storage/debug; the being
  *conversationally* explaining them needs a model-facing read tool — **deferred** (a small
  follow-up / slice 3), not claimed here.

## 10. Review log

**codex `019f6a75` (2026-07-16), verified.** Reshaped v1 → v2:
- **Async orchestration (critical):** the non-delivered call has no executor — added the
  adapter-owned `InternalCognitionRunner` (§3.1), a foundation bead (lm-705.6) built first.
- **Frame-launch strand (critical):** a bare `ASYNC_COMPLETION` frame also wakes
  `CognitionLauncher` and `CoreLoop` returns the launch separately → a generic
  launch-dispatching executor (§3.2).
- **FR20 ≠ aux slot (high):** the slot is routing; added a durable shared quota (§3.4).
- **Buffer keying/in-flight (high):** process-owned service keyed by `session_id+turn_id`,
  `pending|complete|abandoned` states, launch only a closed prefix (§4.1–4.2).
- **Source lineage (high):** extended the seed/capture contract + a durable consumed-id ring,
  immutable provenance preserved (§4.3).
- **Privacy (high):** FR26 classifier **removed** per owner decision — the being may remember
  what the owner writes to it (§2).
- **FR24 (medium):** narrowed to storage/debug; conversational read tool deferred (§9).
- **Obs enum (medium):** closed reason + counts/ids as fields (§5).
- **Retained (codex-affirmed sound):** no-`llm`-in-graph premise; non-delivered aux is
  possible; separate correlation is correct; segment+backlog+top-K is reasonable; aux call
  off the lock; ASYNC_COMPLETION+intent path consistent with single-store/snapshot.

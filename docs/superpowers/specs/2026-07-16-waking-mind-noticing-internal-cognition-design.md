# Waking mind — noticing, on a shared internal-cognition seam (design)

**Bead:** lm-705.5 (the being *notices* what a conversation left worth carrying),
which introduces the **shared internal-cognition seam** that lm-705.2 (private
processing/rumination) also consumes. Under epic lm-705 (Phase 5a — waking mind);
builds on lm-705.1 (capture pipeline, done).
**Date:** 2026-07-16
**Status:** design under review
**Product source:** BRD FR4 (inner life — thoughts, Zeigarnik), FR5 (becoming), FR20
(configurable hard cost ceiling), FR24 (the being explains itself), FR26 (retention/
consent), S5 (idle → 0-LLM), NFR1 (cheap layers first). HLA §4.1 (Thought + provenance
lineage), D10 (rebuild discipline). Owner principle: **appraisal is judgment, never a
keyword heuristic** (bd memory `appraisal-is-judgment-not-heuristic`).

## 1. Context and the load-bearing finding

Slice 1 (lm-705.1) built the capture *pipeline* — the `Thought` object, the
`thought_seed` signal, the 0-LLM `ThoughtCapture` component, the intent-bus write path,
the real-code sim. It is on `main` but **dormant**: the placeholder keyword appraiser was
removed (owner rejected pattern-matching as the being's judgment), so nothing seeds
thoughts. This bead makes the being genuinely notice.

**The finding that reshapes everything (verified in source):** lifemodel has **no internal
LLM seam.** The `LifeModel` graph (`composition.py`) wires `state/clock/delivery/registry/
coreloop/tracer` — **no `llm`**. Its *only* cognition is `LaunchProactive`
(`core/cognition.py`): a **delivered** Hermes turn whose verdict is read back through
`post_llm`. There is no way today to run a model *without delivering a turn to the human*.

Noticing needs exactly that — a private judgment pass that reads the conversation and
writes thoughts, delivering nothing. So does slice 2 (processing a thought). **The real
foundation is therefore not "the appraiser" but a shared *internal, non-delivered
cognition seam*.** Hermes provides the primitives to build it (verified):
- `ctx.llm` — `PluginContext.llm` (`hermes_cli/plugins.py:351`);
- `agent.auxiliary_client.async_call_llm(...)` — the canonical aux-model call, already
  used by the reference plugin `plugins/teams_pipeline/pipeline.py`;
- **aux-slot routing** — `auxiliary.<task>.{provider,model}` (the curator uses
  `auxiliary.curator`): a cheap side-model, config-selectable, our **FR20** cost lever.

This design specifies that seam and the first consumer (noticing) that validates it.

## 2. Invariants (do not reopen — these are the decisions this conversation settled)

- **Appraisal is JUDGMENT, not a heuristic.** No keyword/substring/pattern matching
  decides what the being notices — a model judges. (Owner, corrected once already.)
- **Noticing is INVOLUNTARY.** Not a `note_to_self` tool the being must remember to call
  mid-reply (that burdens the reply, is unreliable, and puts a voluntary action at an
  involuntary layer). Agency belongs to what the being *does* with what stuck (slices 2–4),
  not to noticing itself.
- **Curator-shaped, idle-triggered batch** — the pattern of the Hermes curator (a
  background aux-model pass gated on inactivity), NOT the curator's subject (it maintains
  *skills*). Fine deterministic parts + an optional aux-model pass, on the being's own idle.
- **The being holds pointers, not the transcript.** A thought = *gist* + **source message
  id(s)**; the full conversation lives in Hermes / the memory provider (hindsight). The
  being never stores or re-reads the whole dialogue. Footprint is bounded by thought count,
  independent of dialogue size.
- **Unit of reading is the recent coherent SEGMENT** — the current "sitting" bounded by a
  lull ∨ a size cap — never a lone message (no context → wrong thought) and never the whole
  dialogue (unbounded). Cross-segment continuity comes from the being's **own live thought
  backlog**, read as context — not from re-reading old conversation.
- **The internal cognition call is NON-DELIVERED and ASYNC** — it never reaches the human;
  it runs *off* the 0-LLM tick (a model call inside a component would block the tick); its
  result commits in its **own `ASYNC_COMPLETION` frame** (the existing async pattern), never
  mid-tick.
- **Cost is bounded by FR20, on a cheap aux slot.** Idle ticks stay 0-LLM (S5); the aux
  pass fires only on a real conversation having happened + budget, bounded output (top-K),
  cheap model. Energy is physiology, **not** the cost ceiling.
- **Thoughts carry source lineage.** The seeding message id(s) land in the thought's
  provenance (`source_signal_ids`/`turn_id`, HLA §4.1) — for the "why did I note this"
  audit (FR24), for source-dedup, and for slice 3 (reaching out about the *specific* thing).
- **Reuse the capture core.** `Thought`, `thought_view`, the intent bus + committer stay;
  a captured thought is still born `active` and just sits (no processing here).
- **The seam is SHARED with lm-705.2** — build it once; processing plugs into the same
  non-delivered async cognition path.

## 3. Design

### 3.1 The shared internal-cognition seam (the foundation)

A non-delivered async model call, mirroring the *shape* of `LaunchProactive` but delivering
nothing:

- **An `LlmPort`** wired into the `LifeModel` graph at composition, over
  `agent.auxiliary_client.async_call_llm` (a thin adapter; the core stays Hermes-free
  behind the port; tests inject a fake). Aux-slot selectable (FR20).
- **A launch intent** (e.g. `LaunchInternalCognition`) distinct from `LaunchProactive`:
  it carries the prompt + a **correlation id** and **pending-idempotency** *separate* from
  the proactive contact ones (`pending_proactive_id` is for delivered contact only — an
  internal pass must not collide with it or be read back as a `[SILENT]`/`SENT` outcome).
- **Delivery suppressed by construction** — the internal pass uses the `LlmPort` directly,
  NOT the gateway egress; there is no `DeliveryPort` call and no `post_llm` outcome path.
- **A typed result → its own frame.** The async call returns structured output
  (deterministic schema + validation); on completion we `run_frame(...,
  trigger=ASYNC_COMPLETION)` seeded with a signal carrying the result, and a core component
  applies it via the intent bus (atomic commit). No hook/pass writes the store directly.
- **Runs off the tick.** The tick (0-LLM) only *decides to launch* and reserves budget; the
  model call happens on the async side, exactly like the proactive path.

### 3.2 The conversation buffer (pointers, bounded)

- **Seeded at inbound** (`pre_gateway_dispatch`, `make_inbound_observer`) — where the stable
  **message id** exists (`event.id`/`event.message_id`, `hooks.py:541`): append
  `(origin_id, user_text)`; the paired assistant text is filled at `post_llm`. Entry =
  `(message_id, user_text, assistant_text, ts)`.
- **Bounded ring**, per session (the owner's live session is long-lived — rolls ~daily by
  `session_reset`), capped by turns/tokens so it can never grow unbounded even if a pass
  fails to fire.
- **Cursor semantics** — a pass surveys only the tail since the last successful pass;
  after it commits, the surveyed entries are cleared. Length is bounded by inter-pass
  interval, not by session length.
- **Retention (FR26):** the buffer is transient working text in our owned layer, discarded
  after the pass. Only *gist + id* persists (in the thought). Sensitive-content policy
  applies to what a thought may hold (a later slice's privacy classifier; flagged, not
  built here).

### 3.3 The idle trigger

A loop component reads the existing silence machinery (`last_exchange_at`/`silence_anchor`,
already read each tick by aggregation): when the conversation has been quiet for `N` minutes
**and** the buffer is non-empty → emit the `LaunchInternalCognition` intent for a noticing
pass. A **size-cap** flush fires the same pass mid-sitting if the buffer crosses a
turns/tokens threshold without a lull (so a marathon chat gets periodic digests). No new
host hook — the being's own idle, curator-style.

### 3.4 The noticing pass

The async aux call reads a bounded context: **the recent segment** (the buffered sitting) +
**the live thought backlog** (`live_thoughts` / a bounded read — the being's carried
threads, for cross-segment continuity). It returns **top-K** typed seeds — each a
first-person *gist* + the **source message id(s)** it sprang from + salience. On completion,
the result frame commits them as `active` thoughts (reusing the `thought_seed` →
`ThoughtCapture` path, or a direct `PutRecord` in the result-applying component — §8),
stamping the source ids into provenance. Then the surveyed buffer tail is cleared.

- **Idempotency at two levels:** content-digest id (as slice 1) *and* source-message-id
  (a message that already produced a thought does not produce a duplicate on a later pass).
- **Bounded output** (top-K) so a rich sitting cannot flood the backlog.

## 4. Flow

`inbound → buffer(user, msg_id)` · `post_llm → buffer(assistant)` · … (0-LLM ticks) … ·
`tick sees idle N min ∨ size-cap, buffer non-empty → LaunchInternalCognition` ·
`async: LlmPort survey(segment + backlog) → typed seeds` ·
`ASYNC_COMPLETION frame → commit thoughts (gist + source ids), clear surveyed tail`.

## 5. Observability (forced, D10)

The launch decision, the pass, and each seeded (or skipped) thought are spans with closed
`reason` codes (`noticed:<n>`, `nothing_lingered`, `idle_launch`, `size_cap_launch`,
`budget_denied`, …); the thought id + source ids ride as fields. The being answers
*«что тебя занимает / почему это заметил»* (FR24) from its own spans + the thought's
source lineage.

## 6. Simulation (mechanism recovery, real code)

Vivify the real seam through the fake-port harness (`testing/harness.py`, extended) with a
**fake `LlmPort`** returning scripted seeds: assert a buffered sitting → thoughts with the
right source ids; cross-segment continuity uses the backlog (a thought references a prior
one, not old raw text); idle ∨ size-cap both fire; cursor clears; top-K bound holds; idle
ticks with an empty buffer stay 0-LLM; the internal correlation never collides with the
proactive `pending_proactive_id`. No "calibrated to humans" claims.

## 7. Boundaries

| This bead (lm-705.5) | Reuses the seam later | Deferred |
|---|---|---|
| the shared internal-cognition seam | lm-705.2 (processing a thought) | deep cross-segment reference (running summary / pull-by-id) |
| conversation buffer + idle/size trigger | | privacy/sensitivity classifier on thoughts |
| noticing aux pass → thoughts (gist + source ids) | | map-reduce for a single huge sitting |
| forced observability + real-code sim | | the arbiter (lm-705.4) |

## 8. Open questions

- **Result-apply path:** does the completion frame re-use the `thought_seed` →
  `ThoughtCapture` component (seed a signal per returned thought) or apply `PutRecord`
  directly in the result component? (Leaning: reuse the signal→component path — one write
  door.)
- **Aux slot naming + FR20 budget:** a dedicated `auxiliary.lifemodel_notice` slot vs a
  shared lifemodel aux slot; the default daily budget and how noticing + (later) processing
  share it.
- **Segment/window size** (context vs cost) and **top-K** default — tune in sim/live.
- **Backlog-as-context shape:** how much of the backlog (top-M by salience) the pass reads.
- **`LlmPort` call shape:** sync-in-a-thread vs the platform loop's async; where the result
  re-enters (a queued `ASYNC_COMPLETION` frame under the state-actor lock).

## 9. Acceptance

- **Felt (owner):** the being carries threads from real conversation and can say what it's
  chewing on and *why* (pointing at what was said); it never nags a "note to self" mid-reply.
- **Structural:** a non-delivered aux pass exists and never delivers / never collides with
  the proactive path; thoughts carry source message ids; nothing seeds from pattern-matching.
- **Measured:** idle default 0-LLM; cost ≤ FR20; footprint independent of dialogue length
  (pointers, bounded segment, top-K); cursor + source-dedup hold.

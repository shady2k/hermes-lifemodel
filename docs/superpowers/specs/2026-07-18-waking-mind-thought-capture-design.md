# Waking-mind thought capture — the being drops a thought in its own turn (lm-705.11)

**Status:** design **v1** (owner brainstorm 2026-07-18). Ready for review → plan. Task **lm-705.11** (epic **lm-705** — Живой ум). Lights up the currently-inert waking-mind vector by giving live conversation a real, judgment-based thought producer — **without** a heuristic classifier (owner principle, memory `appraisal-is-judgment-not-heuristic`).

> **One-line intent:** add ONE tool — `create_thought` — that lets the being, mid-reply, drop a thought as an **impulse onto the signal bus**, which the **aggregation layer** dedups and persists as a durable thought; downstream rumination/crystallization (already built) then turn it into commitments/beliefs. The being's reply is its thinking; this only drops a mental bookmark.

## 1. Goal

Make **live conversation create thoughts** on the live being. Today the whole waking-mind machine (capture → rumination → crystallization → injection) is built and wired but **inert**: nothing produces a thought from a reactive exchange, so the processing selector finds an empty backlog every tick and the being has ~0 thought rows (lm-705.11 description, verified against `__init__.py:599`).

There are two intended thought producers (spec §8 of the epic left the form open — "classifier vs ride-the-tail"):
- **Curator / `noticing`** (batch, idle) — already built + registered.
- **Ride-the-tail** (in-the-moment) — the being flags a thought during its own reply turn. **Not built.** This spec builds it.

**Owner decision (2026-07-18):** keep **both** producers and compare their effectiveness on the live being; possibly keep both permanently. The in-the-moment path has genuine, non-redundant value — during its turn the being is *in context and doing reasoning*, so it captures live context + affect that the curator only reconstructs later from a transcript.

## 2. Background — the seam that already exists

- **The impulse type + the aggregation consumer are built and tested.** `thought_seed_signal(...)` (`core/taxonomy.py:192`) is the impulse; `ThoughtCapture` (`core/thought_capture.py:25`) is the AGGREGATION component that consumes it from the bus, dedups, and emits a `PutRecord` — never a direct store write. The only production emitter of `thought_seed_signal` today is the post-hoc appraiser seam (`hooks.py:780`, inside `_maybe_capture_thought`).
- **The post-hoc appraiser is a no-op.** `make_post_llm_observer(appraiser=...)` (`hooks.py:581`) defaults `appraiser=None`; `__init__.py:599` passes no appraiser ("that appraiser is not wired yet"). `core/appraisal.py` holds only the `Appraiser` protocol + `ThoughtSeed` — no concrete implementation. So `_maybe_capture_thought` (`hooks.py:752`) returns early, always.
- **The bus is a general transport, not tick-bound.** `FrameTrigger` (`core/frame.py:36`) is a closed set of FOUR occasions — `HEARTBEAT`, `EVENT`, `ASYNC_COMPLETION`, `ADMIN`; the heartbeat is one of four. `SignalFrame` (`core/frame.py:53`) is *"the in-memory ephemeral signal bus for ONE ExecutionFrame"*. `run_frame` (`core/frame.py:115`) conducts any signals under the one process-wide re-entrant `_STATE_ACTOR_LOCK` (`core/frame.py:85`), so *every* frame — heartbeat, event, async-completion, admin — is strictly serialized; two frames can never interleave their snapshot→commit. `_maybe_capture_thought` already conducts an `EVENT` frame from a hook. There is **no** race with the live tick and **no** tick-coupling.
- **Downstream is wired and functional** (verified 2026-07-18). The internal-cognition LLM lane goes through the sanctioned `ctx.llm.acomplete_structured` facade → the user's **main** model (`adapters/plugin_llm_adapter.py`; the cheap-aux-model routing is a known gap, lm-705.10, not a blocker). `InternalCognitionRunner` is driven by `being_platform._tick` (`adapters/being_platform.py:238`, `report.internal_launches → runner.launch`) whenever a `LlmPort` is injected. Live emitters now exist (processing selector + noticing trigger, `composition.py:395`+). The adapter's "no live emitter yet" note is **stale** (pre-dates noticing/processing landing). Whether the curator is *currently producing* on the live being is a runtime question (read-only `python -m lifemodel.activity` baseline), not a design blocker.

## 3. The mechanism — impulse → bus → aggregation → durable memory

`create_thought` is a normal `lifemodel`-toolset tool (mirrors `commitment`/`check_in` registration, `__init__.py:687`/`:867`). Its handler does **not** write the store. It builds one `thought_seed_signal` per captured text and conducts them through the bus exactly as `_maybe_capture_thought` does today:

```
create_thought(["…", "…"])            # the being, mid-reply
  → run_frame(coreloop, [thought_seed_signal(content=…), …], trigger=EVENT)
      → SignalFrame carries the impulses; components run AUTONOMIC → AGGREGATION → COGNITION
      → ThoughtCapture (aggregation) reads the impulses, DEDUPS, emits PutRecord(s)
      → frame commits intents atomically under _STATE_ACTOR_LOCK → durable Thought row(s), state=ACTIVE
```

The impulse is **transient** (the `SignalFrame` is discarded on commit — "lost consciousness → don't replay stale impulses", `core/frame.py:59`); the **thought is durable** (a persisted row that survives restart, has a status model, is ruminated on later ticks, and can crystallize). Durability is NOT a discriminator between "bus" and "direct write" — the frame commits durably either way; routing through aggregation is correct because that is where dedup + provenance + (future) semantic dedup belong.

**Why the bus, not a direct `memory.put` (like the commitment tool did).** The commitment tool's direct write (`hooks.py:1677`) is a CRUD-style lifecycle op. Thought *creation* is an **impulse** and must flow the nervous-system path so the aggregation layer owns integration/dedup. This also aligns the codebase toward the owner's invariant (§10): impulses and events go through the bus; nothing bypasses it. (`create_thought` moves us toward that invariant; removing the commitment tool, §9, removes one bypass.)

### 3.1 Dedup — done by aggregation, from the bus

`ThoughtCapture` already dedups: the thought id is a content digest (`seed_thought_id`, `core/thought_view.py:66` — `sha256(strip(content))[:16]`); an in-frame `seen` set collapses same-frame duplicates (`core/thought_capture.py:48`); against the store, an existing row's provenance is preserved (`existing_by_id`, `:51`) so a re-notice / host retry upserts ONE row, never a growing pile.

**Honest limitation (accepted for v1):** this is **exact-content** dedup only. Near-duplicates in different words become separate thoughts, and downstream does not merge them (the processing pass sees one thought in isolation). This is acceptable at the low capture volume of v1. The *right home* for better-than-hash (semantic) dedup is the aggregation layer itself — it already sees both the incoming impulse and the full live thought set — so upgrading is a localized change there, filed as a follow-up (§13).

### 3.2 The one safety check (verify in plan, not assert)

`_maybe_capture_thought` conducts its `EVENT` frame from `post_llm` — *after* the turn. A tool handler runs *mid*-reply (inside the host's tool-dispatch loop). Conducting `run_frame(EVENT)` from there is a **re-entrant nested frame** relative to whatever frame the host turn sits in. The `_STATE_ACTOR_LOCK` is an `RLock` (safe against self-nesting, `core/frame.py:85`) and the processing selector's `LaunchInternalCognition` is *dropped* on non-heartbeat frames (only `being_platform._tick` reads `internal_launches`, `core/thought_processing.py:249`), so no spurious rumination fires mid-dialogue. The plan MUST confirm — with a test, not prose — that a nested `EVENT` frame from a tool handler commits cleanly and has no unintended component side effects on the live turn's state.

## 4. The tool contract

- **Name:** `create_thought`. Toolset `lifemodel`. Registered like the others (`ctx.register_tool`, `__init__.py`).
- **Schema:** ONE parameter — a list of thought texts. Accept `1..N` strings so the being can drop several notions in one call (cheaper than N calls, and natural — a mind surfaces a couple of things at once). A single notion is a one-element list. **No** salience / actionability / importance fields — the being does not self-score at capture (owner: capture is a cheap reflex, not a micro-rating; §5).
- **Description (English prose — the ride-the-tail instruction lives here, 0 extra LLM, language-agnostic content).** Draft:
  > *Capture a thought you want to return to later. When something in this exchange leaves a thread worth revisiting — a question you want to sit with, something you noticed about them or about yourself, an idea not yet finished — write it here in a sentence (in whatever language is natural). Your reply is your thinking; this only drops a bookmark your quieter, later mind will pick up and think through. Not every turn — only when something genuinely tugs. You may capture more than one at once.*
- **Hermes contract (exactly like `check_in`/`commitment`, `hooks.py:1621`):** the handler returns a `json.dumps` STRING, returns `{"error": …}` on failure, and **never raises**. A throw is logged + counted + swallowed. The being gets an honest result (e.g. `{"captured": N, "deduped": M}`).
- **Metric:** a per-call outcome counter, mirroring the other tools' folded metrics (e.g. `lifemodel_thought_tool_total{outcome=captured|deduped|empty|error}`), so "did the being drop a thought, how often" is answerable from `metrics.sqlite`, not a grep (observability-first).

## 5. Honest thought schema — appraisal is absent until judged

The current `Thought` carries flat valuation fields born at `0.0` (`build_thought` defaults `salience=0.0`, `attention_score=0.0`, `actionability=0.0`, `other_regarding_value=0.0`, `core/thought_view.py:81`). This conflates two different meanings: **"weighed, and genuinely unimportant (zero)"** vs **"not yet weighed"**. That is a lie in the model — a placeholder zero masquerading as a measurement (owner: *"как мы поймём, это ноль потому что пофиг или потому что ещё не оценили?"*).

**Decision — a raw thought carries no valuation; the valuation is a separate, optional structure produced later:**
- Introduce an `Appraisal` value object holding `salience` (+ `actionability`, `other_regarding_value`).
- `Thought` gains `appraisal: Appraisal | None`, **replacing** the three flat valuation fields. Born `None` — honestly "not yet weighed". A present `Appraisal` (even with a low salience) honestly means "weighed, and this is its weight". Absence ≠ zero.
- `attention_score` gets the same honesty: it is not intrinsic to a thought — it is the arbiter's dynamic competition weight — so it becomes optional/absent until the arbiter (lm-705.4) computes it, rather than a persisted `0.0`.
- **Nested sub-structure, NOT a child durable row** (the owner's "дочернюю структуру" realized without object proliferation). A separate durable row would split identity: dedup keys off the raw thought, provenance/lineage descend from it, crystallization references it — a second row doubles all of that for no gain. `appraisal` hangs *on* the thought. The store is JSON-payload (`domain/objects/thought.py` `req_float(payload, "attention_score")`, `:111`), so an optional nested object is a straightforward `opt_*` decode.

**Who fills `appraisal`, and when.** Importance is *discovered by attention*, not declared at birth. The natural moment is the **deliberate rumination pass** (`thought_processing`, `core/thought_processing.py`) — the being, thinking a thought over with budget, forms and records its salience. This is the "moment salience is born" the owner asked about. **Both** producers (tool and curator) therefore create thoughts with `appraisal=None`; a single locus fills it.

**v1 scope of the appraisal change.** v1 ships the honest *schema* (appraisal optional, born absent) so no zero-placeholders ever land, and the selection sort tolerates absence. v1 does **not** yet build the appraisal-fill pass — so all v1 thoughts are unappraised and the selector picks among them FIFO (honestly "nothing to rank on yet", see §6). Filling `appraisal` in processing is the immediate follow-up (§13), landing where we already touch that code.

## 6. Selection with unappraised thoughts

The processing selector picks the top-salience ACTIVE thought (`live_thoughts` sorts `(-salience, id)`, `core/thought_view.py:141`; `_pick` returns `actives[0]`, `core/thought_processing.py:336`). With `appraisal` optional, the sort MUST handle absence:
- **v1:** absent appraisal ranks as the lowest salience (sentinel), ties broken by id. Since all v1 thoughts are unappraised, selection is uniformly **FIFO-by-id** — honest: there is nothing to prioritize on until appraisal-fill lands. This matches the accepted v1 tradeoff (low volume, one thought chewed per heartbeat, all eventually processed).
- **Follow-up (when appraisal-fill exists):** decide the ordering of unappraised vs appraised. Recommended: **unappraised-first** (a fresh, unweighed thought demands first evaluation — novelty captures attention), so a new thought is appraised promptly, then competes normally on re-processing by its now-real salience. Deadlock-avoidance note for the plan: unappraised thoughts must NOT sort last, or they would never be picked to be appraised.

## 7. Retire the post-hoc appraiser seam (keep the transport + consumer)

Because the producer is now the tool, the post-hoc judgment is dead:
- **Remove:** the `Appraiser` protocol (`core/appraisal.py`), `_maybe_capture_thought` (`hooks.py:752`), and the `appraiser=` parameter on `make_post_llm_observer` (`hooks.py:581`) + its no-op wiring intent in `__init__.py`.
- **Keep:** `ThoughtSeed` (repurposed as the tool→signal payload shape), `thought_seed_signal` (`core/taxonomy.py:192`), and `ThoughtCapture` (`core/thought_capture.py`) — the tool now feeds them. `ThoughtSeed.salience` (currently required) is dropped/made optional in step with §5; the tool supplies content only.
- The `post_llm` observer keeps its OTHER jobs unchanged (proactive read-back resolution, buffer close, turn-observability close).

## 8. Remove the `commitment` tool (keep the object)

The being's single reply-time creative act becomes "notice a thought" (owner: *"дать существу только один инструмент… обязательства, суждения и прочие вещи — что ж теперь, для каждого создавать отдельный инструмент?"*). Commitments are born by **crystallization** from a thought (already built: the processing pass's `crystallize_commitment` outcome → `_crystallize`, `core/thought_processing.py:383`), not by a dedicated tool.
- **Remove:** the `commitment` tool — `make_commitment_tool` (`hooks.py:1609`), its schema/description, the `commitment` `register_tool` (`__init__.py:867`), and the tool-only metric `COMMITMENT_TOOL_TOTAL` (`core/tick_metrics.py:62`). Update the creation-boundary safety prose that lived in the tool description + `PROCESSING_INSTRUCTIONS` accordingly (crystallization remains the sole boundary).
- **Keep, untouched:** the `Commitment` domain object, `commitment_view`, `read_active_commitments`, and `make_commitment_injector` (commitments still exist and are surfaced into live turns).
- **Accepted consequence:** with the tool gone and no "thought → discharge/defer" path yet, the being cannot manually retire a live commitment mid-life; the injector keeps surfacing active ones (bounded, `max_surfaced=8` + overflow notice). Acceptable for v1 (commitments are rare and deliberate); "thought → discharge/defer" folds into the commitment-lifecycle bead (lm-705.12).
- **Final being tool surface:** `check_in`, `write_soul`, `create_thought`.

## 9. Observability — comparable producers

Both producers write to the same thought store, so to later judge "which path is more effective" (owner: *"посмотрим на деле какой из них эффективней"*) the source MUST be distinguishable. `build_thought` already takes a `source` + provenance; the tool tags its thoughts with a distinct source (e.g. `source="create-thought-tool"` / a `thought:live:` id namespace, mirroring commitment's `live`/`seed` split) vs the curator's `noticing` source. That lets a read-only query attribute captures, crystallizations, and drops per producer — the empirical comparison the owner wants, from durable data, no live instrumentation.

**Cross-producer dedup is intentionally per-producer.** Because tool thoughts carry a `thought:live:` id namespace and curator thoughts their own, the SAME content noticed by BOTH the being (in the moment) and the curator (later batch) yields TWO rows — one per producer. This is deliberate: at v1's low volume it is what makes the effectiveness comparison possible (you can see each producer's independent catch), and the cost is negligible. Dedup remains exact within a producer. If cross-producer duplication ever becomes noisy, it folds into the semantic-dedup follow-up (§13), which is the layer that can reconcile across namespaces.

## 10. Alignment with the "everything through the bus" invariant

Owner principle (2026-07-18): **all impulses and events flow through the signal bus, in layer order (AUTONOMIC → AGGREGATION → COGNITION); nothing bypasses it or writes state directly.** The frame pipeline already enforces layer order under one lock. This work advances the invariant (`create_thought` uses the bus; the commitment tool's direct `memory.put` is removed). Technical channels that are neither impulses nor events — notably `write_soul` writing identity to disk (`core/frame.py` `state_actor_lock` docstring) — are a **separate category**, out of the invariant's scope, not exceptions to force through the bus. Enforcing the invariant in code (a lint/test that only the frame committer + sanctioned lock-holders call the mutating store methods) is a follow-up hardening task (§13), not part of v1.

## 11. Thought lifecycle (recap — mostly already built)

`ACTIVE` (born) → selector picks one per heartbeat under gates (single-flight, min-interval, FR20 budget) → non-delivering rumination pass → `ThoughtProcessingApply` transitions it:
- `resolve` → `RESOLVED`; `park` → `PARKED` (6h/24h/72h backoff ladder, re-armed when due, `EXPIRED` after 3 park cycles); `drop` / 3 no-progress → `DROPPED`;
- `crystallize_commitment` → a `Commitment` is born (durable child, inherits the thought's salience + lineage) and the thought → `RESOLVED` (`core/thought_processing.py:420`).

The only v1 changes to this recap: the thought is born with `appraisal=None` (§5); a v1 thought that crystallizes passes an absent/flat salience to its commitment (real salience arrives once appraisal-fill lands).

## 12. Scope

**In v1 (lm-705.11):**
1. `create_thought` tool — text-only, array-capable, English description, Hermes contract, folded metric (§4).
2. Handler conducts `thought_seed_signal` impulses via `run_frame(EVENT)`; aggregation dedups + persists (§3); nested-frame safety verified by test (§3.2).
3. Honest schema — `Thought.appraisal: Appraisal | None`, born absent; `attention_score` optional; selector sort tolerates absence, v1 = FIFO (§5, §6).
4. Retire the post-hoc appraiser seam; keep transport + consumer (§7).
5. Remove the `commitment` tool; keep the object/view/injector (§8).
6. Source-tag tool thoughts for producer comparison (§9).

**Explicitly NOT in v1 (follow-ups, §13).**

## 13. Follow-up beads (filed, not built here)

- **Appraisal-fill in processing** — the rumination pass records the being's salience judgment (the "moment salience is born"). Lands where §5 leaves off; decides unappraised-vs-appraised sort ordering (§6).
- **Belief unification** — add a `crystallize_belief` outcome to processing (mirror `crystallize_commitment`) and make `noticing` produce only thoughts (retire the direct `belief` seed, `core/noticing.py:864`), so *all* durable types are born via thought processing (owner principle). Touches shipped belief-track — its own reviewable bead.
- **"Everything through the bus" invariant + linter** — enforce single write-path; `write_soul` exempt as a technical channel (§10).
- **Semantic (better-than-hash) dedup in aggregation** (§3.1).
- **Cheap-aux-model routing** — already tracked (lm-705.10).
- **Arbiter fills dynamic `attention_score`** — already tracked (lm-705.4), last by design.
- **Live baseline read** — read-only `python -m lifemodel.activity` on the live being to record whether the curator is already producing, as the comparison's starting point (verification-phase, not a code bead).

## 14. Testing

- **Handler contract:** returns a JSON string; `{"error": …}` on a forced failure; never raises; empty/whitespace input handled; array of N; folded metric increments per outcome.
- **Impulse → durable thought (real-code, mirrors `test_thought_capture.py`):** a `create_thought` call results in N `ACTIVE` thoughts in the store with the tool source; identical content across two calls upserts ONE row (dedup); two identical strings in one array collapse to one.
- **Nested-frame safety (§3.2):** an `EVENT` frame conducted from a tool-handler context commits cleanly, emits no `LaunchInternalCognition` into a live-turn dispatch, and does not corrupt in-flight turn state.
- **Honest schema:** a thought round-trips with `appraisal=None` (absent, not zero) through encode/decode; the selector picks unappraised thoughts (FIFO) without a sort crash; a legacy row (pre-refactor, with flat fields) decodes without loss.
- **Removals:** `make_post_llm_observer` with no `appraiser=` still resolves proactive read-backs / closes the buffer / closes the turn; the `commitment` tool is gone from the toolset while the commitment injector still surfaces active commitments.
- **`make check` green** (ruff format+check, mypy, pytest) before commit; deploy is its own gated step.

## 15. Non-goals

- No heuristic/keyword appraiser (owner principle — rejected form).
- No self-scoring of salience/importance at capture.
- No merge of the reactive inbound EVENT frame and the tool's capture frame.
- No semantic dedup, no appraisal-fill, no belief unification, no arbiter — all §13.
- No touching `write_soul` / `check_in`.

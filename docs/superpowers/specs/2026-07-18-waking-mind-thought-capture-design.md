# Waking-mind thought capture — the being drops a thought in its own turn (lm-705.11)

**Status:** design **v2** (owner brainstorm 2026-07-18 + codex design review `019f767c`, findings adjudicated). Ready for review → plan. Task **lm-705.11** (epic **lm-705** — Живой ум). Lights up the currently-inert waking-mind vector by giving live conversation a real, judgment-based thought producer — **without** a heuristic classifier (owner principle, memory `appraisal-is-judgment-not-heuristic`).

> **One-line intent:** add ONE tool — `create_thought` — that lets the being, mid-reply, drop a thought (with its own rough sense of how much it matters) as an **impulse** the aggregation layer dedups and persists as a durable thought, via a **restricted capture path** that touches ONLY the thought store; downstream rumination/crystallization (already built) then turns it into commitments/beliefs. The being's reply is its thinking; this only drops a mental bookmark.

## 1. Goal

Make **live conversation create thoughts** on the live being. Today the whole waking-mind machine (capture → rumination → crystallization → injection) is built and wired but **inert**: nothing produces a thought from a reactive exchange, so the processing selector finds an empty backlog every tick and the being has ~0 thought rows (verified against `__init__.py:599`).

Two intended producers (the epic's §8 open question — "classifier vs ride-the-tail"):
- **Curator / `noticing`** (batch, idle) — already built + registered.
- **Ride-the-tail** (in-the-moment) — the being flags a thought during its own reply turn. **Not built.** This spec builds it.

**Owner decision (2026-07-18):** keep **both** producers and compare their effectiveness on the live being; possibly keep both permanently. The in-the-moment path has genuine, non-redundant value — during its turn the being is *in context and doing reasoning*, so it captures live context + affect that the curator only reconstructs later from a transcript.

## 2. Background — the seam that already exists

- **The impulse type + the aggregation consumer are built and tested.** `thought_seed_signal(...)` (`core/taxonomy.py:192`) is the impulse; `ThoughtCapture` (`core/thought_capture.py:25`) is the AGGREGATION component that consumes it, dedups, and emits a `PutRecord` — never a direct store write. Its only production emitter today is the post-hoc appraiser seam (`hooks.py:780`, inside `_maybe_capture_thought`).
- **The post-hoc appraiser is a no-op.** `make_post_llm_observer(appraiser=...)` (`hooks.py:581`) defaults `appraiser=None`; `__init__.py:599` passes none. `core/appraisal.py` holds only the `Appraiser` protocol + `ThoughtSeed`. So `_maybe_capture_thought` (`hooks.py:752`) returns early, always. **Retiring it is safe** (codex-confirmed: unwired in production).
- **The bus is a general transport, not tick-bound.** `FrameTrigger` (`core/frame.py:36`) is a closed set of FOUR occasions — `HEARTBEAT`, `EVENT`, `ASYNC_COMPLETION`, `ADMIN`. `SignalFrame` (`core/frame.py:53`) is the in-memory ephemeral signal bus; `run_frame` (`core/frame.py:115`) commits under the one re-entrant `_STATE_ACTOR_LOCK` (`core/frame.py:85`), so every frame is strictly serialized (codex-confirmed).
- **But `run_frame` runs the WHOLE registry** (`core/coreloop.py:315`, `for component in self._registry.enabled()`), regardless of trigger — this is the crux the mechanism must respect (§3).
- **Downstream is wired and functional** (verified 2026-07-18, codex-confirmed). The internal-cognition LLM lane goes through the sanctioned `ctx.llm.acomplete_structured` → the user's **main** model (cheap-aux routing is a known gap, lm-705.10, not a blocker). `InternalCognitionRunner` is driven by `being_platform._tick` (`adapters/being_platform.py:234`). Live emitters now exist (processing selector + noticing trigger). Whether the curator is *currently producing* on the live being is a runtime question (read-only baseline), not a design blocker.

## 3. The mechanism — a RESTRICTED capture path (NOT a full frame)

`create_thought` is a normal `lifemodel`-toolset tool (mirrors `commitment`/`check_in`, `__init__.py:687`/`:867`). Its handler must persist thoughts through the aggregation layer (so aggregation owns dedup), but it must **NOT** run a full `run_frame`.

**Why not `run_frame(EVENT)` (codex Critical #1, self-verified).** `CoreLoop.tick` runs *every* enabled component regardless of trigger (`core/coreloop.py:315`); `TickContext` does not even carry the trigger. So a full frame conducted from a tool handler would ALSO run physiology (`core/personality.py`), affect (persists felt state), contact/solitude drives, **desire aggregation and proactive cognition**, and advance tick bookkeeping (a spurious double-tick, `core/coreloop.py:392`). Worst case: with an already-active desire, `CognitionLauncher` writes an intention, reserves energy, sets `pending_proactive_id`, and returns a `LaunchProactive` in one batch (`core/cognition.py`) — but the heartbeat path DISPATCHES launches (`being_platform._tick`: `run_frame` **then** `dispatch_launches`, `:236-237`), and a tool handler does not. The launch is **stranded** while its pending marker + energy reservation commit → future outreach can **deadlock**. A test cannot make this safe; the architecture must.

**The restricted capture path.** Conduct the thought-seed impulse(s) through a path that runs **only** the aggregation capture component and commits **only** its intents, under `_STATE_ACTOR_LOCK`, with **no** physiology/desire/cognition components and **no** tick bookkeeping:

```
create_thought([{content, salience}, …])       # the being, mid-reply
  → capture_thoughts(coreloop, seeds)           # NEW restricted entrypoint (not run_frame)
      → under _STATE_ACTOR_LOCK:
          run ONLY ThoughtCapture.step over a minimal context (the seeds + current object snapshot)
          collect its PutRecord intents
          commit ONLY those intents (the same committer)
      → durable Thought row(s), state=ACTIVE; NO launches produced, NO tick advance, NO other State mutated
```

This preserves the owner's principle — **impulse → aggregation dedups → durable memory** — with zero side effects. The impulse is transient; the thought is durable. The plan settles the exact entrypoint shape (a scoped committer call over a filtered intent set, or a dedicated `CoreLoop.capture(...)` method); §12/§14 name the required guarantee + test.

**Alignment note (§9 invariant).** This is the bus discipline done *correctly*: capture still flows through the aggregation layer under the one lock, but scoped to the one component that subscribes to the impulse — not a bypass, and not a full-brain side-effect.

### 3.1 Dedup — done by aggregation, hardened against resurrection

`ThoughtCapture` dedups by content digest: the thought id is `seed_thought_id` (`core/thought_view.py:66` — `sha256(strip(content))[:16]`); an in-frame `seen` set collapses same-call duplicates; a re-notice / host retry upserts ONE row.

**Must-fix (codex Major #5, a real pre-existing bug the tool would expose).** `ThoughtCapture` currently checks only the **live** snapshot (`live_thoughts`, `core/thought_capture.py:40`). Re-capturing content whose prior row is **terminal** (`resolved`/`dropped`/`expired`) finds nothing live → it upserts a fresh `active` row with the same id, **resurrecting** the dead thought and overwriting its provenance. Noticing already avoids this by checking the authoritative store across ALL states (`core/noticing.py:759`). Fix: the capture path checks the authoritative store for **any-state** existence of the id and treats a duplicate as a **no-op** (never an active upsert, never a provenance overwrite).

**Accepted limitation (v1):** dedup is exact-content only; near-duplicates in different words become separate thoughts, and downstream does not merge them. Acceptable at v1 volume; semantic (better-than-hash) dedup — the aggregation layer's job, since it sees the whole live set — is a follow-up (§12).

## 4. The tool contract

- **Name:** `create_thought`. Toolset `lifemodel`. Registered like the others.
- **Schema:** a list of `1..N` thoughts (so the being can drop several notions in one call — cheaper than N calls, and natural). Each entry:
  - `content` (required string) — what to return to.
  - `salience` (the being's **own rough sense** of how much it matters). **Owner decision A (2026-07-18):** the being provides this even though it is imprecise — an in-the-moment estimate beats a flat `0` placeholder, and it avoids a cross-kind envelope refactor (see §5). Optional; a neutral non-zero default if omitted. Coarse is fine (a small felt scale mapped to the envelope `salience`); the plan picks the exact shape, biased to LOW cognitive load on the reply turn.
- **No** other valuation fields (`actionability`/`other_regarding_value` stay at their envelope defaults; the arbiter/processing refine later).
- **Description (English prose — the ride-the-tail instruction, 0 extra LLM, language-agnostic content).** Draft:
  > *Capture a thought you want to return to later. When something in this exchange leaves a thread worth revisiting — a question you want to sit with, something you noticed about them or about yourself, an idea not yet finished — write it here in a sentence (in whatever language is natural), with a rough sense of how much it tugs at you. Your reply is your thinking; this only drops a bookmark your quieter, later mind will pick up and think through. Not every turn — only when something genuinely tugs. You may capture more than one at once.*
- **Hermes contract (like `check_in`/`commitment`, `hooks.py:1621`):** returns a `json.dumps` STRING, `{"error": …}` on failure, **never raises**.
- **Honest result (codex Minor #8).** Because the restricted path (§3) runs `ThoughtCapture` directly, the handler CAN see real created-vs-duplicate counts and return `{"accepted": N, "deduped": M}` truthfully (unlike `run_frame`, whose `TickReport` exposes no component result).
- **Metric:** a per-call outcome counter (e.g. `lifemodel_thought_tool_total{outcome=captured|deduped|empty|error}`), so "did the being drop a thought, how often" is answerable from `metrics.sqlite`, not a grep.

## 5. Salience — the tool provides it; NO envelope refactor

The earlier plan (v1 spec) proposed replacing the flat thought valuation with an optional nested `appraisal`, born absent. **Codex Major #2 (self-verified) killed that as scoped:** `salience` is not a thought-payload field — it is a `BaseObject` **envelope** field (`domain/objects/base.py`), copied into `MemoryDraft`, persisted as a `REAL NOT NULL DEFAULT 0` column and used for cross-kind ordering (`state/sqlite_store.py`, `salience_desc = "salience DESC, id ASC"`). Making it optional is a storage refactor across desires/intentions/commitments/beliefs + a row migration + noticing/renderer changes — far beyond this task, and `_crystallize` requires a concrete `float` anyway (`core/thought_processing.py:401`).

**Resolution (owner decision A):** do **not** refactor. The being provides a rough `salience` through the tool (§4); the thought is born with a real value on the existing envelope column. The envelope `salience` is best understood as a **scheduling projection** (0 = "not yet placed in the queue"), not a claim that the being judged the thought worthless — so a thought born with the being's own estimate is honest, and a thought that somehow lacks one sits at a neutral default, not at the degenerate bottom. Noticing already supplies `salience` the same way (`core/noticing.py:775`), so **both producers are uniform** — no asymmetry, no noticing change, no migration, no renderer change.

**Deferred (own follow-up, §12):** a deliberate appraisal pass that *refines* the being's rough in-the-moment number with a more considered judgment (the "importance is discovered by attention" idea), and — only if ever wanted — separating the being's semantic judgment from the scheduling projection. Neither is needed to light up the vector.

## 6. Retire the post-hoc appraiser seam (keep the transport + consumer)

Because the producer is now the tool, the post-hoc judgment is dead:
- **Remove:** the `Appraiser` protocol (`core/appraisal.py`), `_maybe_capture_thought` (`hooks.py:752`), and the `appraiser=` parameter on `make_post_llm_observer` (`hooks.py:581`) + its no-op wiring intent in `__init__.py`.
- **Keep:** `ThoughtSeed` (the tool→signal payload shape — now carrying content + salience + producer), `thought_seed_signal` (`core/taxonomy.py:192`), and `ThoughtCapture` (`core/thought_capture.py`) — the tool + the restricted path now feed them.
- The `post_llm` observer keeps its OTHER jobs unchanged (proactive read-back resolution, buffer close, turn-observability close — codex-confirmed independent).

## 7. Commitment tool + curator KEPT — this change is purely additive

**Owner decision B (2026-07-18):** keep the `commitment` tool **entirely** (create / discharge / defer) and keep `noticing`. Codex Major #7 (self-verified) showed removing the tool would **leak** live commitments: the tool is the ONLY path for `honoured`/`dropped`/`deferred` transitions (`hooks.py:1687`); crystallization only *creates* commitments, cannot close them — so completed/obsolete commitments would stay `active` and be injected every turn forever, degrading the injector's accuracy. So `create_thought` is **additive only**: nothing about commitments, beliefs, `noticing`, the injectors, `PROCESSING_INSTRUCTIONS`, or the creation-boundary prose changes.

**Final being tool surface:** `check_in`, `write_soul`, `commitment` (unchanged), `create_thought` (new).

## 8. Observability — comparable producers

Both producers write to the same thought store; to later judge "which path is more effective" (owner: *"посмотрим на деле какой из них эффективней"*) the source MUST be distinguishable. Add a validated `producer` field to the thought-seed (`create-thought-tool` vs `noticing`) that `ThoughtCapture` records into the thought's `source`/provenance (it currently hardcodes `source="thought-capture"`, `core/thought_capture.py:63` — codex Minor #5a). Then a read-only query attributes captures, crystallizations, and drops per producer, from durable data, no live instrumentation. (Ids stay content-global; a producer that re-notices content another already stored is a dedup no-op per §3.1, and provenance records the first creator — sufficient for a rough effectiveness comparison. Per-producer id namespacing, if ever needed for exact overlap counts, is a follow-up.)

## 9. Alignment with the "everything through the bus" invariant

Owner principle (2026-07-18): impulses and events flow through the signal bus, in layer order, under one lock; nothing writes state directly or bypasses aggregation. The restricted capture path (§3) honors this — capture flows through the aggregation layer under the lock — while avoiding the full-brain side-effects a naive `run_frame` would cause. Technical channels that are neither impulses nor events (e.g. `write_soul` writing identity to disk) are a separate category, out of the invariant's scope. Enforcing the invariant in code (a lint/test that only the committer + sanctioned lock-holders mutate the store) is a follow-up hardening task (§12).

## 10. Thought lifecycle (recap — unchanged, already built)

`ACTIVE` (born, with the being's rough salience) → selector picks the top-salience ACTIVE thought per heartbeat under gates (single-flight, min-interval, FR20 budget) → non-delivering rumination pass → `ThoughtProcessingApply` transitions it: `resolve`→`RESOLVED`; `park`→`PARKED` (6h/24h/72h ladder, re-armed when due, `EXPIRED` after 3 cycles); `drop`/3-no-progress→`DROPPED`; `crystallize_commitment`→ a `Commitment` is born (inherits the thought's salience + lineage) and the thought→`RESOLVED` (`core/thought_processing.py:420`). No v1 change here — real salience from the tool means the selector's salience-desc ordering is meaningful from day one (no FIFO-among-zeros).

## 11. Scope (v1, lm-705.11)

1. `create_thought` tool — `content` + rough being-`salience`, array-capable, English description, Hermes contract, honest `{accepted, deduped}` result, folded metric (§4).
2. **Restricted capture path** — a scoped entrypoint that runs only `ThoughtCapture` + commits only its intents under the lock; NO full-registry frame, NO tick advance, NO launches (§3). The must-fix that makes the tool safe.
3. **Resurrection fix** — the capture path checks the authoritative store for any-state existence and no-ops on a duplicate (§3.1).
4. **Producer tagging** — thought-seed carries `producer`; `ThoughtCapture` records it into source/provenance (§8).
5. Retire the post-hoc `Appraiser` seam; keep transport + consumer (§6).
6. **No** removal of the commitment tool; **no** envelope/appraisal refactor; **no** noticing change (§5, §7).

## 12. Follow-up beads (filed, not built here)

- **Deliberate salience refinement** — a processing-side pass that refines the being's rough in-the-moment salience with a considered judgment ("importance discovered by attention"); optionally separates semantic judgment from the scheduling projection.
- **Belief unification** — add a `crystallize_belief` outcome to processing (mirror `crystallize_commitment`) and make `noticing` produce only thoughts, so all durable types are born via thought processing (owner principle). Touches shipped belief-track — its own reviewable bead.
- **"Everything through the bus" invariant + linter** — enforce single write-path; `write_soul` exempt as a technical channel (§9).
- **Semantic (better-than-hash) dedup in aggregation** (§3.1).
- **Cheap-aux-model routing** — already tracked (lm-705.10).
- **Arbiter fills dynamic `attention_score`** — already tracked (lm-705.4), last by design.
- **Live baseline read** — read-only `python -m lifemodel.activity` on the live being to record whether the curator already produces, as the comparison's starting point (verification-phase, not a code bead).

## 13. Testing

- **Handler contract:** returns a JSON string; `{"error": …}` on a forced failure; never raises; empty/whitespace input handled; array of N; folded metric increments per outcome; honest `{accepted, deduped}` counts.
- **Restricted-path SAFETY (codex-required, the load-bearing test):** a `create_thought` call creates the thought rows AND leaves **everything else unchanged** — assert every `State` field, `tick_count`/`last_tick_at`, all non-thought rows, and BOTH launch collections (`launches`, `internal_launches`) are untouched, and that with an already-active desire NO `LaunchProactive` is produced or stranded.
- **Dedup + resurrection:** identical content across two calls → ONE row; two identical strings in one array → one; re-capturing content whose prior row is `resolved`/`dropped`/`expired` is a **no-op** (the terminal row is NOT resurrected and its provenance is intact) — tested against active, parked, AND terminal rows in the real SQLite store.
- **Salience:** a thought is born with the being-provided salience on the envelope; the selector's salience-desc ordering reflects it.
- **Producer tagging:** tool thoughts carry the tool producer/source; a coexistence test with a noticing-created thought shows distinguishable sources.
- **Retire:** `make_post_llm_observer` with no `appraiser=` still resolves proactive read-backs / closes buffer / closes turn.
- **`make check` green** (ruff format+check, mypy, pytest) before commit; deploy is its own gated step.

## 14. Non-goals

- No heuristic/keyword appraiser (owner principle — rejected form).
- No full `run_frame` from the tool handler (codex Critical #1 — the reason for the restricted path).
- No envelope/`salience` refactor, no nested `appraisal`, no migration (§5).
- No removal or change of the commitment tool, the injectors, or noticing (§7).
- No semantic dedup, no salience-refinement pass, no belief unification, no arbiter, no bus-invariant linter — all §12.
- No touching `write_soul` / `check_in`.

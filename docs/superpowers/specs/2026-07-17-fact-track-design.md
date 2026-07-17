# Belief-track — the being's fallible knowledge shapes its replies (and, next, its memory)

**Status:** design **v2** (revised after codex design-review `019f70d2` + owner decisions 2026-07-17). Ready for review → plan. Epic **lm-705** (Phase 5a — waking mind). First shippable slice of **lm-705.16** (the inner life shapes live dialogue), the *knowledge* half; the *actionable-thought → desire* half is a separate, parallel track (§4, §11).

> **v1 → v2 changes (why this was revised):** codex flagged that a fallible inference must not be called a `fact` (unearned epistemic status), that processing cannot judge epistemics from a lossy gist (only noticing has the evidence), that an every-turn scan-all injector reinforces false beliefs and is materially heavier than the felt-state template, and that inferred psychological traits are sensitive (FR26) — especially once exported. All accepted. It also corrected a modelling error the owner and I both made: `UserModel` is **not** "the user's current mood/energy" — it is the durable, per-field-TTL cache of the user's **receptivity/norms** (cadence, good/bad hours, boundaries, styles, explicit prefs; `domain/objects/user_model.py`). A belief is distinct from it (an open proposition, not one of those closed facets).

## 1. Goal

Give the being a first-class **`belief`** — a *defeasible* proposition it has inferred about the person or world (*"they seem to get anxious before a loss of status"*) — carrying **confidence** and **evidence**, and never authoritative merely because it is stored. Make it *matter* through one channel now and one next:
- **v1:** a **gated, sensitivity-aware `pre_llm_call` injector** surfaces a few of the being's held beliefs into its live turn so its understanding shapes the reply — the main turn model judges relevance, we do not.
- **v1.5:** an **autonomous "write-and-stay-silent" turn** — the being persists a matured belief into whatever memory provider the user configured, via the provider's *own* tool, and returns silence.

**Owner-identified live (2026-07-17):** the being noticed *"a deep psychological pattern about the owner"* (salience 0.85) and it went nowhere — parked, never touching a conversation, never remembered. Diagnosis (owner): a noticed pattern is *knowledge*, and knowledge is inert unless it reaches the reply and is remembered.

## 2. Object model (corrected)

Four distinct durable notions — **do not conflate**:

| Entity | What it is | Role | Status |
|---|---|---|---|
| **`AgentState`** (our `State`) | the **being's own** vitals/control — `u`, energy, fatigue, mood, contact bookkeeping | drives the being | exists |
| **`UserModel`** | our model of the **user** — durable inferred **receptivity/norms** (cadence, good/bad hours, boundaries, acceptable styles, explicit prefs), per-field `{value, inferred_at, ttl}` | gates contact; "how to relate to them" | exists |
| **`thought`** | the being's first-person reasoning stream — *"do I want to ask…?"* | → child thought \| **desire candidate** \| resolved \| dropped \| parked | exists (kind); → desire path unbuilt (§4) |
| **`belief`** (NEW) | a **defeasible** inferred proposition about person/world, with confidence + evidence | shapes replies (v1); remembered in the provider (v1.5) | **this spec** |

A `belief` is **not** an `Opinion` (an evaluative *stance* the being holds — *"I dislike status games"* — a planned but unbuilt kind), **not** a `Prediction` (future-oriented), **not** a `UserModel` facet (those are a closed receptivity schema, not open propositions), **not** a `Thought` (bounded first-person reasoning). `Observation` stays provenance, not a kind.

## 3. Two tracks of noticing (why this spec is one half)

- **Actionable question/curiosity** (*"why did they react like that?"*) → a `thought` → **desire candidate** → the existing `Drive→Desire→Intention→act-gate` engine → *behaviour*. HLA §4.1 calls this top-down path *"the cure for `[SILENT]`."* — **separate track (§11).**
- **A fallible understanding** (*"they seem anxious before a loss of status"*) → a **`belief`** → shapes the reply now, remembered next. — **this spec.**

## 4. The `belief` entity — defeasible by construction

New `domain/objects/belief.py`, mirroring `domain/objects/commitment.py` (the last catalog extension). The catalog is closed at construction (`domain/objects/registry.py`); adding a kind = the dataclass + a transitions table + **one line in `_CATALOG`** + exports + a `core/belief_view.py`.

- `BeliefState(StrEnum)`: `ACTIVE`, `SUPERSEDED`, `DROPPED`, `EXPIRED`. Live = `{ACTIVE}`.
- `BELIEF_TRANSITIONS`: `active → {superseded, dropped, expired}`; the three terminals sealed. **Revision = supersession** via `BaseObject.supersedes`/`superseded_by` — but see §9: v1 does **not** wire the revision *operation*; those fields are populated only when a future reconciliation pass exists. v1 is honest that it can hold near-duplicate active beliefs.
- `@dataclass(frozen=True, kw_only=True) class Belief(BaseObject)`:
  - `content: str` — the proposition (*"They tend to …"*).
  - `subject: str` — who/what it is about; v1 uses `"owner"` (a plain entity/topic string; canonical structured subjects are a later concern).
  - `source_message_ids: tuple[str, ...]` — **evidence** (the turn/message ids the belief is grounded in — the same cited-ids contract noticing already enforces, §5).
  - `source_thought_ids: tuple[str, ...]` — thought lineage.
  - `KIND: ClassVar[str] = "belief"`, `SCHEMA_VERSION: ClassVar[int] = 1`.
  - From `BaseObject`: **`confidence: float` is mandatory here** (validated to `[0,1]` in the view builder — the registry does not enforce range), `salience` (attention, **not** truth — kept distinct from confidence), `sensitivity` (§9), `supersedes`/`superseded_by`, `provenance`, etc.
- **Id — deterministic content-digest, scoped to source** (mirror `crystallized_commitment_id`): `belief_id(source_thought_id, content) = derive_id("belief", "seed", sha256(f"{source_thought_id}\x00{content.strip()}")[:16])`. Exact re-derivation upserts one row; a reworded belief is a new row (§9 owns dedup honesty). `UnicodeEncodeError` on a lone surrogate → `InvalidPayload`.
- `core/belief_view.py`: `build_belief(...)` (one constructor, born `active`), `belief_from_seed_fields(...)` (strict parse of untrusted model JSON → `Belief`, `[0,1]` confidence validation, narrow `InvalidPayload`), `encode_belief`, `read_active_beliefs(memory, *, min_confidence, exclude_private, limit)` — **a bounded, filtered store query, not a scan-all** (§6).

## 5. Production — beliefs are born in **noticing** (the grounded pass)

Codex's load-bearing point: **only noticing sees the evidence** (the actual conversation segment + cited source ids); processing sees a lossy `gist`. So beliefs are formed where the grounding is — at noticing — not by a context-poor processing crystallise.

Extend the noticing seam (`core/noticing.py`), not processing:
- `NOTICING_JSON_SCHEMA`: each seed gains `"kind": "thought" | "belief"` (default `"thought"`) and, for a belief, `"confidence": number` (required) and optional `"sensitivity": "normal"|"sensitive"|"private"` (§9 applies a conservative floor regardless).
- `NOTICING_INSTRUCTIONS`: add the distinction in the being's own judgment-first voice — *a belief is a fallible understanding of the person/world you'd act on across conversations, stated with how sure you are; most exchanges yield none; do not inflate a one-off into a standing belief.* No keyword rules.
- `validate_noticed_seeds`: unchanged anti-hallucination (cited `source_message_ids` ⊆ segment) — this **is** the evidence contract; a belief with no grounded ids is dropped. Validate `confidence ∈ [0,1]`.
- `NoticingApply`: branch on the validated seed's `kind` — a `belief` seed → `build_belief(... source_message_ids=<cited>, confidence=<self-assessed>, sensitivity=<floored>)` → `PutRecord(encode_belief(...))`; a `thought` seed → `Thought` exactly as today. Both ride the existing atomic apply.
- **No `crystallize_belief` in processing** (dropped from v1). Processing keeps `resolve/park/drop/crystallize_commitment` unchanged — commitments are owed follow-ups and genuinely benefit from the deliberate second pass; beliefs need the evidence noticing already has.
- **Epistemic caution without a second pass:** a belief is born `active` but **only surfaced above a confidence threshold** (§6). A single low-confidence inference is held (and can be reinforced/superseded later, a follow-up) but does not colour replies. Reinforcement (a second independent observation raising confidence) is a follow-up (§11), not v1.

## 6. Surfacing — a gated, sensitivity-aware `pre_llm_call` injector

A third `pre_llm_call` injector (`hooks.py`), registered in `__init__.py` like `make_felt_state_injector`/`make_genesis_injector` (Hermes concatenates every hook's non-`None` return). Fixing codex's cost + repetition findings:

- **Bounded store read, not a scan-all.** `read_active_beliefs(memory, min_confidence=θ, exclude_private=True, limit=N)` issues a `kind='belief' AND state='active' ORDER BY … LIMIT N` query — cost independent of lifetime history (contrast the `live_thoughts` decode-all-then-slice shape, which must not be copied here). Reads via the same fresh-`LifeModel` path the other injectors use, fail-soft.
- **Gated presentation (mechanical, not a keyword/LLM relevance judge).** Surface at most a **small N** (start **2**), preferring **new / recently-formed / not-recently-surfaced** beliefs over static top-salience, with a **per-belief cooldown** so the same belief is not re-injected every turn (repetition reinforces a possibly-false premise — codex High-4). The cooldown rides a bounded `surfaced_belief_ids` ring in `AgentState`, stamped atomically like `affect_display` (no heavy hot-path write). *Which* of the surfaced set is relevant is left to the main turn model — but *whether* to re-present is decided here first.
- **Sensitivity filter:** `PRIVATE` beliefs are never surfaced (§9); `sensitive` ones may be, since the reply is to the owner themselves.
- **Framing:** a compact first-person block that marks these as **fallible, held lightly, and not instructions** — *"Some things I think I've come to understand about them (my own read, I could be wrong): …"*. This blunts the prompt-injection-amplification risk of re-feeding user-derived prose as standing context.
- **Cache-prefix-safe & fail-soft:** identical to felt-state — the `{"context": …}` is spliced onto a copy of the outgoing message only (never the cached system prompt / rolling history), and any raise → recorded on `BrainHealth`/`MetricRegistry` under a `pre_llm_call` observer → return `None`.

## 7. Provider-write (v1.5) — the being writes the belief and stays silent

Deferred to a **v1.5 bead, immediately after v1** (not to sleep — honouring "сразу"). Confirmed viable: the proactive path is a **full, tool-enabled agent turn** — `ReachInEgress.reach_out` injects a turn into the live session (`inject_proactive_turn`), which runs the gateway's normal agent loop *with tools*. So a matured belief can trigger an **autonomous action turn** whose act-gate outcome is *action, not message*: the being calls whatever memory tool the active provider injected (provider-agnostic — a public plugin cannot assume `fact_store`/holographic and has no sanctioned programmatic recall/retain API) and returns **silence** (`[SILENT]`). Our store stays source of truth (D7/NG5 — we do not build our own memory graph, only write via the provider's tool); the provider is a recall layer.

**Export contract (part of v1.5, per codex):** a **sensitivity gate before export** (`PRIVATE`/"do-not-remember"/third-party-secret beliefs are fail-closed, never exported), **attribution** to this being (FR14/FR22), an **idempotency key** (the belief id) so re-runs don't duplicate, an **export-status** marker on the belief, and honest handling when the provider cannot later delete. This mechanism also naturally unifies with the §4 desire track (a matured belief → a desire to record it → the autonomous action turn) — but v1.5 can wire it directly first.

## 8. What noticing produces vs what already exists

Unchanged: noticing still produces `thought` seeds and the whole claim/finalize durable-buffer path (shipped). New: a `belief` seed kind and its apply branch. Processing untouched.

## 9. Privacy & sensitivity (v1 — this is correctness, not polish)

Inferred psychological/relational traits are among the most sensitive content the plugin creates, surfaced into replies (v1) and exported (v1.5). Per **FR26** (retention rules: "do not remember", third-party secrets, inferred private traits):
- A belief carries `sensitivity` (`normal`/`sensitive`/`private`), **model-proposed under a conservative floor**: inferred psychological/health/sexual/financial traits default to at least `sensitive`; anything the owner flagged "don't remember" or that is a third party's secret is `private` and **fail-closed** (never surfaced, never exported).
- `PRIVATE` never enters the injector (§6) or the export (§7).
- **Observability redaction:** spans/logs carry the belief **id + subject + confidence + sensitivity category**, and the reflection — **not** the full `content` verbatim by default (the aux_raw already holds the model's words once; do not fan sensitive plaintext across more stores). This tightens the D10 rule for this kind specifically.
- A `drop`/"forget" path exists via `active → dropped`; the owner-driven forget UX is refined with v1.5's export.

## 10. Observability (forced — D10)

The noticing span already carries aux_raw + reflection. Add, per surfaced/created belief: the `belief` id, `subject`, `confidence`, `sensitivity` category, and the source evidence ids — never the full content string in the reason field (§9). The injector logs how many beliefs it surfaced + their ids + a latency metric (not just a count).

## 11. Scope

- **v1 (this plan):** the `belief` kind; noticing produces beliefs (grounded, confidence, sensitivity floor); the gated sensitivity-aware injector; observability/redaction.
- **v1.5 (next bead):** the autonomous write-and-silent provider export + the export contract (§7).
- **Deferred (honest — follow-up beads):** belief **reconciliation** (find & supersede a revised belief; dedup near-duplicates; conflict detection) — v1 may hold near-duplicate active beliefs, bounded by the small-N confidence-gated surfacing; **expiry/decay** (nothing auto-expires a stale belief today; a sweep is future); belief **reinforcement** (a second observation raising confidence); the **actionable-thought → desire** behaviour track; injector cooldown/rotation tuning against live traces.

## 12. Testing approach

- **`belief` object:** encode/decode round-trip; deterministic id; `confidence` `[0,1]` validation (out-of-range → `InvalidPayload`); transition legality (`active → superseded/dropped/expired`; terminals sealed); registry rejection of `_`-prefixed field / unknown transition; sensitivity round-trips.
- **Production (noticing):** a `belief` seed with cited (in-segment) source ids + confidence → a `Belief` row with evidence + validated confidence + floored sensitivity; a belief seed with **ungrounded** ids → dropped (anti-hallucination); a `thought` seed → `Thought` unchanged; a below-θ belief is stored but the injector won't surface it; a `private`-floored trait never surfaces.
- **Injector:** bounded query returns ≤ N; `PRIVATE` excluded; cooldown prevents re-injecting the same belief on consecutive turns; no live beliefs → `None`; a raising read → `None` + recorded failure; the block is ephemeral; framing marks fallibility.
- **Observability:** the creation/surface spans carry id/subject/confidence/sensitivity (not full content); injector logs count/ids/latency.
- **Real-code sim:** a noticing pass forms a belief from a segment; a later turn's `pre_llm_call` surfaces it (once), framed as fallible.

## 13. Decisions resolved (owner delegated, 2026-07-17)

- **(a) `belief`, not `fact`** — defeasible semantics (mandatory confidence + evidence; stored ≠ authoritative). Epistemic honesty so the being holds these lightly, not as truths.
- **(b) Born in noticing**, not a processing `crystallize_belief` — only noticing has the evidence.
- **(c) Provider-write = v1.5**, immediately after v1, via the autonomous write-and-silent turn (not sleep) — keeps v1 shippable while honouring "сразу".
- **(d) Sensitivity/privacy in v1** — the injector already exposes inferred traits; sensitivity is correctness, and a prerequisite for the v1.5 export.

## 14. Open questions (small, for the plan)

1. **Confidence threshold θ** for surfacing — start conservative (e.g. 0.6) and tune on live traces; a config knob, not a code constant.
2. **N** surfaced per turn — start 2.
3. **`subject` in v1** — kept as `"owner"` (free string, minimal); structured subjects deferred to v1.5/export where the provider's entity model matters.

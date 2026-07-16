# The waking mind — thinking in an attention economy (design)

**Phase:** Phase 5a — a **new intermediate phase** carved out of lm-adz (Фаза 5 —
Желания и взросление) and realizing epic **lm-egg** (inner economy of competing
drives) as a shippable slice. Roadmap chain:
`lm-4fv (Genesis, done) → [Phase 5a: waking mind] → lm-adz (rest of Phase 5) → lm-0od (Phase 6)`.
**Date:** 2026-07-16
**Status:** **v3.1.1 — ready to plan.** Slice-3 redesigned by the owner + codex-reviewed twice (`019f6c40`, 2026-07-16); v2 was codex `019f69d3` (§10)
**Product source:** BRD FR3 (желания — plural, compete by salience, resolve, leave
residue), FR4 (внутренняя жизнь — thoughts, Zeigarnik), FR5 (взросление — opinions as
residue), FR20 (configurable hard cost ceiling), S5 (idle → 0-LLM). Principle §9.2 (model
on humans, cut baroque). HLA §4.1 (Thought as a bounded generative stream; Desire's two
sources; the energy layer), D8 (BDI core), D10 (rebuild discipline: sim on real code,
forced observability, safety fail-closed, certified wake).

## 1. Context and goal

Today the being has exactly **one** internal drive: contact (`u` — loneliness,
integrated by an AUTONOMIC drive-integrator; `u ≥ θ` wakes cognition). One drive plus a
threshold is a **timer**, however it is dressed. Worse: because the being has nothing else
to want, the only way to stop it pestering its human is to *forbid* contact with hard
gates. A human is silent because they are **busy living**; our being is silent because it
is **forbidden**. That asymmetry — silence imposed rather than emergent — is the
mechanistic feel the owner named on 2026-07-05 (epic lm-egg).

The cure is to give the being **other things to want**. The being has no world yet
(text-only until Phase 7), so its "other things" cannot be errands — they must be its
**inner life**, and the honest inner life it can have now is **thinking**.

**Goal of this phase:** the being acquires a **bounded waking inner life**. It *creates*
thoughts from what happens and *processes* them later, and a processed thought **crystallizes
into a durable "reason" object** — most naturally a `Commitment` (a follow-up it owes) — as
opposed to the bare "I'm lonely". *(v3.1 — codex: Phase 5a builds these durable **initiation
reasons**; turning one into an actual warm outreach is the later **contact** work — §7. The
end-state is proactive contact with a content-bearing source, but this phase produces the
**source**, not the send.)* After this phase the being is richer to talk to in *every*
conversation.

### 1.1 Why a separate phase, not the first task of Phase 5

Teaching the being to think and giving it competing drives are **one foundation seen from
two ends**:

- **Thinking without an economy runs away.** The previous mechanism (`ThoughtGeneration`)
  generated thoughts freely and produced — live — a **silence spiral and duplicate
  thoughts**; it was torn out in the D9/D10 rebuild (and the thought machinery was
  explicitly **moved to Phase 6**, see `core/cognition.py:122`). Rumination without cost
  either spirals (Nolen-Hoeksema) or is strangled by ad-hoc limits — a timer on thoughts.
- **An economy over hollow drives is noise in a costume.** An arbiter over abstract
  "curiosity" with no real object and no real satiation is a random number suppressing
  contact — the "garnish pretending to be the meal" the epic warns against.

They need each other: the shared budget *bounds* thinking; thinking is what the arbiter
*arbitrates*. Everything else in Phase 5 (full SDT vector + temperament weights, trigger/
commitment neurons, opinions, open loops, receptivity, learned set-point) is
**superstructure** on this foundation. The roadmap's own rule sanctions carving it out:
*«Фазы 3–8 ещё крупноваты — дробим перед планированием каждой»*.

## 2. Invariants (do not reopen)

- **Create ≠ process.** A thought is *created* cheaply, from a real event; it is
  *processed* — ruminated on — later, and only under budget. **The reply is the thinking**
  during a live turn; no rumination inside a dialogue turn.
- **Snapshot-per-tick** (HLA §4.1) prevents in-frame recursion. It does **not**, by
  itself, bound rumination across ticks — that needs explicit per-thought attempt/park
  bounds (§4.1, a **required** contract, not a hope).
- **Competition is the meal; noise is garnish.** Variety must come from a genuinely
  competing vector, not from randomness on one threshold. Honest limit (§4.4): with
  **event-only** thoughts a dormant relationship has an empty backlog, so competition
  alone **cannot permanently** dissolve timer behaviour — it makes contact
  context-sensitive *while a backlog exists*.
- **The safety floor is unchanged and fail-closed** — including `repeat_pure_longing`
  (`core/aggregation.py:346`), which HOLDs a *pure-longing* (DRIVE-spring) bid once an
  earlier one is unanswered, **regardless of `u`**. The arbiter never weakens the floor.
- **Liveness is an invariant over real outcomes, not over arbitration.** (Replaces the v1
  "certified ceiling wins unconditionally" claim, which was false against the floor
  above — §4.2.) Winning arbitration ≠ a desire ≠ a launched turn ≠ a delivered message.
  The invariant is stated on the **pipeline output** (§4.2), and the answer to prolonged
  deprivation is the **thought-origin** spring (not a bigger `u`). *(v3.1 — codex: this is a
  **contact** invariant, and slice 3 no longer builds contact; it moves with the deferred
  contact/arbiter work. Slice 3 makes **no** liveness claim — only crystallization. Slice 4's
  arbiter preserves contact-liveness **only once** the deferred object→contact adapter exists.)*
- **Cost is bounded by a hard FR20 ceiling, independent of energy** (§4.5). Energy is
  physiology, not the billing boundary (`core/personality.py:47` refills every tick,
  cheaply). A day with nothing to process still trends to **$0** (S5 preserved for the
  genuinely-idle case); a day with a live backlog spends **at most** the FR20 ceiling.
- **Event-seeded thoughts only.** Spontaneous mind-wandering (thoughts from nothing) is
  **deferred to Phase 6**.
- **Observability is forced** (D10), with a **closed** reason enum: the thought id is a
  span *field*, never embedded in the reason string; positive choices (`rest`/`think`/
  `crystallize`) are distinct from suppressions (§5).
- **Sim runs the real code** (D10) through the existing real-code harness
  (`testing/harness.py`). No parallel model of the tick.
- **Text-only holds.** No world actions.

## 3. Scope and build order

The value is delivered in **ordered slices**. The stochastic arbiter — the riskiest,
least-validated part — is the **last** slice and is **evidence-gated**: we do not put D10
liveness behind an unvalidated controller before live traces prove a healthy backlog
exists. This is codex's re-decomposition and it matches the project's "observe live, don't
certify theory" ethos.

1. **Thought capture** — event-seeded creation with a bounded, durable lifecycle. Needs
   the **appraisal seam** (§4.1) — no such seam exists today.
2. **Private thought processing** — one-shot, under a hard FR20 quota, via a **new
   non-delivering cognition path** (§4.1) — the existing `CognitionLauncher` only delivers.
3. **Thought crystallization** *(redesigned v3 — owner, 2026-07-16)* — a processed thought
   **crystallizes into a durable catalog object** (§4.2): rumination decides what the thought
   *becomes* — a **registered** catalog `kind` (a closed, discriminated set; this slice ships
   `Commitment`) — emitted through the registry + `PutRecord` door, and the slice **stops
   there** (no contact, no send). The first new type built is **`Commitment`** (a
   follow-up the being holds — HLA's "strongest non-intrusive reason, serving the other").
   Turning a crystallized object into an actual outreach is a **separate, later** concern (the
   contact pipeline / arbiter reads it as a source). The old "thought mints a contact desire
   and delivers" framing is **superseded** — that is now merely *one* possible crystallization
   (`kind=desire`, `spring=THOUGHT`), and its delivery is out of this slice.
4. **The arbiter** *(only after live traces from 1–3 show a reliably populated, healthy
   backlog)* — the 3-axis homeostatic selection, with the liveness + cost + feasibility
   contracts of §4.4. May be split off into its own bead if slices 1–3 teach us it should.

**Deferred → Phase 6 (lm-0od):** spontaneous mind-wandering, deep/multi-branch thought
trees, sleep/consolidation.
**Deferred → rest of Phase 5 (lm-adz):** full SDT vector + temperament weights (FR7),
trigger/commitment *neurons* (the `Commitment` **type** now ships in slice 3, v3 — but
*arbitrating / acting on* commitments, and the contact pipeline reading them, is deferred),
**opinions/predictions as first-class** (added later by slice 3's generic crystallization
mechanism — same door, new builder), open loops + receptivity, learned set-point (lm-ocx).
**Not reworked:** the v1 contact drive (lm-x43) is correct as an isolated organ and
becomes **one axis** — untouched.

## 4. Design

### 4.1 The thought lifecycle

**Creation (cheap) needs a new appraisal seam.** No seam appraises an ordinary completed
exchange today: `make_post_llm_observer` returns immediately unless the turn is a pending
*proactive* one (`hooks.py:386`), and the inbound observer emits only actor/quality/
timestamp, no content. **Required:** a bounded appraisal of a completed dialogue turn that
seeds a Thought via the intent bus (`PutRecord`, `kind=thought`, `trigger=event`). A hook
**must never write the store directly**; it seeds a frame with a bounded appraisal result,
and a core component emits the `PutRecord`. **Appraisal form:** *not* a keyword heuristic — a
cognition-grade judgment; its exact form is **settled in lm-705.11** (the single authority — this
spec does not pick classifier-vs-tail vs the noticing design's idle pass). Creation is **not**
slice 3's concern (slice 3 *processes* existing thoughts); and it is **dormant live** until
lm-705.11 lands, so slice 3 is exercised by **seeded** thoughts meanwhile.

**Processing (expensive) needs a new non-delivering cognition path.**
`CognitionLauncher` only launches a *delivered* proactive turn and reads back `SENT` /
`[SILENT]` (`core/cognition.py:100–214`) — unsafe for private rumination. **Required:** a
distinct internal-cognition protocol:

- a **non-delivering** launch intent/port (delivery suppressed by construction);
- correlation + pending-idempotency **distinct** from outbound contact;
- an async completion frame carrying a **typed thought outcome** (deterministic schema +
  validation);
- atomic application of the thought transition **and** the crystallized object's `PutRecord`
  (§4.2 — a *registered* kind via its builder, not a hard-wired desire and not an arbitrary
  runtime kind) in one commit.

**Bounded lifecycle (required — snapshot-per-tick is not enough).** Processing develops
**one** selected thought by one layer (top-K, **K=1**, by salience). The Thought schema
already carries `no_progress_count` / `park_count` / `parked_until` (`domain/objects/
thought.py`); this phase **defines their rules**:

- **max total processing attempts** per thought → terminal `drop`;
- **durable increment** of `no_progress_count` on a failed/malformed/no-progress outcome
  (else an async failure is an unbounded retry loop);
- **park backoff** and a **max park cycles** bound;
- terminal behaviour after repeated malformed LLM output.

Outcomes *(v3, tightened v3.1 — codex)* — a **closed, discriminated** set, **not** a
runtime-arbitrary kind: **`crystallize_<kind>`** (§4.2 — a per-kind *trusted builder* produces
the object, `default_registry().encode()` validates it, and the completion emits the `PutRecord`;
the source thought transitions terminal with a provenance link, in **one** atomic commit) ·
**park** · **drop** · **resolve** (plain — nothing durable). This slice registers exactly one
variant, **`crystallize_commitment`**; a new target is a **new registered variant + builder**,
never an accepted arbitrary `kind` string (`PutRecord` does **not** validate a kind — only
`KindRegistry.encode` of a typed object does, so a free kind+payload would bypass the typed
boundary or overwrite a singleton). **A crystallization whose payload fails the builder/registry
is a no-progress outcome** — caught and routed through the same `no_progress_count` bound as a
malformed result (never an uncaught exception that strands the thought without incrementing its
cap); only an empty-`raw` transport failure stays transient/unpenalized. Processing discharges
the nag; an unprocessed thought decays slowly. *(v3: "mint a contact-desire" is gone as a
distinct outcome — a contact `Desire` is just one crystallization variant, and producing it
delivers nothing here.)*

### 4.2 Crystallization — a processed thought becomes durable objects *(redesigned v3 — owner, 2026-07-16)*

**The model.** Processing a thought is **rumination that crystallizes**: the being thinks a
thought over and it *becomes* a durable catalog object (D8: "cognition mints Desire / Thought /
Intention"). The typed processing outcome is a **closed, discriminated** choice —
`crystallize_<kind>` for a kind this slice has *registered a builder for* — **not** an arbitrary
runtime `kind` (**codex v3.1**: `PutRecord` carries a mutation but does **not** validate a kind;
only a *trusted per-kind builder* + `default_registry().encode()` yields the typed object the
registry will accept, so a free kind+payload could bypass the typed boundary or clobber a
singleton — `desire`/`intention`/`user_model`). `ThoughtProcessingApply` runs the builder,
encodes, emits the `PutRecord`, and transitions the source thought terminal, provenance-linked
(`source_thought_ids` / `parent_id`), in **one atomic commit** (`StateActor` batches the
thought-transition + object-`PutOp` under one `BEGIN IMMEDIATE`). A thought crystallizes into
**one** object here (K=1). "A thought can produce any type" is the **direction**: the mechanism
is extensible by *registering another variant*, so adding `Opinion` / `Prediction` later is a new
builder, not a mechanism change. **The slice stops at producing the object — no contact, no send,
no arbiter.**

**First new type — `Commitment`.** The catalog (D8) declares `Commitment · Opinion ·
Prediction` as extensions; **none exists in code yet** (`domain/objects/` has only
`Desire / Intention / Thought / UserModel`). This slice builds the first: **`Commitment`** —
*what the being decided it **owes**, having thought it over* (a follow-up: "ask how their
interview went", "come back to the moving-house topic"). HLA §4.1 names it "the strongest
non-intrusive reason, **serving the other**", so it is the natural crystallization for the
epic's goal (content-bearing *initiation*) — whereas `Opinion` / `Prediction` shape *replies*,
not initiation (added later by the same mechanism).

**`Commitment` model** *(finalized here — codex v3.1; the plan only fills field types)*:
- **Non-singleton** — the being holds *many* commitments at once (unlike the contact `Desire`
  singleton); a **deterministic** id from the source episode **+ a semantic fingerprint**
  (mirroring `seed_thought_id`), never random (HLA), never a bare global content hash (distinct
  episodes must not conflate).
- Semantic payload: `content` (1st-person), a **typed trigger** (`trigger_kind ∈ time|event|
  condition`, `trigger_value`, optional `due_at` — Gollwitzer if-then), `source_thought_ids`,
  `other_regarding_value`, and **`basis`** (`promised | follow_up | self_assumed`, so an ordinary
  interesting thought cannot masquerade as a debt). `salience` / `expires_at` ride the **base
  envelope**, not the payload.
- State machine (full, registry-guarded): `active → honoured | dropped | expired | deferred`,
  `deferred → active | honoured | dropped | expired`; terminals `honoured` / `dropped` /
  `expired`.
- **Boundary vs `Intention`** (must not blur): a `Commitment` is an **enduring owed follow-up /
  source object**; an `Intention` (Bratman) is an executable, send-gating plan. A commitment only
  *later* becomes a `Desire`, which crystallizes into an `Intention` — that whole chain is the
  deferred contact work, not this slice.
- **Catalog registration:** register `Commitment` in the registry `_CATALOG` + export from
  `domain.objects`; update the exact-kind tests and **extend the terminal/live-state consistency
  test** (`CoreLoop` snapshots by the union of all live-state strings — no string may be terminal
  for one kind and live for another; prefer moving that check into registry construction).
- **Provenance direction:** the link is **target → source** — the `Commitment` carries
  `source_thought_ids` (+ qualified `provenance.source_object_ids`); the source Thought gets **no**
  new `crystallized_object_id` field (the registry rejects unknown payload keys).

**The `[SILENT]` cure is now one crystallization, and it is decoupled from delivery.** A thought
crystallizing into a contact `Desire` (`kind=desire`, `spring=THOUGHT` / `MIXED`) is still the
structural `[SILENT]` cure, and the domain + floor are already pre-wired for it
(`build_contact_desire`, the **DRIVE-only** `repeat_pure_longing` hold — `aggregation.py:346` —
and the unanswered-counter's THOUGHT/MIXED "materially-new reason" branch). But **producing an
object and turning it into an outreach are separate**: this slice enables neither the
desire-crystallization nor any send. The contact-side questions it raised — collision with a
live DRIVE desire (**merge → `MIXED`, lifting the hold** — the non-pestering "reach with
something to say"), *when* a source object is discharged, and the liveness invariant (under
clean silence the real pipeline must launch at least one contact judgment; a `THOUGHT`/`MIXED`
spring is the reachable path, else it falls back to the certified DRIVE wake *subject to*
`repeat_pure_longing`) — all move to the deferred **contact / arbiter** work. *(This supersedes
v2's §4.2, which had slice 3 mint a contact desire and deliver it through the existing pipeline.)*

**The "no send" guarantee, stated precisely** *(codex v3.1)*. "No send" is **causal**, not a
literal claim about the completion frame: `run_internal_completion` runs every registered
component and still dispatches an *incidental* `LaunchProactive` another component returns (the
birth-voice-threaded strand-fix), so a completion frame **can** send if an unrelated active
contact `Desire` already existed. The structural guarantee this slice makes is narrower:
**(1) crystallization emits no `LaunchProactive`**, and **(2) no current component consumes a
`Commitment` as a contact source** — aggregation reads only the contact `Desire`/`Intention`
(`aggregation.py:149`), `CognitionLauncher` launches only from the singleton active contact
`Desire` (`cognition.py:100`), and snapshot-per-frame means a freshly-inserted object is not read
in the same frame. **Acceptance test:** a completion frame with **no** pre-existing contact
desire produces **zero** `LaunchProactive`.

### 4.3 Internal state: a 3-axis vector (built with slice 4)

`H = (energy, curiosity, contact)`, each a bounded deviation on the §4 scale:

| Axis | Deviation grows with | Reduced by |
|---|---|---|
| **energy** | thinking / acting | resting (later, sleep) |
| **curiosity** | unprocessed thoughts nagging (Zeigarnik) | processing a thought |
| **contact** (`u`) | silence (existing integrator) | a real exchange |

**Curiosity is a derived projection, never a persisted scalar** (a second ledger would
drift and violate the single-store rule, HLA §4.1). It is computed by a **bounded query**
over the existing thought backlog in `memory_records`, with explicit semantics: **exclude
parked** thoughts until `parked_until` (they are in the live snapshot but must not nag),
and **deterministic truncation** (the per-state snapshot is capped at 256 — a naive sum
could omit the true top thought). Axis **weights** are fixed defaults here; per-being
temperament (FR7) is deferred.

### 4.4 The arbiter (slice 4 — evidence-gated; required contracts)

Each idle tick with budget, choose among `{rest, process-a-thought, reach-out}` by
**homeostatic soft selection** — score each action by how much it reduces the **total**
drive across `H` (Keramati–Gutkin), select softly (not argmax) with a small noise term.
Required contracts (v1 omitted these; without them the arbiter is a stochastic timer):

- **A no-action baseline + feasibility masks.** A softmax always chooses *something*; with
  empty backlog, full energy and `u≈0`, `reach-out` must **not** get ~⅓ probability. Score
  **expected drive-reduction minus action + opportunity cost**, with an explicit
  do-nothing option, so zero need → rest.
- **Contact eligibility below `θ`** must be defined; if eligible, the minimum evidence
  that prevents random low-`u` wakes.
- **Liveness + cost invariants (§4.2, §4.5) hold *through* the arbiter** — it selects
  above the floor and within the FR20 quota; it can never bypass either. *(v3.1 — codex: the
  **contact**-liveness half is conditional — it holds only once the deferred object→contact
  adapter (a `Commitment`/thought-origin `Desire` becoming an outreach) exists; until then the
  arbiter preserves only the invariants actually implemented — cost, and the fixed floor. Do
  not gate slice-4 acceptance on a contact path that is not yet built.)*
- **Narrowed claim:** the arbiter makes contact **context-sensitive while a backlog
  exists**; it does not abolish timer behaviour for a dormant relationship (that would
  need spontaneous thoughts — Phase 6). Acceptance tests assert on **conditional hazard
  distributions**, not merely "same `u` gave varied outcomes".

### 4.5 Cost governance (FR20) + S5

Rumination is real LLM spend, and energy does not bound it. **Required:** a hard **FR20
quota** — a token/call budget, configurable, **independent of energy** — shared by
rumination *and* proactive contact, with a conservative default. Define **min inter-
processing interval** and **max attempts per thought** (§4.1). **S5 is amended honestly:**
an idle tick with no budget-permitted work is **0-LLM**, and a fully dormant day (empty
backlog, no events) trends to **$0**; a day with a live backlog costs **≤ the FR20
ceiling**. *(This amendment is a product decision — see §10 hand-off.)*

**lm-705.2 implementation notes (what shipped vs. what deferred).** Slice 2 caps the
FR20 quota over **internal rumination only** — the genuinely-unbounded new spend. Proactive
contact is left bounded by its **own** drive/backstop dynamics (`repeat_pure_longing` HOLDs
after one unanswered outreach, §2), so total spend is bounded even though the two paths do
not yet share **one** counter; unifying them is **lm-705.7** (deferred: it touches the
load-bearing proactive path, so it waits on live traces). Cheap-model routing is **blocked
on the host** — `ctx.llm.acomplete_structured` hard-codes `task=None`, so the sanctioned
plugin lane cannot reach a registered aux-model slot; internal cognition therefore routes to
the **main** model and the FR20 **call** ceiling (not a token budget) is the cost bound,
with `model=`/`provider=` override tracked as **lm-705.10**. The min inter-processing
interval + **single-flight** gate pace the spend below the daily ceiling in practice.

## 5. Observability

Every arbiter/processing decision is a span with a **closed** `reason` enum
(`rested_low_energy`, `rested_nothing_salient`, `chose_think`, `chose_reach_out`,
`floor_held_*`, …); the thought id rides as a **field** (`thought_id=…`), never inside the
reason string. Positive choices are **not** suppressions (the existing `SuppressionReason`
enum in `core/suppression.py` stays for genuine holds; a new positive-decision reason set
is added). **Crystallization (v3.1 — codex):** the closed reason set gains
**`processed_crystallize`**, and *what the thought became* rides as **separate span fields**
— `thought_id`, `crystallized_kind`, `crystallized_id` (never a reason-per-object-id; the
qualified provenance carries the same source-thought link) — stamped once the builder + encode
succeed. **Encode ≠ durable commit** (codex): the component span persists **before** the
end-of-frame `StateActor` commit, so `processed_crystallize` records the crystallization
*decision*, not proof of durability — a rolled-back frame leaves the span but **no** row.
Durable success is the committed row + the tick's commit span; read them together (do not treat
`processed_crystallize` alone as "it landed"). The being answers *«почему промолчал / что тебя занимает»*
(FR24) by reading its own recent spans, not by confabulating.

## 6. Simulation (mechanism recovery, not human calibration)

The sim vivifies the real code through the existing fake-port harness
(`testing/harness.py`). For the **crystallization slices (1–3)** it asserts: **backlog health**
(seeded thoughts get processed; no starve, no spiral; no-progress decays; attempt/park bounds
terminate) · **crystallization** (v3 —
slice 3: a processed thought produces the right durable object via `PutRecord`, provenance-
linked, and the source thought discharges; the `[SILENT]`-cure **contact liveness** invariant
moves with the deferred contact/arbiter work) · **safety** (every fixed-floor invariant, incl.
`repeat_pure_longing`, holds throughout) · **cost** (idle default 0-LLM; a bounded backlog
stays within the FR20 quota). **Deferred to the contact/arbiter slices (4 + the object→contact
adapter):** **emergence** (contact hazard varies with inner context, not a clock) — *not* a
crystallization-slice measure. No "calibrated to humans" claims.

## 7. Boundaries (recap)

| Phase 5a (this) | → Phase 6 (lm-0od) | → rest of Phase 5 (lm-adz) |
|---|---|---|
| event-seeded thought create+process | spontaneous mind-wandering | full SDT vector + temperament weights |
| private (non-delivering) cognition path | deep multi-branch thought trees | trigger / commitment neurons |
| thought **crystallization** → durable object; `Commitment` type *(v3)* | sleep / consolidation | thought→**contact** + arbiter (the deferred `[SILENT]` delivery); opinions/predictions + Thought **residue** field |
| FR20 cost quota; forced observability | | open loops + receptivity |
| arbiter *(slice 4, evidence-gated)* | | learned set-point (lm-ocx) |

## 8. Open questions (genuinely undecided after the review)

- ~~Appraisal form: cheap classifier vs riding the dialogue turn's tail.~~ *(decided — §4.1:
  a cognition-grade judgment, not a keyword heuristic; unwired live → **lm-705.11**.)*
- Homeostatic score shape and default axis weights / normalization (commensurability).
- Arbiter selection form: softmax-over-drive-reduction vs basal-ganglia WTA — simplest
  that shows emergence.
- FR20 default budget and how rumination and proactive contact share it.
- Noise magnitude (tune in sim).

*(Seams, rumination bounds, collision semantics, curiosity-as-projection, closed enum, and
the liveness/cost invariants moved from "open" to **required contracts** above — the review
showed they are architecture, not plan-time detail.)*

## 9. Acceptance

- **Felt (owner's judgment):** the being no longer reads as a timer; it can tell you what
  it has been chewing on; silences feel like a life elsewhere.
- **Structural:** every silence has a logged reason; no fixed-floor invariant is weakened.
  *(v3.1 — codex: acceptance is split by slice.)* **Slice 3 (crystallization):** a processed
  thought produces the right typed durable object (atomically, provenance-linked, bounds
  terminate) and crystallization causes **zero** launch. **Deferred contact work:** proactive
  contact can originate from a *thought* (via the object→contact adapter) — the goal, not a
  slice-3 assertion.
- **Measured (sim + early live):** idle default 0-LLM and daily cost ≤ FR20 *(slice-3 measure)*.
  The §4.2 contact-liveness invariant and "contact hazard context-sensitive while a backlog
  exists" are **deferred** measures — they hold only once the contact path is built, not for the
  crystallization-only slices.

## 10. Review log

**codex `019f69d3` (2026-07-16), verified against source.** Reshaped v1 → v2:
- **Liveness (critical):** v1's "certified `u` ceiling wins unconditionally" was **false**
  against `repeat_pure_longing` (`aggregation.py:346`, HOLDs pure longing regardless of
  `u`). Replaced with a liveness invariant over real pipeline outcomes; the thought-origin
  spring is the real answer to prolonged silence (§4.2).
- **Cost/S5 (critical):** v1 processed thoughts on idle, violating S5, with energy as a
  non-existent cost bound (`personality.py:47`). Added a hard FR20 quota independent of
  energy and an honest S5 amendment (§4.5). **← product decision handed to the owner.**
- **Scope (critical):** v1 bundled ~11 subsystems and put D10 liveness behind an
  unvalidated arbiter. Re-decomposed into ordered, evidence-gated slices; the arbiter is
  last (§3).
- **Seams:** the appraisal seam and a non-delivering cognition path **do not exist**
  (`hooks.py:386`, `cognition.py:100`) — promoted from open question to required
  architecture (§4.1).
- **Rumination bounds, thought→desire collision, curiosity-as-projection, closed
  observability enum:** promoted from hopes to required contracts (§4.1, §4.2, §4.3, §5).
- **Retained (codex-affirmed sound):** thought records fit the typed BDI store;
  `PutRecord`/committer + end-of-frame atomicity are real seams; the real-code harness
  exists; thought-origin Desire is the strongest idea; event-seeded-only is an honest cut.

**lm-705.2 build (2026-07-16) — slice 2 shipped (§3.2/§4.1/§4.5).** Private thought
processing over the lm-705.6 seam: `ThoughtProcessingSelector` (heartbeat: re-arm expired
parks `parked→active`, emit one FR20/interval/single-flight-gated `LaunchInternalCognition`
for the top-salience active thought) + `ThoughtProcessingApply` (completion: typed outcome →
guarded transition). Outcome schema `{resolve|park|drop}`; bounds `no_progress_count`
(cap→drop) and `park_count` (6h/24h/72h backoff, cap→expire); a **transient** call failure
(empty raw) never penalizes the thought. As the **first live emitter**, it also landed the
birth-voice thread (prereq #1) and the single-flight gate (prereq #2). Deferred as beads:
shared FR20 (**lm-705.7**), launch↔completion trace-weave (**lm-705.8**), refund-on-transient
(**lm-705.9**), cheap-model routing — host-blocked (**lm-705.10**). No residue/opinion is
written (that stays lm-adz); slice 3 (lm-705.3) extends the outcome to **crystallization** (v3).

**v3 slice-3 redesign (owner, 2026-07-16) — "a thought can produce any type".** The owner
replaced v2's fixed slice-3 outcome ("a processed thought mints a *contact desire* and delivers
it through the existing pipeline") with a **generic crystallization** model: rumination decides
*what durable catalog object the thought becomes* — **any `kind`** — via the generic `PutRecord`
door, and the slice **stops at producing the object** (no contact, no send, no arbiter). The
first new type built is **`Commitment`** (HLA's follow-up/obligation, "serving the other" — the
initiation-reason, where Opinion/Prediction shape replies). Consequences threaded through the
doc: §3 build-order item 3, §4.1 outcomes, §4.2 (rewritten), §6 sim, §7 boundaries. The
`[SILENT]` cure is **not abandoned** — a thought→contact-`Desire` crystallization is now *one*
case, and its collision/discharge/liveness questions + the actual send move to the deferred
**contact / arbiter** work (the domain + `repeat_pure_longing` floor are already pre-wired for
it). Motivation: the being's inner life should crystallize thinking into durable BDI objects
generically (D8: "cognition mints Desire/Thought/Intention"), not be hard-wired to one outcome
coupled to delivery. Bead **lm-705.3** re-scoped to match.

**codex `019f6c40` (2026-07-16), verified against source — reshaped v3 → v3.1.** Verdict:
*ready-with-changes*; the atomic crystallization seam (`StateActor` batches thought-transition +
object-`PutOp` under one `BEGIN IMMEDIATE`) and `Commitment`-as-first-type are affirmed sound.
Fixes folded in:
- **Critical — stranded contact/liveness acceptance:** §2/§4.4/§9 still demanded contact-liveness
  that v3 defers. Acceptance is now **split by slice** — slice 3 = crystallization + zero-launch;
  contact-liveness is a **deferred** measure, conditional on the object→contact adapter (§2, §4.4,
  §9).
- **"Any kind" was too dynamic** → a **closed, discriminated** outcome (`crystallize_<kind>`,
  this slice only `crystallize_commitment`); `PutRecord` does not validate a kind — a trusted
  per-kind builder + `default_registry().encode()` does, so a free kind+payload could clobber a
  singleton (§4.1, §4.2).
- **Invalid crystallization must not evade the bounds** → a builder/registry failure is a
  no-progress outcome routed through `no_progress_count`, never an uncaught strand (§4.1).
- **`Commitment` model finalized:** non-singleton, deterministic id (source + fingerprint), full
  transition table (`+deferred`), typed trigger (`trigger_kind`/`trigger_value`/`due_at`),
  `basis`, explicit `Intention` boundary, base-envelope `salience`/`expires_at`, `_CATALOG`
  registration + terminal/live-consistency test, target→source provenance (§4.2).
- **"No send" stated precisely:** crystallization emits no `LaunchProactive` and no component
  consumes a `Commitment` as a contact source (a completion frame can still dispatch an
  *incidental* proactive launch from an unrelated live desire) (§4.2).
- **Observability:** `processed_crystallize` reason + `crystallized_kind`/`crystallized_id` span
  fields, stamped only after encode (§5).
- **Minor:** appraisal form decided (judgment, not heuristic; unwired → lm-705.11) (§4.1, §8).

**codex `019f6c40` re-review → v3.1.1 consistency pass.** The re-review confirmed I2/I3/I4/I5/M2
**closed** and flagged 4 residual slips (C1/I1/I6/M1 partial), all now fixed: the phase-goal
promise reframed to "durable initiation reasons, not the send" (§1); the last "any kind" wording
tightened to "a registered kind" (§3 item 3, §4.1); the appraisal form deferred to lm-705.11 as
the single authority (§4.1); `processed_crystallize` marked as the crystallization *decision*
(component span persists **before** the `StateActor` commit — encode ≠ durable) (§5); and
**emergence / contact-hazard** relabelled a deferred contact/arbiter measure, not a
crystallization-slice one (§6). **Verdict: ready to hand to the implementation plan.**

**lm-705.3 build (2026-07-17) — slice 3 shipped (§4.1/§4.2/§5, v3.1.1).** Thought crystallization:
a new `Commitment` BDI type (non-singleton, deterministic source-scoped id, typed
`basis`/`trigger`, full registry-guarded state machine — `domain/objects/commitment.py`) +
`commitment_view`; the processing outcome gained the closed `crystallize_commitment` variant;
`ThoughtProcessingApply` **strictly parses** the model's commitment fields
(`commitment_from_crystallize_fields` — no coercion; every bad-data mode → `InvalidPayload`),
builds the `Commitment`, and emits its `PutRecord` **plus** the source-thought
`TransitionRecord(active→resolved)` atomically; a validation failure is a bounded
`crystallize_malformed` no-progress; provenance is target→source (qualified). Reviewed by opus
(whole-branch) + codex `019f6cf8` ×2 (0 Critical; the coercion/overflow/surrogate bounded-failure
holes and the qualified-provenance link all fixed). Host-integration ran a real crystallization
against real Hermes. **Deferred as beads:** registry-transition enforcement at the persistence
boundary (**lm-1w2.1**, all kinds), and `Commitment` expiry/discharge + snapshot-eviction
(**lm-705.12**). **Not built:** commitment→contact (the deferred contact/arbiter work);
`Opinion`/`Prediction` (same mechanism, later); thought creation/appraiser (**lm-705.11**).

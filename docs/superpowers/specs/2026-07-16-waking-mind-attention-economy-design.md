# The waking mind — thinking in an attention economy (design)

**Phase:** Phase 5a — a **new intermediate phase** carved out of lm-adz (Фаза 5 —
Желания и взросление) and realizing epic **lm-egg** (inner economy of competing
drives) as a shippable slice. Roadmap chain:
`lm-4fv (Genesis, done) → [Phase 5a: waking mind] → lm-adz (rest of Phase 5) → lm-0od (Phase 6)`.
**Date:** 2026-07-16
**Status:** design under review — **v2, revised after codex review `019f69d3`** (§10)
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
thoughts from what happens, *processes* them later, and a processed thought can become a
warm, specific **reason to reach out** — as opposed to the bare "I'm lonely". After this
phase the being is richer to talk to in *every* conversation, and its proactive contact
has a content-bearing source, not only a threshold.

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
  deprivation is the **thought-origin** spring (not a bigger `u`).
- **Cost is bounded by a hard FR20 ceiling, independent of energy** (§4.5). Energy is
  physiology, not the billing boundary (`core/personality.py:47` refills every tick,
  cheaply). A day with nothing to process still trends to **$0** (S5 preserved for the
  genuinely-idle case); a day with a live backlog spends **at most** the FR20 ceiling.
- **Event-seeded thoughts only.** Spontaneous mind-wandering (thoughts from nothing) is
  **deferred to Phase 6**.
- **Observability is forced** (D10), with a **closed** reason enum: the thought id is a
  span *field*, never embedded in the reason string; positive choices (`rest`/`think`/
  `reach`) are distinct from suppressions (§5).
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
3. **Thought-origin contact desire** — a processed thought mints a contact desire through
   the **existing** safety/cognition pipeline (§4.2). This is the structural `[SILENT]`
   cure and it ships **without** the arbiter.
4. **The arbiter** *(only after live traces from 1–3 show a reliably populated, healthy
   backlog)* — the 3-axis homeostatic selection, with the liveness + cost + feasibility
   contracts of §4.4. May be split off into its own bead if slices 1–3 teach us it should.

**Deferred → Phase 6 (lm-0od):** spontaneous mind-wandering, deep/multi-branch thought
trees, sleep/consolidation.
**Deferred → rest of Phase 5 (lm-adz):** full SDT vector + temperament weights (FR7),
trigger/commitment neurons, **opinions/predictions as first-class** (see §4.1 — the
"resolve → opinion" residue is *not* built here; Thought has no residue field yet), open
loops + receptivity, learned set-point (lm-ocx).
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
and a core component emits the `PutRecord`. Design decision (before plan): whether the
appraisal is a cheap classifier or rides the dialogue turn's own tail.

**Processing (expensive) needs a new non-delivering cognition path.**
`CognitionLauncher` only launches a *delivered* proactive turn and reads back `SENT` /
`[SILENT]` (`core/cognition.py:100–214`) — unsafe for private rumination. **Required:** a
distinct internal-cognition protocol:

- a **non-delivering** launch intent/port (delivery suppressed by construction);
- correlation + pending-idempotency **distinct** from outbound contact;
- an async completion frame carrying a **typed thought outcome** (deterministic schema +
  validation);
- atomic application of the thought transition **and** any minted desire in one commit.

**Bounded lifecycle (required — snapshot-per-tick is not enough).** Processing develops
**one** selected thought by one layer (top-K, **K=1**, by salience). The Thought schema
already carries `no_progress_count` / `park_count` / `parked_until` (`domain/objects/
thought.py`); this phase **defines their rules**:

- **max total processing attempts** per thought → terminal `drop`;
- **durable increment** of `no_progress_count` on a failed/malformed/no-progress outcome
  (else an async failure is an unbounded retry loop);
- **park backoff** and a **max park cycles** bound;
- terminal behaviour after repeated malformed LLM output.

Outcomes (a `TransitionRecord`): **mint a contact-desire** (§4.2) · **park** · **drop** ·
**resolve** (plain — *no* opinion/residue is written here; the residue field and the
"opinion" outcome belong to the deferred becoming work, lm-adz). Processing discharges the
nag; an unprocessed thought decays slowly.

### 4.2 The top-down contact desire — the structural `[SILENT]` cure (ships without the arbiter)

A processed thought can mint a **thought-origin** contact desire ("I thought about X and
want to share it"). The domain already supports `DRIVE` / `THOUGHT` / `MIXED` springs
(`domain/objects/desire.py`). This is the heart of non-intrusive contact (HLA §4.1) and it
goes through the **existing** pipeline, so it is deliverable in slices 1–3 with no arbiter.

**Collision semantics (required — the contact desire is a singleton).** A thought may want
to mint contact while a DRIVE-origin desire is already live. Decide before plan:

- **merge** the thought reason into the live desire, converting it to `MIXED` (leaning
  this — preserves provenance and lifecycle);
- vs queue vs replace.

**Interaction with `repeat_pure_longing` (the liveness answer).** The floor HOLDs
*pure-longing* (DRIVE) bids after one unanswered outreach, regardless of `u`. A
`THOUGHT`/`MIXED` spring is **not** pure longing, so it is the legitimate way a
long-silent being reaches again — *with something to say*, which is exactly the
non-pestering form we want. **Decide:** does a thought reason merging into a held
DRIVE desire lift the hold (becoming `MIXED`)? And **when** is the source thought
discharged — at proposal, desire creation, launch, or actual send? (Leaning: discharge on
**delivered send**, so a lost/`[SILENT]` outcome does not silently consume the thought.)

**Liveness invariant (stated on real outcomes).** Under clean silence — no in-flight turn,
no active backoff, FR20 budget available — the real pipeline must launch **at least one**
contact judgment by a deadline. With the floor as-is, that judgment must be **reachable
via a thought-origin spring**; if the backlog is empty, liveness falls back to the
existing certified DRIVE wake **subject to** `repeat_pure_longing` (i.e. after the *first*
unanswered bid, further pure-longing HOLDs by design — the being waits for a real event or
the human, which is the intended non-pestering behaviour, **not** a regression). This is
the honest statement; v1's "u ceiling always fires" was not true.

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
  above the floor and within the FR20 quota; it can never bypass either.
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

## 5. Observability

Every arbiter/processing decision is a span with a **closed** `reason` enum
(`rested_low_energy`, `rested_nothing_salient`, `chose_think`, `chose_reach_out`,
`floor_held_*`, …); the thought id rides as a **field** (`thought_id=…`), never inside the
reason string. Positive choices are **not** suppressions (the existing `SuppressionReason`
enum in `core/suppression.py` stays for genuine holds; a new positive-decision reason set
is added). The being answers *«почему промолчал / что тебя занимает»* (FR24) by reading
its own recent spans, not by confabulating.

## 6. Simulation (mechanism recovery, not human calibration)

The sim vivifies the real code through the existing fake-port harness
(`testing/harness.py`) and asserts: **emergence** (contact hazard varies with inner
context, not a clock) · **backlog health** (seeded thoughts get processed; no starve, no
spiral; no-progress decays; attempt/park bounds terminate) · **liveness** (the §4.2
invariant holds; the thought-origin spring reaches under prolonged silence) · **safety**
(every fixed-floor invariant, incl. `repeat_pure_longing`, holds throughout) · **cost**
(idle default 0-LLM; a bounded backlog stays within the FR20 quota). No "calibrated to
humans" claims.

## 7. Boundaries (recap)

| Phase 5a (this) | → Phase 6 (lm-0od) | → rest of Phase 5 (lm-adz) |
|---|---|---|
| event-seeded thought create+process | spontaneous mind-wandering | full SDT vector + temperament weights |
| private (non-delivering) cognition path | deep multi-branch thought trees | trigger / commitment neurons |
| thought-origin contact desire (`[SILENT]` cure) | sleep / consolidation | opinions/predictions + Thought **residue** field |
| FR20 cost quota; forced observability | | open loops + receptivity |
| arbiter *(slice 4, evidence-gated)* | | learned set-point (lm-ocx) |

## 8. Open questions (genuinely undecided after the review)

- Appraisal form: cheap classifier vs riding the dialogue turn's tail.
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
- **Structural:** proactive contact can originate from a *thought*; every silence has a
  logged reason; no fixed-floor invariant is weakened.
- **Measured (sim + early live):** the §4.2 liveness invariant holds; idle default 0-LLM
  and daily cost ≤ FR20; contact hazard is context-sensitive while a backlog exists.

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

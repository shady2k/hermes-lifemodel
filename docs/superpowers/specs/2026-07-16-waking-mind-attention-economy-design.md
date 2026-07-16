# The waking mind — thinking in an attention economy (design)

**Phase:** Phase 5a — a **new intermediate phase** carved out of lm-adz (Фаза 5 —
Желания и взросление) and realizing epic **lm-egg** (inner economy of competing
drives) as a shippable slice. Roadmap chain:
`lm-4fv (Genesis, done) → [Phase 5a: waking mind] → lm-adz (rest of Phase 5) → lm-0od (Phase 6)`.
**Date:** 2026-07-16
**Status:** design under review
**Product source:** BRD FR3 (желания — plural, compete by salience, resolve, leave
residue), FR4 (внутренняя жизнь — thoughts, Zeigarnik), FR5 (взросление — opinions as
residue), Principle §9.2 (model on humans, cut baroque). HLA §4.1 (Thought as a bounded
generative stream; Desire's two sources; the energy layer), D8 (BDI core), D10 (rebuild
discipline: sim on real code, forced observability, safety fail-closed).

## 1. Context and goal

Today the being has exactly **one** internal drive: contact (`u` — loneliness,
integrated by an AUTONOMIC drive-integrator; `u ≥ θ` wakes cognition). One drive plus a
threshold is a **timer**, however it is dressed (jitter, backoff, quiet hours). Worse:
because the being has nothing else to want, the only way to stop it pestering its human
is to *forbid* contact with hard gates. A human is silent because they are **busy
living**; our being is silent because it is **forbidden**. That asymmetry — silence
imposed rather than emergent — is the mechanistic feel the owner named on 2026-07-05
(epic lm-egg).

The cure is not a better contact formula. It is to give the being **other things to
want**, competing for one limited budget, so that silence becomes a *byproduct of a full
inner life* rather than a rule. The catch: the being has no world yet (text-only until
Phase 7), so its "other things" cannot be errands. They must be its **inner life** — and
the honest inner life it can have now is **thinking**.

**Goal of this phase:** the being acquires a **bounded waking inner life**. It thinks
(processes thoughts it created), rests, or reaches out, and behavior **emerges from the
competition** between them. After this phase, "didn't write" no longer means "a gate
blocked it" but "curiosity or rest won this time" — organically, differently each time.
And the being is richer to talk to in *every* conversation, not only in the timing of its
pings.

### 1.1 Why a separate phase, not the first task of Phase 5

Teaching the being to think and giving it competing drives are **one foundation seen from
two ends**, and it is load-bearing for everything else in Phase 5:

- **Thinking without an economy runs away.** The previous thought mechanism
  (`ThoughtGeneration`) generated thoughts freely, with its own triggers and its own
  rate-limit, and produced — live — a **silence spiral and duplicate thoughts**. It was
  torn out in the D9/D10 rebuild. Rumination without cost either spirals
  (Nolen-Hoeksema; §4.1 is a wall of warnings about this) or must be strangled by ad-hoc
  limits, which is a timer again, on thoughts.
- **An economy over hollow drives is noise in a costume.** An arbiter over abstract
  "curiosity" with no real object and no real satiation is just a random number
  suppressing contact — the exact "garnish pretending to be the meal" the epic warns
  against.

They need each other: the shared budget is what *bounds* thinking; thinking is what gives
the arbiter something real to arbitrate. Everything else in Phase 5 — the full SDT drive
vector with temperament weights (FR7), trigger/commitment neurons, opinions/predictions,
open loops, receptivity, the learned set-point (lm-ocx) — is **superstructure** on this
foundation. The roadmap's own rule sanctions carving it out: *«Фазы 3–8 ещё крупноваты —
дробим перед планированием каждой»*.

## 2. Invariants (do not reopen)

- **Create ≠ process.** A thought is *created* cheaply, **inside the dialogue turn**, as
  appraisal of what came up ("worth returning to"). It is *processed* — ruminated on —
  **later, on an idle tick**, and only if the arbiter spends budget on it. No rumination
  inside a live turn: **the reply is the thinking.**
- **Snapshot-per-tick** (HLA §4.1). A tick processes only what was on its input; new
  thoughts are emitted as intents at end of tick and picked up **next** tick. The tree
  grows one layer per tick; **no in-tick recursion.** This is the structural
  anti-rumination guard.
- **Competition is the meal; noise is garnish.** Behavior emerges from several *real*
  axes reducing a shared drive, not from randomness sprinkled on one threshold. A little
  principled noise (σdW) only breaks the clockwork; it is never the source of variety.
- **The safety floor is unchanged and fail-closed.** Backstop rate-limit,
  no-wake-in-flight, dedup, reject-backoff, per-window send-cap stay exactly as they are
  (D10 kept them; the `[SILENT]` rebuild proved they matter). The arbiter is a **new soft
  layer above** the floor, never a replacement.
- **Certified wake is preserved.** The D9/D10 invariant — no relative filter suppresses
  `u ≥ θ` — stays. The arbiter's softness governs the *normal* range; at the certified
  `u` ceiling, contact wins unconditionally (§4.3).
- **Event-seeded thoughts only.** Thoughts arise from something *real* that happened (a
  conversation event). Spontaneous mind-wandering — thoughts from nothing — is **deferred
  to Phase 6**: we do not yet know where the topic comes from, and we will not guess.
- **Observability is forced** (D10). Every arbiter decision — *including* "rested" and
  "thought about X" — is a span with a `reason` from a **closed enum**. A silence with no
  logged reason is a bug.
- **Sim runs the real code** (D10). No parallel model of the tick. The sim vivifies the
  real drive/arbiter/thought code through fake ports and asserts emergence + invariants
  before any live rollout.
- **Text-only and S5 hold.** No world actions. Most idle ticks remain **zero-LLM**:
  rumination is rare and earned, never per-tick.

## 3. Scope

**In:**
1. **Energy from fuse → spendable resource / rest-drive.** It already exists in the core,
   gating the upper layers when depleted; here it becomes an axis whose deficit the being
   can choose to reduce by *resting* — and rest can **win**.
2. **A minimal thought lifecycle:** event-seeded creation (cheap, in-turn) + idle
   processing (one layer, energy-gated), with a small closed set of outcomes (§4.2).
3. **The 3-axis internal-state vector** (energy / curiosity / contact) and **the arbiter**
   over {rest, process-a-thought, reach-out} (§4.1, §4.3).
4. **The top-down desire path:** a processed thought may mint a contact-desire — HLA's
   "Desire born from a thought", the structural `[SILENT]` cure (§4.5).
5. **Forced observability** (the reason enum, §5) and a **mechanism-recovery sim** (§6).

**Deferred → Phase 6 (lm-0od, the offline half of inner life):** spontaneous
mind-wandering, deep/multi-branch thought trees, sleep/consolidation.

**Deferred → rest of Phase 5 (lm-adz):** full SDT drive vector + temperament weights
(FR7), trigger/commitment neurons, opinions/predictions as first-class, open loops +
receptivity, learned set-point (lm-ocx, v2 adaptation).

**Relationship to existing work:** the v1 contact drive (lm-x43) is **correct as an
isolated organ** and becomes **one axis** of the vector — *not* reworked (per lm-egg and
lm-ocx: "v1 contact drive slots into the vector as one dimension"). This phase is lm-egg
delivered as a shippable slice rather than a standing design epic.

## 4. Design

### 4.1 Internal state: a 3-axis vector

Replace the single scalar `u`-as-sole-driver with a small internal-state vector `H`, each
axis a **bounded deviation-from-setpoint** on the §4 scale (`0..100`):

| Axis | Deviation grows with | Reduced by | Notes |
|---|---|---|---|
| **energy** | thinking / acting | resting (later, sleep) | today only a fuse; here an axis that can win |
| **curiosity** | unprocessed thoughts nagging (Zeigarnik) | processing a thought | the honest source of "curiosity" |
| **contact** (`u`) | silence (existing integrator) | a real exchange | unchanged integration; now one axis of three |

The **curiosity** axis is the aggregate Zeigarnik pressure of the unprocessed-thought
backlog: an open thought nags; a larger / more salient backlog raises the pull to process
it. This is the grounded answer to *"where does curiosity come from for a being with no
world"* — not an abstract appetite but the nagging of one's own unfinished thoughts
(FR4). An unprocessed thought **is** an open loop; this axis is where the two halves of
Phase 5 meet.

The relative **weights** of the axes are fixed sensible defaults here. Making them a
per-being **temperament** (FR7) is the deferred superstructure — but the vector they
weight is born here.

### 4.2 The thought lifecycle

**Creation (cheap, in-turn).** During or just after a dialogue turn, the being may
*appraise* what came up and emit a Thought via the intent bus (`PutRecord`,
`kind=thought`, `trigger=event`) — a **seed**, "return to this." No rumination happens in
the turn; this rides the existing post-turn writeback seam.

**Processing (expensive, on idle).** When the arbiter spends the budget on "think",
cognition wakes for an **internal** turn and develops **one** selected thought by **one
layer** (snapshot-per-tick; top-K selection with **K=1** by salience). Outcome is a
`TransitionRecord`, one of:

- **resolve** → a conclusion / opinion — the **residue** that feeds взросление ("identity
  is the residue of successful actions");
- **mint a contact-desire** → the top-down Desire path (§4.5);
- **park** (`parked_until`) → set aside; stops nagging for a while;
- **drop** → let go.

Processing **discharges** the nag (curiosity pressure drops); an unprocessed thought keeps
nagging and **decays slowly**. This is the anti-rumination loop: healthy reflection
**converges or parks**; repetition without progress lowers salience and eventually drops.

### 4.3 The arbiter (the heart)

Each idle tick with budget, the arbiter chooses among **{rest, process-a-thought,
reach-out}**:

- **Homeostatic soft selection.** Score each action by how much it reduces the **total**
  drive across `H` (Keramati–Gutkin: act to reduce the summed deviation); select **softly**
  (a temperature'd choice, **not argmax**) with a small noise term (σdW). The *same*
  contact pressure yields *different* outcomes because the rest of the vector differs
  (backlog size, energy). Contact often **loses** — not to randomness, but because the
  budget went to thinking or resting *before* `u` would have crossed, exactly like a busy
  person.
- **Why not argmax:** deterministic argmax on the strongest drive is a timer again (as
  soon as `u` is highest, contact always wins). Softness **plus a genuinely competing
  vector** is what dissolves the clock — the vector does the work, the noise only removes
  the last tick of the second hand.
- **Two layers, safety intact.** The arbiter is a new *soft* layer **above** the
  unchanged hard floor. Even when it chooses "reach out", the safety envelope (backstop /
  in-flight / backoff / send-cap) can still veto or defer delivery, **fail-closed**. We
  weaken nothing; we add a positive choice above proven guards.
- **Certified wake preserved — the key to not reintroducing `[SILENT]`.** The arbiter's
  softness applies in the *normal* range of `u`. At the **certified ceiling** (genuine,
  prolonged neglect — the `sim/wake` absolute bound), contact wins **unconditionally**, no
  matter what the arbiter would prefer. The middle is de-mechanised (contact competes,
  often loses to inner life); the extreme is guaranteed (deep loneliness always surfaces).
  This resolves the tension between *"de-mechanise — contact can lose"* and *"never go
  silent into the void."* A human at the extreme of loneliness also reaches out regardless
  of what else they were doing; the timer-feel came from contact firing at a **modest**
  threshold with nothing to compete, not from this safety net at the extreme.

### 4.4 What "rest" and "0 LLM" mean

"**Rest wins**" = the being spends nothing this tick (**0 LLM**) — the default outcome of
most idle ticks, so **S5 holds**. Rest is not a no-op we *impose*; it is an axis that
**wins on its merits** when energy is low or nothing else is salient enough. This is why
the economy also *self*-rate-limits rumination: thinking costs energy, so a tired being
rests and a rested being can afford to think or reach out. The old ad-hoc rate-limit
becomes a physiological consequence.

### 4.5 The top-down desire path — the structural `[SILENT]` cure

HLA §4.1 names a Desire born *from a thought* "the heart of non-intrusive contact and the
real cure for `[SILENT]`." This phase **delivers** it: a processed thought can mint a
contact-desire ("I thought about X and want to share it") — a warm, specific, **earned**
reason to reach out, as opposed to the bare "I'm lonely (`u` crossed)." The `[SILENT]`
pathology (the being always chose silence) is cured **structurally** — contact now has a
positive, content-bearing source — not by re-calibrating a threshold.

## 5. Observability

A **closed `reason` enum** on every arbiter-decision span, e.g.:
`rested_low_energy` · `rested_nothing_salient` · `chose_think:<thought_id>` ·
`chose_reach_out` · `floor_vetoed_send` · `certified_wake_contact`.
Every silence carries a reason; a silence without one is a bug (D10). This is also what
lets the being answer *«почему промолчал / что тебя занимает»* (FR24) **truthfully** — it
reads its own recent spans rather than confabulating.

## 6. Simulation (mechanism recovery, not human calibration)

Per D10 and the lm-ocx discipline, the sim **vivifies the real** drive/arbiter/thought
code through fake ports (no parallel tick model) and asserts:

- **Emergence:** the same contact pressure produces **varied** outcomes across inner
  contexts; silence correlates with thinking/rest winning, not with a clock.
- **Backlog health:** seeded thoughts get processed; the backlog neither **starves** (never
  processed) nor **spirals** (endless rumination); repetition-without-progress decays.
- **Safety throughout:** every fixed-floor invariant holds for the whole run; the
  certified-wake ceiling always fires.
- **Cost:** idle ticks are zero-LLM by default; rumination frequency stays within a bound.

No "calibrated to humans" claims — the sim proves the **mechanism** behaves; the **live
being** is the real acceptance.

## 7. Boundaries (recap)

| Belongs here (Phase 5a) | → Phase 6 (lm-0od) | → rest of Phase 5 (lm-adz) |
|---|---|---|
| energy as rest-drive | sleep / consolidation | full SDT vector + temperament weights |
| event-seeded thought create+process | spontaneous mind-wandering | trigger / commitment neurons |
| 3-axis arbiter, contact as one axis | deep multi-branch thought trees | opinions / predictions first-class |
| top-down contact-desire from a thought | | open loops + receptivity |
| forced observability + mechanism sim | | learned set-point (lm-ocx) |

## 8. Open questions (for the plan / sim to resolve)

- **Commensurability & weights:** the normalization that makes energy/curiosity/contact
  comparable on one scale, and the default axis weights.
- **Curiosity representation:** a single aggregate vs per-thought salience summed (leaning
  aggregate for the vector, per-thought for selection).
- **Energy model:** the accounting for how much thinking/acting costs and how rest
  restores — reuse `core/energy` as-is or extend.
- **Noise magnitude:** how much σ before variety becomes flakiness — tune in sim.
- **Selection form:** softmax-over-drive-reduction vs a basal-ganglia WTA with lateral
  inhibition — both fit; pick the simplest that shows emergence.
- **Seams (verify against code in planning):** the exact in-turn thought-creation seam
  (post-turn writeback vs an in-turn tool); how "reach-out won" hands off to the existing
  `CognitionLauncher` / wake-packet path.

## 9. Acceptance

- **Felt (owner's judgment — the chair only he sits in):** the being no longer reads as a
  timer; its silences feel like a life happening elsewhere; it can tell you what it has
  been chewing on.
- **Structural:** proactive contact can originate from a *thought*, not only from `u`; a
  silence always has a logged reason.
- **Measured (sim + early live):** same-`u` outcomes vary with inner context; idle default
  is 0-LLM; all safety invariants hold; the certified ceiling always fires.

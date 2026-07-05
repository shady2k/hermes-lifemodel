# Proactive-contact desire model — design spec

**Bead:** [lm-x43](../../) — Mathematical model of proactive-contact desire (dynamics + simulation + calibration)
**Status:** draft — **rev.3 (2026-07-05)**. rev.1 folded Codex instrument-honesty fixes (thread `019f2eec`); rev.2 folded the desire-lifecycle + bounded-scale design session; **rev.3 reframes the whole model as a homeostatic controller with a (v2) learned set-point, grounded in four convergent literatures, and answers the one-vs-two-variable question with that grounding** (web literature sweep + Codex adversarial review, thread `019f316e`). Consolidation bead: [lm-27h].
**Scope:** the DESIRE model + its simulation only. No Hermes plugin code in this task. Product "what" lives in `business-requirements.md` (FR2/FR3/FR6); architecture "how" in `hla.md` (§2.1, §11); this spec formalises and *simulates* them.

**rev.3 headline.** We could never *derive* a constant answer to "after a conversation, how many hours until it proactively writes again?" — because there is **no universal constant**. The right rhythm is individual and must be *learned per relationship*. This is not our failure; it is consistent with established findings (Niv et al.: optimal action latency is context-dependent, not fixed; Matthews & Tye: the social set-point is individual and plastic). Consequently the model splits cleanly:

- **v1 (this task — mechanics built & certified; one calibration step remains):** the *fast drive* + the *desire lifecycle* + the *hard safety envelope* — a well-mannered being with a **fixed, conservative set-point** (a documented prior, explicitly **not** a calibrated truth). It never drums, never interrupts. The 54-test mechanics are green; the last v1 step is the shared-prior feasibility check (§9).
- **v2 (separate epic, [lm-27h]/new):** *adaptation* — a bounded **learned set-point** and a separate **rhythm/availability estimator** that make the being learn *this* person, slowly, inside hard clamps. Designed here (§10) so it is not lost; **not implemented here**.

---

## 1. Goal

A minimal, science-grounded, **simulatable** model of how the *desire to make proactive contact* arises and is resolved, validated by simulation **before** any plugin code.

Framed precisely (rev.3): the being is a **homeostatic controller** for social connection (Matthews & Tye 2019). A cheap drive senses the deficit between perceived connection and a *set-point* and produces an urge; at threshold it *wakes* cognition; cognition (the LLM) *judges*. In **v1** the set-point is a fixed conservative prior. In **v2** it is *learned* — which is the honest resolution of the one-vs-two-variable question (§8).

The live system today (HEAD `0881ebc`) ships the proactive-egress **delivery** layer; its **decision** layer misbehaved (real Telegram log 2026-07-04: proactive message one minute after the user greeted, then a "nothing to add" drum every 30 min). This task rebuilds the *when to reach out / stay silent / hold* decision as a model provable in simulation. It does not touch delivery.

## 2. Non-negotiable principles

1. **URGE ≠ ACTION.** A cheap, zero-LLM drive produces pressure; at threshold the aggregation layer *creates a desire* and wakes cognition. An expensive, context-aware LLM then *judges* — **fulfill / defer / reject**. Threshold-crossing **never** sends a message.
2. **Logic lives in a layer, not in a neuron** (project convention). The neuron is a thin sensor: it *measures* (bounded pressure + how long over threshold) and *emits*; it does not *decide*.
3. **Hard structural gates are separate from the drive.** "Don't interrupt", "don't wake in-flight", "don't re-wake right after a reject" are *policy*, not *dynamics*.
4. **A desire has a lifecycle and is remembered.** Repeated crossings do not spawn duplicate wakes (dedup / `ack`); a *deferred* desire is held and re-presented, never forgotten, never drummed.
5. **The safety envelope is fixed; only comfort is learned** (rev.3). Whatever v2 learns, a hard-coded envelope never moves: never-interrupt (`W`), no-wake-in-flight, dedup, growing reject-backoff, internal-impulse exclusion, a per-window send cap, and "threshold wakes cognition, never sends." Learning adjusts *latency/eagerness inside* the envelope — never *permission* to violate it. This is the "something basic" underneath adaptation.
6. **Calibrate honestly, do not guess, do not overclaim.** v1 ships a *conservative prior* to be verified for feasibility across every scenario (not truth) — the mechanics are certified, the shared-prior check is the last v1 step (§9). Fitting the set-point to a real person is v2, and even there the simulation proves the *mechanism*, never "calibrated to humans" (§10).
7. **Minimal set first (YAGNI).** v1 is one continuous drive with a fixed set-point. The second (learned) variable is added in v2 *because the science and the "no universal constant" result point to it* — argued from the literature, not assumed, and not certified by v1 (which ships one variable).

## 3. Science grounding

Four convergent literatures describe systems of this shape and map closely onto ours. **All are inspiration and constraint, not validation** (Codex `019f2f26`/`019f316e`): they establish *that* a mechanism is biologically real and give us its *shape/equations*; they do not certify our constants, nor do they prove that *our* second variable must be exactly `s`. Constants are a prior (v1) or learned (v2), never taken from a citation.

- **Social homeostasis — Matthews & Tye 2019; individual differences — Frontiers 2023.** The *structure*: a **detector → control-center (holding a set-point) → effector**; loneliness is the *error signal* between perceived contact and the set-point; the effector adjusts effort to seek contact. This is our neuron / aggregation / cognition stack almost 1:1. Crucially the set-point is **individual and plastic** — "calibrated via complex and continuous interactions between genetics and life experiences," and "extreme acute or chronic exposures could result in a new set point." → grounds *both* the v1 controller *and* the v2 learned set-point.
- **Homeostatic Reinforcement Learning — Keramati & Gutkin 2011/2014.** The *math* of drive + reward. Drive is distance from set-point, `D(H)=Σ|hᵢ*−hᵢ|^m`; **reward = drive reduction**, `r = D(Hₜ) − D(Hₜ₊₁)`; with discounting this favours *earlier* drive reduction — a formal basis for latency pressure (the temporal "when"). Their base set-point is **fixed**; making it *adaptive/learned* is their explicitly-named future extension (it explains tolerance/addiction). → grounds our urge semantics (v1) and names our v2 second variable.
- **Allostasis / active inference — (interoception-as-modeling 2022; social-allostasis 2025).** Set-points are **adjusted predictively, ahead of need** — the organism learns a model and pre-tunes regulation. Formally, shifting a *prior preference* = shifting the set-point. → motivates predictive pre-tuning; the v2 *rhythm* hazard estimator (§10) is our engineering formalism for it, not something this literature prescribes.
- **Average-reward-rate / tonic dopamine — Niv, Daw & Dayan 2007.** The optimal **latency to act** is set by the long-run average reward rate as an *opportunity-cost-of-time*. → the formal statement that "when to reach out" is **context-dependent, not a constant** — the reason a fixed `T_urge` cannot exist.

Retained as **analogy only**, with caveats (from rev.2): Hull drive-reduction (borrow the shape, not the behaviourist ontology); Baumeister & Leary "need to belong" (quality not frequency → the `q_event` classifier); Cacioppo/Hawkley loneliness-as-aversive-signal and the chronic-withdrawal twist (grounds the *optional* v2 load term `a(t)`, §10); allostatic load (sustained wanting has a cost). Opponent-process and habituation are engineering analogies, never proof that "unmet desire decays on its own." Ego-depletion is avoided (replication crisis). The Abplanalp double-well is **not** adopted — a leaky integrator (or, per Codex, a hazard model) is the right first tool, not a bistable-with-hysteresis model that needs dense longitudinal data.

## 4. Architecture (where things live)

```
                  ┌──────────────────────────────────────────────────────────┐
 inbound exchange │  NEURON (autonomic, 0 LLM) — a thin sensor (DETECTOR)    │
 (user message ──►│   u = deviation of perceived connection from set-point s  │
  via plugin hook)│   u ∈ [0,100]: rises in silence, satiates on real contact │
                  │   emits {intensity u, duration_over_θ}; measures, no decide│
                  │   v1: set-point s = fixed conservative prior (on disk).    │
                  └──────────────────────────────────────────────────────────┘
                                          │ signal
                                          ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  AGGREGATION (0 LLM) — the desire lifecycle (CONTROL CTR) │
                  │   first crossing + structural gates pass → create ONE     │
                  │   contact-desire → WAKE. Repeat crossings deduped (ack).  │
                  │   deferred desire: held; re-woken on release condition.   │
                  └──────────────────────────────────────────────────────────┘
                                          │ WAKE (desire)
                                          ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  COGNITION (LLM) — the verdict (EFFECTOR gate)            │
                  │   FULFILL (send) / DEFER (hold) / REJECT (nothing to say) │
                  └──────────────────────────────────────────────────────────┘

   v2 (separate epic, dashed — NOT built here):
     • learned set-point s   ← bounded slow adaptation, inside hard clamps
     • rhythm/availability estimator h(t) ← a hazard model, feeds release/defer
     • optional withdrawal/load a(t)      ← Cacioppo suppressive term, later
```

The neuron and aggregation layer are **ours** (cheap, zero-LLM, simulatable without Hermes). Cognition is the LLM, **out of scope** — the harness *scripts* its verdict (§9). The model guarantees the *machinery* (when it wakes, that a held desire is remembered, that it never drums); verdict *quality* is the soul/prompt concern (bead `lm-pbm`).

## 5. The v1 model (one continuous drive, fixed conservative set-point)

**Neuron state** (per relationship lane):
- `u ∈ [0, 100]` — the contact-pressure intensity, read as `u = D(deficit; s)`, the deviation of perceived connection from the set-point `s`. A drive/deficit neuron → unipolar `0..100` (valenced neurons like mood are `−100..100`, out of scope). In v1 `s` is a **fixed conservative prior**; the dynamics below evolve `u`.
- `duration_over_θ` — how long `u` has been ≥ `θ`, a separate field the neuron owns (a pinned-near-100 `u` no longer carries "how long"). Feeds deferred-desire escalation (§7).

**Dynamics** (discrete time, tick `Δt`; "genuine silence" = no exchange event in the lane this tick):
- **Rise in genuine silence:** `u ← u + Δt·α` (linear-with-clip), or saturating `u ← u + Δt·α·(1 − u/100)` — kept as a swept knob (§11).
- **Satiation on a positive exchange:** `u ← max(0, u − β·q_event)`, only for `q_event > 0` (§6).
- **Reset on real contact:** when cognition **fulfills** (a message is sent), that is contact → `u` satiates and `duration_over_θ` resets. **Deferring or rejecting does NOT reset `u`** — no contact happened, the deficit honestly persists.

**Threshold + desire creation (aggregation):** when `u` first crosses `θ` and the structural gates (§7) pass, aggregation **creates one desire and wakes cognition**. No per-tick "drain": the anti-drum guarantee is the desire **dedup** (`ack`) + reject-backoff, not zeroing `u`.

**Desire lifecycle (aggregation owns it):** create → wake once; dedup/`ack` while live; fulfill → satiate + reset + resolve; defer → *held*, re-woken on a release condition (§7); reject → cleared + growing backoff.

**v1 constants (a documented prior, not a truth):** `α` (rise rate), `θ` (threshold), `β` (satiation), plus the §7 policy constants. All are **disk config, hot-reloadable** (BRD NFR5). v1 calibration = confirm *one shared conservative set* satisfies every invariant on every scenario (§9); it is a safe starting envelope that v2 learning adapts, **never claimed as the right rhythm for any real person**.

> **Scale & sim note.** The product/plugin scale is `u ∈ [0,100]` (this section's narrative). The *certified simulation* (`src/lifemodel/sim/`) uses a **normalized** scale (`θ = 1`, `β = 1`, `u_max = ∞`) — the mechanics are scale-free, so the normalized harness and the 0..100 plugin are the same model at different units (a v1 open question, §12, is confirming the ceiling never binds). `Drive.drain()` remains in the sim as a retained primitive but is **not** used by the rev.3 lifecycle: the anti-drum guarantee is dedup + growing backoff, never a drain (rev.2 removed the drain-fork).

## 6. The `q_event` exchange-quality classifier

An exchange event in a lane is classified into a quality `q`:

| event in the lane                                          | `q`   |
|------------------------------------------------------------|-------|
| user message following assistant activity (genuine two-way)| +1.0  |
| user low-effort acknowledgement / reaction                 | +0.5  |
| assistant monologue / internal proactive wake              |  0.0  |
| conflict / explicit rejection / "busy"                     | −0.5  |

For the simulation, `q` is read from the trace label. For the plugin (out of scope), `q` is inferred from hooks. **Internal proactive impulses never count as user contact** (`q = 0`) — load-bearing: it stops the being satiating itself. Satiation uses `β · max(q, 0)`. First contact after long silence still counts `+1.0`. An **ignored question** is the *absence* of an event (an `awaiting_answer` timer at the wake-decision, scenario 4), never a `q_event`.

## 7. Hard structural gates + release conditions (the fixed safety envelope)

**Structural gates** decide whether an urge may *wake* cognition. They are policy, outside the drive, and (rev.3) **hard-coded forever** — the learnable v2 parameters live *inside* these, never replace them:

1. **`active_silence_window` (`W`):** no wake if `now − last_exchange_at < W` (most recent exchange of any role). The anti-drum / don't-interrupt lever.
2. **`no-wake-while-in-flight`:** no wake if a turn is running/queued.
3. **`reject-backoff` (`R`) — growing, not fixed.** After a **reject**, suppress a new desire for `R_n = min(R_max, R₀·kⁿ)`, `n = consecutive rejects`, unless a new exchange occurs. A fixed `R` merely relabels the drum period; growth is the requirement. A new user exchange clears the reject record and satiates. (`reject_count` is policy memory, not a drive variable — §8.)
4. **internal-impulse exclusion:** internal proactive impulses never satiate, never update the clock, never reset gates.
5. **(rev.3) per-window send cap + minimum inter-send interval — a *plugin/runtime* safety, not a certified sim gate.** An absolute ceiling on visible proactive sends per window; cheap insurance that survives any learned parameter. It lives at the egress/delivery layer, **outside** this model's harness — listed here so the envelope reads complete, but it is **not** among the §9 certified invariants and is implemented in the plugin epic, not this task.

**Release of a *deferred* desire** (held, not dropped) is met by **any** of: observed presence (the user is demonstrably active), high learned availability (v2 rhythm estimator; v1 scripts it), or deprivation escalation (`duration_over_θ` past a bar → re-present, anti-neglect).

**Timing/appropriateness is the cognition verdict, not a gate.** Whether *now* is a good moment (mood, the user's rhythm) needs knowledge in the LLM/Hermes; the cheap gates only decide *whether to wake*.

## 8. The one-vs-two-variable question — answered for the product architecture

**What "one variable" honestly means.** The v1 drive is one *continuous* variable `u` (plus the neuron's own `duration_over_θ`), accompanied by bounded **policy memory** (`last_exchange_at`, desire status, `reject_count`, `declined_at`, `in_flight`). Policy memory is **not** a second drive variable; it does not accumulate a deficit or feed the threshold. So the question is precise: *does the drive need a second **continuous** variable?*

**Answer (rev.3, grounded): yes — a slow, LEARNED set-point `s` — and it is deferred to v2.** The evidence is not "v1 invariants failed" (they pass, §9). The evidence is stronger and comes from the science + the problem itself:

1. **"When" is not a universal constant** (Niv, Daw & Dayan 2007). A single fixed `α/θ` encodes one universal rhythm; the optimal latency is context/relationship-dependent. So a *fixed* drive cannot be *right* for two different people — it can only be *safe* (conservative). v1 is deliberately safe, not right.
2. **The social set-point is individual and plastic** (Matthews & Tye; Frontiers 2023) and **making it adaptive is HRRL's named extension** (Keramati & Gutkin). The correct second variable is therefore a **learned set-point**, not the rev.2 guess of a leaky "load `a(t)`."

**Three separated quantities (Codex correction — do NOT collapse into one).**

| quantity | plain meaning | learns? | phase |
|---|---|---|---|
| `u` (fast) | urge = deviation of connection from set-point `s` | no (dynamics) | v1 ✅ |
| `s = τ*` (slow) | *how much silence this bond tolerates* — a contact half-life (one orientation, fixed) | yes, **inside hard clamps** | v2 |
| `h(t)` | *when* contact is welcome — a rhythm/availability hazard | yes (observational) | v2 |
| `a(t)` | withdrawal/load after repeated rejection (Cacioppo) | yes, separate & optional | v2+ (only if evidence demands) |

`s = τ*` answers "how long a silence is comfortable for this bond?"; `h(t)` answers "is *now* a welcome moment?"; they correlate but differ (a chatty user may tolerate short silences yet hate a 3am ping). `a(t)` answers "has trying become costly?" and is a *separate* suppressive term — the set-point does **not** replace it (both may be needed eventually; neither is in v1). `s` is pinned to **one orientation** (tolerated-silence, so a *larger* `s` = *rarer* contact) to keep its clamps unambiguous (§10).

**Two learning signals, treated differently (Codex):** the user's own inter-contact intervals are **dense observational** evidence → update `h(t)` directly and `s = τ*` only through a slower aggregate (§10). Outreach outcomes (warm/ack/reject/ignored) are **sparse, censored, confounded** causal evidence → update policy value slowly, with **rejection updating faster (suppressive) than warmth updates upward**. User silence after outreach is not clean credit assignment and must not be treated as reward.

**Direction-conflict note (kept from rev.2):** withdrawal (`a`) *raises* the effective threshold; a deteriorating-bond intuition *lowers* it. They are opposite and must be modelled separately or as an explicit non-monotone effect, decided on real traces in v2 — never hidden under one variable.

## 9. Simulation harness (v1 mechanics — built & certified)

**Pure Python, no Hermes**, under `src/lifemodel/sim/` (inside the ruff/mypy-strict/pytest/coverage gate). Built: `drive.py`, `quality.py`, `wake.py`, `aggregation.py`, `harness.py`; **54 tests green**, ruff+mypy clean. Trace rows:

```
time, lane, actor=user|assistant|proactive_internal, text, label, cognition_verdict, user_available
```

`label ∈ {two_way, ack, monologue, rejection}` drives `q_event`. `cognition_verdict ∈ {—, fulfill, defer, reject}` scripts the (out-of-scope) LLM per wake. `user_available ∈ {—, yes, no}` scripts presence/availability. The harness ticks the neuron, lets aggregation create/dedup/hold/re-present, applies the scripted verdict, and records per tick `u`, `duration_over_θ`, `last_exchange_at`, desire status, `reject_count`, `exchange_clock`, and the wake outcome.

**Model invariants (asserted numerically over every scenario):**

```
no_proactive_within_active_window          no wake within W of any exchange
no_wake_while_in_flight                     no wake while a turn runs/queued
user_message_satiates_and_resets            inbound exchange drops u, clears desire + reject record
acked_urge_does_not_refire                  while a desire is live/deferred, repeat crossings create NO new wake
deferred_intention_releases                 a deferred desire re-presents on presence OR deprivation-escalation
inter_wake_intervals_grow_without_evidence  repeated rejects, no new exchange → non-decreasing wake gaps
eventual_wake_after_long_deprivation        ≥1 wake within a declared deadline Y of clean silence
threshold_means_wake_not_send               each wake = one verdict; sends ≤ wakes
internal_impulse_is_not_user_contact        proactive_internal rows never satiate / never reset gates
reject_or_defer_delivers_nothing            a scripted defer/reject emits no visible message
no_contact_when_scripted_unavailable        no fulfill while user_available=no
```

**Scenarios (all 8 green):** (1) the 2026-07-04 failing log — no wake inside `W`, reject backoff grows 30→60, **no drum** *(the regression)*; (2) active back-and-forth → never wakes; (3) dormant-healthy bond → one wake then growing backoff, eventual re-wake; (4) question-then-disappear → no drum; (5) overnight/multi-day silence → `eventual_wake` within Y; (6) bad-moment defer + presence release; (7) gateway restart persists a deferred desire, no spurious wake; (8) user returns after a reject → backoff cleared + satiated.

**v1 calibration = feasibility of a conservative prior, not a search for truth — and it is the one piece of v1 still to do.** The 54 tests certify the *mechanics*, but each scenario currently uses its own `α` (`0.5`, `0.25`, `0.01`, …); they do **not** yet prove one shared prior. Remaining v1 deliverable: confirm a single shared conservative set `{α, θ, β, W, r0, k}` satisfies **every** invariant on **every** scenario. Because most gates are `α`-independent once `u ≥ θ`, "widest margin" would otherwise collapse to the degenerate never-wake optimum (`θ→100`); to constrain it, add per-scenario **wake-latency bands** (a *should-wake-by* deadline and a *should-stay-silent-until* floor) as soft targets *alongside* the hard invariants. Report the chosen shared prior and its margins. The grid-search-for-the-one-true-constant is *retired* — there is no such constant (§8); this is only a feasibility + latency-band check documenting a safe starting envelope.

## 10. v2 adaptation design (deferred epic — designed here, NOT implemented)

The learning layer that makes the being learn *this* person. **Bounded, slow, and outside the safety envelope.**

**What is learned, and its clamps.**
- **Set-point `s = τ*`** (tolerated silence / contact half-life — the interval of silence this bond finds comfortable before contact is welcome; larger `τ*` ⇒ rarer contact): a slow parameter moved by evidence, hard-clamped to `[τ_min = polite minimum interval, τ_max = neglect bound]`. Learning adjusts the comfortable interval *inside* the clamp; it can never drop below `τ_min` (spam) or exceed `τ_max` (abandonment).
- **Rhythm/availability `h(t)`** — a **constrained contextual hazard/survival model** (Codex's cleaner formalism for "when"): `P(contact welcome | time-since-exchange, local time-of-day, weekday pattern, recent warmth/rejects, relationship prior)`. Learned observationally from *when the user themselves is active*; feeds the release/defer decision, never the safety gates.
- **`a(t)`** (optional, latest): a separate slow suppressive load after repeated failure/rejection; added only if traces demand it.

**Learning rules (honest about signal quality).**
- Observational (dense): user-initiated inter-contact intervals update `h(t)` **directly**, and update `s = τ*` only through a *slower aggregate* (so one busy week does not reset the bond's comfort).
- Causal (sparse/censored): update policy value from outreach outcomes slowly; **asymmetric** — rejection suppresses faster than warmth emboldens; treat post-outreach silence as censored, not as reward.
- Reward = drive reduction (HRRL): a fulfilled outreach that produced a genuine two-way is rewarding *iff* it reduced the deficit without friction.

**Simulation = mechanism recovery, not human calibration (Codex).** An **oracle** generates a *known* latent user rhythm (archetypes: chatty / reserved / nocturnal / bursty). Assert, over the learning run: (a) the estimator **recovers** the latent rhythm within bounds; (b) it **reduces regret** versus the fixed conservative prior; (c) **all v1 invariants still hold throughout learning** (the envelope is never violated while adapting). We validate the *mechanism*; we make **no** claim of being "calibrated to real humans."

**Out of the v2 epic's own scope:** the real receptivity/availability estimator's production wiring, and the cognition/soul quality — those are their own beads.

## 11. Out of scope (explicit)

- Hermes plugin implementation (the `pre_gateway_dispatch` inbound hook, neuron/aggregation wiring into the tick, per-neuron bounded-scale + duration state, `NO_REPLY` reject wiring, streaming-disable for internal events, busy-owner fix). → the plugin epic consuming this model.
- The cognition verdict + wake-packet framing (LLM/soul quality) — bead `lm-pbm`.
- **All of §10 (v2 adaptation) implementation** — a separate epic; only its *design* is in scope here.
- Multi-transport continuity (bead `lm-pin`); component fault-isolation (BRD FR28, a runtime concern).

## 12. Open questions (swept by the harness / decided in v2, not by opinion)

- **Rise form:** linear-with-clip (v1 default) vs saturating `α(1−u/100)` — swept; decide by which meets invariants at calmer constants.
- **Reject-backoff shape:** growing is required; `k`, `R_max` from the v1 feasibility check.
- **v1 conservative prior:** which shared `{α, θ, β, W, r0, k}` gives the widest invariant **+ latency-band** margin (§9) — the documented starting envelope for v2.
- **v2 set-point clamps:** where exactly `s_min`/`s_max` sit; how `h(t)` combines with observed presence (OR vs weighted).
- **v2 second-variable direction:** does `a(t)` (withdrawal, raises threshold) get added alongside `s`, and does any effect *lower* the threshold — resolved on real traces, never assumed.
- **Global vs per-lane:** desire is per-lane; a global daily-cap (BRD FR6) is cheap multi-lane insurance — add only if a multi-lane scenario shows lanes co-firing.

## References

- **Keramati & Gutkin** 2011 (NIPS), 2014 (eLife) — Homeostatic reinforcement learning — https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/
- **Matthews & Tye** 2019 — Neural mechanisms of social homeostasis — https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/nyas.14016 ; **Individual differences in social homeostasis**, Frontiers 2023 — https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2023.1068609/full
- **Niv, Daw & Dayan** 2007 — Tonic dopamine: opportunity costs and the control of response vigor — https://link.springer.com/article/10.1007/s00213-006-0502-4
- **Interoception as modeling, allostasis as control** 2022 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9270659/
- Hull 1943 (drive-reduction); Baumeister & Leary 1995 (need to belong); Hawkley & Cacioppo 2010 (loneliness); Sterling & Eyer 1988 / McEwen 1998 (allostatic load) — analogy/constraint only.
- Codex consults: capability facts `019f2ed7`/`019f2ed9`; instrument-honesty `019f2eec`; science + v2-form `019f2f26`; **rev.3 adversarial review `019f316e`**.
- Prior egress design: `docs/superpowers/specs/2026-07-04-lifemodel-proactive-egress-design.md`.

# Proactive-contact desire model — design spec

**Bead:** [lm-x43](../../) — Mathematical model of proactive-contact desire (dynamics + simulation + calibration)
**Status:** draft — rev.2 (2026-07-05). Rev.1 folded Codex instrument-honesty fixes (thread `019f2eec`); rev.2 folds the full design session: desire-lifecycle with `ack`/dedup, timing as a cognition verdict, bounded `0..100` neuron scale, neuron-owned deprivation duration, and corrected science + v2 form (Codex thread `019f2f26`). Consolidation bead: [lm-27h].
**Scope:** the DESIRE model + its simulation only. No Hermes plugin code in this task — that is a separate epic that consumes whatever this task certifies. Product "what" lives in `business-requirements.md` (FR2/FR3/FR6); architecture "how" in `hla.md` (§2.1, §11); this spec formalises and *simulates* them.

---

## 1. Goal

A minimal, science-grounded, **simulatable** model of how the *desire to make proactive contact* arises and is resolved, validated by simulation **before** any plugin code. Output: a certified model + a pure-Python simulation harness + calibrated constants + an empirical answer to *one vs. two state variables*.

The live system today (HEAD `0881ebc`) ships the proactive-egress **delivery** layer but its **decision** layer misbehaves (real Telegram log 2026-07-04: the being proactive-messaged one minute after the user greeted, then drummed every 30 min, each time voicing "nothing to add"). This task does not touch delivery. It rebuilds the *when to reach out, when to stay silent, when to hold* decision as a model provable in simulation.

## 2. Non-negotiable principles

1. **URGE ≠ ACTION.** A cheap, zero-LLM drive produces pressure; at threshold the aggregation layer *creates a desire* and wakes cognition. An expensive, context-aware LLM then *judges* — **fulfill / defer / reject** — from the full session context and what it knows about the user. Threshold-crossing **never** deterministically sends a message. The model cannot speak; only cognition can.
2. **Logic lives in a layer, not in a neuron** (project convention). The neuron is a thin sensor: it *measures* (bounded pressure + how long it has been over threshold) and *emits*, but it does not *decide*. Desire creation, dedup, hold, and escalation live in the aggregation layer; the verdict lives in cognition.
3. **Hard structural gates are separate from the drive.** "Don't interrupt an active conversation", "don't wake while a turn is in flight", "don't re-wake right after a reject" are *policy*, not *dynamics*. They gate **whether to wake cognition**, not whether it is a good moment — that (mood, the user's rhythm) is cognition's verdict, because the knowledge lives in the LLM/Hermes, not in a cheap neuron.
4. **A desire has a lifecycle and is remembered.** Repeated threshold-crossings do **not** spawn duplicate wakes: the aggregation layer holds one desire and **dedups** further signals against it (`ack`). A desire cognition chose to **defer** is *held*, not dropped, and re-presented later — so the being never forgets an intention and never drums.
5. **Calibrate, do not guess.** Constants are fit by grid-search against labelled traces with an explicit objective. No coefficient is taken on faith.
6. **Minimal set first.** Start with one continuous drive variable. Add a second only if simulation proves the invariants cannot otherwise be met.

## 3. Science grounding

Grouped by the exact phenomenon each model licenses. **All are inspiration and constraint, not validation** (Codex review, thread `019f2f26`): they establish *that* a mechanism is biologically real; they do not certify our equations. Constants are earned from simulation (§9), not from citations.

- **Bounded rise from deprivation** — **Hull, drive-reduction / homeostatic drive (1943).** Deprivation builds a drive; the drive motivates reducing behaviour; satisfaction discharges it. The cleanest historical template for "silence accumulates a bounded urge, contact discharges it." *Caveat:* old behaviourist theory; physiological drives fit better than social longing — we borrow the shape, not the ontology.
- **Quality, not just frequency** — **Baumeister & Leary 1995, "The need to belong."** A fundamental motivation requiring *frequent, positive* interactions in stable bonds. Motivates the `q_event` classifier (§6): a genuine two-way exchange satiates more than a low-effort ack. *Caveat:* not dynamics.
- **Deficit as an aversive signal; chronic deficit → withdrawal** — **Cacioppo / Hawkley & Cacioppo 2010.** Loneliness is an aversive signal of perceived disconnection: acutely it motivates reconnection; chronically it biases toward social-threat hypervigilance, worse sleep/health, and **withdrawal**. Closest domain fit. Grounds both the rise *and* the deferred v2 withdrawal twist (§8). *Caveat:* not an equation for proactive-contact desire.
- **Cost of sustained wanting; threshold/mood shift** — **Allostasis & allostatic load (Sterling & Eyer 1988; McEwen 1998).** Stability-through-change: chronic regulatory activation carries cumulative "wear and tear" that shifts physiology, mood, and resilience. This is the grounding for "sustained wanting depletes energy and moves later thresholds" and ties the drive to the Energy/dreaming layer (BRD FR21). *Caveat:* not social-contact-specific; does not by itself predict a rise-then-fall curve.
- **Analogies only, with explicit caveats:** **opponent-process (Solomon & Corbit 1974)** — delayed counter-regulation of an affective state; native paradigm is a stimulus *present then removed*, **not** a sustained unmet deficit. **Habituation / neural adaptation (Thompson & Spencer 1966)** — response decrement to a *repeated/persistent* stimulus; silence is the *absence* of a stimulus, so this fits only if we reframe repeated *failed outreach* as the stimulus. Cite as engineering analogies, never as proof that "unmet desire decays on its own." **Ego-depletion is deliberately avoided** (replication crisis).
- **Bistable relationship climate (v2 escalation only)** — **Abplanalp, Maimone & Green 2025, *NPP*, "Viewing social isolation as a complex dynamical system."** Latent connectedness `Z` on a double-well potential `V(Z)=¼Z⁴−½aZ²−bZ`, `dZ = (−Z³+aZ+b)dt + σ dW` — two basins (connected/isolated), a tipping point, hysteresis. Powerful but **overpowered** for our v2: it needs dense longitudinal data (~365 d × 5 obs/d) to identify and only earns its keep if bistability/tipping is *proven* necessary. **Reserved as a *further* escalation, not the first second-variable** (§8).

## 4. Architecture (where things live)

```
                  ┌──────────────────────────────────────────────────────────┐
 inbound exchange │  NEURON (autonomic, 0 LLM) — a thin sensor               │
 (user message ──►│   u ∈ [0,100]: +α in silence (saturating), satiates on   │
  via plugin hook)│   real contact; emits {intensity u, duration_over_θ}      │
                  │   measures, does NOT decide. Params on disk, hot-reload.  │
                  └──────────────────────────────────────────────────────────┘
                                          │ signal
                                          ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  AGGREGATION (0 LLM) — owns the desire lifecycle          │
                  │   first crossing + structural gates pass → create ONE     │
                  │   contact-desire → WAKE. Repeat crossings deduped (ack).  │
                  │   deferred desire: held; re-woken on release condition.   │
                  └──────────────────────────────────────────────────────────┘
                                          │ WAKE (desire, not a drain)
                                          ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  COGNITION (LLM) — the verdict, by what Hermes knows      │
                  │   FULFILL (send) / DEFER (hold, wrong moment) /           │
                  │   REJECT (nothing to say → NO_REPLY + backoff)            │
                  └──────────────────────────────────────────────────────────┘
```

The neuron and the aggregation layer are **ours** (cheap, zero-LLM, simulatable without Hermes). Cognition is the LLM and is **out of scope for this model** — the harness *scripts* its verdict (§9). The model guarantees the *machinery* (when it wakes, that a held desire is remembered and re-presented, that it never drums); the *quality* of the verdict and of the wake-packet framing is the soul/prompt concern (bead `lm-pbm`), tested by real logs, not here.

## 5. The model (v1 — one continuous drive)

**Neuron state** (per relationship lane):
- `u ∈ [0, 100]` — the contact-pressure intensity (a **drive/deficit** neuron → unipolar `0..100`; valenced neurons like mood are `−100..100`, out of scope here). The scale is a property of the neuron type, declared on the base `Neuron`.
- `duration_over_θ` — how long `u` has been ≥ `θ`, in time. A **separate field the neuron owns** (its own measurement), because a saturated `u` pinned near 100 no longer carries "how long." This duration, not the pinned value, feeds deferred-desire escalation (§7).

**Dynamics** (discrete time, tick `Δt`; "genuine silence" = no exchange event in the lane this tick):

- **Rise in genuine silence:** saturating toward the ceiling, `u ← u + Δt·α·(1 − u/100)` (default), or linear-with-clip `u ← min(100, u + Δt·α)` — kept as a swept knob (§11). Saturating is the default because the bound is real (you cannot want infinitely *hard*), and marginal urgency flattens near the ceiling.
- **Satiation on a positive exchange:** `u ← max(0, u − β·q_event)`, only for `q_event > 0` (§6). `β` is calibrated in scale units so a genuine two-way exchange (`q=1`) drops `u` well below `θ` (a real conversation discharges the drive); an ack (`q=0.5`) discharges half as much.
- **Reset on real contact:** when cognition **fulfills** (a message is actually sent), that is contact → `u` satiates as an exchange and `duration_over_θ` resets. **Deferring or rejecting does NOT reset `u`** — no contact happened, so the deficit honestly persists.

**Threshold + desire creation (aggregation):** when `u` first crosses `θ` and the structural gates (§7) pass, aggregation **creates one contact-desire and wakes cognition**. There is **no per-tick "drain"**: the anti-drum guarantee is the desire **dedup** (`ack`) + the reject-backoff, not zeroing `u`. (This *removes* rev.1's drain-fork entirely — see §11.)

**Desire lifecycle (aggregation owns it):**
- **create** → wake cognition (once).
- **dedup / `ack`** → while a desire is live (active or deferred), further threshold-crossings are absorbed: no duplicate desire, no re-wake.
- **fulfill** → message sent → `u` satiates, `duration_over_θ` resets, desire resolved (leaves residue, BRD FR3).
- **defer** → desire *held*; `u` not reset; re-woken when a **release condition** holds (§7): observed presence, high learned availability, or long-enough deprivation.
- **reject** → desire cleared + a **growing backoff** so high `u` does not immediately recreate it.

**v1 constants:** `α` (rise rate), `θ` (threshold on the `0..100` scale), `β` (satiation magnitude). Plus the policy constants in §7. All are **disk config, hot-reloadable** (BRD NFR5) — the sim *calibrates* them; the plugin *loads* them.

## 6. The `q_event` exchange-quality classifier

An *exchange event* in a lane is classified into a quality `q`:

| event in the lane                                          | `q`   |
|------------------------------------------------------------|-------|
| user message following assistant activity (genuine two-way)| +1.0  |
| user low-effort acknowledgement / reaction                 | +0.5  |
| assistant monologue / internal proactive wake              |  0.0  |
| conflict / explicit rejection / "busy"                     | −0.5  |

For the simulation, `q` is read from the trace's label column. For the plugin (out of scope), `q` is inferred from the `pre_gateway_dispatch` hook + `post_llm_call` heuristics. **Internal proactive impulses never count as user contact** (`q = 0`, not positive) — load-bearing: it is what stops the being satiating itself.

Satiation uses `β · max(q, 0)`. A genuine two-way (`q=1`) drops `u` below `θ`; an ack (`q=0.5`) drops half as much; `β` is calibrated (§9).

> **First contact after long silence** (a `two_way` with no immediately-preceding assistant turn) still counts `+1.0` — the "following assistant activity" wording is descriptive, not a gate; any genuine user inbound satiates.
>
> An **ignored question** (the being asked, the user never answered) is the *absence* of an event, not a `q_event` — handled by an `awaiting_answer` timer at the wake-decision (scenario 4), never by this classifier.

## 7. Hard structural gates + release conditions

**Structural gates** decide whether an urge is allowed to *wake* cognition (create/re-present a desire). They are policy, outside the drive dynamics:

1. **`active_silence_window` (`W`):** no wake if `now − last_exchange_at < W` in the lane (most recent exchange of **any** role). The "don't interrupt an active conversation" gate and primary anti-drum lever.
2. **`no-wake-while-in-flight`:** no wake if a turn is running or queued in the lane.
3. **`reject-backoff` (`R`) — growing, not fixed.** After cognition **rejects** (NO_REPLY, "nothing to say"), suppress a new contact-desire for `R_n = min(R_max, R₀·kⁿ)`, `n = consecutive rejects` (+ optional jitter), *unless* a new exchange occurs. A fixed `R` merely relabels the drum period to `max(recross-time, R)` and still drums; growing is the v1 requirement. A new user exchange clears the reject record and satiates `u`. (`reject_count` is **policy memory, not a second drive variable** — §8.)
4. **internal-impulse exclusion:** internal proactive impulses are never exchanges (§6): they neither satiate nor update the clock nor reset gates.

**Release of a *deferred* desire** (distinct from reject-backoff — a deferred desire is *held*, not dropped) is met by **any** of:
- **observed presence** — the user messaged / is demonstrably active (we wait for a sign of life, we do not guess a schedule);
- **high learned availability** — the being's learned estimate of when this person is reachable (part of receptivity, BRD FR5) is high now;
- **deprivation escalation** — `duration_over_θ` has grown past a bar, so a held intention is re-presented rather than forgotten (the anti-neglect fallback).

**Timing/appropriateness is NOT a structural gate — it is the cognition verdict.** Whether *now* is a good moment (mood, the user's rhythm) needs knowledge that lives in the LLM/Hermes: mood is read from the conversation context, rhythm from observed activity; with no data yet, common social norms. The cheap gates only decide *whether to wake*; *is it appropriate* is decided by the woken cognition. Dedup/`ack` keep wake frequency low, so an occasional LLM wake that concludes "not now → defer" is rare and justified.

## 8. The core tension the simulation must resolve

Two failure modes trade off:

- **Drumming** (today's bug): firing on a fixed cadence regardless of conversation. Killed by gates 1 + 3, satiation-on-exchange, and desire **dedup** (`ack`).
- **Neglect**: never reaching out even after long genuine silence. Guarded by `eventual_wake_after_long_deprivation` and by deferred-desire deprivation-escalation.

**What "one variable" honestly means.** The drive is exactly one *continuous* variable `u` (plus the neuron's own `duration_over_θ` measurement). It is accompanied by **policy memory** — bounded discrete bookkeeping the aggregation needs: `last_exchange_at`, `desire status` (active/deferred), `reject_count`, `declined_at`, `in_flight`. Policy memory is **not** a second drive variable; it does not accumulate a deficit or feed the threshold rule. Conflating the two would let the harness "certify one variable" while steering on hidden state. So the §8 question is precise: *does the **drive** need a second **continuous** variable* over and above bounded policy memory?

**If v1 fails, the second variable is a slow leaky "allostatic load / withdrawal" `a(t)`, NOT the Abplanalp double-well** (Codex `019f2f26`). Recommended form:

```
du/dt = α·(1 − u/100)·silence − β·q⁺(t)
da/dt = ( load(u, reject_count, q⁻) − a ) / τ_a          # slow leaky integrator
θ_eff = θ₀ + λ·a          # or   effective_urge = u − γ·a ;   energy = E₀ − η·a
```

`a` rises when `u` stays high without resolution / after repeated rejects / after negative events, and recovers after good contact — a lagged counter-regulation that opponent-process, adaptation, and allostasis all point to, and that a leaky integrator captures directly (the double-well is only for *bistable climate with tipping/hysteresis*, added later if proven). `a` also couples to the Energy layer (BRD FR21).

> **Direction conflict to resolve on traces (Codex catch).** The **withdrawal** intuition **raises** the threshold (chronic unmet wanting → withdraw/suppress). The original climate-`Z` intuition **lowers** it (deteriorating bond → reach out more to avoid neglect). These are **opposite**. They must be modelled separately or as an explicit non-monotone `θ_eff(a)`, and decided by simulation — never hidden under one variable untested.

**This is decided by simulation, not opinion.** The user's "rise-then-fall + energy cost" is a strong *candidate* v2 fork, not grounds to pre-commit `a` (or `Z`) into v1.

## 9. Simulation harness (the deliverable that certifies the model)

**Pure Python, no Hermes dependency**, under `src/lifemodel/sim/` (Hermes-free, but inside the ruff/mypy-strict/pytest/coverage gate; the certified drive + aggregation later inform the real `core/Neuron` + `Aggregator`). Trace rows:

```
time, lane, actor=user|assistant|proactive_internal, text, label, cognition_verdict, user_available
```

- `label ∈ {two_way, ack, monologue, rejection}` drives `q_event`.
- **`cognition_verdict ∈ {—, fulfill, defer, reject}` is the scripted stand-in for the LLM** on a generated wake — cognition is out of scope, so the trace scripts what it decided. Without it, the fulfill/defer/reject paths and every backoff/hold scenario are untestable.
- **`user_available ∈ {—, yes, no}`** scripts the presence/availability the release conditions (§7) and cognition read — so "don't contact at a bad moment" and "release when present" are drivable.

The harness ticks the neuron, lets aggregation create/dedup/hold/re-present desires, applies the scripted verdict, and records per tick: `u`, `duration_over_θ`, `last_exchange_at`, desire status, `reject_count`, gate/release states, and the wake outcome (`no_wake_silence_window` / `no_wake_in_flight` / `no_wake_reject_backoff` / `deduped_ack` / `WAKE`).

**Model invariants** (this harness owns them; asserted numerically over every scenario):

```
no_proactive_within_active_window          # no wake within W of any exchange in the lane
no_wake_while_in_flight                     # no wake while a turn is running/queued
user_message_satiates_and_resets            # inbound exchange drops u, clears desire + reject record
acked_urge_does_not_refire                  # while a desire is live/deferred, repeat crossings create NO new wake
deferred_intention_releases                 # a deferred desire is re-presented on presence OR deprivation-escalation
inter_wake_intervals_grow_without_evidence  # repeated rejects with no new exchange → non-decreasing wake gaps
eventual_wake_after_long_deprivation        # ≥1 wake within a declared deadline Y of clean silence
threshold_means_wake_not_send               # each wake == one cognition verdict; sends ≤ wakes (defer/reject send nothing)
internal_impulse_is_not_user_contact        # proactive_internal rows never satiate u / never reset gates
```

**Integration stubs** (delivery/cognition layer, out of scope; asserted against the *scripted* verdict/availability, kept here to keep traces honest, moved to the plugin epic):

```
reject_or_defer_delivers_nothing            # a scripted defer/reject emits no visible message
no_contact_when_scripted_unavailable        # no fulfill while user_available=no (cognition's call, scripted)
```

`inter_wake_intervals_grow_without_evidence`, `eventual_wake_after_long_deprivation`, and `deferred_intention_releases` pin the drum-vs-neglect tension numerically. `Y` is a **declared constant per scenario**, never implicit.

**Scenarios** (traces the harness drives):

1. **The failing 2026-07-04 log** — user at 21:57, (broken) proactive at 21:58 must be **forbidden** (within `W`); 22:28 / 22:58 "nothing to add" scripted as `reject` must yield growing backoff, no drum. *The regression test.*
2. **Active back-and-forth**, gaps under `W`: never a proactive wake.
3. **Dormant-but-healthy bond:** after clean silence, exactly one wake; scripted `reject` → no fixed-period re-wake (growing backoff); eventual re-wake if silence continues.
4. **Question-then-disappear:** at most one gentle follow-up after a separate `awaiting_answer` delay; must not drum.
5. **Overnight / multi-day silence:** `eventual_wake_after_long_deprivation` holds.
6. **Bad-moment defer + presence release:** wake while `user_available=no` → scripted `defer` → **held, no re-fire** (`acked_urge_does_not_refire`); later `user_available=yes` → `deferred_intention_releases` fires and it is delivered.
7. **Gateway restart** with persisted `u` / `duration_over_θ` / desire status: state survives, no spurious wake on reload.
8. **User returns after a reject:** new exchange clears backoff and satiates; conversation resumes.

**Calibration — feasibility first, objective second.** A naive "minimise weighted violations, false-interrupt ×10" has a trivial optimum: push `θ→100` / `α→0` / `W→∞` and never wake. So:

1. **Feasible region = every model invariant passes** (hard constraints), including `eventual_wake_after_long_deprivation` within `Y` and `inter_wake_intervals_grow_without_evidence`. Never-waking **violates** `eventual_wake` → *infeasible*, not merely costly. Parameter ranges are bounded (`0 < θ < 100`, `α > 0`, finite `W`, `R₀`, `k`, `R_max`, release bars).
2. **Within the feasible region**, minimise the soft objective: false-interrupt count weighted an order of magnitude over outreach *latency* (whether it ever happens is guaranteed by stage 1).

Grid-search sweeps `{α, θ, β, W}` **× the residual knobs** (`rise ∈ {saturating, linear}`, `backoff ∈ {growing, fixed}` — fixed retained only to *demonstrate* it drums). **Report precision/recall separately** on labelled should-wake / should-stay-silent moments, the chosen constants, per-scenario invariant pass/fail, and residual margin. If the feasible region is **empty** for the single-`u` model, that emptiness is the §8 evidence for the slow variable `a`.

**The empirical answer, with evidence:** does single-`u` satisfy every invariant on every scenario at some calibrated setting? Yes → v1 ships one variable, `a`/`Z` deferred. No → document the failing invariant/scenario and show that the leaky-load `a(t)` (§8) resolves it → v2 design input.

## 10. Out of scope (explicit)

- Hermes plugin implementation: the `pre_gateway_dispatch` inbound hook, the neuron/aggregation wiring into the tick, per-neuron bounded-scale + duration state (Phase 2, replacing the single global `State.pressure`), the `NO_REPLY` reject instruction + scoped `transform_llm_output` safety net, disabling streaming for internal proactive events, fixing the split `busy` ownership. → separate epic consuming this model.
- **The cognition verdict itself and the wake-packet framing** (first-person self-attribution + recent context + options) — the LLM/soul quality, bead `lm-pbm`; tested by real logs, not simulated here.
- The **learned-availability / receptivity model** (how the being infers when the user is reachable) — the sim scripts `user_available`; the real estimator is its own component (BRD FR5).
- Multi-transport continuity (bead `lm-pin`).
- **Component fault-isolation (BRD FR28)** — a plugin-runtime concern: a failing neuron is caught and skipped (zero contribution), never crashing the tick; fail-open for sensors/drives, fail-closed for safety. In the pure model a "failed" neuron is simply no-signal; the being adapts on the remaining drives.
- The slow variable `a(t)` / `Z(t)` *implementation* — only its *necessity* is in scope (§8), decided by simulation.

## 11. Open questions to resolve during implementation

Swept by the harness, not decided by opinion:

- **Removed: the rev.1 drain-fork.** The anti-drum guarantee is now desire **dedup** (`ack`) + reject-backoff, not zeroing `u` on wake — so `drain_on_write / drain_on_wake / no_drain_on_decline` are gone. `u` resets only on **real contact** (fulfill).
- **Rise form:** saturating `α(1−u/100)` (default) vs linear-with-clip — swept; decide by which meets invariants at calmer constants.
- **Reject backoff shape:** growing is the v1 requirement (§7 gate 3); `fixed` retained only to demonstrate it drums. Sub-question: `k`, `R_max` from calibration.
- **Release-condition tuning:** the deprivation-escalation bar for a deferred desire, and how learned-availability combines with observed presence (OR vs weighted) — swept; the sim scripts `user_available`, calibration sets the bar.
- **Negative-event handling:** a `rejection` (`q=−0.5`) — merely fail to satiate (current), or feed the v2 load `a` / raise the threshold for a window (Cacioppo withdrawal)? Defer to v2 unless scenario evidence demands it.
- **One vs. two variables / threshold-direction conflict (§8):** single-`u` vs `u + a(t)`; and if `a`, whether it **raises** (withdrawal) or **lowers** (climate) the threshold — resolved on traces, never assumed.
- **Global vs per-lane (near-moot today):** desire is per-lane; a global daily-cap / cooldown (BRD FR6 "бюджет обращений") is cheap insurance against multi-lane spam but the live being has one lane. Add only if a multi-lane scenario shows lanes co-firing.

## References

- Hull, C. L. 1943, *Principles of Behavior* (drive-reduction / homeostatic drive).
- Baumeister & Leary 1995, *The need to belong* — https://psycnet.apa.org/record/1995-29052-001
- Hawkley & Cacioppo 2010, *Loneliness matters* (Ann. Behav. Med.); Cacioppo et al., *Evolutionary Mechanisms for Loneliness* — https://pmc.ncbi.nlm.nih.gov/articles/PMC3855545/
- Sterling & Eyer 1988, *Allostasis: a new paradigm to explain arousal pathology*; McEwen 1998, *Protective and damaging effects of stress mediators* (NEJM) — allostatic load.
- Solomon & Corbit 1974, *An opponent-process theory of motivation* (Psychol. Rev. 81(2):119-145) — analogy only. Thompson & Spencer 1966, *Habituation* (Psychol. Rev. 73(1):16-43) — analogy only.
- Abplanalp, Maimone & Green 2025, *Viewing social isolation as a complex dynamical system* — https://www.nature.com/articles/s44277-025-00051-y (v2 bistability escalation only).
- Codex design consults: capability facts `019f2ed7` / `019f2ed9`; instrument-honesty review `019f2eec`; science + v2-form review `019f2f26`.
- Prior egress design: `docs/superpowers/specs/2026-07-04-lifemodel-proactive-egress-design.md`.

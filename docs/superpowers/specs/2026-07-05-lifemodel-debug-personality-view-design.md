# `/lifemodel debug` — structured, self-explaining personality view

**Date:** 2026-07-05
**Status:** design (intent, not an implementation plan)
**Bead:** `lm-zmf` (P2) — tracker for this work
**Side-finding:** `lm-zhz` (P3) — `duration_over_theta` latent/unused (do NOT fix here)
**Reviewed by:** codex, verdict REVISE → all must-fixes folded in (thread `019f3351-735a-7db3-9e4c-5fc734cb255f`)

> This document fixes **what** and **why**. The **how** (files, interfaces, tests) is the next step (writing-plans / worker).
> All output is **English** (localization is deferred). Prose and identifiers alike are English.

---

## 1. Context and goal

`/lifemodel debug` is the owner's read-only introspection surface (HLA §12, NFR9). Today it prints raw `State`
fields with a one-word label each. That is enough to prove the engine persists, but it does **not** let a human
answer the question that matters: *what is happening with the personality right now, and what will it do next?*

To read the current dump you must hold the thresholds in your head (θ, α, the silence window `w`, the decline
backoff) and do arithmetic by hand. The constants that define the being's *temperament* are invisible — buried in
code. Runtime-relevant distinctions (gate verdict vs. actual launch, stale-pending recovery) are absent.

**Goal.** Redesign the dump (in place — same `/lifemodel debug` command) into a **structured, self-explaining**
view at **level C**: every section shows raw value(s) + derived quantities + one terse, visibly-derived
interpretation line. Show **both** the fixed *temperament* constants (who this being is by nature) and the current
dynamic *state* (what it feels now). The reader should understand the being without a calculator and without
reading the engine source.

Non-goals: localization (later), colour/ANSI, and any mutation of state (the surface stays strictly read-only).

---

## 2. Invariants (load-bearing)

1. **Read-only (HLA §9).** No commit, no signal consumption, no writes, no logging. The renderer keeps the narrow
   read-only protocols it already uses (`StateReader.load`, `UnprocessedPeek.peek_unprocessed`, `EventReader.read`).
   The new readings module operates on a **deep copy** of `State`, so running live logic mutates nothing on disk.
2. **No drift by construction.** Every displayed quantity is either read from `State` or **imported from its owning
   module** — never restated. The wake verdict is produced by the **real** engine code, not reconstructed. If a
   constant or formula changes, the dump changes with it automatically.
3. **stdlib-only, Hermes-free.** `core/introspect.py` imports only `core.decision`, `sim.*`, and `state.model`
   (mirrors the existing Hermes-free submodules). No new dependencies.
4. **Honesty over tidiness.** Anything not derivable from persisted state (notably runtime `in_flight`) is shown as
   `n/a (runtime-only)` with an explicit caveat — never silently defaulted into a rosier verdict.

---

## 3. Architecture

### 3.1 New module `core/introspect.py` — pure readings

```
compute_readings(state: State, *, now: datetime, busy: bool = False) -> PersonalityReadings
```

`PersonalityReadings` is a frozen dataclass holding everything the renderer needs, already computed. The module
**reuses the live primitives** rather than re-deriving them:

- **Wake verdict = the real decision, run on a copy.** Deep-copy `state` and call the genuine
  `core.decision.decide_reachout(copy, now=now, busy=busy)`. The returned `ReachoutDecision(wake, reason)` is the
  honest answer, and it captures three things a naive `evaluate_wake` call would miss:
  - **Stale-pending recovery** — `decide_reachout` first converts a stale `active`+pending desire
    (`pending_proactive_since` ≥ `PENDING_TIMEOUT_MIN`) into a `REJECT` before rise/gates. The copy runs that path,
    so the verdict reflects it.
  - **Drive rise** — the copy's `u` is risen by elapsed time since `last_tick_at`, i.e. "as of now".
  - **Aggregator dedup** — `decision.wake` already folds in `Aggregator.on_urge()`, which suppresses a second wake
    while a desire is live (the anti-drum guarantee). So `wake` ≠ "gate said URGE".
- **`reason`** is the gate `WakeOutcome` value (e.g. `no_wake_below_threshold`, `URGE`); **`wake`** is whether an
  outreach would actually launch. The readings expose **both**, distinctly (must-fix 4).
- **Derived display quantities** are simple arithmetic composed from **imported** constants and helpers (no restated
  formulas): time-to-θ `= (θ − u)/α`; silence-window remaining `= w − since_last_exchange`; backoff remaining
  `= backoff_interval(decline_count, …) − since_declined` (reusing `sim.wake.backoff_interval`). Reuse
  `core.decision._minutes_between` for every "N ago"/"until" so timezone/naive behavior matches the engine exactly.
- **Stale-pending detection** is exposed as a boolean + age so the renderer can print an explicit note when the next
  live tick would recover a stale pending as `REJECT`.

### 3.2 `debug.py` — thin renderer

`render_debug_dump` keeps its DI shape and read-only protocols; it gains a `PersonalityReadings` input (produced in
`render_dump_for_dir` from `lm.state.load()` + `now`) and formats the six sections below. Pure formatting: it
computes nothing behavioral, logs nothing, and degrades to `n/a` / `<unreadable: …>` on empty or corrupt stores
(the existing defensive per-section reads stay).

### 3.3 Module boundary & imports (drift owners)

| Displayed quantity | Imported from |
|---|---|
| θ, α, β, `U_MAX`, `PENDING_TIMEOUT_MIN`, `BASE_PARAMS` (w, r0, k, r_max) | `core/decision.py` |
| `backoff_interval` (backoff schedule) | `sim/wake.py` |
| `Drive` dynamics | `sim/drive.py` (reused, not restated) |
| service-liveness max age | `tick.py` (`SERVICE_LIVENESS_MAX_AGE`) |
| proactive loop interval | `egress_service.py` |
| event category names | `events.py` (already imported) |

No circular imports: `debug.py → composition.py → core.*` and `debug.py → core.introspect → core.decision`;
`decision.py` imports neither `debug` nor `introspect`.

---

## 4. Rendered sections (mockup on current live values)

```
lifemodel debug dump  (read-only)
==================================

META
  schema_version:  1
  tick_count:      668

TEMPERAMENT  (fixed nature — how this being is calibrated)
  wake threshold θ:        1.0
  loneliness rate α:       0.00417 /min      (0 → θ in ~240 min of silence)
  silence window w:        15 min            (won't reach out within w of a real exchange)
  decline backoff:         30 → 60 → 120 …   (×2 per decline, cap 1440 min)
  urge ceiling U_MAX:      100.0
  pending-verdict timeout: 30 min            (stale proactive turn recovers as REJECT)

DRIVE  (the contact urge right now)
  u:                    0.0097   (1% of θ)   ← strength of the pull to reach out
  duration_over_theta:  0.0 min              ← time u has sat ≥ θ (tracked; not currently gating) [lm-zhz]
  energy:               1.0 (placeholder)    ← body charge slot; recovery is a later phase
  → calm & satiated; needs ~3h 58m of continued silence to feel a pull

DESIRE LIFECYCLE
  desire_status:   none                      ← no live desire  (none → active → deferred)
  pending:         no                        ← outstanding proactive turn awaiting a verdict
  decline_count:   0                         ← consecutive rejects (grow the backoff)
  declined_at:     n/a
  → nothing pending; not in backoff

TIMING
  last_exchange:   2026-07-05T17:08:48Z   (2.2 min ago)   ← last real exchange (satiates u)
  last_contact:    2026-07-04T21:29:01Z   (19.7 h ago)    ← last time IT reached out (outbound bookkeeping)
  last_tick:       2026-07-05T17:10:49Z   (0.2 min ago)   ← heartbeat alive

WAKE READINESS  (what the next heartbeat would decide — run on a copy, read-only)
  urge now (risen):        0.0097   (1% of θ=1.0)   [+237.7 → θ]
  gate verdict:            no_wake_below_threshold   (assuming in_flight=false)
  would launch outreach:   no                        (gate=URGE AND new-desire allowed)
  stale-pending recovery:  none this eval
  gate ladder (precedence):
    below_threshold   BLOCKS HERE   u 0.0097 < θ 1.0
    in_flight         n/a           (u < θ → cannot matter)
    silence_window    —             (not reached; ~12.8 min of w would remain)
    decline_backoff   —             (not reached; no active decline)
    urge              —             conditional on in_flight=false

AUTONOMOUS LOOP
  signal bus unprocessed:  0
  in-proc egress service:  alive  (stamp 0.3 min ago)
  last tick outcome:       deferred=service_alive
  events:                  wake_decision n/a · act_gate n/a · dream_run n/a
  lock status:             n/a  (no lock held in Phase 1)
```

**When risen `u ≥ θ`** the WAKE READINESS block changes shape to stay honest (must-fix 1 & 2):

```
  urge now (risen):        1.20   (120% of θ=1.0)
  gate verdict:            URGE                      (assuming in_flight=false)
  ⚠ in_flight is runtime-only: if a turn is executing now, actual verdict = no_wake_in_flight
  would launch outreach:   no                        (desire already active → Aggregator dedup)
  gate ladder (precedence):
    below_threshold   clear
    in_flight         UNKNOWN       runtime-only; if true, BLOCKS HERE
    silence_window    clear         (last exchange > w ago)
    decline_backoff   would-block   ~Y min of R_n left
    urge              reached       conditional on in_flight=false
```

---

## 5. The `in_flight` / `busy` decision (why it is `n/a`, not `false`)

`busy`/in_flight is a **runtime-only** fact — "is a turn executing this instant" — passed into `decide_reachout(…,
busy)` by the host. It is deliberately **not** in persisted `State`: persisting it risks a stale "stuck busy" flag
across restart/crash (the same bug class already handled by `pending_proactive_since` stale-recovery). So the debug
path computes as of now from `State` and cannot know true `in_flight`.

Codex must-fix (1 & 2): because `evaluate_wake` gives `in_flight` **higher precedence** than silence/backoff once
`u ≥ θ`, passing `busy=false` can turn a true runtime `no_wake_in_flight` into a rosier verdict. Therefore:

- The verdict is always labeled **"assuming in_flight=false"**.
- When **`u < θ`**, `in_flight` is shown `n/a` (precedence means it cannot matter — `below_threshold` decides first).
- When **`u ≥ θ`**, `in_flight` is shown `UNKNOWN (runtime-only; if true, BLOCKS HERE)` plus a ⚠ line that the real
  runtime verdict may be `no_wake_in_flight`.
- `pending: yes/no` (from persisted `pending_proactive_id`) is shown as the honest, persisted cousin of in_flight.

**Deferred alternative — "snapshot-event" (NOT in scope).** Enrich the per-tick event in `events.jsonl` with the
full decision snapshot (u, busy, verdict, blocking gate) and have debug read the last snapshot to show true
`in_flight`. Deferred as YAGNI: `proactive_service_loop` currently hardcodes `busy=false` (no precise upstream
in-flight signal exists yet), so a snapshot would not buy true in-flight today; it adds a hot-path write and shows
"what was" not "what is now". Revisit when Hermes exposes a per-session in-flight primitive.

---

## 6. Time handling

Persisted timestamps are ISO-8601 UTC strings; the decision layer works in "minutes relative to now (now=0.0)".
Reuse `core.decision._minutes_between` for all display deltas so behavior is identical to the engine:
`last_tick_at` is intentionally forgiving (malformed/naive → treated as 0, never raises); the other timestamp fields
are validated on load. `now` must be timezone-aware UTC. The dump must not crash on a bad `last_tick_at`.

---

## 7. Testing (TDD)

- **`compute_readings` (unit, offline, no Hermes):** one crafted `State` per gate outcome — (a) below-threshold;
  (b) over-θ inside the silence window; (c) over-θ inside decline backoff; (d) clean URGE that launches; plus (e)
  over-θ + desire already `active` → gate=URGE but `would launch = no` (dedup); (f) stale `active`+pending older than
  `PENDING_TIMEOUT_MIN` → recovery flagged and verdict reflects the REJECT/backoff. Assert derived quantities
  (time-to-θ, window/backoff remaining) against hand-computed values. Assert the copy is untouched-on-disk (no
  commit) and the input `State` is not mutated.
- **Renderer (unit):** every section present with expected labels; `n/a` for empty categories; `<unreadable: …>`
  on a store that raises; the `u ≥ θ` branch renders the in_flight ⚠ caveat; the `u < θ` branch renders `in_flight
  n/a`.
- **Offline discipline:** tests run with Anthropic env unset (repo rule); explicit `encoding=` on any file I/O
  (PLW1514).

---

## 8. Out of scope (YAGNI)

- Snapshot-event / true runtime `in_flight` (§5) — future bead when a host primitive exists.
- Localization — later.
- Colour/ANSI, paging, JSON output mode.
- Fixing `duration_over_theta` — tracked separately as `lm-zhz`; here it is only **displayed** (labeled
  "tracked; not currently gating").

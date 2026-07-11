# Phase-2 close: last-wake-outcome in `/lifemodel debug` + adapter smoke probe

**Date:** 2026-07-11
**Beads:** `lm-9zj` (residual), `lm-dte`
**Epic:** `lm-fib` — Фаза 2, Контакт-драйв (closing)
**Status:** design, approved for spec review

## Goal

Close Phase 2 by finishing the two remaining genuine blockers, both small:

1. **`lm-9zj` residual** — surface *why the being stayed silent* in `/lifemodel debug`
   without log-diving. Owner's hard requirement: the model's actual reasoning must be
   visible. (The reachability half of the original bead is **dropped** — see below.)
2. **`lm-dte`** — a cheap pre-deploy smoke probe that catches adapter-shell regressions
   (a missing abstract method / construction drift) which `make check` structurally
   cannot see, because the dev venv lacks the Hermes `gateway` package.

Everything else once parented to `lm-fib` has been reparented out (`lm-37t` → Phase 6
`lm-0od`; the rest → hardening epic `lm-1w2`). After these two, `lm-fib` is closeable.

---

## Feature 1 — `lm-9zj`: last-wake-outcome block in `/lifemodel debug`

### Foundation already shipped (verified, do NOT rebuild)

The model's reasoning is already captured and viewable:

- `_log_proactive_reasoning` (`hooks.py:200`) writes the turn's chain-of-thought
  (`reasoning`) through the origin-trace `SpanBoundLogger`.
- `SpanBoundLogger._emit` (`log.py:195`) submits **every** event to the durable trace
  writer regardless of level — the DEBUG `proactive_reasoning` event is **always**
  persisted to `trace_events`. Level only gates the human console tail.
- `render_trace` (`trace_view.py:237`) already renders all of a trace's events, so
  `/lifemodel trace <id>` and `/lifemodel trace last N` already show the full reasoning.
- Privacy is already handled (`lm-jh6` closed): the trace carries the being's own CoT,
  not the raw `conversation_history` with owner PII.

**Consequence:** full reasoning lives in `/lifemodel trace`. `/lifemodel debug` shows
only the *decision* (per owner: "полное — в trace, в последнем решении — только решение").

### What we build

A compact **"Last wake outcome"** section in the `/lifemodel debug` dump:

- The outcome of the **last time the being actually woke cognition** — not the
  every-tick resting gate. One of:
  - `delivered` — a `proactive_delivery` span was emitted.
  - `act_gate_silent` — the async turn returned `[SILENT]` (a conscious silence).
  - `backstop_rate_limited` — the fail-closed rate backstop held fire.
  - `egress_unavailable` / `egress_failed` — a delivery was attempted and the channel
    did not accept it (this is the useful diagnostic the dropped reachability half was
    reaching for — we get it for free).
  - `energy_unaffordable` / `repeat_pure_longing` — the launch was gated at wake time.
- Plus its **timestamp** and **`trace_id`**, so the owner can run
  `/lifemodel trace <trace_id>` to read the reasoning behind it.
- **No reasoning rendered inside `debug`.**

**Excluded from "wake outcome"**: the mundane pre-wake gates that fire on quiet ticks
(`below_threshold`, `silence_window`, `in_flight`, `pending_proactive`, `decline_backoff`).
The being is *usually* below threshold; surfacing that as "the last decision" would be
noise. The existing drive/gates section of `debug` already explains the current resting
state (u vs θ, backoff, silence window). The new block is strictly about the last time
it *woke and decided*, which is complementary.

### Data source & taxonomy

Read from the **trace store** (`observability.sqlite`) — the system of record where all
outcome types are already unified as spans/events. The outcome markers:

- delivery → `proactive_delivery` span.
- post-wake suppressions → suppression spans whose `reason` ∈
  {`act_gate_silent`, `backstop_rate_limited`, `egress_unavailable`, `egress_failed`,
  `energy_unaffordable`, `repeat_pure_longing`}.

Take the **most recent** such marker by timestamp; read `reason`/label + `ts` + `trace_id`
(every suppression span is contract-guaranteed to carry `trace_id` — `SUPPRESSION_MIN_FIELDS`,
`core/suppression.py:88`, whose comment already reserves them for `/lifemodel debug`).

**Reuse, don't duplicate:** lift the existing read helpers from `trace_view.py`
(`connect`, the `trace_spans`/`trace_events` readers) into a small shared reader rather
than writing new SQL in `debug.py`. `debug.py` is already a read-only aggregating view and
`trace_view.py` already reads this same store Hermes-free/stdlib, so the coupling is
consistent with the codebase.

**Why not a stored `State` field:** the outcomes are produced in two places
(`proactive_tick` for egress/backstop; `_emit_async_outcome` for silent/delivered).
The trace store already unifies them; a `State` summary would need writes at 2–3 sites,
new schema fields, and would risk drift. Read from the one place the truth already lives.

### Rendering, placement, fail-soft

- New section near the timing/gates area of the dump (exact position during
  implementation). One-datum-per-line, Hermes-local timestamp, matching the existing
  `/status`-style layout (`lm-fib.3`/`lm-25t` conventions).
- **Fail-soft:** no `observability.sqlite` yet, an unreadable store, or no wake recorded
  yet → a single friendly line (`(no wake outcome recorded yet)` / `(trace store
  unavailable)`), never an exception. `/lifemodel debug` must stay a read-only dump that
  never crashes.

### Out of scope (dropped) — reachability health-stamp

Original bead part (3), "delivery-lane reachability in HEALTH", is **cut**:

- A down transport is not the plugin's fault and is not actionable by the being.
- It is transport-coupled (a Telegram-specific liveness stamp); the owner may use a
  different messenger — fragile.
- The actual diagnostic value ("the last outreach failed to deliver") is already covered:
  `egress_unavailable`/`egress_failed` are among the last-wake-outcome values above.

Tracked as dropped in the `lm-9zj` scope note.

### Testing

- Unit tests over the reader + renderer with a fake trace store containing:
  a delivered outcome; an `act_gate_silent`; an `egress_failed`; a `backstop_rate_limited`;
  and the "no wake yet" / "only below-threshold ticks" cases (→ friendly line).
- Assert the block shows outcome + ts + trace_id and **never** the reasoning text.
- Assert fail-soft on an unreadable/missing store.

---

## Feature 2 — `lm-dte`: cheap adapter smoke probe

### Problem

The being is hosted as `BeingAdapter(BasePlatformAdapter)`; `BasePlatformAdapter` lives in
Hermes `gateway`, absent from the dev/test venv. So `make check` (ruff/mypy/pytest under
`uv`) cannot see the real base class — mypy treats it as `Any`, pytest cannot instantiate
the adapter. There is a blind spot between "make check green" and "the being boots on the
gateway". It has already bitten once: a missing abstract method (`get_chat_info`, in the
then-installed gateway) passed every check and failed only at gateway connect with
`TypeError: can't instantiate abstract class`.

The Hermes docs' recommended validation (`tests/gateway/test_<plat>.py`) assumes in-tree
development where `gateway` is importable; as an external plugin we cannot run it off-host.
The docs prescribe no smoke test and no abstract-method check — the closest is a manual
"parity audit". The `get_chat_info` incident was almost certainly **version skew**: the
docs list only `connect`/`disconnect`/`send` as abstract, so the truth about "what is
abstract / does it construct" lives only in the **actually-installed gateway**, not our
assumptions or the docs. That is exactly what a probe against the real venv checks.

### What we build

`lifemodel/smoke.py` — a `run_smoke(...)` function plus a thin `__main__`, run by the
**Hermes venv** interpreter:

1. **Import** `BeingAdapter` (pulls the real `gateway.*`) — an import/shell regression
   surfaces here.
2. **Assert** `BeingAdapter.__abstractmethods__ == frozenset()` — every abstract method of
   the installed base is implemented; this is the automated parity-audit / version-skew
   guard that catches the class of bug that hit us.
3. **Construct** `BeingAdapter(fake_config, base_dir=<temp>, target={})` — catches
   config-shape / construction drift (the doc-endorsed "construction from config"). The
   constructor only sets fields + calls `super().__init__` (no loop start, no disk writes),
   so this is cheap and side-effect-free.
4. Exit `0` on success; on failure print the specific failure and exit non-zero.

**Explicitly NOT built:** the full `connect()`/`disconnect()` lifecycle (starts the brain
loop, acquires stores). Not in the docs, heavier, and higher false-failure risk. Dropped.

### Makefile wiring

- New `make smoke` target: resolve the Hermes venv python (default
  `~/.hermes/hermes-agent/venv/bin/python`, overridable via an env var), run
  `PYTHONPATH=<repo-parent> $PY -m lifemodel.smoke` so it imports the **working tree**
  against the real gateway. Non-zero exit fails the target.
- `make deploy` gains `smoke` as a **pre-flight dependency** — smoke runs before push, so
  a shell regression aborts the deploy *before* `hermes plugins update` + restart, closing
  the bead's "caught before update+restart" wording. (`make deploy` already targets
  `~/.hermes`, where the Hermes venv exists.)

### Safety

- Construction uses a **throwaway `tempfile.mkdtemp()`** base_dir, cleaned up after — never
  `~/.hermes`. Even though construction touches no disk today, this keeps the probe safe if
  it ever grows. The live being is never touched (cf. CLAUDE.md: integration checks use an
  isolated `HERMES_HOME`, never the live being).

### Testing

- The probe itself is **not** in `make check` (it needs the gateway venv).
- The pure logic is unit-testable in the dev venv: factor `run_smoke` to take the adapter
  class (or a construction thunk) so a test can pass a fake class with a non-empty
  `__abstractmethods__` and assert the probe fails, and an empty one and assert it passes —
  no `gateway` needed for that unit test.

---

## Risks / notes

- **Fake config shape (`lm-dte`)**: construction calls the real `BasePlatformAdapter.__init__`,
  whose config contract we don't fully control. If a minimal duck-typed fake proves too
  coupled, the fallback is abstract-methods-only (drop step 3); the version-skew guard
  (step 2) is the load-bearing part and stands alone.
- **Reader coupling (`lm-9zj`)**: extracting shared trace-store readers must not change
  `/lifemodel trace` behavior — keep `trace_view.py`'s output byte-identical; only lift
  helpers.
- Neither change is auto-committed/deployed by these tasks; deploy remains the owner's
  explicit action.

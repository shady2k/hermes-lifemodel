# lifemodel: the being as a supervised platform adapter

**Date:** 2026-07-06
**Status:** codex-reviewed + owner-approved (2026-07-06) — proceeding to plan

## Problem

The being's proactive brain runs as a self-spawned in-process asyncio task
(`egress_service.proactive_service_loop`) started via
`gateway_core.register_gateway_service`, which reaches into the gateway's private
`runner._gateway_loop` and is armed off the `on_session_start` hook. That hook can
fire in worker subprocesses that have no `GatewayRunner` (so
`default_runner_accessor()` returns `None` → `no_loop`), and nothing supervises the
task once started. So it fails to start or dies silently, and the being goes
permanently mute while a deliberately-neutered cron heartbeat (`tick.py`) only
stamps "alive" without ever waking (`tick.py:129`).

Evidence: no bundled Hermes plugin uses `register_gateway_service` (only core
`hermes_cli/profiles.py`); it is an unsupported pattern. 20+ bundled plugins host
long-lived gateway work as **platform adapters** (`register_platform` +
`BasePlatformAdapter.connect()`), supervised by the gateway's
`_platform_reconnect_watcher`.

## Goal

Host the autonomic brain the supported, supervised way — as a gateway platform
adapter — so the gateway owns its lifecycle and restarts it on failure. Delete the
self-hosted service, the cron heartbeat, and all two-brain machinery. No backward
compatibility, no dead code.

## GLOBAL CONSTRAINT — Hermes stays behind an abstract boundary

All Hermes coupling lives in a thin adapter layer and must NOT bleed into the
plugin core. The plugin is hexagonal and already partly obeys this; this work
extends the discipline and fixes existing leaks.

- **Core (zero Hermes imports, unit-testable with fakes):** `core/*`
  (physiology, drive, aggregation, cognition, coreloop, backstop), the
  decision+delivery tick logic (talks only to ports), `domain/*`, `state/*`,
  `ports/*`, `paths.py`, `log.py`.
- **Ports (abstract seams):** `ProactiveEgressPort` (deliver a reach-out to a
  target), clock. No new port required.
- **Hermes boundary (`adapters/` + thin wiring in `__init__.py`):** the ONLY place
  Hermes types/internals (`BasePlatformAdapter`, `MessageEvent`, `_set_fatal_error`,
  `register_platform`, runner internals) may appear.
- **Dependency direction:** `core → ports ← adapters → Hermes`. Enforced by the
  existing "no Hermes on sys.path" core unit tests.
- **Leak to fix:** `egress_service.run_proactive_tick` imports
  `gateway_core.reachin_available` — remove; readiness is implicit once the
  gateway calls `connect()`.

## Architecture

The being registers as a virtual platform:
`ctx.register_platform("lifemodel", label=..., adapter_factory=..., check_fn=lambda: True)`.

**Precedent:** `plugins/platforms/homeassistant` — a non-chat "platform" whose
`connect()` starts a background loop (`asyncio.create_task(self._listen_loop())`,
`adapter.py:131`) and injects turns from *external* events by building a
`MessageEvent` and calling `self.handle_message(...)` (`adapter.py:310-318`). Ours
injects from an *internal* event: drive-over-threshold.

Adapter (the only new Hermes-coupled module, e.g. `adapters/being_platform.py`):

- `connect(is_reconnect=False) -> bool`: start the brain loop as an asyncio task;
  return `True`. The loop, every `INTERVAL` (60s, from the core), calls the
  **Hermes-free** decision+delivery tick (today's `run_proactive_tick` logic,
  cleaned to be port-only: `coreloop.tick()` → backstop → `egress.reach_out(...)`).
- **Delivery:** on a surfaced launch, deliver into the user's real Telegram lane
  via the existing `ProactiveEgressPort` impl (`adapters/reachin.py::ReachInEgress`
  → `inject_proactive_turn`). The adapter is the supervised host; delivery
  targeting is unchanged and proven. The adapter's own `send()` is not the reach-out
  path (see open question 2).
- **Supervision (load-bearing — verified against source).** Gateway supervision is
  **notification-based, not task-based**: once `connect()` returns `True` the
  gateway stores the adapter as connected (`gateway/run.py:6896-6906`) and does NOT
  watch the tasks the adapter spawned. If the loop task dies by return/exception,
  nothing notices — a "connected" adapter is silently dead (the current failure
  mode, and exactly ntfy's bug: `plugins/platforms/ntfy/adapter.py:217-221`).
  Therefore the adapter MUST convert task-death into a fatal notification:
  `connect()` attaches `Task.add_done_callback(...)`; the callback, on *unexpected*
  completion, calls `self._set_fatal_error(code, msg, retryable=True)` then
  `_notify_fatal_error()`. That path is confirmed to drive a reconnect:
  handler installed pre-connect (`run.py:6875-6878`) → `_set_fatal_error` flips
  `_running`/records (`base.py:2682-2687`) → `_notify_fatal_error` invokes handler
  (`base.py:2721-2727`) → handler removes+`disconnect()`s+queues retryable in
  `_failed_platforms` (`run.py:3962-3980`) → watcher re-dials `connect(is_reconnect=True)`
  (`run.py:7701-7725`), backoff 30s→5min. Reference pattern: IRC
  (`plugins/platforms/irc/adapter.py:388-391`).
- `disconnect()`: mark an **intentional-shutdown** flag first, then cancel the loop
  task — so the done-callback distinguishes normal stop (do nothing) from crash
  (notify fatal) and never requeues during a clean shutdown.
- **Enable/config:** register with a `validate_config`/`is_connected` so
  `platforms.lifemodel.enabled: true` is treated as configured without fake tokens.
  The plugin must be enabled in `plugins.enabled` (our `plugin.yaml` is
  `kind: standalone`, which does NOT block `register_platform` —
  `hermes_cli/plugins.py:882-929` — but an un-enabled plugin's platform entry is
  silently skipped).

## Deleted (no compat, no dead code)

- `egress_service.py::proactive_service_loop` (the fragile host).
- `gateway_core.py::register_gateway_service` + `install_core_shim`
  (keep `inject_proactive_turn` — it is the delivery mechanism).
- `heartbeat.py` (cron registration + launcher shim).
- `tick.py` (neutered cron brain, `service_is_alive`, `SERVICE_LIVENESS_MAX_AGE`).
- The `on_session_start` deferred-arm block in `__init__.py`.
- State field `egress_service_alive_at` (liveness stamp; also drop its
  introspect reading at `core/introspect.py:190-193`); the `service_is_alive` /
  `SERVICE_LIVENESS_MAX_AGE` freshness logic; the inert `busy` /
  `reachin_available` yield-to-cron dance.

**KEEP `last_tick_at`** — it is the core's dt clock, not a cron artifact: read by
`core/aggregation.py:136-138`, `core/personality.py:42-49`,
`core/contact_neuron.py:31-37`, and stamped by `core/coreloop.py:113-115` inside
`tick()`. The adapter loop drives `coreloop.tick()`, which keeps it fresh.

## Kept / reused

- Core engine (`coreloop.tick()` pipeline, physiology/aggregation/cognition),
  `state/*`, `composition.py`, `ports/*`.
- Decision+delivery tick logic (today's `run_proactive_tick`, minus the liveness
  stamp and the `reachin_available` leak).
- Delivery: `ReachInEgress` + `inject_proactive_turn` into the Telegram lane.
- Ears: `pre_gateway_dispatch` + `post_llm_call` hooks (unchanged) that satiate the
  drive and stamp last-exchange state.

## Cron: external cron removed; supervision moves in-adapter

Remove the external cron heartbeat as a second brain: if the gateway is down the
being cannot reach Telegram anyway, so a cron "brain while the gateway is down"
cannot deliver. One clean brain, no two-brain race, no liveness handshake.

But gateway supervision is notification-based (see Supervision above), so cron's
liveness role is **not** replaced by "the gateway watches us" — it is replaced by
**in-adapter task supervision**: the loop's `Task.add_done_callback` is the local
watchdog that turns any loop death into a fatal notification → gateway reconnect.
Removing cron without that watchdog would reproduce the silent-death bug.

## Observability (make the being legible)

Rewrite the debug `TIMING` section into an honest `HEALTH` view driven by real
adapter state: is the adapter connected? is the loop alive (age of last tick)? is
there a fatal-error pending? — plus the existing would_wake/reason/backstop. The
health readings are computed in the Hermes-free core from state + injected values;
the adapter supplies the live values. No Prometheus (no scraper exists — YAGNI).

Operator-health caveat (from review): a connected virtual platform counts as a
live platform, so lifemodel being "connected" keeps the gateway out of its "no
connected messaging platforms remain" branch (`run.py:3986-4010`) and can mask a
real "Telegram down" at the coarse `/platform list` level (`slash_commands.py:1118`).
Acceptable (reach-in delivery depends on Telegram being up regardless), but the
being's own HEALTH view should surface delivery-lane reachability, not just "loop
alive", so a down Telegram is visible.

## Out of scope (separate tasks)

- `lm-67g`: `/lifemodel` subcommand list (help/discoverability).
- Per-layer event stats / metrics.

## Testing (TDD, RED-first)

- Core stays Hermes-free and unit-tested with fakes (fake clock, fake
  `ProactiveEgressPort`): a surfaced launch delivers via the port; a blocked
  backstop holds the desire; a failed delivery rolls pending back. (These largely
  exist for `run_proactive_tick`; keep green through the refactor.)
- New adapter tests with a fake `BasePlatformAdapter` surface: `connect()` starts
  the loop; loop death triggers `_set_fatal_error(retryable=True)`; `disconnect()`
  cancels cleanly. No real Hermes import in the tested seam.
- Work on a branch, never on `main`.

## Resolved by codex review (2026-07-06)

Verdict: **sound-with-changes.** The four open questions, resolved against source:

1. **Virtual platform — viable.** `register_platform` works regardless of manifest
   kind (`hermes_cli/plugins.py:882-929`); no remote-endpoint assumption once
   `connect()` returns `True` (`run.py:6896-6906`). Caveat: it counts as a connected
   platform (masking risk — see Observability). Must be enabled in `plugins.enabled`.
2. **Delivery lane — keep reach-in; never self-`handle_message`.** `handle_message`
   derives the session from `event.source`, and `build_source` stamps the adapter's
   OWN platform (`base.py:4604-4608`, `5432-5455`) — so injecting via our own adapter
   would create a lifemodel lane and reply through our `send()`. Correct path stays
   the current reach-in, which builds a Telegram source and calls the Telegram
   adapter (`gateway_core.py:93-115`); lifemodel `send()` is a clear no-op/failure.
   No cleaner public "originate on A, reply on B" API exists — it remains a private
   reach-in (kept under `adapters/`).
3. **Cron — remove as brain, replace with in-adapter task supervision** (not gateway
   auto-supervision). See the Cron section.
4. **Fatal→reconnect contract — holds if called explicitly.** Trace verified
   (`run.py:6875` → `base.py:2682/2721` → `run.py:3962/7701`). Adapter must call it
   itself; there is no implicit task-death detection.

### Required changes folded into this spec

- `connect()` starts the loop AND attaches `Task.add_done_callback` → on unexpected
  completion `_set_fatal_error(retryable=True)` + `_notify_fatal_error()`.
- `disconnect()` sets an intentional-shutdown flag so the callback doesn't requeue.
- Register `validate_config`/`is_connected` so `enabled: true` needs no fake tokens.
- Delivery via `ReachInEgress`→Telegram; lifemodel `send()` no-op; never
  self-`handle_message`.
- KEEP `last_tick_at` (core dt clock); delete only `egress_service_alive_at` +
  cron liveness machinery.
- Port-only tick must not retain `reachin_available`/runner checks; all reach-in
  stays under `adapters/`.

### Implementation-review nits (codex, verdict: ship-with-nits) — decisions

- **Applied:** track/log the `_notify_fatal_error()` task (a failed notify would
  otherwise silently strand the reconnect); call `_mark_disconnected()` on a clean
  `disconnect()` so status is not left stale-"connected".
- **Deliberate non-goal (YAGNI):** no watchdog for a "wedged-forever" synchronous
  tick. The tick is disk-read + pure computation; a truly hung sync tick would
  block the whole gateway event loop (loudly visible), not fail silently. The
  realistic death mode is an exception, which `SupervisedLoop` already converts to
  a fatal + reconnect. Revisit only if a hang is ever observed.
- **Untested seam (accepted):** the adapter shell (`connect`/fatal/`disconnect`,
  `register_platform`) is not off-host unit-tested (the test venv lacks `gateway`).
  All logic is in tested Hermes-free units; the shell is verified at runtime and a
  Hermes-venv integration probe is a possible later addition.

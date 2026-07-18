# Turn observability — the turn-hook path becomes a first-class traced unit (lm-hg7)

**Status:** design **v1** (after codex design-consult `019f7449` + owner decisions 2026-07-18). Ready for review → plan. Task **lm-hg7**. Closes the observability asymmetry between the richly-traced **tick** path and the log-only **turn-hook** path surfaced live 2026-07-18 while checking commitment-track.

> **One-line intent:** make a *turn* (a Hermes LLM exchange) a first-class **traced** unit — a per-turn root span with a child span per `pre_llm_call` injector and per tool — written to the SAME `observability.sqlite` as the tick, so the being's live turn is answerable from one durable place, not by grepping `agent.log`.

## 1. Goal

Give the being **one coherent observability surface** that covers BOTH execution paths, so debugging "what did the being perceive and decide this turn" is a single read, not a hunt across four surfaces (state / metrics / traces / logs) + a `grep`.

The concrete, non-negotiable framing (owner, 2026-07-18): **the primary consumer is the debugging *agent* reading the live being's *durable* stores from a shell** — `python3 -m lifemodel.activity …` over `~/.hermes/workspace/lifemodel/*.sqlite`, read-only. The in-being slash commands (`/lifemodel trace|why|stats|debug`) are secondary: the agent cannot invoke them (they run inside the gateway process), so we design the *reader function + its CLI façade*, not the command UX.

## 2. The gap (verified live, 2026-07-18)

- **Tick path — richly traced.** `CoreLoop.tick()` mints one root span per frame (`start_root`), a child span per component (`child_of`), stamps decision-attrs, and persists to `observability.sqlite` (`core/coreloop.py:198`/`:309`/`:380`). Universal + domain metrics auto-emit into the `MetricRegistry` singleton-per-base_dir. Readers: `trace_for_dir` / `why_for_dir` / `stats_for_dir` / `render_dump_for_dir`.
- **Turn-hook path — log-only.** The four `pre_llm_call` injectors (felt-state / genesis / belief / commitment, `hooks.py`) and the tools (`check_in` / `commitment` / `write_soul`) write ONLY structured logs. They mint **no span**, so `trace`/`why` cannot see them; metrics are patchy (`FELT_DISPLAY_TOTAL` for felt; `COMMITMENT_INJECTOR_OVERFLOW` only on overflow; belief/genesis emit nothing per turn).
- **Root cause — verified.** Only the `post_llm_call` registration holds a live writer: `_outcome_writer = acquire_trace_writer(observability_db_path(sdir))` and passes it into its graph (`__init__.py:574`/`:580`). Every OTHER turn hook — inbound `pre_gateway_dispatch` (`:609`), felt (`:629`), genesis (`:707`), belief (`:741`), commitment (`:757`) — calls `build_lifemodel(base_dir=sdir)` with **no** writer, so its graph gets `NULL_TRACE_SINK` (`composition.py:245`). The injectors *physically cannot* write spans as wired today.
- **Live evidence.** Reading the live `observability.sqlite` read-only from a shell: every span component is a tick component (`contact-aggregation`, `solitude-drive`, `affect-sense`, … + `None` tick-roots); the entire turn path (injectors + tools) has **0 spans**; the only 22 `tick IS NULL` rows are `hooks` — the `post_llm_call` proactive read-back. `trace last N` returns only ticks (drowned in minute-heartbeats).

## 3. Consumer & read interface

**Primary interface — a stable CLI over durable stores** (new): `python3 -m lifemodel.activity last N` (unified timeline) and `python3 -m lifemodel.activity turn <trace_id>` (single-turn tree), run from the repo root against `~/.hermes/workspace/lifemodel/`. This replaces the fragile `PYTHONPATH=… python -c "from lifemodel.trace_view import …"` one-liner with one documented command — the "one place" for the agent-debugger.

**Cross-process reality (drives the design):** the agent's CLI is a SEPARATE process from the live gateway. It sees only **durable** state on disk — `observability.sqlite` (spans), `metrics.sqlite` (sampled series), `lifemodel.sqlite` (vitals/BDI). The in-memory `MetricRegistry` NOW view and the `EventRing` freshness tail live in the gateway and are invisible to the CLI. Therefore: (a) the `TurnRecorder` MUST write spans to durable `observability.sqlite` (it will); (b) the turn metric MUST reach `metrics.sqlite` via the existing sampler, not live only in NOW; (c) reads are read-only, WAL-safe, concurrent with the live writer.

**Reader = code-over-data (decoupling win):** the `activity` reader is pure code reading sqlite. The agent iterates it **from the repo working tree over LIVE data with no deploy**. But turn *spans* only EXIST in the live store after the **recording** half is deployed (`git push` → `make deploy` → gateway restart); until then `activity` shows tick-only. The reader MUST tolerate old spans without the new `frame_kind` attr (16 MB of them live today).

## 4. Design spine — a `TurnRecorder`, not an `ExecutionFrame`

Introduce a **`TurnRecorder`**: a **process-lifetime service** built once in `register()`, holding the already-acquired live writer (`_outcome_writer`), the shared `tracer` (`StdlibTracer`), the shared `MetricRegistry`, and a bounded in-memory **turn ledger**. It is threaded into every turn hook (all four injectors, the tools, the `post_llm_call` observer). The injectors STOP rediscovering the graph via `build_lifemodel(base_dir=sdir)` for tracing — that is exactly why they have null writers today.

**Not an `ExecutionFrame` (codex).** `ExecutionFrame`/`run_frame` means a serialized snapshot → component pipeline → atomic commit under the one state-actor lock (`core/frame.py:115`). A turn is an *asynchronous observability scope* spanning host work across threads; it must **not** take that lock or masquerade as a core frame. `TurnRecorder` only reads/writes the trace store + metric registry + its own ledger — never `State`.

**Reuse the acquired writer, not `peek_trace_writer`.** `register()` already owns `_outcome_writer` for the plugin's lifetime (`__init__.py:574`); `peek_trace_writer` (`state/trace_store.py:958`) is for callers with no writer of their own — we have one.

## 5. Correlation & lifecycle

**Key = `(session_id, turn_id)`.** Hermes already passes `turn_id` to `pre_llm_call` (`agent/turn_context.py:525`) and `post_llm_call` (`agent/turn_finalizer.py:406`); the injectors discard it (`**_ignored`) today. It is the exact turn identity — no session-key adoption or heuristics needed.

**Lifecycle:**
1. **Open (first `pre_llm_call`).** A small first-registered `pre_llm_call` observer calls `ensure_turn(session_id, turn_id, model, platform, origin)`. It mints a root `TraceContext` and **persists the root span immediately** with `component="turn"`, `tick=NULL`, `ended_at=NULL`, `status=NULL`, and bounded root attrs (`frame_kind=turn`, `turn_id`, `session_id`, `origin=reactive|proactive`, `model`, `platform`). Persisting eagerly means a crash still leaves a discoverable parent (a child whose parent never landed is only *tolerated* by the reader, not *discoverable* by `trace last`).
2. **Injector children (each `pre_llm_call`).** Each injector opens ONE child of the turn root, does its work, and closes the child synchronously with its typed outcome + bounded attrs (see §7/§8). The child is opened **inside the injector factory**, so the existing `_record_observer_failure` exception branch closes it `failed` (an external decorator would misclassify a fail-soft `None`-return as an ordinary skip).
3. **Tool children (`pre_tool_call`/`post_tool_call`).** Open an open child keyed by `tool_call_id` on `pre_tool_call`; close it on `post_tool_call` with the host-provided `status`/`duration_ms` (`model_tools.py:974`/`:1176`). Prefer these host hooks over wrapping handlers — direct handler dispatch drops `turn_id`.
4. **Close (`post_llm_call`).** Write a **completion child** carrying the turn's actual final output + the current turn's available reasoning (see §8), then close the root `ok`.
5. **Open-root reconciliation.** `post_llm_call` is **not guaranteed** — Hermes fires it only for a non-empty, non-interrupted turn (`agent/turn_finalizer.py:395`). Empty / interrupted / cancelled / crashed turns leave the root open. So on the **next** `ensure_turn` for the same session, mark any older still-open turn `abandoned`/`failed`. No sweeper thread — bounded lazy cleanup on ledger ops (TTL + max entries).

**Threading.** Store only the **immutable** root `TraceContext` in the ledger (not a shared mutable `ActiveSpan`). Injectors / tools / `post_llm_call` may run on different threads within one turn; a single lock guards the ledger map + summary counters, and tool children are independently keyed by `tool_call_id`. Within a session turns are FIFO-serialized, so one live turn per session.

**Proactive continuation (the one correlation we keep).** A proactive turn skips `pre_gateway_dispatch` and carries a `pending_proactive_origin_traceparent`; `ensure_turn` **continues** that trace (via `start_root(upstream_traceparent=…)` / the existing `open_correlated_span`, `core/correlate.py:54`) so launch → injection → tools → output → resolution stay under one `trace_id` ("one proactive attempt = one trace"). **Reactive** turns get a fresh trace. We do **not** (v1) merge the reactive inbound EVENT frame and the turn into one trace — that needs threading a root out of `CoreLoop.tick()` (which mints internally, `coreloop.py:198`) and couples to a pre-auth hook that may never become a turn.

## 6. Storage

- **Same `observability.sqlite`.** No new table, no parallel turn store (that would re-create the original lifemodel's "another durable truth + another reader" mistake). The schema already fits: `trace_spans(trace_id, span_id, parent_span_id, component, tick, started_at, ended_at, status, attrs_json)` with **nullable `tick`** (`state/trace_store.py:115`).
- **Component naming:** root `turn`; injector children `turn.injector.{felt_state,genesis,belief,commitment}`; tool children `turn.tool.{check_in,commitment,write_soul}`; completion child `turn.completion`.
- **Stamp the tick roots too.** Persist `frame_kind=execution` + `trigger` on the CoreLoop root span (today `trigger` lives in `TickReport` but is not on the span, `coreloop.py:417`). Without it the unified timeline cannot distinguish a tick from a turn.
- **No `trace_correlations` rows for ordinary turns.** Unresolved correlation rows are protected from pruning **indefinitely** (`state/trace_store.py:315`); a crash would turn every abandoned turn into permanent retention. Open turn roots simply age out under the normal 14-day / 5000-trace / 256 MiB policy (`:262`). (Proactive turns reuse the launch's existing correlation; they add no new row.)

## 7. Metrics

- **One shared counter** `lifemodel_turn_injector_total{component, outcome}`, `component ∈ {felt_state, genesis, belief, commitment}`. Fits `MetricSpec`'s closed label set `{component, layer, phase, reason, outcome, model}` (`core/metrics.py:62`) exactly. Exactly **one increment per injector invocation**.
- **Retire `FELT_DISPLAY_TOTAL`** — it is precisely the felt injector's per-call verdict, now a duplicate. **Keep `COMMITMENT_INJECTOR_OVERFLOW`** — overflow is orthogonal to the primary outcome (a turn can both `surfaced` and overflow).
- **Typed outcome enums + tests.** The registry validates label *keys* but NOT *values* (`core/metrics.py:228`) — a typo silently forks a new series. Define constants:
  - felt: `light` · `not_warmed` · `not_salient` · `task` · `cooldown_unchanged`
  - genesis: `injected` · `born` · `carried_by_impulse` · `own_impulse` · `not_due` · `stale_identity` · `error`
  - belief: `surfaced` · `empty` · `unavailable` · `error`
  - commitment: `surfaced` · `empty` · `unavailable` · `error`
- **Not "free".** Parity does not fall out of closing a span — CoreLoop persists the span AND separately emits the metric (`coreloop.py:380`). The `TurnRecorder` deliberately does both at its per-injector close choke-point.

## 8. What the turn spans must carry (to actually answer "what did it decide")

- **Injector children:** the typed outcome + bounded decision-attrs — surfaced count, redacted record ids (belief/commitment ids, NEVER content — §9-style redaction preserved), latency. `overflow=true` on the commitment child when the cap trips.
- **Completion child:** the turn's **actual final output** + the **current turn's** available reasoning (not accumulated history) — otherwise the trace still cannot answer "what did the being decide". Today proactive-only reasoning capture lives in `_log_proactive_reasoning()` (`hooks.py:258`); generalize it onto the completion child for every turn.
- **Never** copy `conversation_history` / the full prompt onto a span (hot-path serialization + retention bloat).

## 9. Reader — `lifemodel.activity`

A new module `activity.py` with `activity_for_dir(base_dir, raw_args)` + a `__main__` CLI façade, read-only over the durable stores. Reuses `trace_view` internals for the tree.

- **`activity last N`** — a **unified timeline** of *activity units* (tick-frames + turn-frames) newest-first, one scannable line each with `frame_kind`, the key attrs, and a header exposing writer-drop health. It queries **by activity unit + `frame_kind`**, NOT raw `trace last N` (which is drowned in heartbeats and truncates attrs to 200 chars, `trace_view.py:262`/`:361`).
- **`activity turn <trace_id>`** — a single turn's full child tree (injectors + tools + completion), with belief/commitment ids **enriched from `lifemodel.sqlite`** (the raw span carries ids only).
- **State header** — current vitals/BDI (reuse `render_dump_for_dir`'s readings) so "current state + recent activity" is one read.
- **Robustness:** render `ended_at IS NULL` turns explicitly as **incomplete** (never as success); tolerate spans without `frame_kind` (old rows); tolerate missing parents (already handled, `trace_view.py:207`).
- **`why` stays object-causality.** `why_for_dir` walks BDI provenance in `lifemodel.sqlite`, not spans (`state_commands.py:902`) — injectors do NOT appear there "for free". A belief/commitment born in a turn already carries its `trace_id`, so a future `why`↔turn join is possible, but v1 does not build it.
- Existing `/lifemodel …` commands are untouched (or gain thin wrappers) — out of scope as a deliverable.

## 10. Fail-soft & overhead

- Every tracing call in its OWN `try/except` — instrumentation failure must never enter or replace the injector/tool result path. The injectors are already fail-soft via `_record_observer_failure`.
- Bounded span attrs (ids/counts, redacted). The writer is async/queued: caller submissions are `put_nowait`, queue-full drops are counted not blocked (`state/trace_store.py:509`), batched with a 200 ms idle commit (`:55`), and the sqlite connection lives only on the writer thread. The only synchronous cost is JSON-serializing bounded attrs before enqueue. Four injector spans + a root open/close + a completion child is negligible against an LLM turn.

## 11. Scope

- **v1 (this design):** the `TurnRecorder` service (ledger + open/child/close lifecycle, open-root reconciliation, proactive continuation); child spans for the four injectors (opened inside their factories, typed outcomes) + tools via `pre/post_tool_call` (incl. `write_soul`) + the completion child; `frame_kind`/`trigger` stamps on both turn and tick roots; the unified `lifemodel_turn_injector_total{component,outcome}` metric (retire `FELT_DISPLAY_TOTAL`, keep overflow); the `lifemodel.activity` reader + CLI.
- **Deferred (honest — follow-ups):** merging the reactive inbound EVENT frame + turn under one `trace_id`; a `why`↔turn-span join; an indexed turn table for "find a turn by id across many sessions"; thin `/lifemodel` wrappers over `activity`; any OTel/contextvar/distributed-tracing expansion.

## 12. Testing approach

- **`TurnRecorder` lifecycle** (`test_turn_recorder.py`, new): `ensure_turn` persists a root with `ended_at=NULL`/`frame_kind=turn`; a second `ensure_turn` for the same session marks the prior open root `abandoned`; a proactive `ensure_turn` continues the origin traceparent (same `trace_id`), a reactive one mints a fresh trace; the ledger is bounded (TTL + max) with lazy cleanup; concurrent injector/tool closes on different threads don't corrupt the map (lock); a raising trace call is swallowed and never reaches the caller.
- **Injector children** (extend each injector's test): every branch closes exactly one child with its typed outcome + one `lifemodel_turn_injector_total` increment; a fail-soft `None`-return still closes the child (`failed` on a raise, the typed skip otherwise); attrs carry ids/counts, never content.
- **Tools** (`test_tool_spans.py`): `pre/post_tool_call` open/close a `turn.tool.*` child keyed by `tool_call_id`; concurrent/repeated calls to one tool don't collide; `write_soul` is instrumented.
- **Stamps** (extend `test_coreloop`/`test_trace_store`): the tick root persists `frame_kind=execution` + `trigger`.
- **Metric** (`test_turn_metrics.py`): outcome enums are closed constants; `FELT_DISPLAY_TOTAL` is gone and felt now increments the unified counter; overflow still increments its own counter.
- **Reader** (`test_activity_view.py`): `activity last N` interleaves tick + turn units newest-first and filters by `frame_kind` (not drowned in heartbeats); `activity turn <id>` renders the child tree with enriched ids; an `ended_at=NULL` turn renders **incomplete**; a span without `frame_kind` (old row) does not crash the reader; the reader is read-only and works cross-process (no live registry/ring).
- **Real-code sim** (`test_turn_observability_harness.py`): drive a reactive turn through the real injectors + a tool via the `TurnRecorder`, then read the same durable store back and assert the turn root + injector/tool/completion children + the metric all landed — end-to-end, the way the agent will actually read it.

## 13. Decisions resolved (owner, 2026-07-18)

- **(a)** Primary consumer = the debugging **agent** reading durable stores from a shell; slash-command UX is not a deliverable ("ты их всё равно не можешь вызвать").
- **(b)** All three mental models are wanted, folded into ONE reader: current-state + timeline (`activity last N`) and per-turn deep-dive (`activity turn <id>`).
- **(c)** The reader is a **CLI façade** (`python -m lifemodel.activity`), not another `-c` one-liner.
- **(d)** Recording + reading ship as needed; slicing is the orchestrator's concern, not the owner's.

## 14. Design consult (codex, 2026-07-18) — adjudicated

Codex (thread `019f7449`) **backed the spine** (turn = a first-class trace scope, one store, one reader) and corrected three of my premises — all **verified in code and accepted**:

- **The inbound EVENT frame is not durably traced today** (its graph gets `NULL_TRACE_SINK`, `__init__.py:609`) — folded into §2/§4 (the writer-wiring root cause).
- **Ordinary `post_llm_call` does not run an ASYNC_COMPLETION frame** (only a pending proactive turn does, `hooks.py:464`) and **is not guaranteed to fire** — folded into §5 (open-root reconciliation).
- **`why` will not cover injectors "for free"** (it walks BDI provenance in `lifemodel.sqlite`, not spans) — folded into §9.

Sharpenings accepted: `TurnRecorder` not `ExecutionFrame` (§4); key `(session_id, turn_id)` not session-key (§5); start at first `pre_llm_call` not `pre_gateway_dispatch` (§5); reuse `_outcome_writer` not `peek_trace_writer` (§4); open children **inside** the injector factories (§5); tools via `pre/post_tool_call` incl. `write_soul` (§5); one `turn_injector_total{component,outcome}` with typed enums + tests, retire `FELT_DISPLAY_TOTAL`, keep overflow (§7); no `trace_correlations` for ordinary turns (§6); stamp `frame_kind`/`trigger` on both roots (§6); completion child records final output + reasoning (§8); bounded lazy ledger cleanup, no sweeper (§5). **Cuts adopted:** no separate turn DB/table; no EVENT+turn merge in v1; no session-key-only correlation; no sweeper thread; no new top-level command; no full-history/prompt in attrs; no OTel expansion (§11).

## 15. Open questions (small — for the plan)

- **Which module registers the "open" observer.** A dedicated first `pre_llm_call` observer that calls `ensure_turn` before the injectors, or fold `ensure_turn` into the first-registered injector (felt-state, which fires every turn)? Leaning a dedicated observer for a clean single responsibility.
- **`activity last N` unit selection.** Does the reader read root spans and group children, or read a lightweight roots-index? Whichever renders the timeline without loading every child of every heartbeat — decide against the live store's size in the plan.
- **Where the completion reasoning comes from for a reactive turn.** Confirm the `post_llm_call` kwargs expose the current turn's reasoning/output the same way `_log_proactive_reasoning` gets it; if not, the completion child carries final output only.

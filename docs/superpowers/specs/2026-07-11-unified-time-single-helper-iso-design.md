# Unified Time — one helper, UTC storage, ISO-8601 everywhere

**Status:** design (codex-reviewed, revised) → implementation
**Bead:** lm-fib.10 (epic)
**Decision (owner):** Option A — store **UTC**, display in **Hermes/owner TZ**; unify all DB time **columns** to **ISO-8601 UTC TEXT** (drop epoch columns); **one single time helper**.

## 1. Why

- **Format split across DBs:** `metrics.sqlite` stores time as INTEGER epoch; `lifemodel.sqlite` and `observability.sqlite` use ISO-8601 TEXT.
- **Dual columns (single-source violation):** `lifemodel.sqlite` stores each instant twice — ISO `x_at` TEXT **and** redundant `x_at_epoch` INTEGER; `sqlite_store.py` does *all ordering/expiry on the epoch columns*.
- **No single helper:** ~98 direct time call-sites — scattered `datetime.now(UTC)`, `.isoformat()`, `fromisoformat`, `domain/memory.epoch_ms`, `time.time()` (sampler), `_utcnow()` (trace), `resolve_owner_tz`, `core/timeutil.minutes_between`.

Storage TZ is already uniform UTC (correct, stays). Gateway logs are local MSK — Hermes' logging framework, a **display** concern, **out of scope** (do not conflate UTC-in-DB with MSK-in-logs).

## 2. Format decision: ISO-8601 UTC TEXT (for DB columns)

Chosen because 2 of 3 DBs already use it, the being's durable state must be human-inspectable, volumes are modest, and a **normalized** serializer makes TEXT ordering provably correct. Trade accepted: TEXT range-scans marginally slower than INT — fine at this scale.

**Scope note (codex #6):** "drop epoch" applies to **DB time COLUMNS only**. Epoch remains legitimate as a **metric VALUE** (e.g. `BRAIN_LAST_TICK_EPOCH`, a gauge whose value is epoch seconds) — those keep working via a helper `to_epoch_seconds(dt)`. We are not banning epoch as a concept, only as a storage column duplicate.

## 3. The one canonical time helper

Two layers (hexagon: core Hermes-free; `now()` is the ONE system-time read, a port):

1. **Source of "now" — the ClockPort ONLY.** `adapters/clock.py:SystemClock.now() -> datetime.now(UTC)` is the single place system time is read. Core takes `ctx.now`; adapters/threads use an **injected** clock. **Threads that currently self-source wall time — the metrics sampler (`time.time()`) and the trace writer (`_utcnow()`) — must be given the clock (or a `now: datetime` arg), not read system time directly** (codex #3/#5).
2. **Pure format/parse/display — `core/timeutil.py`** (Hermes-free, no I/O). Exactly these, and nothing else in the codebase generates/parses time strings:
   - `to_iso(dt) -> str` — **reject tz-naive** (raise), convert to UTC, return `dt.astimezone(UTC).isoformat(timespec="microseconds")`. This yields fixed-width `YYYY-MM-DDTHH:MM:SS.ffffff+00:00` (µs always 6 digits — defeats Python's omission of `.000000`), lexically sortable. The serializer for **all** time columns and any time string written.
   - `from_iso(s) -> datetime` — **strict** parse to aware UTC (raise on malformed). The one storage parser.
   - `to_epoch_seconds(dt) -> float` — for legitimate epoch-VALUED metrics only (codex #6).
   - `to_display(dt_or_iso, tz) -> str` — owner-local render for human surfaces; `tz` injected from the adapter (`resolve_owner_tz()`). **Fail-open** (codex #7): accepts a datetime or an ISO string; on a malformed/naive stored value it returns the raw value + logs, so a bad row never blanks the debug view. This is the ONLY fail-open path; `from_iso` stays strict. Keep `minutes_between`.
   - **Remove** `domain/memory.epoch_ms` and the `*_epoch` parse helpers.

**Normalization invariant (load-bearing).** Correctness of ordering/expiry now rests on `to_iso`. Tests MUST assert the regex `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$` and correct TEXT sort for `microsecond ∈ {0,1,500000,999999}`, whole-second vs sub-second, `datetime.min/max` (tz=UTC), and mixed inputs — TEXT order == datetime order.

## 4. DDL unification + every consuming query

Being resettable → **no data migration**. But do NOT rely on `CREATE TABLE IF NOT EXISTS` (an old file keeps the old columns). Each store bumps its own `SCHEMA_VERSION` and, on a version/shape mismatch, takes a **destructive fresh-DB path** (codex #4/#10). Acceptance runs `PRAGMA table_info` on all three DBs after boot.

**`lifemodel.sqlite`** (state/sqlite_store.py, domain/memory.py):
- DROP `created_at_epoch`, `updated_at_epoch`, `expires_at_epoch` (memory_records) and `updated_at_epoch` (runtime_state). Index `idx_memory_records_expires_at_epoch` → `idx_memory_records_expires_at` on `expires_at` (TEXT).
- **Normalize on write** (codex #1): caller-provided `expires_at` (and every `_at`) is passed through `to_iso` before storage — not just validated. No raw caller strings reach a column.
- Ordering keys: `updated_at_epoch DESC`→`updated_at DESC`, `created_at_epoch DESC`→`created_at DESC`.
- Expiry sweep: `expires_at_epoch <= ?`→`expires_at <= to_iso(now)`.
- **`read_pressure_index(now)`** (sqlite_store.py:~735) — currently `expires_at_epoch IS NULL OR expires_at_epoch > ?` → `expires_at IS NULL OR expires_at > to_iso(now)`; preserve strict `>` active / `<=` expired (codex #2).
- **`summarize_pressure_index()`** (domain/memory.py:~301) — rewrite off `epoch_ms` to compare via `from_iso`/normalized ISO, same `>`/`<=` semantics (codex #2).

**`metrics.sqlite`** (state/metrics_store.py) — a small API redesign, not just a column flip (codex #3):
- `metric_samples.ts`, `metric_defs.created_at/updated_at` INTEGER → TEXT ISO; keep index `ix_samples_name_label_ts`.
- `MetricSample.ts: int` → `str` (or add a parsed `datetime`); remove all `int(ts)` casts (read_samples, `_delete_oldest_ts_cohort`'s `MIN(ts)` cast).
- `MetricsSampler.sample_once(ts: int)` → accept `datetime`/ISO; the sampler stamps **one `ts` per sample cycle** and reuses it for every row in that cycle (µs would otherwise make each row a unique cohort and break whole-snapshot pruning).
- Retention: `cutoff = to_iso(from_iso(now) - timedelta(seconds=max_age))`, `DELETE WHERE ts < ?`. Cohort prune keeps deleting by exact `ts`.
- Sampler thread gets the injected clock (see §3.1).

**`observability.sqlite`** (state/trace_store.py) — columns already TEXT, but (codex #9): normalize at the `submit_*`/enqueue boundary via `to_iso` (not only at SQL apply); retention's lexical `MIN(ts)` is only safe once all rows are normalized. Fix `_parse_ts` to stop silently treating naive as UTC for **new** writes (reject/normalize at ingress); the reader may stay defensive for legacy rows but new writes are always normalized.

## 5. Route all time through the helper + enforce

- **Audit-then-replace** (codex #8): before touching the ~98 sites, classify each — `source` / `storage-serialize` / `parse-strict` / `parse-defensive/display` / `metric-epoch-value` / `test-fixture` — then replace per class. Prevents drift (sites that truncate to seconds for display, IDs embedding `isoformat()`, fixtures on int `ts`).
- **Lint test** (AST, like `test_no_absolute_self_imports.py`), precise (codex #5): in runtime dirs ban `datetime.now(`, `datetime.utcnow(`, `datetime.fromtimestamp(`, `datetime.timestamp(`, and `.isoformat(`/`.timestamp(` where the receiver is a `datetime` or a name in `{now, dt, *_at, ts, started, ended}`. Allowlist: `adapters/clock.py` (only `datetime.now(UTC)`), `core/timeutil.py` (isoformat/fromisoformat/timestamp). The sampler/trace threads are NOT allowed to self-source wall time — they route through the injected clock + `to_iso`.

## 6. Acceptance
1. DDL: every time column in all three DBs is TEXT; zero `*_epoch` columns; `PRAGMA table_info` confirms after boot on fresh DBs.
2. `to_iso` regex-normalized; TEXT sort == datetime sort across the tricky set (§3); `to_iso` rejects naive.
3. Ordering/expiry regressions (SAME results as the old epoch logic): `updated_desc`/`created_desc`, memory expiry sweep, **`read_pressure_index`**, **`summarize_pressure_index`** — strict `>`/`<=` preserved.
4. Metrics: sampler stamps one ISO `ts` per cycle; retention deletes `ts < cutoff_iso`; whole-snapshot cohort prune still works; `read_samples` returns ISO; `BRAIN_LAST_TICK_EPOCH` still emits epoch-seconds via `to_epoch_seconds`.
5. Round-trip: `from_iso(to_iso(dt)) == dt` (µs); `to_iso(from_iso(s)) == s` for normalized `s`.
6. `to_display` renders owner-local and is fail-open on a bad row; `/lifemodel debug` still shows `+03:00`.
7. Lint fails on a planted `datetime.now()` / datetime `.isoformat()` in a runtime file (prove by temporary edit); no false-positive on non-datetime `.isoformat()` / the epoch-valued metric.
8. `make check` green. Deploy (fresh/reset DBs): being connects + ticks; all three DBs on disk show normalized ISO, no `*_epoch`.

## 7. Non-goals
- Gateway log timezone (Hermes framework). A general datetime library. Preserving epoch **columns**. (Epoch metric *values* stay.)

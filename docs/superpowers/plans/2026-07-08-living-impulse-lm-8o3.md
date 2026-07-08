# Living Impulse (lm-8o3 / lm-8o3.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagents MUST follow superpowers:test-driven-development for every task.

**Goal:** Replace the synthetic, drive-only wake-packet ("Внутри у тебя сейчас: {feeling}") with a first-person *situational brief* sourced from real state (human-worded elapsed since last exchange, recent-rebuff tone, energy restraint, an orientation to mine in-context history, graceful degradation on a history-less wake), and add a first-class unanswered-outbound gate so the being does not nag after one unanswered pure-longing bid.

**Architecture:** Two hexagon layers, no new ports. Slice 1 stays in the COGNITION framing layer: a word-only humanizer in `core/timeutil.py`, an enriched `core/wake_packet.py::build_wake_packet`, and one wiring change in `core/cognition.py` to thread already-available `ctx.state` fields + `ctx.now`. Slice 3 (bead lm-8o3.1) adds one `State` scalar (`unanswered_outbound_count`) with a bounded increment-on-longing-send / reset-on-exchange rule and a HOLD gate in `core/aggregation.py`, overridable only by a top-down (thought-crystallized) high-value reason. Slice 2 is a verify-not-rebuild pass over the already-shipped `STRONG_LONGING` Rubicon gate plus a small deterministic jitter.

**Tech Stack:** Python 3 **stdlib only** (runtime runs inside Hermes's venv — no third-party runtime deps). Dev tooling: `uv`, `ruff`, `mypy --strict`, `pytest`. Russian is the content language for all being-facing strings.

## Global Constraints

- **stdlib-only at runtime.** No new imports beyond the Python standard library in any `core/`/`state/` module. (Dev/test may use pytest.)
- **No raw numbers in the wake packet.** The assembled proactive prompt MUST contain zero digit characters (`re.search(r"\d", prompt)` is `None`). This invariant is asserted in `tests/test_wake_packet.py`, `tests/test_projection.py`, and `tests/test_cognition.py` and MUST stay green. All elapsed/energy/tone framing is word-only.
- **Content language: Russian.** Every new being-facing string is Russian, matching the tone of existing `GUIDANCE`/`RECENT_THOUGHTS_HEADER`.
- **Additive state only.** New `State` fields carry a default and load via the existing tolerant `from_dict` (missing key → default). Bump `SCHEMA_VERSION` only when a field is added (Slice 3).
- **Mutation goes through intents.** State changes are emitted as `UpdateState(changes={...})` intents from a component's `step`, never written directly (snapshot → intents → atomic commit). Records are emitted as `PutRecord`.
- **Determinism.** No wall-clock reads inside pure builders; time enters as an explicit `now: datetime` / ISO string argument. Any jitter is seeded deterministically (e.g. off `correlation_id`), never `random.random()`.
- **Gate command:** `make check` (runs `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy -p lifemodel`, `uv run pytest`). Single file: `uv run pytest tests/test_x.py -v`. Auto-format: `make fmt`.

---

## File Structure

- `core/timeutil.py` — **modify.** Add `humanize_elapsed(minutes: float | None) -> str`, a word-only Russian rendering of an elapsed duration. Sits beside the existing `minutes_between`.
- `core/wake_packet.py` — **modify.** Extend `build_wake_packet` with optional situational-context kwargs and a `render_situational_brief` helper; assemble the brief into the prompt.
- `core/cognition.py` — **modify.** Thread `state.last_exchange_at`, `state.decline_count`, `state.energy`, and `ctx.now` into the `build_wake_packet` call.
- `state/model.py` — **modify (Slice 3).** Add `unanswered_outbound_count: int = 0`; bump `SCHEMA_VERSION` 1 → 2; validate in `from_dict`.
- `core/aggregation.py` — **modify (Slice 3).** HOLD when `unanswered_outbound_count >= 1` and the bid is pure-longing (drive-urge only, no top-down proposal); allow a top-down proposal to override.
- The increment/reset of `unanswered_outbound_count` lives wherever `proactive_send_log`/`last_exchange_at` are already mutated on FULFILL / on a genuine exchange (Slice 3 Task locates it from the dossier of the commit path).
- Tests: `tests/test_timeutil.py`, `tests/test_wake_packet.py`, `tests/test_cognition.py`, `tests/test_state_model.py` (or the existing State test file), `tests/test_aggregation.py`.

---

# Slice 1 — Situational brief (bead lm-8o3, the payoff)

### Task 1: `humanize_elapsed` — word-only elapsed renderer

**Files:**
- Modify: `core/timeutil.py`
- Test: `tests/test_timeutil.py`

**Interfaces:**
- Consumes: nothing new (pure function of a float).
- Produces: `humanize_elapsed(minutes: float | None) -> str` — a Russian phrase describing how long ago something was, containing **no digit characters**. `None` (or a non-positive/unknown elapsed) → the "we have not really talked yet" phrase.

Band mapping (minutes → phrase), boundaries in minutes:

| Range (min) | Phrase |
|---|---|
| `None` | `"вы ещё толком не общались"` |
| `< 60` | `"совсем недавно"` |
| `< 180` | `"пару часов назад"` |
| `< 480` | `"несколько часов назад"` |
| `< 1440` | `"сегодня, но уже порядочно прошло"` |
| `< 2880` | `"со вчерашнего дня"` |
| `< 5760` | `"уже несколько дней"` |
| `< 11520` | `"около недели"` |
| `< 43200` | `"не одну неделю"` |
| `>= 43200` | `"очень давно"` |

- [ ] **Step 1: Write failing tests**

```python
# tests/test_timeutil.py — add
import re
from lifemodel.core.timeutil import humanize_elapsed

def test_humanize_elapsed_never_talked() -> None:
    assert humanize_elapsed(None) == "вы ещё толком не общались"

def test_humanize_elapsed_bands() -> None:
    assert humanize_elapsed(0.0) == "совсем недавно"
    assert humanize_elapsed(59.0) == "совсем недавно"
    assert humanize_elapsed(60.0) == "пару часов назад"
    assert humanize_elapsed(179.0) == "пару часов назад"
    assert humanize_elapsed(180.0) == "несколько часов назад"
    assert humanize_elapsed(479.0) == "несколько часов назад"
    assert humanize_elapsed(480.0) == "сегодня, но уже порядочно прошло"
    assert humanize_elapsed(1439.0) == "сегодня, но уже порядочно прошло"
    assert humanize_elapsed(1440.0) == "со вчерашнего дня"
    assert humanize_elapsed(2880.0) == "уже несколько дней"
    assert humanize_elapsed(5760.0) == "около недели"
    assert humanize_elapsed(11520.0) == "не одну неделю"
    assert humanize_elapsed(43200.0) == "очень давно"

def test_humanize_elapsed_has_no_digits() -> None:
    for m in (None, 0.0, 60.0, 500.0, 1440.0, 3000.0, 50000.0):
        assert re.search(r"\d", humanize_elapsed(m)) is None

def test_humanize_elapsed_negative_is_recent() -> None:
    # a clock-skew negative elapsed is treated as "just now", never crashes
    assert humanize_elapsed(-5.0) == "совсем недавно"
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_timeutil.py -v` → FAIL (`humanize_elapsed` undefined).

- [ ] **Step 3: Implement**

```python
# core/timeutil.py — add (keep existing minutes_between)
_ELAPSED_BANDS: tuple[tuple[float, str], ...] = (
    (60.0, "совсем недавно"),
    (180.0, "пару часов назад"),
    (480.0, "несколько часов назад"),
    (1440.0, "сегодня, но уже порядочно прошло"),
    (2880.0, "со вчерашнего дня"),
    (5760.0, "уже несколько дней"),
    (11520.0, "около недели"),
    (43200.0, "не одну неделю"),
)


def humanize_elapsed(minutes: float | None) -> str:
    """Render an elapsed duration as a word-only Russian phrase (no digits).

    ``None`` means "no prior exchange to measure from". A negative value
    (clock skew) is clamped to "just now" rather than raising."""
    if minutes is None:
        return "вы ещё толком не общались"
    for upper, phrase in _ELAPSED_BANDS:
        if minutes < upper:
            return phrase
    return "очень давно"
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_timeutil.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add core/timeutil.py tests/test_timeutil.py
git commit -m "feat(core): word-only humanize_elapsed for the situational brief (lm-8o3)"
```

---

### Task 2: `render_situational_brief` + enriched `build_wake_packet`

**Files:**
- Modify: `core/wake_packet.py`
- Test: `tests/test_wake_packet.py`

**Interfaces:**
- Consumes: `humanize_elapsed` (Task 1); `minutes_between(a_iso, b)` (existing, `core/timeutil.py`).
- Produces: `build_wake_packet` new keyword-only params (all defaulted, so existing callers/tests still compile):
  `last_exchange_at: str | None = None`, `now: datetime | None = None`, `decline_count: int = 0`, `energy: float = 1.0`.
  New helper `render_situational_brief(*, last_exchange_at, now, decline_count, energy) -> str`.

Brief assembly rules (word-only, Russian):
- **Elapsed line** (always, when `now` is given): `f"Вы общались {humanize_elapsed(elapsed)}."` where `elapsed = minutes_between(last_exchange_at, now)`, or `None` when `last_exchange_at is None` (→ the "не общались" phrase). When `now is None`, the whole brief degrades to empty (back-compat for callers that pass no time — keeps old tests that call `build_wake_packet(value=, theta=, correlation_id=)` byte-comparable to a no-brief prompt).
- **Fresh-history line** (only when `last_exchange_at is None`): append `"Конкретики под рукой нет — не выдумывай повод: если сказать нечего настоящего, честно промолчи."` (graceful degrade — do NOT fabricate).
- **Tone line** (only when `decline_count > 0`): `"Недавно ты уже тянулся и промолчал — тем более не дави, потянись только если есть что-то настоящее."`
- **Energy line** (only when `energy < 0.3`): `"Сил сейчас немного — коротко и мягко, без длинных заходов."`
- **Orient line** (always, when the brief is non-empty): `"Прежде чем писать, вспомни, на чём вы остановились в прошлый раз — есть ли живая нить, которую хочется продолжить."`  — skip this line when `last_exchange_at is None` (nothing to mine; the fresh-history line already governs).

Prompt assembly order in `build_wake_packet`: `desire_frame line` → (blank) → `situational brief` (if non-empty) → (blank) → `GUIDANCE` → (blank) → `thoughts block` (if any).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_wake_packet.py — add
import re
from datetime import UTC, datetime
from lifemodel.core.wake_packet import build_wake_packet, render_situational_brief

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)

def test_brief_frames_elapsed_in_words() -> None:
    brief = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=1.0
    )
    assert "несколько часов назад" in brief  # 180 min
    assert "вспомни, на чём вы остановились" in brief

def test_brief_fresh_history_does_not_fabricate() -> None:
    brief = render_situational_brief(last_exchange_at=None, now=NOW, decline_count=0, energy=1.0)
    assert "вы ещё толком не общались" in brief
    assert "не выдумывай повод" in brief
    assert "вспомни, на чём вы остановились" not in brief  # nothing to mine

def test_brief_rebuff_tone_only_when_declined() -> None:
    hot = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=2, energy=1.0
    )
    cold = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=1.0
    )
    assert "не дави" in hot
    assert "не дави" not in cold

def test_brief_energy_restraint_only_when_low() -> None:
    low = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=0.1
    )
    assert "Сил сейчас немного" in low

def test_wake_packet_weaves_brief_and_keeps_no_digits() -> None:
    p = build_wake_packet(
        value=2.0, theta=1.0, correlation_id="c",
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=1, energy=0.1,
    )
    assert "несколько часов назад" in p.prompt
    assert "не дави" in p.prompt
    assert "Сил сейчас немного" in p.prompt
    assert re.search(r"\d", p.prompt) is None  # global invariant

def test_wake_packet_without_now_is_brief_free() -> None:
    # back-compat: no `now` -> no situational brief in the prompt
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="c")
    assert "Вы общались" not in p.prompt
    assert "вспомни, на чём вы остановились" not in p.prompt
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_wake_packet.py -v` → FAIL (`render_situational_brief` undefined / new kwargs unknown).

- [ ] **Step 3: Implement**

```python
# core/wake_packet.py
# add imports at top:
from datetime import datetime
from .timeutil import humanize_elapsed, minutes_between

# add helper:
def render_situational_brief(
    *, last_exchange_at: str | None, now: datetime | None, decline_count: int, energy: float
) -> str:
    """First-person situational context for the wake, word-only (no digits).

    Empty string when ``now`` is None (caller passed no time → no brief)."""
    if now is None:
        return ""
    lines: list[str] = []
    if last_exchange_at is None:
        # word-only, and MUST contain the lowercase substring the Task-2 test
        # asserts ("вы ещё толком не общались")
        lines.append("С ним вы ещё толком не общались.")
        lines.append(
            "Конкретики под рукой нет — не выдумывай повод: если сказать нечего "
            "настоящего, честно промолчи."
        )
    else:
        elapsed = minutes_between(last_exchange_at, now)
        lines.append(f"Вы общались {humanize_elapsed(elapsed)}.")
    if decline_count > 0:
        lines.append(
            "Недавно ты уже тянулся и промолчал — тем более не дави, потянись "
            "только если есть что-то настоящее."
        )
    if energy < 0.3:
        lines.append("Сил сейчас немного — коротко и мягко, без длинных заходов.")
    if last_exchange_at is not None:
        lines.append(
            "Прежде чем писать, вспомни, на чём вы остановились в прошлый раз — "
            "есть ли живая нить, которую хочется продолжить."
        )
    return "\n".join(lines)
```

Then extend `build_wake_packet`:

```python
def build_wake_packet(
    *,
    value: float,
    theta: float,
    correlation_id: str,
    thoughts: Sequence[Thought] = (),
    last_exchange_at: str | None = None,
    now: datetime | None = None,
    decline_count: int = 0,
    energy: float = 1.0,
) -> ProactivePrompt:
    desire_frame, projection_id = project_contact(value, theta=theta, seed=correlation_id)
    prompt = f"Внутри у тебя сейчас: {desire_frame}."
    brief = render_situational_brief(
        last_exchange_at=last_exchange_at, now=now, decline_count=decline_count, energy=energy
    )
    if brief:
        prompt = f"{prompt}\n\n{brief}"
    prompt = f"{prompt}\n\n{GUIDANCE}"
    if thoughts:
        prompt = f"{prompt}\n\n{render_thoughts_block(thoughts)}"
    return ProactivePrompt(
        prompt=prompt, projection_id=projection_id, correlation_id=correlation_id
    )
```


- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_wake_packet.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add core/wake_packet.py tests/test_wake_packet.py
git commit -m "feat(core): situational brief woven into the wake packet (lm-8o3)"
```

---

### Task 3: Wire real state + time into the cognition launch

**Files:**
- Modify: `core/cognition.py` (the `build_wake_packet(...)` call, ~lines 91-96)
- Test: `tests/test_cognition.py`

**Interfaces:**
- Consumes: enriched `build_wake_packet` (Task 2); `ctx.state.last_exchange_at`, `ctx.state.decline_count`, `ctx.state.energy`, `ctx.now` (all already on `TickContext`).
- Produces: no new symbol; the launched proactive prompt now carries the situational brief.

- [ ] **Step 1: Write failing test** — a wake launched with a known `last_exchange_at` a few hours before `NOW` puts the humanized elapsed into `launch.prompt`; and the existing byte-identical test is updated to reflect the new (brief-carrying) expected prompt.

```python
# tests/test_cognition.py — add
def test_launch_prompt_carries_situational_brief(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0,
                  last_exchange_at="2026-07-06T09:00:00+00:00", decline_count=0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert launch is not None
    assert "несколько часов назад" in launch.prompt   # NOW is 2026-07-06 12:00, 180 min
    assert "вспомни, на чём вы остановились" in launch.prompt
    assert re.search(r"\d", launch.prompt) is None

# UPDATE the existing byte-identical test (it must now build the expected prompt
# WITH the situational context the cognition path passes):
def test_launch_prompt_has_no_thoughts_block_without_thoughts(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0,
                  last_exchange_at="2026-07-06T09:00:00+00:00", decline_count=0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert RECENT_THOUGHTS_HEADER not in launch.prompt
    expected = build_wake_packet(
        value=2.0, theta=1.0, correlation_id=launch.correlation_id,
        last_exchange_at="2026-07-06T09:00:00+00:00", now=NOW, decline_count=0, energy=1.0,
    ).prompt
    assert launch.prompt == expected
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_cognition.py -v` → the new test FAILs (brief absent) and the updated byte-identical test FAILs against the old call.

- [ ] **Step 3: Implement** — thread the fields through the existing call:

```python
# core/cognition.py — replace the build_wake_packet call
packet = build_wake_packet(
    value=state.u,
    theta=1.0,
    correlation_id=correlation_id,
    thoughts=live_thoughts(ctx.objects),
    last_exchange_at=state.last_exchange_at,
    now=ctx.now,
    decline_count=state.decline_count,
    energy=state.energy,
)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_cognition.py -v` → PASS. Then run the whole suite: `make test` → GREEN (watch for any other test that hard-coded the old prompt).

- [ ] **Step 5: Commit**

```bash
git add core/cognition.py tests/test_cognition.py
git commit -m "feat(core): thread real state+time into the wake packet (lm-8o3)"
```

---

# Slice 2 — Longing→reason coupling (verify) + jitter (bead lm-8o3, tail)

### Task 4: Verify the STRONG_LONGING Rubicon gate already lowers the reason bar; add a note or a minimal tune

**Files:**
- Read: `core/thought_crystallization.py` (the `should_crystallize` gate + `STRONG_LONGING_*` / `STRONG_EVENT_*` constants), `core/aggregation.py` (the `top_down_admissible` path).
- Test: `tests/test_thought_crystallization.py` (existing).

**This is a verify-first task, not a rebuild.** The design's "sliding reason-threshold keyed to accumulated longing" was largely delivered by lm-27n.9. The deliverable is a decision + evidence, not new behavior unless a gap is proven.

- [ ] **Step 1:** Read `should_crystallize` and the `STRONG_LONGING_*` constants. Write down, in the task report, the exact rule by which accumulated longing (high `u` / strong-longing salience) lowers the actionability/other-regarding bar for crystallizing a contact desire. Confirm whether it is keyed to accumulated `u`/elapsed (design intent) or only to a single-tick salience.
- [ ] **Step 2:** Add ONE characterization test that pins the current behavior: a strong-longing thought crystallizes at a lower actionability than a neutral one (assert the existing thresholds). If the current gate already couples longing→reason: the test documents it, no production change — report "point (2) satisfied by lm-27n.9".
- [ ] **Step 3:** If (and only if) Step 1 proves the coupling is absent (longing does NOT lower the bar), STOP and report `BLOCKED` with the specific gap — do NOT invent a new threshold curve; the controller will escalate this design decision. Do not modify certified `sim/` gate params.
- [ ] **Step 4:** Commit the characterization test.

```bash
git add tests/test_thought_crystallization.py
git commit -m "test(core): pin longing→reason coupling in the Rubicon gate (lm-8o3)"
```

### Task 5: Deterministic launch jitter

**Files:**
- Modify: `core/cognition.py`
- Test: `tests/test_cognition.py`

**Interfaces:**
- Consumes: `correlation_id` (already built per tick from `ctx.now.isoformat()`).
- Produces: no signature change; a small deterministic gate that occasionally holds a launch by one tick so the being does not fire on a perfectly predictable clock edge.

Rule: derive a stable byte from `sha256(correlation_id)`; when `desire.state == ACTIVE` and all other gates pass, HOLD (return `[]`) on a small deterministic fraction of correlation-ids (e.g. `digest[0] % 5 == 0` → ~20% of ticks deferred). The desire persists (not resolved), so the next admissible tick launches. Word this in a comment as "human unpredictability, not a timer". Because it is seeded off `correlation_id` (which is `ctx.now.isoformat()`), it is fully deterministic and testable.

- [ ] **Step 1: Write failing test** — pick a `NOW` whose `correlation_id` hashes to the hold bucket, assert no `LaunchProactive`; pick one that does not, assert a launch. (The implementer computes which `NOW` values fall in/out of the bucket and hard-codes them.)
- [ ] **Step 2:** Run to verify fail.
- [ ] **Step 3:** Implement the seeded hold in `Cognition.step`, placed AFTER the receptivity/energy gates (so jitter never overrides a respect gate — it only ever delays an otherwise-permitted launch).
- [ ] **Step 4:** Run to verify pass; `make test`.
- [ ] **Step 5: Commit**

```bash
git add core/cognition.py tests/test_cognition.py
git commit -m "feat(core): deterministic launch jitter so wakes are not clock-edge predictable (lm-8o3)"
```

---

# Slice 3 — Unanswered-outbound gate + contact digest (bead lm-8o3.1)

### Task 6: `unanswered_outbound_count` state field

**Files:**
- Modify: `state/model.py` (add field, bump `SCHEMA_VERSION` 1→2, validate in `from_dict`)
- Test: `tests/test_state_model.py` (or the existing State test file — implementer locates it)

**Interfaces:**
- Produces: `State.unanswered_outbound_count: int = 0`; `SCHEMA_VERSION == 2`.

- [ ] **Step 1: Write failing tests** — a default `State()` has `unanswered_outbound_count == 0`; `from_dict` fills the default when the key is absent (old file loads clean); `to_dict`/`from_dict` round-trips a set value; a non-int present value raises `StateCorruptError`; `SCHEMA_VERSION == 2`.
- [ ] **Step 2:** Run to verify fail.
- [ ] **Step 3:** Add the field after `proactive_send_log`, bump `SCHEMA_VERSION`, add `unanswered_outbound_count=_as_int(data.get("unanswered_outbound_count", 0), "unanswered_outbound_count")` to `from_dict` (match the existing `_as_int` pattern).
- [ ] **Step 4:** Run to verify pass; `make test` (a schema bump can ripple — fix any test asserting `schema_version == 1`).
- [ ] **Step 5: Commit**

```bash
git add state/model.py tests/
git commit -m "feat(state): unanswered_outbound_count scalar, schema v2 (lm-8o3.1)"
```

### Task 7: increment on longing-send / reset on exchange

**Files:**
- Modify: the commit/verdict path where `proactive_send_log` is appended on FULFILL and where `last_exchange_at`/`decline_count` are reset on a genuine exchange (implementer locates via `grep -rn "proactive_send_log" core/ adapters/` and `grep -rn "last_exchange_at" core/` — Task begins by reporting the exact file:line of both mutation sites).
- Test: the test file covering that path.

**Interfaces:**
- Consumes: `State.unanswered_outbound_count` (Task 6).
- Rule: on a FULFILLED **pure-longing** proactive send (drive-driven, no top-down proposal backing it), emit `unanswered_outbound_count = state.unanswered_outbound_count + 1`. On any genuine inbound exchange (the same place `decline_count`/`last_exchange_at` reset), emit `unanswered_outbound_count = 0`. A top-down (thought-crystallized) send does NOT increment (it is a materially new reason, not a repeat longing bid).

- [ ] **Step 1: Write failing tests** — after a pure-longing FULFILL, count is 1; after a genuine exchange, count resets to 0; a top-down send does not increment.
- [ ] **Step 2:** Run to verify fail.
- [ ] **Step 3:** Implement via `UpdateState` intents at the located sites (never a direct write).
- [ ] **Step 4:** Run to verify pass; `make test`.
- [ ] **Step 5: Commit**

```bash
git commit -am "feat(core): track unanswered pure-longing outreach (lm-8o3.1)"
```

### Task 8: HOLD gate — no second pure-longing bid until engagement or a new high-value reason

**Files:**
- Modify: `core/aggregation.py` (the `create_desire` decision, ~lines 236-272) — or `core/cognition.py` launch gate, whichever the implementer's Step-1 analysis shows is the correct single chokepoint.
- Test: `tests/test_aggregation.py` (or `tests/test_cognition.py`).

**Interfaces:**
- Consumes: `State.unanswered_outbound_count`; the existing `drive_urge` vs `top_down_admissible` distinction in `aggregation.py`.
- Rule: when `unanswered_outbound_count >= 1`, a **pure-longing** urge (`drive_urge and not top_down_admissible`) does NOT create/activate a new contact desire (HOLD). A `top_down_admissible` proposal (a crystallized high-value reason) is still allowed — it overrides the gate. This sits alongside, never replaces, the fixed respect gates in `appraise_receptivity` (those still hard-veto independently).

- [ ] **Step 1:** Report the exact current `create_desire` predicate. Confirm where `top_down_admissible` and `drive_urge` are available together.
- [ ] **Step 2: Write failing tests** — with `unanswered_outbound_count == 1` and only a drive-urge (no proposal): no desire created; with `unanswered_outbound_count == 1` and a top-down proposal: desire created; with `unanswered_outbound_count == 0`: unchanged (drive-urge creates desire as before).
- [ ] **Step 3:** Run to verify fail.
- [ ] **Step 4:** Add the `unanswered_outbound_count >= 1 and not top_down_admissible` HOLD to the predicate.
- [ ] **Step 5:** Run to verify pass; `make test`.
- [ ] **Step 6: Commit**

```bash
git commit -am "feat(core): hold repeat pure-longing outreach until engagement or a new reason (lm-8o3.1)"
```

### Task 9: Surface the unanswered-bid state in the situational brief

**Files:**
- Modify: `core/wake_packet.py` (`render_situational_brief`), `core/cognition.py` (pass `unanswered_outbound_count`).
- Test: `tests/test_wake_packet.py`, `tests/test_cognition.py`.

**Interfaces:**
- Consumes: `State.unanswered_outbound_count`.
- Produces: `render_situational_brief`/`build_wake_packet` gain a `unanswered_outbound_count: int = 0` kwarg; when `>= 1`, the brief adds a word-only line, e.g. `"Ты уже потянулся и пока без ответа — не повторяйся ради самого жеста; пиши, только если появилось что-то по-настоящему новое."`

- [ ] **Step 1: Write failing tests** — brief contains the line when count ≥ 1, absent when 0; no digits; cognition threads the field through.
- [ ] **Step 2:** Run to verify fail.
- [ ] **Step 3:** Implement (add param + line; thread through cognition).
- [ ] **Step 4:** Run to verify pass; `make test`.
- [ ] **Step 5: Commit**

```bash
git commit -am "feat(core): surface the unanswered-bid state in the wake brief (lm-8o3.1)"
```

---

## Self-Review notes (for the controller, not a task)

- **Spec coverage:** design point (1) situational brief → Tasks 1-3; (2) sliding reason-threshold → Task 4 (verify, escalate if absent); (3) fixed respect gates → already in `appraise_receptivity`, untouched (Task 8 explicitly composes with it, does not replace); (4) unanswered-outbound gate → Tasks 6-8; (5) minimal contact-digest → collapses into the situational brief (Tasks 2/9) reading existing scalars + the one new `unanswered_outbound_count`; (6) jitter → Task 5.
- **No-digits invariant** re-asserted in every wake-packet/cognition test that touches the prompt.
- **Schema bump** (Task 6) is the one ripple risk — `make test` after Task 6 catches any `schema_version == 1` assertion elsewhere.
- **Certified code** (`sim/`) is never modified; Task 4 escalates rather than tuning drive params.
```

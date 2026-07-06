# Core Rebuild — Phase D1: Cognition Layer (the Voice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the COGNITION layer's *pure, isolatable* machinery (spec §13, model A): the **prompt-safety projection** (a woken contact-desire → a human, desire-framed phrasing — never raw numbers/timers), the **output-lint** (catch mechanical timer-justification + contentless filler in a would-be message), the **wake-packet builder** (desire-frame + guidance the being sees when it wakes proactively), and the **Cognition component** (when a desire is live and un-acted, gate on energy, then emit a `LaunchProactive` intent carrying the wake-packet). **No live cutover:** the `LaunchProactive` intent, the `post_llm` verdict feedback, and applying the output-lint at send-time are wired into Hermes in Phase E; here everything is exercised with fakes.

**Architecture (spec §13, model A):** Cognition does **not** call an LLM — it decides *when* to wake the being's native Hermes turn and *how* to frame the pull. The being's own turn is the act-gate (message = FULFILL, `[SILENT]`/`NO_REPLY` = REJECT), fed back by the `post_llm` hook in Phase E. The Cognition component runs each tick: if `desire_status == "active"` and no proactive turn is already in flight (`pending_proactive_id is None`), it reserves the energy for a proactive turn (spec §8; if unaffordable it holds — the emergent shutoff), builds a wake-packet, and emits `LaunchProactive(prompt, correlation_id)` while stamping `pending_proactive_id`/`pending_proactive_since` (idempotence — no second launch while one is pending). The **projection** maps the drive's value-band to synonymic human phrasings chosen pseudo-randomly by a seed derived from the `correlation_id` (deterministic, no `random`). The **output-lint** is a pure filter applied to the being's produced message at send-time (Phase E), built and unit-tested here.

**Tech Stack:** Python 3.11 stdlib-only (`hashlib` for the seeded choice); the C2 energy helpers (`reserve`, `cost_real`). `uv run ruff format/check`, `uv run mypy -p lifemodel`, `uv run pytest`.

## Global Constraints

- **Flat root-layout; core imports no Hermes.**
- **No raw mechanism in the wake-packet (spec §13, rev.6):** the packet carries a **desire-frame** phrasing + guidance, never pressure/threshold numbers or "N hours elapsed". Time/history *awareness* is fine (the native turn sees it); the barrier is on raw mechanism only.
- **Deterministic, reproducible (spec §2.8):** the projection's synonym choice is a pure function of the seed (`hashlib`, no `random`/`Math.random`/`uuid4`). The `correlation_id` is derived deterministically from `now` (injected clock).
- **Only cognition pays energy (spec §8):** the Cognition component's *check* is cheap; it **reserves** the proactive turn's energy before launching (via C2's `reserve`). If unaffordable → **do not launch** (hold the desire; emergent shutoff). No `if energy < X`.
- **Idempotent launch (Codex):** launch only when `desire_status == "active"` **and** `pending_proactive_id is None`; stamping `pending_proactive_id` prevents a second launch (a deduped still-active desire never re-launches).
- **Wake-packet phrasing language:** the being is a Russian-speaking companion, so the default desire-frame/guidance strings are **Russian content** (configurable/localizable later — they are data, not logic). Code, comments, identifiers stay English. Output-lint patterns are bilingual (the being's messages are Russian).
- **`mypy -p lifemodel` strict.**
- **Do NOT modify** `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`. Do NOT wire the CoreLoop into any live loop. Do NOT push/merge/touch `main`. `tests/sim/` must stay green.
- **Branch:** `core/rebuild`. One commit per task.

## File Structure

- Create `core/projection.py` — desire→human-phrasing projection (Task 1).
- Create `core/output_lint.py` — mechanical-timer/filler lint (Task 2).
- Create `core/wake_packet.py` — the `ProactivePrompt` builder (Task 3).
- Create `core/cognition.py` — `Cognition` component + `LaunchProactive` intent (Task 4).
- Modify `core/intents.py` (add `LaunchProactive`), `core/__init__.py` (re-exports), `composition.py` (register cognition).
- Tests: `tests/test_projection.py`, `tests/test_output_lint.py`, `tests/test_wake_packet.py`, `tests/test_cognition.py`, extend `tests/test_composition.py`.

**Interfaces produced (Phase E consumes):**
- `core/projection.py`: `project_contact(value: float, *, theta: float, seed: str) -> tuple[str, str]` → `(phrasing, projection_id)`.
- `core/output_lint.py`: `LintResult(ok: bool, reason: str)`; `lint_proactive(text: str, *, patterns: Sequence[str] = DEFAULT_MECHANICAL_PATTERNS) -> LintResult`.
- `core/wake_packet.py`: `ProactivePrompt(prompt: str, projection_id: str, correlation_id: str)`; `build_wake_packet(*, value: float, theta: float, correlation_id: str) -> ProactivePrompt`.
- `core/intents.py`: `LaunchProactive(prompt: str, correlation_id: str)`.
- `core/cognition.py`: `Cognition(*, fast_cost, send_cost, alpha, id="cognition")`.

---

### Task 1: Prompt-safety projection (desire → human phrasing)

**Files:**
- Create: `core/projection.py`
- Modify: `core/__init__.py`
- Test: `tests/test_projection.py`

**Interfaces:**
- Produces: `project_contact`.

**Behavior (spec §13):** map the contact drive's value to a **band** (light / medium / strong pull) and return a **synonymic human phrasing** for that band, chosen deterministically from the band's synonyms by a seed (`hashlib.sha256(seed)` → index). Returns `(phrasing, projection_id)` where `projection_id` identifies the exact phrasing chosen (for observability). **No numbers** appear in the phrasing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_projection.py
from __future__ import annotations

import re

from lifemodel.core.projection import project_contact

THETA = 1.0


def test_bands_map_to_distinct_phrasings() -> None:
    light, _ = project_contact(1.1, theta=THETA, seed="a")
    strong, _ = project_contact(3.0, theta=THETA, seed="a")
    assert light != strong


def test_choice_is_deterministic_in_seed() -> None:
    assert project_contact(2.0, theta=THETA, seed="corr-1") == project_contact(2.0, theta=THETA, seed="corr-1")


def test_different_seed_can_vary_phrasing() -> None:
    # across several seeds within a band, more than one synonym is reachable
    outs = {project_contact(2.0, theta=THETA, seed=f"s{i}")[0] for i in range(20)}
    assert len(outs) >= 2


def test_phrasing_contains_no_raw_numbers() -> None:
    for v in (1.1, 2.0, 3.5):
        phrasing, _ = project_contact(v, theta=THETA, seed="x")
        assert not re.search(r"\d", phrasing)  # no digits — never leaks values/hours


def test_projection_id_identifies_choice() -> None:
    phrasing, pid = project_contact(2.0, theta=THETA, seed="k")
    assert isinstance(pid, str) and pid
    # same seed+value -> same id
    assert project_contact(2.0, theta=THETA, seed="k")[1] == pid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_projection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.projection'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/projection.py
"""Prompt-safety projection: a woken drive → a human, desire-framed phrasing
(spec §13).

Raw affect never reaches the LLM. A drive value is bucketed into a band, and a
band maps to a small set of *synonymic* human phrasings; the choice is
pseudo-random but deterministic — a stable hash of the seed (the desire's
correlation id) — so a preamble is neither monotonous nor unreproducible. The
phrasing carries feeling, never numbers. Default strings are Russian (the being's
language); they are content, localizable later.
"""

from __future__ import annotations

import hashlib

# (low_inclusive_multiple_of_theta, synonyms) — bands over u/theta.
_CONTACT_BANDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (
        2.5,
        (
            "заметно соскучился — тянет написать первым",
            "давно хочется на связь, скучаешь по нему",
        ),
    ),
    (
        1.5,
        (
            "ловишь себя на мыслях о нём",
            "хочется услышать, как он там",
        ),
    ),
    (
        1.0,
        (
            "тихое желание побыть на связи",
            "лёгкая тяга черкнуть пару слов",
        ),
    ),
)


def _seed_index(seed: str, n: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def project_contact(value: float, *, theta: float, seed: str) -> tuple[str, str]:
    """Return ``(phrasing, projection_id)`` for a contact-drive value."""
    ratio = value / theta if theta else value
    for band_index, (low, synonyms) in enumerate(_CONTACT_BANDS):
        if ratio >= low:
            choice = _seed_index(seed, len(synonyms))
            projection_id = f"contact.b{band_index}.s{choice}"
            return synonyms[choice], projection_id
    # below threshold — no pull worth framing (defensive; cognition gates on wake)
    return "нет заметной тяги к контакту", "contact.none"
```
Re-export `project_contact` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_projection.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/projection.py core/__init__.py tests/test_projection.py
uv run ruff check core/projection.py core/__init__.py tests/test_projection.py
uv run mypy -p lifemodel
git add core/projection.py core/__init__.py tests/test_projection.py
git commit -m "feat(core): prompt-safety projection — desire→human phrasing, seeded, no numbers (spec §13)"
```

---

### Task 2: Output-lint (mechanical-timer + filler guard)

**Files:**
- Create: `core/output_lint.py`
- Modify: `core/__init__.py`
- Test: `tests/test_output_lint.py`

**Interfaces:**
- Produces: `LintResult`, `DEFAULT_MECHANICAL_PATTERNS`, `lint_proactive`.

**Behavior (spec §13, rev.6):** a pure filter over a candidate proactive message. It flags **mechanical self-justification** ("шесть часов тишины", "инициирую проверку", "6 hours of silence") and **contentless filler** ("мне нечего сказать", "nothing to add") — but **passes** natural human time references ("давно тебя не слышал"). Case-insensitive substring/regex over a bilingual default pattern list (patterns are injectable/extensible).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_output_lint.py
from __future__ import annotations

from lifemodel.core.output_lint import LintResult, lint_proactive


def test_passes_a_warm_natural_message() -> None:
    r = lint_proactive("Саш, привет! Давно тебя не слышал, как ты?")
    assert isinstance(r, LintResult)
    assert r.ok is True


def test_flags_mechanical_timer_justification() -> None:
    assert lint_proactive("Прошло шесть часов тишины, решил проверить.").ok is False
    assert lint_proactive("6 hours of silence detected — checking in.").ok is False


def test_flags_contentless_filler() -> None:
    assert lint_proactive("Молчу, мне нечего сказать, но решил написать.").ok is False
    assert lint_proactive("Nothing to add, just checking in.").ok is False


def test_flag_gives_a_reason() -> None:
    r = lint_proactive("инициирую проверку статуса")
    assert r.ok is False and r.reason


def test_natural_time_mention_is_not_flagged() -> None:
    # a human "it's been a while" must pass — barrier is on mechanism, not time
    assert lint_proactive("Сто лет не общались, скучаю по нашим разговорам.").ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_output_lint.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.output_lint'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/output_lint.py
"""Output-lint: a send-time safety filter on a proactive message (spec §13).

Catches the two anti-patterns of the old monolith — mechanical self-justification
by the clock/monitor, and contentless filler — without touching natural human
time references ("давно не виделись" is fine; "обнаружено 6ч тишины" is not).
Pure and language-agnostic: it matches a bilingual list of *mechanical* phrases,
so a warm message passes while a timer-narrated one is flagged. Applied at
send-time in Phase E; unit-tested here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Mechanical self-justification / filler markers (RU + EN). Deliberately narrow —
#: they target the *mechanism* narration, not any mention of time.
DEFAULT_MECHANICAL_PATTERNS: tuple[str, ...] = (
    r"\d+\s*(час|hour|минут|minute)\w*\s+тишин",
    r"\d+\s*(hours?|minutes?)\s+of\s+silence",
    r"тишин\w*\s+\d+",
    r"инициир\w*\s+проверк",
    r"检查|checking in\b",
    r"нечего\s+(сказать|добавить)",
    r"nothing\s+to\s+(say|add)",
    r"silence\s+detected",
    r"обнаружен\w*\s+\d",
)


@dataclass(frozen=True)
class LintResult:
    ok: bool
    reason: str = ""


def lint_proactive(
    text: str, *, patterns: Sequence[str] = DEFAULT_MECHANICAL_PATTERNS
) -> LintResult:
    """Flag mechanical timer-justification / contentless filler. Warm natural
    messages (incl. human time references) pass."""
    low = text.lower()
    for pat in patterns:
        if re.search(pat, low):
            return LintResult(ok=False, reason=f"mechanical_pattern:{pat}")
    return LintResult(ok=True)
```
Re-export `LintResult, DEFAULT_MECHANICAL_PATTERNS, lint_proactive` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_output_lint.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/output_lint.py core/__init__.py tests/test_output_lint.py
uv run ruff check core/output_lint.py core/__init__.py tests/test_output_lint.py
uv run mypy -p lifemodel
git add core/output_lint.py core/__init__.py tests/test_output_lint.py
git commit -m "feat(core): output-lint — flag mechanical timer/filler, pass natural time refs (spec §13)"
```

---

### Task 3: Wake-packet builder (`ProactivePrompt`)

**Files:**
- Create: `core/wake_packet.py`
- Modify: `core/__init__.py`
- Test: `tests/test_wake_packet.py`

**Interfaces:**
- Consumes: `project_contact` (Task 1).
- Produces: `ProactivePrompt`, `build_wake_packet`, `GUIDANCE`.

**Behavior (spec §13):** assemble the prompt injected into the being's proactive turn: the projected **desire-frame** + a fixed **guidance** block (own the wish; you know the time/history and may note it humanly, but you reach out because you *want* to, not because "N hours passed"; if there's nothing genuine, staying silent with `[SILENT]` is fine). Carries `projection_id` and `correlation_id` for observability. **No raw numbers.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wake_packet.py
from __future__ import annotations

import re

from lifemodel.core.wake_packet import GUIDANCE, ProactivePrompt, build_wake_packet


def test_packet_carries_desire_frame_and_guidance() -> None:
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="corr-1")
    assert isinstance(p, ProactivePrompt)
    assert GUIDANCE in p.prompt
    # the desire-frame phrasing for this band appears in the prompt
    assert "мыслях о нём" in p.prompt or "услышать, как он" in p.prompt
    assert p.correlation_id == "corr-1"
    assert p.projection_id.startswith("contact.")


def test_packet_has_no_raw_numbers() -> None:
    p = build_wake_packet(value=3.4, theta=1.0, correlation_id="c")
    assert not re.search(r"\d", p.prompt)  # never leaks the value/hours


def test_guidance_permits_silence_and_owns_the_wish() -> None:
    # the guidance must invite [SILENT] and frame the motive as desire, not a timer
    assert "[SILENT]" in GUIDANCE
    assert "хочешь" in GUIDANCE.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wake_packet.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lifemodel.core.wake_packet'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/wake_packet.py
"""The wake-packet: the desire-framed prompt for a proactive turn (spec §13).

Injected into the being's native Hermes turn (model A). It carries the projected
*desire-frame* (the feeling, never numbers) plus fixed *guidance*: own the wish;
you know the time and how the last talk ended and may note it humanly, but you
reach out because you WANT to — not because a timer fired; if there is nothing
genuine to say, staying silent is right. Default strings Russian (content).
"""

from __future__ import annotations

from dataclasses import dataclass

from .projection import project_contact

GUIDANCE = (
    "Это твоё собственное желание потянуться — присвой его. Ты знаешь, сколько "
    "прошло и чем закончился прошлый разговор; можешь по-человечески это учесть, "
    "но тянешься ты потому что ХОЧЕШЬ, а не потому что «прошло сколько-то времени». "
    "Есть что-то настоящее сказать — скажи тепло и коротко. Если это будет "
    "наполнитель — совершенно нормально промолчать (ответь [SILENT])."
)


@dataclass(frozen=True)
class ProactivePrompt:
    prompt: str
    projection_id: str
    correlation_id: str


def build_wake_packet(*, value: float, theta: float, correlation_id: str) -> ProactivePrompt:
    """Build the proactive-turn prompt from the projected desire-frame + guidance."""
    desire_frame, projection_id = project_contact(value, theta=theta, seed=correlation_id)
    prompt = f"Внутри у тебя сейчас: {desire_frame}.\n\n{GUIDANCE}"
    return ProactivePrompt(prompt=prompt, projection_id=projection_id, correlation_id=correlation_id)
```
Re-export `ProactivePrompt, build_wake_packet, GUIDANCE` from `core/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wake_packet.py -q`
Expected: PASS.

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/wake_packet.py core/__init__.py tests/test_wake_packet.py
uv run ruff check core/wake_packet.py core/__init__.py tests/test_wake_packet.py
uv run mypy -p lifemodel
git add core/wake_packet.py core/__init__.py tests/test_wake_packet.py
git commit -m "feat(core): wake-packet builder — desire-frame + guidance, owns the wish (spec §13)"
```

---

### Task 4: Cognition component (`LaunchProactive` intent, energy gate) + composition

**Files:**
- Modify: `core/intents.py` (add `LaunchProactive`), `core/__init__.py`
- Create: `core/cognition.py`
- Modify: `composition.py`
- Test: `tests/test_cognition.py`, `tests/test_composition.py` (extend)

**Interfaces:**
- Consumes: `build_wake_packet` (Task 3), `energy.{cost_real, reserve}` (C2), `timeutil` — and `State`/`TickContext`/`UpdateState`.
- Produces: `LaunchProactive`; `Cognition`.

**Behavior (spec §8, §13, model A):** each tick, the `Cognition` component:
1. If `state.desire_status != "active"` **or** `state.pending_proactive_id is not None` → return `[]` (nothing to do / a turn is already in flight — idempotent).
2. `correlation_id = f"proactive-{now.isoformat()}"` (deterministic).
3. Reserve the proactive turn's energy: `estimate = cost_real(fast_cost + send_cost, state.fatigue, alpha=alpha)`; `reserve(state.energy, estimate)`. If `None` (unaffordable) → return `[]` (**hold** — emergent shutoff; the desire stays active and will be retried when energy recovers).
4. Else build the wake-packet and emit: `LaunchProactive(prompt, correlation_id)` **plus** `UpdateState({"energy": energy_after_reserve, "pending_proactive_id": correlation_id, "pending_proactive_since": now.isoformat()})`.

`LaunchProactive` is consumed by the egress in Phase E; here it is asserted in isolation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cognition.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.cognition import Cognition
from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchProactive, UpdateState
from lifemodel.state.model import State

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _cog() -> Cognition:
    return Cognition(fast_cost=0.02, send_cost=0.03, alpha=2.0)


def _ctx(state: State, *, tmp_path) -> TickContext:
    return TickContext(state=state, now=NOW, bus=FileSignalBus(tmp_path), signals=())


def _launch(intents):
    return next((i for i in intents if isinstance(i, LaunchProactive)), None)


def _update(intents):
    return next((i for i in intents if isinstance(i, UpdateState)), None)


def test_no_active_desire_does_nothing(tmp_path) -> None:
    intents = _cog().step(_ctx(State(desire_status="none", u=2.0), tmp_path=tmp_path))
    assert list(intents) == []


def test_active_desire_launches_proactive_turn(tmp_path) -> None:
    state = State(desire_status="active", u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    launch = _launch(intents)
    assert launch is not None
    assert launch.correlation_id == f"proactive-{NOW.isoformat()}"
    assert launch.prompt  # carries the wake-packet prompt
    upd = _update(intents)
    assert upd.changes["pending_proactive_id"] == launch.correlation_id
    assert upd.changes["pending_proactive_since"] == NOW.isoformat()
    assert upd.changes["energy"] < 1.0  # reserved


def test_pending_turn_is_not_relaunched(tmp_path) -> None:
    state = State(desire_status="active", u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    assert _launch(intents) is None  # idempotent — a turn is already in flight


def test_insufficient_energy_holds_no_launch(tmp_path) -> None:
    # estimate = (0.02+0.03)*(1+2*1.0)=0.15 at max fatigue; energy 0.05 can't afford
    state = State(desire_status="active", u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, tmp_path=tmp_path))
    assert _launch(intents) is None  # emergent shutoff — hold
    assert _update(intents) is None  # energy untouched, desire stays active


def test_prompt_has_no_raw_numbers(tmp_path) -> None:
    import re

    state = State(desire_status="active", u=3.2, energy=1.0)
    launch = _launch(_cog().step(_ctx(state, tmp_path=tmp_path)))
    assert not re.search(r"\d", launch.prompt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cognition.py -q`
Expected: FAIL — no `LaunchProactive` / `lifemodel.core.cognition`.

- [ ] **Step 3: Implement**

Add to `core/intents.py`:
```python
@dataclass(frozen=True)
class LaunchProactive(Intent):
    """Launch a proactive turn (the being's native Hermes turn) with this
    desire-framed prompt. Consumed by the egress in Phase E."""

    prompt: str
    correlation_id: str
```
Re-export `LaunchProactive` from `core/__init__.py`.

```python
# core/cognition.py
"""Cognition — decides WHEN to wake the being's native turn and HOW to frame it
(spec §13, model A).

Cognition does not call an LLM: it emits a ``LaunchProactive`` intent carrying a
desire-framed wake-packet, and the being's own Hermes turn is the act-gate
(message = FULFILL, ``[SILENT]`` = REJECT — fed back by the ``post_llm`` hook in
Phase E). It launches only for a live, un-acted desire, and only if the proactive
turn's energy is affordable — otherwise it holds (emergent shutoff, spec §8).
"""

from __future__ import annotations

from collections.abc import Sequence

from .component import TickContext
from .energy import cost_real, reserve
from .intents import Intent, LaunchProactive, UpdateState
from .wake_packet import build_wake_packet


class Cognition:
    """The cognition layer: launch a proactive turn for a live desire, gated by
    energy. Idempotent via ``pending_proactive_id``."""

    def __init__(self, *, fast_cost: float, send_cost: float, alpha: float, id: str = "cognition") -> None:
        self.id = id
        self._fast_cost = fast_cost
        self._send_cost = send_cost
        self._alpha = alpha

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        if state.desire_status != "active" or state.pending_proactive_id is not None:
            return []

        estimate = cost_real(self._fast_cost + self._send_cost, state.fatigue, alpha=self._alpha)
        reserved = reserve(state.energy, estimate)
        if reserved is None:
            return []  # can't afford a proactive turn -> hold (emergent shutoff)
        energy_after, _reservation = reserved

        correlation_id = f"proactive-{ctx.now.isoformat()}"
        packet = build_wake_packet(value=state.u, theta=1.0, correlation_id=correlation_id)
        return [
            LaunchProactive(prompt=packet.prompt, correlation_id=correlation_id),
            UpdateState(
                {
                    "energy": energy_after,
                    "pending_proactive_id": correlation_id,
                    "pending_proactive_since": ctx.now.isoformat(),
                }
            ),
        ]
```
Re-export `Cognition` from `core/__init__.py`.

In `composition.py`: add module constants (reuse the C2 prices) `COGNITION_FAST_COST = 0.02`, `COGNITION_SEND_COST = 0.03` (or reuse existing `COST_*` if defined in C2 — check and reuse; otherwise define here), import `from .core.cognition import Cognition`, and register it **after** the aggregation (so a desire born this... note: components see the pre-tick snapshot, so cognition acts on the desire from the *previous* tick's checkpoint — correct: birth on tick T, launch on tick T+1; acceptable 60s latency). Use the same `UnknownComponent` guard:
```python
    cognition = Cognition(
        fast_cost=COGNITION_FAST_COST, send_cost=COGNITION_SEND_COST, alpha=COST_ALPHA
    )
    try:
        registry.manifest(cognition.id)
    except UnknownComponent:
        registry.register(cognition, ComponentManifest(id=cognition.id, type="cognition"))
```
(Use the C2 `COST_ALPHA` constant if present in `composition.py`; otherwise add `COST_ALPHA = 2.0`.)

- [ ] **Step 2b: Add the composition test (append to `tests/test_composition.py`)**

```python
def test_cognition_registered_after_aggregation(tmp_path) -> None:
    from lifemodel.core.cognition import Cognition

    lm = build_lifemodel(base_dir=tmp_path)
    ids = [c.id for c in lm.registry.enabled()]
    assert ids.index("contact-aggregation") < ids.index("cognition")
    assert any(isinstance(c, Cognition) for c in lm.registry.enabled())
```

- [ ] **Step 4: Run the full suite to verify green**

Run: `uv run pytest -q`
Expected: PASS — cognition + composition tests pass; every prior test (incl. `tests/sim/`) still passes. (Cognition runs only via `coreloop.tick()`; the live path is unchanged — nothing consumes `LaunchProactive` yet.)

- [ ] **Step 5: Format, type-check, commit**

```bash
uv run ruff format core/intents.py core/cognition.py core/__init__.py composition.py tests/test_cognition.py tests/test_composition.py
uv run ruff check core/intents.py core/cognition.py core/__init__.py composition.py tests/test_cognition.py tests/test_composition.py
uv run mypy -p lifemodel
git add core/intents.py core/cognition.py core/__init__.py composition.py tests/test_cognition.py tests/test_composition.py
git commit -m "feat(core): Cognition component — energy-gated LaunchProactive with wake-packet (spec §8/§13)"
```

---

## Phase-D1 Definition of Done

- [ ] `make check` fully green — paste the tail.
- [ ] Four commits on `core/rebuild`, one per task.
- [ ] No modification to `core/decision.py`, `egress_service.py`, `tick.py`, `heartbeat.py`, `hooks.py`, `impulse.py`.
- [ ] `tests/sim/` scenarios still green.
- [ ] Do **not** push, merge, or touch `main`. Send `orca orchestration send --type worker_done --message "<summary + make check tail>"` (or `--type escalation` if blocked).

## Self-Review (author check against the spec)

- **Spec coverage:** §13 prompt-safety projection (drive→phrasing, seeded, no numbers) → Task 1; §13 output-lint (mechanical/filler, not all time) → Task 2; §13 wake-packet (desire-frame + guidance, owns the wish, permits `[SILENT]`) → Task 3; §8 energy gate on the proactive turn + §13 model-A launch (native turn is the act-gate) + Codex idempotent launch (`pending_proactive_id` guard) → Task 4. **Deferred to Phase E (cutover):** consuming `LaunchProactive` → real proactive turn via egress; `post_llm` verdict feedback; applying output-lint at send-time; `pending` stale-recovery reconciliation. **Deferred to D2/D3:** backstop, async invalidation, tick-discipline, observability, bus-pruning.
- **Type consistency:** `project_contact(value, *, theta, seed) -> (phrasing, projection_id)` identical in Tasks 1, 3. `build_wake_packet(*, value, theta, correlation_id) -> ProactivePrompt` identical in Tasks 3, 4. `lint_proactive(text, *, patterns=…) -> LintResult` self-consistent. `Cognition(*, fast_cost, send_cost, alpha)` identical in Task 4 + composition.
- **Determinism:** projection uses `hashlib` (not `random`); `correlation_id` from the injected clock — reproducible.
- **Coma-safety:** cognition's *check* is cheap; it only reserves (never blocks the cheap layers); at low energy it holds rather than locking anything.
- **No placeholders:** every step ships real code + an exact command with expected output.

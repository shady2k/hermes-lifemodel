"""Reactive felt-state display — the pure gate + composers behind lm-ukc.4 / .4.1.

The being's core affect already colours its PROACTIVE reach (lm-ukc.5) and shows in
debug/trace (lm-ukc.6/.3). What is missing is mood proving through the being's
manner in an ORDINARY reactive turn — but **not in every reply**, and **invisibly
during real work**. This Hermes-free module is where that decision and its prose
live; the adapter boundary (``hooks.py`` / ``__init__.py``) only reads committed
state and wires it to the host.

Two channels, both grounded here:

* **Ambient (push)** — a light "manner cue" the ``pre_llm_call`` injector emits per
  turn. :func:`decide` gates it suppression-first with **zero language detection**:
  ``warmed`` → (``is_salient`` AND ``not is_task_context`` AND
  (``felt_changed`` OR ``cooldown_elapsed``)) → LIGHT, else a typed suppression
  reason. Task suppression reads only robust BEHAVIORAL signals
  (:func:`is_task_context`) — never keywords — so the mood is never wrongly muted
  on a relational reply that merely mentions a file, and never leaks onto real work.
* **On-demand (pull)** — :func:`compose_self_read` renders the felt prose the
  ``check_in`` tool returns when the being reads itself (spec §4b). It lives outside
  the gate (the model calls the tool itself) and is honest at any state.

**The first-class "feeling, not sensor" guarantee (spec §4b, risk #1):** neither
composer ever emits a raw axis (a valence/arousal number). Only felt WORD / TEXTURE
prose (from :mod:`lifemodel.core.affect`), an energy bucket, and the strongest live
desire — never ``"you are at 0.3 valence"``. That is a guarantee, not a format rule.

**One-way invariant (Phase 3):** everything here is READ-ONLY over affect — it
colours the manner, and never writes into the drive / wake path.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..domain.objects import Desire
from ..state.model import State
from .affect import FELT_WORD_PARAMS, FeltWordParams, felt_texture, felt_word
from .timeutil import minutes_between


@dataclass(frozen=True)
class FeltDisplayParams:
    """Calibratable knobs for the ambient gate — tunable on disk (like
    :class:`~lifemodel.core.affect.FeltWordParams`, spec NFR5/§11).

    Starting values are calibration seeds — deliberately "visible but not
    flaunted" (spec §11: start slightly expressive; too high a threshold is the
    same latent failure as ``[SILENT]``), docked live via lm-ukc.7.
    """

    #: Magnitude cut (off the neutral centre) below which affect reads as empty
    #: retrieval and no cue surfaces — ``max(|v|, |a − neutral_a_center|)`` (§5).
    salience_threshold: float = 0.30
    #: Minutes between ambient shows when the felt word has NOT changed — cures a
    #: repeated cue on a long non-neutral stretch (§5).
    cooldown_min: float = 45.0
    #: RESERVED for the lm-z2e time-based cold-start gate (spec §12). Until it
    #: lands, the local :func:`warmed` uses :attr:`cold_start_epsilon` on affect
    #: magnitude; this stays declared so the disk-config surface is stable.
    warmup_min: float = 15.0
    #: ``|v| + |a|`` within this of the origin reads as the un-warmed cold-start
    #: 0/0 — the local warmth check pending lm-z2e (§5/§12).
    cold_start_epsilon: float = 0.08
    #: A user message longer than this is a paste (code/log/doc dump) → task (§5).
    long_paste_chars: int = 600
    #: How many recent conversation messages the task detector scans (natural
    #: decay — replaces a task-streak flag without ever getting stuck, §5).
    task_window: int = 6


#: The default seeds — one shared frozen instance the injector and tool read alike,
#: calibratable on disk later (spec NFR5). Mirrors ``FELT_WORD_PARAMS``.
DEFAULT_FELT_DISPLAY_PARAMS = FeltDisplayParams()


class Decision(enum.Enum):
    """The ambient gate's verdict for one turn.

    Spec §5 sketches ``NONE``/``LIGHT``/``RICH``; this enum refines the ``NONE``
    case into its four suppression REASONS so observability (spec §9) can count
    *why* the mood stayed quiet. ``RICH`` (the on-demand self-read) lives OUTSIDE
    this gate — the model calls the ``check_in`` tool itself — so it is not a
    :func:`decide` outcome. Each value is exactly the closed ``outcome`` metric
    label the injector emits.
    """

    LIGHT = "light"
    NOT_WARMED = "not_warmed"
    NOT_SALIENT = "not_salient"
    TASK = "task"
    COOLDOWN_UNCHANGED = "cooldown_unchanged"

    @property
    def shows(self) -> bool:
        """True only when an ambient cue should be injected this turn."""
        return self is Decision.LIGHT


# --------------------------------------------------------------------------- #
# TurnSignals — a typed, defensive snapshot of the pre_llm_call hook input
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecentMessage:
    """One prior conversation entry, reduced to what the task detector needs."""

    role: str
    text: str
    has_tool_calls: bool


@dataclass(frozen=True)
class TurnSignals:
    """The slice of the ``pre_llm_call`` input the gate reads (spec §6).

    Built from the host's ``user_message`` + ``conversation_history`` via
    :meth:`from_hook`, which is defensive about the untrusted host payload shape
    (a non-dict entry, a missing key, a multimodal ``content`` list) — it never
    raises, so a malformed history can never crash the hot dispatch path.
    """

    user_message: str
    recent_messages: tuple[RecentMessage, ...] = field(default_factory=tuple)

    @classmethod
    def from_hook(
        cls, user_message: object, conversation_history: object, *, window: int
    ) -> TurnSignals:
        history = conversation_history if isinstance(conversation_history, list) else []
        tail = history[-window:] if window > 0 else []
        return cls(
            user_message=user_message if isinstance(user_message, str) else "",
            recent_messages=tuple(_recent_from_entry(entry) for entry in tail),
        )


def _extract_text(content: object) -> str:
    """Best-effort text of a message ``content`` (str, or a multimodal part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _recent_from_entry(entry: object) -> RecentMessage:
    if not isinstance(entry, dict):
        return RecentMessage(role="", text="", has_tool_calls=False)
    role = entry.get("role")
    tool_calls = entry.get("tool_calls")
    return RecentMessage(
        role=role if isinstance(role, str) else "",
        text=_extract_text(entry.get("content")),
        has_tool_calls=isinstance(tool_calls, list) and bool(tool_calls),
    )


# --------------------------------------------------------------------------- #
# Detectors — all language-independent (spec §5)
# --------------------------------------------------------------------------- #


def warmed(state: State, params: FeltDisplayParams) -> bool:
    """True when core affect has warmed off the cold-start 0/0 (spec §5).

    Local, self-contained (no lm-z2e dependency): the affect must have been
    derived at least once (a stamp) AND moved a real distance from the origin
    (``|v| + |a| ≥ cold_start_epsilon``). Arousal eases up from 0 over the first
    minutes, so a fresh being reads NOT warmed until its body genuinely tints.
    """
    if state.affect_updated_at is None:
        return False
    magnitude = abs(state.affect_valence) + abs(state.affect_arousal)
    return magnitude >= params.cold_start_epsilon


def is_salient(
    valence: float,
    arousal: float,
    params: FeltDisplayParams,
    *,
    word_params: FeltWordParams = FELT_WORD_PARAMS,
) -> bool:
    """True when affect is bright enough to surface — magnitude off the neutral
    centre, not merely non-neutral (spec §5). A soft ``content``/``wistful`` tint
    is low-salience (empty retrieval); a deep ``lonely`` or a keyed ``restless``
    (high arousal at neutral valence) both clear the one calibratable threshold."""
    metric = max(abs(valence), abs(arousal - word_params.neutral_a_center))
    return metric >= params.salience_threshold


#: Unified-diff frames (line-anchored so ``---``/``+++`` in prose don't false-fire).
_DIFF_RE = re.compile(r"^(@@ |diff --git |\+\+\+ |--- |index [0-9a-f]{7,})", re.MULTILINE)
#: Stack-trace shapes across a few ecosystems (Python/JS) — high precision.
_STACK_RE = re.compile(
    r"(Traceback \(most recent call last\)"
    r'|^\s*File "[^"]+", line \d+'
    r"|^\s+at [\w.$<>]+ ?\([^)]*:\d+"
    r"|^[\w.]*(?:Error|Exception)(?:: .+)?$)",
    re.MULTILINE,
)
#: A shell prompt line (``$ cmd`` / ``# cmd``) — a pasted command session.
_SHELL_RE = re.compile(r"^\s*[$#] \S", re.MULTILINE)
#: A log line: an ISO-ish timestamp prefix, or a bracketed/level-prefixed record.
_LOG_RE = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"
    r"|(?:ERROR|WARN|WARNING|DEBUG|TRACE|CRITICAL|FATAL|INFO)\s*[:\]])",
    re.MULTILINE,
)


def _looks_like_json(text: str) -> bool:
    """True when *text* is (non-trivially) a JSON object/array — a data paste."""
    stripped = text.strip()
    if not (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return False
    try:
        parsed = json.loads(stripped)
    except (ValueError, RecursionError):
        return False
    return isinstance(parsed, (dict, list)) and bool(parsed)


def _has_work_markers(text: str) -> bool:
    if not text:
        return False
    return bool(
        "```" in text
        or _DIFF_RE.search(text)
        or _STACK_RE.search(text)
        or _SHELL_RE.search(text)
        or _LOG_RE.search(text)
        or _looks_like_json(text)
    )


def is_task_context(turn: TurnSignals, params: FeltDisplayParams) -> bool:
    """True when the turn is focused WORK, not relational talk (spec §5).

    Robust behavioral signals ONLY — never keywords, never a language-biased
    work-intent classifier (which would wrongly mute the mood on a relational
    reply): a recent assistant turn with ``tool_calls``; a long paste (a
    code/log/doc dump anywhere in the window, so a paste followed by a short
    "continue"/"what about part 2" still reads as task); or structural markers
    (code fences, unified diffs, stack traces, shell sessions, log lines, JSON
    blocks) anywhere in the window. A warm reply that merely NAMES a file carries
    none of these, so it is not task.
    """
    if any(msg.has_tool_calls for msg in turn.recent_messages):
        return True
    texts = [turn.user_message, *(msg.text for msg in turn.recent_messages)]
    if any(len(text) > params.long_paste_chars for text in texts):
        return True
    return any(_has_work_markers(text) for text in texts)


def felt_changed(state: State) -> bool:
    """True when the current felt WORD differs from the last ambiently shown one.

    A first-ever show (``affect_display_last_word is None``) counts as a change,
    so a warmed, salient being surfaces its first cue immediately.
    """
    current = felt_word(state.affect_valence, state.affect_arousal)
    return current != state.affect_display_last_word


def cooldown_elapsed(state: State, params: FeltDisplayParams, now: datetime) -> bool:
    """True when enough time has passed since the last ambient show (spec §5).

    Never shown (``affect_display_last_at is None``) → trivially elapsed. The
    timestamp is parsed defensively (:func:`minutes_between` returns 0 on a bad
    value → NOT elapsed, the safe "don't repeat" default)."""
    if state.affect_display_last_at is None:
        return True
    return minutes_between(state.affect_display_last_at, now) >= params.cooldown_min


def decide(
    state: State,
    turn: TurnSignals,
    params: FeltDisplayParams,
    now: datetime,
) -> Decision:
    """The ambient gate (spec §5) — suppression-first, zero language detection.

    Order is load-bearing: cold-start silence first, then salience, then task
    suppression, then the change/cooldown throttle. Returns :attr:`Decision.LIGHT`
    to inject, else the typed suppression reason (for the metric, spec §9).
    """
    if not warmed(state, params):
        return Decision.NOT_WARMED
    if not is_salient(state.affect_valence, state.affect_arousal, params):
        return Decision.NOT_SALIENT
    if is_task_context(turn, params):
        return Decision.TASK
    if felt_changed(state) or cooldown_elapsed(state, params, now):
        return Decision.LIGHT
    return Decision.COOLDOWN_UNCHANGED


# --------------------------------------------------------------------------- #
# Composers — felt PROSE only, never a raw axis (spec §4a/§4b guarantee)
# --------------------------------------------------------------------------- #

#: The ambient system-note — mirrors ``agent/memory_manager.build_memory_context_block``
#: (a semantic tag + a ``[System note: …]``), so felt-state reads uniformly with
#: memory/hindsight context. The "do not mention unless asked" line is REQUIRED
#: (else the mood becomes the topic — the "manner, not subject" invariant, §4a).
_LIGHT_CUE_NOTE = (
    "[System note: This is private, per-turn context about your present inner state,\n"
    "not new user input. Let it color only the manner of your reply when appropriate:\n"
    "tone, pace, softness, brevity. Do not mention or explain it unless the user\n"
    "directly asks how you are. If the user is asking for focused work, let it pass.]"
)


def compose_light_cue(state: State, *, word_params: FeltWordParams = FELT_WORD_PARAMS) -> str:
    """The ambient LIGHT cue block for the ``pre_llm_call`` inject (spec §4a).

    A ``<felt-state>`` envelope (tag + system-note + prose) mirroring the memory
    context block. Prose ONLY — the felt TEXTURE, never a ``(v, a)`` number; the
    voice renders the English internal representation.
    """
    texture = felt_texture(state.affect_valence, state.affect_arousal, word_params)
    return (
        "<felt-state>\n"
        f"{_LIGHT_CUE_NOTE}\n"
        "\n"
        f"Right now, the feeling in you is {texture}.\n"
        "</felt-state>"
    )


def _energy_bucket(energy: float) -> str:
    """Coarse felt read of the being's energy — never the number (spec §4b).

    The top bucket is "full", NOT "bright": ``bright`` is already a felt WORD
    (pleasant + keyed-up), so it collided in the read — "You feel bright: … Energy
    is bright." — which reads like a stutter rather than two distinct facts.
    """
    if energy < 0.34:
        return "low"
    if energy < 0.67:
        return "steady"
    return "full"


def _pull_phrase(desire: Desire | None) -> str:
    """The strongest live desire as felt prose (spec §4b).

    The only live desire kind today is the contact desire, so its presence reads
    as the pull to stay close; absence reads calm. Never a salience/urgency
    number — the "feeling, not sensor" guarantee holds here too."""
    if desire is not None:
        return "The strongest live pull is wanting to stay close."
    return "Nothing in particular is pulling at you right now."


def compose_self_read(
    state: State,
    *,
    desire: Desire | None,
    params: FeltDisplayParams = DEFAULT_FELT_DISPLAY_PARAMS,
    word_params: FeltWordParams = FELT_WORD_PARAMS,
) -> str:
    """The honest first-person self-read the ``check_in`` tool returns (spec §4b).

    ``felt_word`` + ``felt_texture`` + an energy bucket + the strongest live
    desire, all prose. Cold-start (not warmed) gives a soft "still settling"
    read, never a garbage "very quiet". **Never a raw axis** (the §4b guarantee):
    no digit, no ``valence``/``arousal`` word ever appears here.
    """
    if not warmed(state, params):
        return "You're still settling in — it's hard to tell right now."
    word = felt_word(state.affect_valence, state.affect_arousal, word_params)
    texture = felt_texture(state.affect_valence, state.affect_arousal, word_params)
    energy = _energy_bucket(state.energy)
    return f"You feel {word}: {texture}. Energy is {energy}. {_pull_phrase(desire)}"


__all__ = [
    "DEFAULT_FELT_DISPLAY_PARAMS",
    "Decision",
    "FeltDisplayParams",
    "RecentMessage",
    "TurnSignals",
    "compose_light_cue",
    "compose_self_read",
    "cooldown_elapsed",
    "decide",
    "felt_changed",
    "is_salient",
    "is_task_context",
    "warmed",
]

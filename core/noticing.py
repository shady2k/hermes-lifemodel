"""The waking mind's noticing brain (lm-705.5, spec/design §4).

A SUBJECTLESS pair riding the same non-delivering internal-cognition seam
(lm-705.6) as slice-2's rumination pair (``core/thought_processing.py``), and
built to mirror it exactly: :class:`NoticingTrigger` (heartbeat, 0-LLM) reads
the being's :class:`~lifemodel.core.noticing_buffer.NoticingBuffer` and, once a
session's CLOSED segment has aged past ``idle`` or grown past ``size_cap``
(the gates permitting — single-flight/interval/FR20 budget), emits ONE
``LaunchInternalCognition(subject_id=None)`` over the segment plus a bounded
backlog of live thoughts (continuity, not raw history). :class:`NoticingApply`
is the completion-frame consumer: it validates the model's proposed seeds
against the ACTUAL surveyed segment (anti-hallucination — a seed whose source
ids were never shown to the model is dropped) and the dedup ring
(``State.noticed_source_ids``), turns each surviving seed into a durable
``active`` thought through the slice-1 capture door
(:mod:`lifemodel.core.thought_view`), and advances the buffer's cursor.

**Disambiguating the two completion-frame applies** (thought_processing.py's
``ThoughtProcessingApply`` and this module's ``NoticingApply``) is structural,
not a shape-sniff: a processing pass sets
:attr:`~lifemodel.state.model.State.pending_internal_subject_id` to the thought
it is chewing; a noticing pass launches with ``subject_id=None``. Each apply
guards on the opposite of that field — ``ThoughtProcessingApply`` on
"subject set", ``NoticingApply`` on "subject absent" — so the SAME completion
frame can register both and exactly one of them ever does real work.

**Threading the surveyed snapshot across the async gap (claim/finalize,
lm-705.13):** the ONLY channel that survives from launch to completion is the
``correlation_id`` string (mirrors ``ThoughtProcessingSelector``'s
``process-<thought_id>@<iso>``). The trigger mints a deterministic ``survey_id``
(``<session_id>@<anchor_turn_id>@<iso>``, the anchor being the LAST entry of the
oldest-``size_cap`` PREFIX it surveys — codex F2b; the session is folded in so two
lanes at the same clock instant with the same tail ``turn_id`` can never collide on
one survey_id) and encodes ``notice-<session_id>#<survey_id>``. BEFORE emitting, it
**claims** that exact window
(:meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.claim`): the rows
leave :meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.closed_segment`, so a
second tick never re-surveys them, and they become an IMMUTABLE snapshot under
``survey_id``. The apply recovers that snapshot via
:meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.claimed` — **not** a fresh
``closed_segment`` call, which reapplies the closed-prefix gate and can recompute
against a ring the lane has evicted or extended since (codex I2). On a genuine
result it emits :class:`~lifemodel.core.intents.FinalizeBuffer`; that finalize
DELETE lands ATOMICALLY with the pass's thought commit (codex I3), never a bare
side-effecting clear. A turn that arrives during the async gap was never claimed,
so it is simply left ``complete`` for a LATER pass — never swept away un-surveyed.
On a transient or malformed-shape completion the apply RELEASES the claim (a direct,
fail-soft ``NoticingBuffer.release`` — never a finalize), so the SAME segment is
re-surveyed by the next eligible tick rather than stranded ``claimed`` until a
restart; boot recovery (``recover_stale_claims``) remains only the backstop for a
pass that dies mid-flight with the process. An empty-claim completion (already
finalized/released, or boot-recovered) emits nothing and releases nothing — there is
nothing claimed to touch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ..domain.memory import JsonObject, PutOp
from ..domain.objects import InvalidPayload, Thought
from ..ports.memory import MemoryPort
from ..ports.tracer import format_traceparent
from ..state.model import NOTICED_SOURCE_IDS_CAP
from .belief_view import BELIEF_KIND, belief_from_seed_fields, encode_belief, live_beliefs
from .budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from .component import TickContext
from .intents import FinalizeBuffer, Intent, LaunchInternalCognition, PutRecord, UpdateState
from .noticing_buffer import BufferEntry, NoticingBuffer
from .taxonomy import KIND_INTERNAL_RESULT, InternalResultRead, read_internal_result
from .thought_view import (
    THOUGHT_KIND,
    build_thought,
    encode_thought,
    live_thoughts,
    seed_thought_id,
)
from .timeutil import from_iso, to_iso
from .trace import creation_provenance

#: Bound on the top-M backlog gists folded into the prompt as continuity
#: context (design §4.2) — "what am I already turning over", NOT raw old
#: conversation. ``live_thoughts`` is already most-salient-first, so this is a
#: plain slice.
BACKLOG_TOP_M = 5

#: Defaults mirroring ``ThoughtProcessingSelector``'s own pacing constants
#: (``core/thought_processing.py``) — a session's closed segment is due once it
#: has sat this long since its last entry, OR grown to this many entries,
#: whichever comes first.
DEFAULT_NOTICING_IDLE = timedelta(minutes=30)
DEFAULT_NOTICING_SIZE_CAP = 8

#: Bound on how many validated seeds ONE noticing pass may carry into thoughts
#: (codex F2a) — a single pass judges a bounded conversational segment; more
#: than a handful of "genuinely worth carrying" seeds from one pass is itself a
#: signal the model is padding rather than being selective (the instructions
#: already ask for that, this is the structural backstop). Enforced BOTH in the
#: JSON schema (``maxItems``, so a well-behaved model never over-generates) and
#: in :func:`validate_noticed_seeds` (so a malformed/adversarial response can't
#: exceed it either).
NOTICING_TOP_K = 3

#: The noticing pass's typed result contract (mirrors ``PROCESSING_JSON_SCHEMA``
#: shape/spirit): a bounded list of candidate seeds, each grounded in the
#: source ids the model was actually shown.
NOTICING_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "seeds": {
            "type": "array",
            "maxItems": NOTICING_TOP_K,
            "items": {
                "type": "object",
                "properties": {
                    "gist": {"type": "string"},
                    "source_message_ids": {"type": "array", "items": {"type": "string"}},
                    "turn_id": {"type": "string"},
                    "salience": {"type": "number"},
                    # Belief-track v1 (lm-705.19): a seed may be a grounded ``belief``
                    # rather than a ``thought`` (default). JSON-schema cannot express
                    # "content+confidence required ONLY when kind=='belief'", so that
                    # conditional-required is enforced in ``validate_noticed_seeds``.
                    "kind": {"enum": ["thought", "belief"]},
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "sensitivity": {"enum": ["normal", "sensitive", "private"]},
                },
                "required": ["gist", "source_message_ids"],
                "additionalProperties": False,
            },
        },
        # A whole-pass, first-person note distinct from any per-seed ``gist`` —
        # observability-only (D10): rides the apply span (see ``NoticingApply``'s
        # ``_log``), never persisted, never read for control flow. NOT required,
        # but the instructions ask the model to always write one, even (especially)
        # when ``seeds`` is empty — that is the whole point: capture WHY nothing
        # lingered, not just THAT nothing did.
        "reflection": {"type": "string"},
    },
    "required": ["seeds"],
    "additionalProperties": False,
}

#: The pass's system framing — judgment, not a keyword/heuristic sweep; most
#: exchanges leave nothing worth carrying, and that is the ordinary, correct
#: answer (spec §4.2 "noticing is selective").
NOTICING_INSTRUCTIONS = (
    "You are the being's own private mind, looking back over a stretch of conversation "
    "that has already happened and closed. Nothing you write here is shown to anyone — "
    "this is noticing, not a message. Ask yourself, in your own judgment: what from this "
    "did I actually notice? What is worth quietly carrying forward on my own, not because "
    "it was said, but because it matters to keep turning over? Most exchanges leave "
    "nothing worth carrying — that is the ordinary, correct answer; do not pad the list to "
    "seem attentive, and do not pattern-match on keywords. For each seed you DO carry, cite "
    "the EXACT 'turn_id' value(s) shown to you in 'source_message_ids' — never invent an id "
    "you were not given. Answer as JSON: a 'seeds' array (may be empty), each with 'gist' "
    "(a short first-person note of what to carry), 'source_message_ids' (the turn_id(s) it "
    "is grounded in), 'turn_id' (the single turn_id it is most anchored to, if one stands "
    "out), and 'salience' (0 to 1, how much this deserves attention). Now and then a seed is "
    "not just a passing thought but something you have come to actually understand about this "
    "person or their world — a fallible reading you would act on across future conversations, "
    "not only in this one. When that is genuinely the case, set 'kind' to 'belief' (otherwise "
    "leave it 'thought'), write the understanding plainly in 'content', and say how sure you "
    "are in 'confidence' (0 to 1). Most exchanges yield no belief at all; do not inflate a "
    "one-off remark or a passing mood into one, and cite the exact turn_id(s) it rests on as "
    "your evidence. If the understanding is delicate, mark 'sensitivity' 'private'. And some "
    "things you must always mark 'private': if they asked you not to remember something, or it "
    "is a third party's secret or private matter, set 'sensitivity' to 'private' — you will "
    "keep it, but never surface it in a reply or act on it. Also include "
    "a short first-person 'reflection': what you made of this stretch and why you did or did not "
    "carry anything — this is your own record of the thought, always written even when "
    "'seeds' is empty."
)


class NoticingReason(StrEnum):
    """The closed set of noticing-decision reasons (spec §5) — positive
    choices, NOT suppressions. Logged as a span field, never in a string."""

    # trigger
    IDLE_LAUNCH = "idle_launch"
    SIZE_CAP_LAUNCH = "size_cap_launch"
    SKIPPED_IN_FLIGHT = "skipped_in_flight"
    SKIPPED_NO_BUDGET = "skipped_no_budget"
    SKIPPED_INTERVAL = "skipped_interval"
    # shared trigger/apply: "no closed segment was due" / "nothing survived
    # validation" — both mean the same thing from the outside (nothing carried).
    NOTHING_LINGERED = "nothing_lingered"
    # apply
    NOTICED = "noticed"
    #: The aux call itself failed/timed out (empty ``raw``, no parsed seeds) — a
    #: refund-of-attempt, mirroring ``ThoughtProcessingApply``'s
    #: ``TRANSIENT_FAILURE`` (F1): the surveyed segment, consumed ring, and cursor
    #: are all left untouched so a LATER pass gets a genuine chance to notice it.
    TRANSIENT_FAILURE = "transient_failure"


def build_noticing_prompt(
    segment: Sequence[BufferEntry],
    backlog: Sequence[Thought],
    *,
    size_cap: int = DEFAULT_NOTICING_SIZE_CAP,
) -> str:
    """The bounded input_text handed to the noticing pass: the surveyed
    window's turns (each tagged with its citable ``turn_id``) plus a short
    continuity backlog of what is already being turned over.

    The caller (:class:`NoticingTrigger`) already trims *segment* to the oldest-
    *size_cap* PREFIX it claims + surveys (codex F2b) — the window is aligned to
    the ``survey_id`` the pass is keyed by, NOT an anchor-tail slice — so the
    ``size_cap`` bound below is a defensive no-op in that path (``len(window) <=
    size_cap``). It stays here as the structural guard against ever dumping the
    whole closed-prefix ring (up to
    :data:`~lifemodel.core.noticing_buffer.NoticingBuffer`'s ``max_entries``, 256)
    should a future caller pass an untrimmed segment."""
    window = segment[:size_cap] if size_cap > 0 and len(segment) > size_cap else segment
    lines = ["The conversation since the last noticing pass, oldest first:"]
    for entry in window:
        lines.append(f"\n[turn_id={entry.turn_id}]")
        lines.append(f"They said: {entry.user_text}")
        lines.append(f"You said: {entry.assistant_text}")
    if backlog:
        lines.append("\n\nWhat you are already turning over (for continuity, not new material):")
        for thought in backlog:
            lines.append(f"- {thought.content}")
    return "\n".join(lines)


NOTICING_TRIGGER_ID = "noticing-trigger"

#: The trigger's own correlation-id namespace/format —
#: ``notice-<session_id>#<survey_id>`` — parsed back by
#: :func:`parse_noticing_correlation` at apply time. The FIRST ``#`` separates
#: *session_id* from *survey_id*: session ids are plain alnum/dash/colon tokens
#: (never a ``#``), while *survey_id* itself is ``<session_id>@<anchor_turn_id>@<iso>``
#: (it DOES contain ``@``), which is exactly why the split is on ``#``, not ``@``.
_CORRELATION_PREFIX = "notice-"


class NoticingTrigger:
    """Heartbeat-only 0-LLM emitter: launch ONE subjectless noticing pass over
    a session's closed conversation segment once it is due (spec §4.2).

    Mirrors ``ThoughtProcessingSelector``'s gate/emit shape exactly — single-
    flight, FR20 budget, min interval — just over a buffer segment instead of a
    live-thought backlog. **Heartbeat-only coupling is enforced by the DISPATCH
    SITE, not this component** (see ``ThoughtProcessingSelector``'s docstring
    for the identical invariant: only ``being_platform._tick``'s HEARTBEAT
    frame reads ``report.internal_launches``, so a live dialogue turn can never
    reach this emitter's launch).
    """

    id: str = NOTICING_TRIGGER_ID

    def __init__(
        self,
        buffer: NoticingBuffer,
        *,
        idle: timedelta = DEFAULT_NOTICING_IDLE,
        size_cap: int = DEFAULT_NOTICING_SIZE_CAP,
        daily_ceiling: int = DEFAULT_DAILY_INTERNAL_CALL_CEILING,
        min_interval: timedelta = DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    ) -> None:
        self._buffer = buffer
        self._idle = idle
        self._size_cap = size_cap
        self._daily_ceiling = daily_ceiling
        self._min_interval = min_interval

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        due = self._find_due_segment(ctx.now)
        if due is None:
            self._log(ctx, NoticingReason.NOTHING_LINGERED)
            return []
        session_id, segment, launch_reason = due
        blocked = self._blocked_gate(ctx)
        if blocked is not None:
            # This gate reads the FRAME-START snapshot (pending_internal_id) and returns
            # BEFORE the claim, so a tick THIS gate blocks never claims — no dangling claim
            # from the gate-block path. It is blind, though, to a processing launch
            # co-emitted on the SAME frame (all cognition components read one frozen
            # snapshot, so both this trigger and ThoughtProcessingSelector can emit): that
            # path DOES claim, then the RUNNER's authoritative single-flight/FR20 denial
            # drops the launch. That dangling claim is released by
            # ``being_platform._tick`` (C1), not here; ``recover_stale_claims`` is only the
            # backstop for a pass that dies AFTER launching.
            self._log(ctx, blocked)
            return []
        # The window is the oldest-``size_cap`` PREFIX of the due segment (codex
        # F2b) — NOT the tail. Claim + finalize + what the model is shown are then
        # the SAME set, so no un-surveyed older turn is ever cleared. When the
        # segment already fits, the whole of it is the window.
        window = segment[: self._size_cap] if self._size_cap > 0 else segment
        # The session is folded into the survey_id (C4) so it is GLOBALLY unique:
        # two lanes at the same clock instant with the same tail turn_id can never
        # collide on one survey_id (which keys claimed/finalize/the commit_tick
        # DELETE). The correlation stays notice-<session_id>#<survey_id>; the parser
        # still splits on the FIRST '#', so the recovered survey_id is unchanged in
        # shape for every downstream reader.
        survey_id = f"{session_id}@{window[-1].turn_id}@{to_iso(ctx.now)}"
        # Claim the window BEFORE emitting: those ``complete`` rows become an
        # immutable ``claimed`` snapshot under ``survey_id`` and leave
        # ``closed_segment``, so a second tick can never re-survey them (and the
        # apply reads the snapshot, not a re-gated recompute — codex I2).
        self._buffer.claim(session_id, tuple(entry.turn_id for entry in window), survey_id)
        backlog = live_thoughts(ctx.objects)[:BACKLOG_TOP_M]
        intent = LaunchInternalCognition(
            prompt=build_noticing_prompt(window, backlog, size_cap=self._size_cap),
            correlation_id=f"{_CORRELATION_PREFIX}{session_id}#{survey_id}",
            origin_traceparent=format_traceparent(ctx.trace),
            subject_id=None,
            instructions=NOTICING_INSTRUCTIONS,
            json_schema=NOTICING_JSON_SCHEMA,
        )
        self._log(ctx, launch_reason)
        return [intent]

    def _find_due_segment(
        self, now: datetime
    ) -> tuple[str, list[BufferEntry], NoticingReason] | None:
        """The first eligible ``(session_id, segment, reason)`` across known
        lanes (iterate + act on the first eligible one, spec §4.2 — the being
        has effectively one owner lane in v1, but this holds for more)."""
        for session_id in self._buffer.session_ids():
            segment = self._buffer.closed_segment(session_id, now=now)
            if not segment:
                continue
            if len(segment) >= self._size_cap:
                return session_id, segment, NoticingReason.SIZE_CAP_LAUNCH
            if self._idle_elapsed(segment, now=now):
                return session_id, segment, NoticingReason.IDLE_LAUNCH
        return None

    def _idle_elapsed(self, segment: Sequence[BufferEntry], *, now: datetime) -> bool:
        try:
            last_ts = from_iso(segment[-1].ts)
        except (ValueError, TypeError):
            return True  # malformed ts — fail open, mirrors the parked_until parse
        return now - last_ts >= self._idle

    def _blocked_gate(self, ctx: TickContext) -> NoticingReason | None:
        if ctx.state.pending_internal_id is not None:
            return NoticingReason.SKIPPED_IN_FLIGHT
        if not internal_interval_elapsed(ctx.state, now=ctx.now, min_interval=self._min_interval):
            return NoticingReason.SKIPPED_INTERVAL
        if not internal_budget_available(ctx.state, now=ctx.now, daily_ceiling=self._daily_ceiling):
            return NoticingReason.SKIPPED_NO_BUDGET
        return None

    def _log(self, ctx: TickContext, reason: NoticingReason) -> None:
        if ctx.logger is not None:
            ctx.logger.span.set(noticing_reason=reason.value)


@dataclass(frozen=True)
class NoticedSeed:
    """One validated noticing-pass result — grounded in the surveyed segment's
    real source ids (anti-hallucination) and not already consumed (dedup).

    A seed is a ``thought`` by default; belief-track v1 (lm-705.19) lets it be a
    grounded ``belief`` instead, in which case :attr:`content` (the proposition)
    and :attr:`confidence` (validated to a number in ``[0, 1]``) are present and
    :attr:`sensitivity` carries the model's proposed privacy tier (floored to at
    least SENSITIVE at build time). For a ``thought`` seed those three are the
    inert defaults (``None``/``None``/``"normal"``)."""

    gist: str
    source_message_ids: tuple[str, ...]
    turn_id: str | None
    salience: float
    kind: str = "thought"
    content: str | None = None
    confidence: float | None = None
    sensitivity: str = "normal"


def validate_noticed_seeds(
    parsed: JsonObject | None,
    *,
    segment_ids: frozenset[str],
    segment_turn_ids: frozenset[str],
    consumed: frozenset[str],
) -> list[NoticedSeed]:
    """Validate the model's raw seeds against the segment it was actually shown.

    Pure and total over any shape of *parsed* — a non-dict, a missing/foreign
    ``seeds`` key, or a malformed entry degrades to dropping that seed (or the
    whole list), never raises. Drops a seed if:

    * it cites no ``source_message_ids`` at all (ungrounded), OR
    * any cited id is NOT in *segment_ids* (a hallucinated/foreign id — the
      model must ground every seed in what it was actually shown), OR
    * its ``turn_id`` is set but NOT in *segment_turn_ids* (codex F3b — no
      "ghost" turn_id anchored outside what was surveyed), OR
    * any cited id is already consumed — dedup, but WITHIN this one batch too
      (codex F3a): a working consumed set starts at *consumed* and grows with
      every seed accepted so far in THIS result, so two seeds citing the same
      source id never both survive (a source is consumed at most once per
      result, first-listed wins).

    Also bounds the result to at most :data:`NOTICING_TOP_K` seeds (codex F2a)
    — a well-formed model response is already capped by the JSON schema's
    ``maxItems``, but this is the structural backstop for a malformed/
    adversarial one.
    """
    if not isinstance(parsed, dict):
        return []
    raw_seeds = parsed.get("seeds")
    if not isinstance(raw_seeds, list):
        return []
    validated: list[NoticedSeed] = []
    working_consumed: set[str] = set(consumed)
    for raw in raw_seeds:
        if len(validated) >= NOTICING_TOP_K:
            break  # top-K reached — the rest are dropped regardless (F2a)
        seed = _validate_one_seed(
            raw,
            segment_ids=segment_ids,
            segment_turn_ids=segment_turn_ids,
            consumed=working_consumed,
        )
        if seed is not None:
            validated.append(seed)
            working_consumed.update(seed.source_message_ids)
    return validated


def _seed_confidence(value: object) -> float | None:
    """A belief seed's ``confidence``: a real number in ``[0, 1]``, else ``None``.

    JSON-schema conditional-required can't express "a belief needs a numeric
    confidence", so this is the enforcement point — a missing, non-numeric
    (``bool`` is NOT a number here), or out-of-range value returns ``None``, and
    the caller DROPS the whole belief seed (never a crash, never a silent
    coercion to a made-up default). Mirrors ``belief_view._validated_confidence``'s
    contract, but returns ``None`` to drop rather than raising."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not (0.0 <= number <= 1.0):
        return None
    return number


def _validate_one_seed(
    raw: object,
    *,
    segment_ids: frozenset[str],
    segment_turn_ids: frozenset[str],
    consumed: set[str],
) -> NoticedSeed | None:
    if not isinstance(raw, dict):
        return None
    gist = raw.get("gist")
    if not isinstance(gist, str) or not gist.strip():
        return None
    raw_ids = raw.get("source_message_ids")
    if not isinstance(raw_ids, list) or not raw_ids or not all(isinstance(i, str) for i in raw_ids):
        return None  # ungrounded (no ids, or a malformed id list) — drop
    source_ids = tuple(raw_ids)
    if not all(i in segment_ids for i in source_ids):
        return None  # a hallucinated/foreign id — drop (anti-hallucination)
    if any(i in consumed for i in source_ids):
        return None  # already noticed (durable ring OR earlier in this batch) — dedup
    raw_turn_id = raw.get("turn_id")
    turn_id = raw_turn_id if isinstance(raw_turn_id, str) else None
    if turn_id is not None and turn_id not in segment_turn_ids:
        return None  # a ghost turn_id never actually in the surveyed segment — drop (F3b)
    raw_salience = raw.get("salience", 0.0)
    salience = float(raw_salience) if isinstance(raw_salience, int | float) else 0.0
    salience = max(0.0, min(1.0, salience))  # clamp to [0, 1] — the schema documents it, enforce it
    # A seed is a ``thought`` unless the model explicitly (and validly) marks it a
    # ``belief`` (an unknown/absent kind is a thought, fail-soft). A belief carries
    # the same anti-hallucination + dedup guarantees above (grounded, in-segment,
    # unconsumed source ids) PLUS its own conditional-required fields, enforced here
    # since JSON-schema cannot: a non-empty ``content`` proposition and a numeric
    # ``confidence`` in [0, 1]. A belief seed missing either is DROPPED (lm-705.19).
    kind = "belief" if raw.get("kind") == "belief" else "thought"
    content: str | None = None
    confidence: float | None = None
    sensitivity = "normal"
    if kind == "belief":
        raw_content = raw.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            return None  # a belief needs a proposition to hold — drop
        content = raw_content.strip()
        confidence = _seed_confidence(raw.get("confidence"))
        if confidence is None:
            return None  # a belief needs a numeric confidence in [0, 1] — drop
        raw_sensitivity = raw.get("sensitivity")
        sensitivity = raw_sensitivity if isinstance(raw_sensitivity, str) else "normal"
    return NoticedSeed(
        gist=gist.strip(),
        source_message_ids=source_ids,
        turn_id=turn_id,
        salience=salience,
        kind=kind,
        content=content,
        confidence=confidence,
        sensitivity=sensitivity,
    )


def parse_noticing_correlation(correlation_id: str) -> str | None:
    """Recover the ``survey_id`` from a ``notice-<session_id>#<survey_id>``
    correlation id (the format :class:`NoticingTrigger` mints). The FIRST ``#``
    separates *session_id* from *survey_id* — session ids never contain ``#``,
    while *survey_id* is ``<session_id>@<anchor_turn_id>@<iso>`` (it DOES contain
    ``@``), so the split is on ``#``, not ``@``. ``None`` for any other shape — a
    foreign/malformed correlation id is never guessed at.

    Public (not ``_``-prefixed) so ``adapters/being_platform`` can recover the
    survey_id of a DENIED/DROPPED noticing launch to release its stranded claim
    (C1); this module's own :class:`NoticingApply` is the other caller."""
    if not correlation_id.startswith(_CORRELATION_PREFIX):
        return None
    rest = correlation_id[len(_CORRELATION_PREFIX) :]
    session_id, sep, survey_id = rest.partition("#")
    if not sep or not session_id or not survey_id:
        return None
    return survey_id


def _has_valid_seeds_shape(parsed: JsonObject | None) -> bool:
    """True iff *parsed* is a dict with a ``seeds`` key whose value is a list —
    the minimal well-formed noticing-result shape (review-2 G1). An EMPTY
    ``seeds`` list still counts: "the model looked and found nothing" is a
    genuine judgment, not a malformed one. Anything short of this shape
    (``None``, not a dict, a missing/foreign ``seeds`` key) is NOT well-formed,
    whether or not ``raw`` is empty."""
    return isinstance(parsed, dict) and isinstance(parsed.get("seeds"), list)


def _is_transient_failure(result: InternalResultRead) -> bool:
    """True when *result* is a transport/provider failure, not a genuine
    judgment (F1, both reviewers): empty ``raw`` — the aux call itself never
    produced text (timeout/provider error, ``adapters/internal_runner.py``) —
    AND *parsed* does not already carry a valid ``{"seeds": [...]}`` shape.
    Mirrors ``ThoughtProcessingApply``'s ``not raw.strip()`` transient-failure
    guard (``core/thought_processing.py``): a real result, even one whose
    ``seeds`` list is genuinely empty ("nothing lingered"), is NOT transient —
    only the "the call never happened" case is.

    NOTE this alone is NOT the full malformed-response guard: a NON-empty
    ``raw`` whose ``parsed`` still isn't :func:`_has_valid_seeds_shape` (the
    model responded, but not into a shape we can validate) returns ``False``
    here — ``step`` below checks :func:`_has_valid_seeds_shape` again,
    separately, to catch that case too (review-2 G1)."""
    if result.raw.strip():
        return False
    return not _has_valid_seeds_shape(result.parsed)


def _append_consumed_ring(
    existing: tuple[str, ...], new_ids: Sequence[str], *, cap: int
) -> tuple[str, ...]:
    """Append *new_ids* to the consumed-source-id ring, deduped and bounded to
    the most-recent *cap* entries (Task 1's bound, enforced HERE where the ring
    is appended — ``State`` itself just persists whatever tuple it is handed)."""
    combined = list(dict.fromkeys((*existing, *new_ids)))
    if len(combined) > cap:
        combined = combined[-cap:]
    return tuple(combined)


def _log_aux_raw(ctx: TickContext, raw: str) -> None:
    """Log the aux call's raw output on the apply span (D10 — every internal-
    cognition pass's completion must make what the model actually said visible
    in the traces, not just a derived reason code). Observability-only: never
    persisted, never read for control flow."""
    if ctx.logger is not None:
        ctx.logger.span.set(aux_raw=str(raw)[:2000])


NOTICING_APPLY_ID = "noticing-apply"


class NoticingApply:
    """Turn a completed noticing pass's typed result into durable thoughts.

    The runner's injected ``apply`` for a SUBJECTLESS completion — guards on
    :attr:`~lifemodel.state.model.State.pending_internal_subject_id` being
    ``None`` (a subject-SET completion is a processing pass, not ours; mirrors
    ``ThoughtProcessingApply``'s guard, inverted) plus a matching
    ``internal_result`` signal. See the module docstring for why the surveyed
    segment is recovered via :meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.claimed`
    (the immutable snapshot keyed by the correlation-id-encoded ``survey_id``)
    rather than a fresh ``closed_segment`` read, and why the cursor advance is an
    emitted :class:`~lifemodel.core.intents.FinalizeBuffer` (atomic with the thought
    commit, codex I3) rather than a direct buffer clear.

    *memory* (F4) is the store's :class:`~lifemodel.ports.memory.MemoryPort` —
    used ONLY to check whether a seed's content-digest id already exists as a
    row in ANY state (not just live), so a terminal (resolved/dropped/expired/
    merged) thought is never resurrected and its immutable creation provenance
    never overwritten. ``None`` (a bare unit-test construction with no store)
    degrades to checking only the live-thoughts snapshot — the composition
    root always wires the real store, so this only matters off-host.
    """

    id: str = NOTICING_APPLY_ID

    def __init__(self, buffer: NoticingBuffer, *, memory: MemoryPort | None = None) -> None:
        self._buffer = buffer
        self._memory = memory

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        if ctx.state.pending_internal_subject_id is not None:
            return []  # a subject-set completion is a processing pass, not ours
        correlation_id = ctx.state.pending_internal_id
        if correlation_id is None:
            return []
        result = next(
            (
                read_internal_result(s)
                for s in ctx.signals
                if s.kind == KIND_INTERNAL_RESULT
                and s.payload.get("correlation_id") == correlation_id
            ),
            None,
        )
        if result is None:
            return []
        _log_aux_raw(ctx, result.raw)
        # Parse the survey_id FIRST (codex): the transient/malformed refund paths
        # below must RELEASE the claim (not merely leave it stranded), and that
        # needs the survey_id in hand. A correlation that isn't ours degrades to
        # nothing-lingered — there is no claim of ours behind it to release.
        survey_id = parse_noticing_correlation(correlation_id)
        if survey_id is None:
            self._log(ctx, NoticingReason.NOTHING_LINGERED, count=0, parsed=result.parsed)
            return []
        if _is_transient_failure(result):
            # The aux call itself failed/timed out (empty raw) — refund the attempt
            # like ThoughtProcessingApply's TRANSIENT_FAILURE (F1), but RELEASE the
            # claim so the SAME segment is re-surveyed by the next eligible tick
            # rather than stranded ``claimed`` until a restart (the plan's promised
            # clean retry). A no-op if the claim was already finalized/released.
            self._release_claim(ctx, survey_id)
            self._log(ctx, NoticingReason.TRANSIENT_FAILURE, count=0, parsed=result.parsed)
            return []
        # The IMMUTABLE snapshot the launch claimed (codex I2) — never a fresh
        # closed_segment recompute against a ring the lane may have evicted/extended
        # during the async gap. Empty means the claim was already finalized/released
        # (a duplicate completion, or a boot-recovered pass): nothing to do, nothing
        # to release, and crucially DO NOT finalize (there is nothing claimed).
        segment = self._buffer.claimed(survey_id)
        if not segment:
            self._log(ctx, NoticingReason.NOTHING_LINGERED, count=0, parsed=result.parsed)
            return []
        if not _has_valid_seeds_shape(result.parsed):
            # The aux call DID respond (raw non-empty, so _is_transient_failure
            # above already said "not transient") but the response never took the
            # {"seeds": [...]} shape — a malformed/adversarial reply, not a genuine
            # "nothing lingered" judgment (review-2 G1). Treated like the transient
            # case: RELEASE the claim so a LATER pass gets a genuine chance to notice
            # this segment instead of it being silently swept away by a parse failure.
            self._release_claim(ctx, survey_id)
            self._log(ctx, NoticingReason.TRANSIENT_FAILURE, count=0, parsed=result.parsed)
            return []
        segment_ids = frozenset(
            {entry.turn_id for entry in segment}
            | {sid for entry in segment for sid in entry.source_ids}
        )
        segment_turn_ids = frozenset(entry.turn_id for entry in segment)
        consumed = frozenset(ctx.state.noticed_source_ids)
        seeds = validate_noticed_seeds(
            result.parsed,
            segment_ids=segment_ids,
            segment_turn_ids=segment_turn_ids,
            consumed=consumed,
        )

        intents: list[Intent] = self._seed_intents(ctx, seeds)
        # A belief seed is not a thought — keep the thought_ids span field honest.
        # Its source ids still ride the consumed ring below (dedup applies to belief
        # evidence too); the belief's own id/subject/confidence/sensitivity are logged
        # separately (redacted, no content) by ``_seed_intents``.
        thought_ids = [seed_thought_id(seed.gist) for seed in seeds if seed.kind != "belief"]
        new_source_ids = [sid for seed in seeds for sid in seed.source_message_ids]
        if new_source_ids:
            updated_ring = _append_consumed_ring(
                ctx.state.noticed_source_ids, new_source_ids, cap=NOTICED_SOURCE_IDS_CAP
            )
            intents.append(UpdateState({"noticed_source_ids": updated_ring}))

        # The segment was genuinely surveyed (a real result matched a real,
        # still-claimed snapshot) whether or not any seed survived validation —
        # advance the cursor either way, so a fruitless pass is never re-shown the
        # same old turns forever. FinalizeBuffer (not a direct clear) so the
        # claimed-row DELETE lands ATOMICALLY with the thought PutRecord + consumed
        # ring in the tick's one commit_tick transaction (codex I3): a rollback
        # leaves neither the thoughts nor the finalize.
        intents.append(FinalizeBuffer(survey_id))
        reason = NoticingReason.NOTICED if seeds else NoticingReason.NOTHING_LINGERED
        self._log(
            ctx,
            reason,
            count=len(seeds),
            thought_ids=thought_ids,
            source_ids=new_source_ids,
            parsed=result.parsed,
        )
        return intents

    def _release_claim(self, ctx: TickContext, survey_id: str) -> None:
        """Return the segment claimed under *survey_id* to ``complete`` so the NEXT
        eligible tick re-surveys it — the transient/malformed refund path (C2).

        **Design asymmetry (deliberate):** ``finalize`` MUST land atomically with
        the thought commit, so it goes through a :class:`~lifemodel.core.intents.FinalizeBuffer`
        intent; a ``release`` has NO companion state mutation, so it needs no
        atomicity and no intent machinery — a direct, fail-soft ``self._buffer.release``
        (a short UPDATE, consistent with the ``self._buffer.claimed`` read this same
        ``step`` already makes off-lock). Fail-soft: the apply runs OFF-lock in the
        completion frame, and a release hiccup must never crash that frame — it rides
        the span (a tick component logs only through its SpanBoundLogger, spec §4.5),
        never stdlib logging."""
        try:
            self._buffer.release(survey_id)
        except Exception as exc:  # noqa: BLE001 - a release hiccup must never crash the frame
            if ctx.logger is not None:
                ctx.logger.span.set(noticing_release_error=str(exc)[:500])

    def _seed_intents(self, ctx: TickContext, seeds: Sequence[NoticedSeed]) -> list[Intent]:
        live_ids = {t.id for t in live_thoughts(ctx.objects)}
        live_belief_ids = {b.id for b in live_beliefs(ctx.objects)}
        seen_thought_ids: set[str] = set()
        seen_belief_ids: set[str] = set()
        belief_log: list[JsonObject] = []
        intents: list[Intent] = []
        for seed in seeds:
            if seed.kind == "belief":
                # A belief seed routes to a Belief (its own kind/row), NOT a Thought.
                intent = self._belief_intent(
                    ctx, seed, seen_belief_ids, live_belief_ids, belief_log
                )
                if intent is not None:
                    intents.append(intent)
                continue
            thought_id = seed_thought_id(seed.gist)
            if thought_id in seen_thought_ids:
                # Two validated seeds can share a gist (→ the same content-
                # digest id) while citing DISJOINT source ids — both pass
                # source-validation independently, but must not both become a
                # PutRecord for the SAME id in one call (review-2 G3): the
                # later would silently win with different provenance. Skip
                # any seed whose id THIS call already scheduled a put for.
                continue
            if self._row_already_exists(thought_id, live_ids):
                # A row for this content-digest id already exists — in ANY
                # state, not just live (F4). Never re-seed it: a terminal
                # (resolved/dropped/expired/merged) row must never be
                # resurrected back to active, and its immutable creation
                # provenance must never be silently overwritten.
                continue
            seen_thought_ids.add(thought_id)
            provenance = creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="cognition",
                reason="noticed",
                source_object_ids=seed.source_message_ids,
                turn_id=seed.turn_id,
            )
            thought = build_thought(
                id=thought_id,
                content=seed.gist,
                trigger="noticed",
                salience=seed.salience,
                source="noticing",
                provenance=provenance,
            )
            intents.append(PutRecord(op=PutOp(draft=encode_thought(thought))))
        # D10 (tightened): a created belief's id/subject/confidence/sensitivity ride
        # the span — NEVER its content (the model's words already live once in
        # ``aux_raw``; a reason/span field must not re-log the proposition).
        if belief_log and ctx.logger is not None:
            ctx.logger.span.set(beliefs=belief_log)
        return intents

    def _belief_intent(
        self,
        ctx: TickContext,
        seed: NoticedSeed,
        seen_belief_ids: set[str],
        live_belief_ids: set[str],
        belief_log: list[JsonObject],
    ) -> Intent | None:
        """Build the :class:`~lifemodel.domain.objects.belief.Belief` PutRecord for a
        validated ``belief`` seed, or ``None`` to drop it.

        The belief id is CONTENT-scoped (a constant ``"noticing"`` anchor + content
        fingerprint, NOT the survey_id): an exact-duplicate belief upserts ONE row
        rather than minting a fresh duplicate on every re-notice (the partial dedup
        v1 affords; broader semantic dedup is deferred). ``belief_from_seed_fields``
        strictly re-parses the untrusted seed and can still raise :class:`InvalidPayload`
        on a case validation above did not (e.g. content that isn't UTF-8 encodable) —
        that is caught and DROPPED, never crashing the off-lock completion frame.
        A within-batch id already scheduled this call is skipped (G3, mirrors the
        thought path): two seeds with identical content would otherwise emit two
        puts for one row.

        A belief whose id already names a row in ANY state is skipped too (F1,
        mirrors the thought path's ``_row_already_exists`` guard EXACTLY): a
        blind ``MemoryPort.put`` upsert would otherwise resurrect a terminal
        (superseded/dropped/expired) belief back to ``active`` once the
        forget/reconcile path lands (lm-705.20), and — reachable already in v1 —
        wholesale-overwrite an active belief's evidence/confidence/creation
        provenance on a re-notice of identical content. A skipped belief still
        consumes its source ids into the dedup ring (its seed rides ``seeds`` in
        ``step``'s ``new_source_ids`` regardless), identical to how a
        skipped-existing thought behaves — the two paths never diverge."""
        provenance = creation_provenance(
            ctx.trace,
            created_by=self.id,
            component="cognition",
            reason="believed",
            source_object_ids=seed.source_message_ids,
            turn_id=seed.turn_id,
        )
        try:
            belief = belief_from_seed_fields(
                source_thought_id="noticing",
                fields={
                    "content": seed.content,
                    "confidence": seed.confidence,
                    "subject": "owner",
                    "sensitivity": seed.sensitivity,
                },
                source_message_ids=seed.source_message_ids,
                salience=seed.salience,
                provenance=provenance,
            )
        except InvalidPayload:
            return None  # bad model data (e.g. non-encodable content) — drop, never crash
        if belief.id in seen_belief_ids:
            return None  # this call already scheduled a put for this belief id (G3)
        if self._belief_row_already_exists(belief.id, live_belief_ids):
            # A row for this content-scoped belief id already exists — in ANY state,
            # not just live (F1). Never re-seed it: a terminal (superseded/dropped/
            # expired) belief must never be resurrected back to active, and an active
            # belief's immutable creation provenance/evidence/confidence must never be
            # silently overwritten. Mirrors the thought path's guard above exactly.
            return None
        seen_belief_ids.add(belief.id)
        belief_log.append(
            {
                "id": belief.id,
                "subject": belief.subject,
                "confidence": belief.confidence,
                "sensitivity": belief.sensitivity.value,
            }
        )
        return PutRecord(op=PutOp(draft=encode_belief(belief)))

    def _row_already_exists(self, thought_id: str, live_ids: set[str]) -> bool:
        """True iff *thought_id* already names a row in ANY state (F4).

        ``ctx.objects`` is the tick's LIVE-only snapshot (active/parked — see
        ``core/coreloop.py``'s snapshot docstring), so it alone can't see a
        terminal row; *live_ids* only shortcuts the common live-match case
        without a store round-trip. When a real store is wired, ``memory.get``
        is the authority — it sees every state, live or terminal."""
        if thought_id in live_ids:
            return True
        if self._memory is not None:
            return self._memory.get(THOUGHT_KIND, thought_id) is not None
        return False

    def _belief_row_already_exists(self, belief_id: str, live_belief_ids: set[str]) -> bool:
        """True iff *belief_id* already names a belief row in ANY state (F1).

        The belief-track sibling of :meth:`_row_already_exists`, built identically:
        ``ctx.objects`` is the tick's LIVE-only snapshot (active — see
        ``core/coreloop.py``), so *live_belief_ids* only shortcuts the common
        live-match case without a store round-trip; when a real store is wired,
        ``memory.get`` is the authority — it sees every state, live or terminal
        (superseded/dropped/expired). ``None`` memory (a bare unit-test construction)
        degrades to the live snapshot only, exactly like the thought path."""
        if belief_id in live_belief_ids:
            return True
        if self._memory is not None:
            return self._memory.get(BELIEF_KIND, belief_id) is not None
        return False

    def _log(
        self,
        ctx: TickContext,
        reason: NoticingReason,
        *,
        count: int,
        thought_ids: Sequence[str] = (),
        source_ids: Sequence[str] = (),
        parsed: JsonObject | None = None,
    ) -> None:
        if ctx.logger is None:
            return
        ctx.logger.span.set(noticing_reason=reason.value, noticed_count=count)
        if thought_ids:
            ctx.logger.span.set(thought_ids=list(thought_ids))
        if source_ids:
            ctx.logger.span.set(source_ids=list(source_ids))
        # The model's whole-pass reflection rides the span (D10/FR24 debug),
        # never the thought — ALWAYS logged when present, including the
        # nothing_lingered path (that is the whole point: capture WHY nothing
        # lingered, not just THAT nothing did).
        if isinstance(parsed, dict) and isinstance(parsed.get("reflection"), str):
            ctx.logger.span.set(reflection=str(parsed["reflection"])[:1000])

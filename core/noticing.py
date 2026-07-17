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

**Threading the session id (and the exact surveyed prefix) across the async
gap:** the ONLY channel that survives from launch to completion is the
``correlation_id`` string (mirrors ``ThoughtProcessingSelector``'s
``process-<thought_id>@<iso>``). The trigger encodes
``notice-<session_id>@<anchor_turn_id>@<iso>``, where ``anchor_turn_id`` is the
LAST entry's ``turn_id`` in the segment it just surveyed. The apply recovers
both and reads the buffer via
:meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.segment_through` —
**not** a fresh :meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.closed_segment`
call. That distinction is load-bearing: ``closed_segment`` reapplies the
closed-prefix gate, which returns ``[]`` for the WHOLE lane the instant a NEW
turn opens on it — entirely plausible during the async call's own window (up
to ``DEFAULT_TIMEOUT_SECONDS``, ``adapters/internal_runner.py``) — and would
then reject every otherwise-good seed from a pass that was itself triggered by
a lively (size-cap) burst on that very lane. ``segment_through`` instead reads
the exact prefix the trigger already gated once at launch time, un-gated by
whatever the lane is doing now, and the apply clears through that SAME anchor
— never a larger, freshly-recomputed segment — so a turn that arrives during
the async gap is left in the ring for a LATER pass, never silently swept away
un-surveyed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ..domain.memory import JsonObject, PutOp
from ..domain.objects import Thought
from ..ports.memory import MemoryPort
from ..ports.tracer import format_traceparent
from ..state.model import NOTICED_SOURCE_IDS_CAP
from .budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from .component import TickContext
from .intents import Intent, LaunchInternalCognition, PutRecord, UpdateState
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
                },
                "required": ["gist", "source_message_ids"],
                "additionalProperties": False,
            },
        },
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
    "out), and 'salience' (0 to 1, how much this deserves attention)."
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
    segment's turns (each tagged with its citable ``turn_id``) plus a short
    continuity backlog of what is already being turned over.

    Bounded to at most the most-recent *size_cap* entries of *segment* (codex
    F2b) — NEVER the whole closed-prefix ring, which can hold up to
    :data:`~lifemodel.core.noticing_buffer.NoticingBuffer`'s ``max_entries``
    (256) turns on an idle-triggered long-lived lane. The window is a plain
    tail slice: the anchor the caller clears/validates against is always the
    segment's true LAST entry, unaffected by this display-only trim."""
    window = segment[-size_cap:] if size_cap > 0 and len(segment) > size_cap else segment
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
#: ``notice-<session_id>@<anchor_turn_id>@<iso>`` — parsed back by
#: :func:`_parse_noticing_correlation` at apply time. Assumes neither
#: *session_id* nor a *turn_id* contains ``@`` (true of every id this codebase
#: mints: platform session keys and buffer ``turn_id``s are plain
#: alnum/dash/colon tokens).
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
            self._log(ctx, blocked)
            return []
        backlog = live_thoughts(ctx.objects)[:BACKLOG_TOP_M]
        anchor_turn_id = segment[-1].turn_id
        intent = LaunchInternalCognition(
            prompt=build_noticing_prompt(segment, backlog, size_cap=self._size_cap),
            correlation_id=f"{_CORRELATION_PREFIX}{session_id}@{anchor_turn_id}@{to_iso(ctx.now)}",
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
    real source ids (anti-hallucination) and not already consumed (dedup)."""

    gist: str
    source_message_ids: tuple[str, ...]
    turn_id: str | None
    salience: float


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
    return NoticedSeed(
        gist=gist.strip(), source_message_ids=source_ids, turn_id=turn_id, salience=salience
    )


def _parse_noticing_correlation(correlation_id: str) -> tuple[str, str] | None:
    """Recover ``(session_id, anchor_turn_id)`` from a
    ``notice-<session_id>@<anchor_turn_id>@<iso>`` correlation id (the format
    :class:`NoticingTrigger` mints). ``None`` for any other shape — a foreign/
    malformed correlation id is never guessed at."""
    if not correlation_id.startswith(_CORRELATION_PREFIX):
        return None
    rest = correlation_id[len(_CORRELATION_PREFIX) :]
    head, sep, _timestamp = rest.rpartition("@")
    if not sep:
        return None
    session_id, sep2, anchor_turn_id = head.rpartition("@")
    if not sep2 or not session_id or not anchor_turn_id:
        return None
    return session_id, anchor_turn_id


def _is_transient_failure(result: InternalResultRead) -> bool:
    """True when *result* is a transport/provider failure, not a genuine
    judgment (F1, both reviewers): empty ``raw`` — the aux call itself never
    produced text (timeout/provider error, ``adapters/internal_runner.py``) —
    AND *parsed* does not already carry a valid ``{"seeds": [...]}`` shape.
    Mirrors ``ThoughtProcessingApply``'s ``not raw.strip()`` transient-failure
    guard (``core/thought_processing.py``): a real result, even one whose
    ``seeds`` list is genuinely empty ("nothing lingered"), is NOT transient —
    only the "the call never happened" case is."""
    if result.raw.strip():
        return False
    return not (isinstance(result.parsed, dict) and isinstance(result.parsed.get("seeds"), list))


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


NOTICING_APPLY_ID = "noticing-apply"


class NoticingApply:
    """Turn a completed noticing pass's typed result into durable thoughts.

    The runner's injected ``apply`` for a SUBJECTLESS completion — guards on
    :attr:`~lifemodel.state.model.State.pending_internal_subject_id` being
    ``None`` (a subject-SET completion is a processing pass, not ours; mirrors
    ``ThoughtProcessingApply``'s guard, inverted) plus a matching
    ``internal_result`` signal. See the module docstring for why the surveyed
    segment is recovered via :meth:`~lifemodel.core.noticing_buffer.NoticingBuffer.segment_through`
    (keyed by the correlation-id-encoded anchor turn id) rather than a fresh
    ``closed_segment`` read.

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
        if _is_transient_failure(result):
            # The aux call itself failed/timed out — refund the attempt exactly
            # like ThoughtProcessingApply's TRANSIENT_FAILURE (F1): leave the
            # surveyed segment, the consumed ring, and the cursor all untouched
            # so a LATER pass gets a genuine chance to notice it, instead of the
            # segment being lost forever to a transient provider hiccup.
            self._log(ctx, NoticingReason.TRANSIENT_FAILURE, count=0)
            return []
        parsed_correlation = _parse_noticing_correlation(correlation_id)
        if parsed_correlation is None:
            self._log(ctx, NoticingReason.NOTHING_LINGERED, count=0)
            return []
        session_id, anchor_turn_id = parsed_correlation
        segment = self._buffer.segment_through(session_id, anchor_turn_id)
        if not segment:
            self._log(ctx, NoticingReason.NOTHING_LINGERED, count=0)
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
        thought_ids = [seed_thought_id(seed.gist) for seed in seeds]
        new_source_ids = [sid for seed in seeds for sid in seed.source_message_ids]
        if new_source_ids:
            updated_ring = _append_consumed_ring(
                ctx.state.noticed_source_ids, new_source_ids, cap=NOTICED_SOURCE_IDS_CAP
            )
            intents.append(UpdateState({"noticed_source_ids": updated_ring}))

        # The segment was genuinely surveyed (a real result matched a real,
        # still-present prefix) whether or not any seed survived validation —
        # advance the cursor either way, so a fruitless pass is never re-shown
        # the same old turns forever.
        self._buffer.clear_through(session_id, anchor_turn_id)
        reason = NoticingReason.NOTICED if seeds else NoticingReason.NOTHING_LINGERED
        self._log(ctx, reason, count=len(seeds), thought_ids=thought_ids, source_ids=new_source_ids)
        return intents

    def _seed_intents(self, ctx: TickContext, seeds: Sequence[NoticedSeed]) -> list[Intent]:
        live_ids = {t.id for t in live_thoughts(ctx.objects)}
        intents: list[Intent] = []
        for seed in seeds:
            thought_id = seed_thought_id(seed.gist)
            if self._row_already_exists(thought_id, live_ids):
                # A row for this content-digest id already exists — in ANY
                # state, not just live (F4). Never re-seed it: a terminal
                # (resolved/dropped/expired/merged) row must never be
                # resurrected back to active, and its immutable creation
                # provenance must never be silently overwritten.
                continue
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
        return intents

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

    def _log(
        self,
        ctx: TickContext,
        reason: NoticingReason,
        *,
        count: int,
        thought_ids: Sequence[str] = (),
        source_ids: Sequence[str] = (),
    ) -> None:
        if ctx.logger is None:
            return
        ctx.logger.span.set(noticing_reason=reason.value, noticed_count=count)
        if thought_ids:
            ctx.logger.span.set(thought_ids=list(thought_ids))
        if source_ids:
            ctx.logger.span.set(source_ids=list(source_ids))

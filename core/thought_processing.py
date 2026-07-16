"""The waking mind's rumination brain (lm-705.2, spec §3.2/§4.1/§4.5).

Two 0-LLM cognition components ride the non-delivering internal-cognition seam
(lm-705.6): :class:`ThoughtProcessingSelector` (heartbeat) picks ONE live thought
and emits a :class:`~lifemodel.core.intents.LaunchInternalCognition`;
:class:`ThoughtProcessingApply` (completion-frame) turns the typed aux result into
the thought's next state. The lifecycle rules — attempt/park bounds — live here
(spec §4.1: "a required contract, not a hope"), so a thought is chewed a bounded
number of times and then terminates (``resolve``/``drop``/``expire``), never spirals.

Non-delivery is structural (the seam calls the ``LlmPort``, never egress). No
residue/opinion is written (spec §4.1) — the ``reflection`` rides the span for FR24
debug, never the thought. Every from-state is ``active`` (the selector re-arms
expired-parked thoughts to ``active`` first), so no transition is a forbidden
``active→active``/``parked→parked`` self-loop (``domain/objects/thought.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from ..domain.memory import JsonObject, MemoryPatch, TransitionOp
from ..domain.objects import Thought, ThoughtState
from ..ports.tracer import format_traceparent
from .budget import (
    DEFAULT_DAILY_INTERNAL_CALL_CEILING,
    DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    internal_budget_available,
    internal_interval_elapsed,
)
from .component import TickContext
from .intents import Intent, LaunchInternalCognition, TransitionRecord
from .taxonomy import KIND_INTERNAL_RESULT, read_internal_result
from .thought_view import live_thoughts
from .timeutil import from_iso, to_iso

#: How many non-progress (malformed/no-parse) attempts a thought tolerates before it
#: is ``drop``ped (spec §4.1 "max total processing attempts"). Distinct from park.
MAX_NO_PROGRESS_COUNT = 3
#: How many park cycles before a thought ``expire``s rather than re-arming (spec §4.1).
MAX_PARK_CYCLES = 3
#: The widening park backoff (spec §4.1) — the 6h/24h/72h ladder the Thought schema's
#: ``park_count`` docstring already names (``domain/objects/thought.py``). Indexed by
#: the pre-increment ``park_count``, clamped to the last rung.
PARK_BACKOFFS: tuple[timedelta, ...] = (
    timedelta(hours=6),
    timedelta(hours=24),
    timedelta(hours=72),
)

THOUGHT_KIND = "thought"

#: The processing pass's typed result contract (spec §4.1 "deterministic schema +
#: validation"). ``outcome`` is the disposition; ``reflection`` is a short first-person
#: note that rides the span (FR24 debug), never persisted.
PROCESSING_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["resolve", "park", "drop"]},
        "reflection": {"type": "string"},
    },
    "required": ["outcome"],
    "additionalProperties": False,
}

#: The pass's system framing — first-person, private, non-delivered.
PROCESSING_INSTRUCTIONS = (
    "You are the being's own private mind, quietly turning over one of your thoughts. "
    "Nothing you write here is shown to anyone — this is rumination, not a message. "
    "Reflect briefly, in the first person, then decide the thought's disposition: "
    "'resolve' if you have thought it through and it needs nothing more; "
    "'park' if it is worth returning to later but not now; "
    "'drop' if it no longer matters. "
    "Answer as JSON: an 'outcome' of resolve/park/drop and a short 'reflection'."
)


class ProcessingReason(StrEnum):
    """The closed set of processing-decision reasons (spec §5) — positive choices, NOT
    suppressions. Logged as a span field alongside ``thought_id``, never in a string."""

    # selector
    CHOSE_PROCESS = "chose_process"
    SKIPPED_EMPTY_BACKLOG = "skipped_empty_backlog"
    SKIPPED_IN_FLIGHT = "skipped_in_flight"
    SKIPPED_NO_BUDGET = "skipped_no_budget"
    SKIPPED_INTERVAL = "skipped_interval"
    # apply
    RESOLVED = "processed_resolve"
    PARKED = "processed_park"
    DROPPED = "processed_drop"
    EXPIRED_PARK_CAP = "processed_expired_park_cap"
    PARKED_NO_PROGRESS = "processed_park_no_progress"
    DROPPED_NO_PROGRESS = "processed_drop_no_progress"
    TRANSIENT_FAILURE = "processed_transient_failure"
    NO_SUBJECT = "processed_no_subject"


@dataclass(frozen=True)
class ProcessingDecision:
    """A pure decision: the guarded transition to apply (or ``None`` for a transient
    failure that leaves the thought untouched) plus the closed reason for the span."""

    transition: TransitionOp | None
    reason: ProcessingReason


def build_processing_prompt(thought: Thought) -> str:
    """The bounded input_text handed to the aux call — the thought and its history."""
    revisited = (
        f"\n\n(You have revisited this {thought.no_progress_count} time(s) without resolving it.)"
        if thought.no_progress_count
        else ""
    )
    return f"The thought you are turning over:\n\n{thought.content}{revisited}"


def _transition(thought: Thought, to: ThoughtState, merge: JsonObject) -> TransitionOp:
    return TransitionOp(
        kind=THOUGHT_KIND,
        id=thought.id,
        from_state=ThoughtState.ACTIVE.value,
        to_state=to.value,
        patch=MemoryPatch(payload_merge=merge),
    )


def _park_or_terminate(thought: Thought, *, now: datetime, no_progress: bool) -> ProcessingDecision:
    """Park with a widening backoff, or terminate at a bound. Bumps ``no_progress_count``
    when *no_progress* (a malformed attempt), always bumps ``park_count``."""
    new_np = thought.no_progress_count + (1 if no_progress else 0)
    new_park = thought.park_count + 1
    if no_progress and new_np >= MAX_NO_PROGRESS_COUNT:
        return ProcessingDecision(
            _transition(thought, ThoughtState.DROPPED, {"no_progress_count": new_np}),
            ProcessingReason.DROPPED_NO_PROGRESS,
        )
    if new_park > MAX_PARK_CYCLES:
        return ProcessingDecision(
            _transition(
                thought,
                ThoughtState.EXPIRED,
                {"no_progress_count": new_np, "park_count": new_park},
            ),
            ProcessingReason.EXPIRED_PARK_CAP,
        )
    backoff = PARK_BACKOFFS[min(thought.park_count, len(PARK_BACKOFFS) - 1)]
    merge: JsonObject = {
        "no_progress_count": new_np,
        "park_count": new_park,
        "parked_until": to_iso(now + backoff),
    }
    reason = ProcessingReason.PARKED_NO_PROGRESS if no_progress else ProcessingReason.PARKED
    return ProcessingDecision(_transition(thought, ThoughtState.PARKED, merge), reason)


def decide_processing_transition(
    thought: Thought, *, parsed: JsonObject | None, raw: str, now: datetime
) -> ProcessingDecision:
    """Map an aux result to the thought's next state (pure; spec §4.1).

    ``resolve``/``drop`` are terminal. ``park`` backs off (or expires at the park cap).
    A malformed result (no valid ``outcome``, but the model DID respond) is a
    no-progress attempt → park+bump (or drop at the no-progress cap). A TRANSIENT
    failure (empty ``raw`` — the call itself failed/timed out) leaves the thought
    untouched, so provider flakiness never drops a good thought (refund-of-attempt)."""
    outcome = parsed.get("outcome") if isinstance(parsed, dict) else None
    if outcome == "resolve":
        return ProcessingDecision(
            _transition(thought, ThoughtState.RESOLVED, {}), ProcessingReason.RESOLVED
        )
    if outcome == "drop":
        return ProcessingDecision(
            _transition(thought, ThoughtState.DROPPED, {}), ProcessingReason.DROPPED
        )
    if outcome == "park":
        return _park_or_terminate(thought, now=now, no_progress=False)
    if not raw.strip():
        return ProcessingDecision(None, ProcessingReason.TRANSIENT_FAILURE)
    return _park_or_terminate(thought, now=now, no_progress=True)


THOUGHT_PROCESSING_SELECTOR_ID = "thought-processing-selector"


class ThoughtProcessingSelector:
    """Pick ONE live thought to ruminate on this tick, and re-arm expired parks (§4.1).

    0-LLM: it only emits intents. Re-arms every parked thought past its ``parked_until``
    (``parked→active``) so parking means "return later", not "shelve till expiry". Then,
    if the gates pass (single-flight, FR20 budget, min interval), emits ONE
    ``LaunchInternalCognition`` for the top-salience ACTIVE thought — the being's private,
    non-delivered pass. Emits no launch (idle 0-LLM, S5) when the active backlog is empty
    or any gate holds; the reason is a span field either way (spec §5).

    **Heartbeat-only coupling is enforced by the DISPATCH SITE, not this component.**
    This selector is a normal registered component: it runs on EVERY frame the
    CoreLoop schedules it for and emits ``LaunchInternalCognition`` on every frame it
    runs, gates permitting — ``TickContext`` carries no trigger/frame-kind, so the
    selector cannot self-restrict to heartbeats. "No rumination during a live dialogue
    turn" holds SOLELY because only ``being_platform._tick`` (the HEARTBEAT tick) reads
    ``report.internal_launches`` and drives it into the runner; the EVENT,
    ASYNC_COMPLETION, and ADMIN callers ignore that field and drop the launch on the
    floor. If a future change wires ``internal_launches`` into a non-heartbeat
    dispatch path, this component will silently start ruminating mid-dialogue — that
    invariant lives entirely at the dispatch site, not here."""

    id: str = THOUGHT_PROCESSING_SELECTOR_ID

    def __init__(
        self,
        *,
        daily_ceiling: int = DEFAULT_DAILY_INTERNAL_CALL_CEILING,
        min_interval: timedelta = DEFAULT_MIN_INTERPROCESSING_INTERVAL,
    ) -> None:
        self._daily_ceiling = daily_ceiling
        self._min_interval = min_interval

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        thoughts = live_thoughts(ctx.objects)
        intents: list[Intent] = []
        actives = []
        rearmed_count = 0
        for t in thoughts:
            if t.state == ThoughtState.PARKED.value:
                if self._parked_is_due(t, ctx.now):
                    intents.append(
                        TransitionRecord(
                            op=TransitionOp(
                                kind=THOUGHT_KIND,
                                id=t.id,
                                from_state=ThoughtState.PARKED.value,
                                to_state=ThoughtState.ACTIVE.value,
                            )
                        )
                    )
                    rearmed_count += 1
            elif t.state == ThoughtState.ACTIVE.value:
                actives.append(t)

        reason, subject = self._pick(ctx, actives)
        if subject is not None:
            intents.append(
                LaunchInternalCognition(
                    prompt=build_processing_prompt(subject),
                    correlation_id=f"process-{subject.id}@{to_iso(ctx.now)}",
                    origin_traceparent=format_traceparent(ctx.trace),
                    subject_id=subject.id,
                    instructions=PROCESSING_INSTRUCTIONS,
                    json_schema=PROCESSING_JSON_SCHEMA,
                )
            )
        if ctx.logger is not None:
            ctx.logger.span.set(processing_reason=reason.value)
            if subject is not None:
                ctx.logger.span.set(thought_id=subject.id)
            if rearmed_count:
                # Re-arms (parked→active) are a separate observable event from the
                # tick's pick *reason* (§5) — a count, not a reason code, since more
                # than one thought can re-arm in the same tick the pick decides.
                ctx.logger.span.set(unparked=rearmed_count)
        return intents

    def _parked_is_due(self, thought: Thought, now: datetime) -> bool:
        if not thought.parked_until:
            return True  # parked with no window set → treat as due (defensive)
        try:
            return from_iso(thought.parked_until) <= now
        except (ValueError, TypeError):
            return True

    def _pick(
        self, ctx: TickContext, actives: list[Thought]
    ) -> tuple[ProcessingReason, Thought | None]:
        if not actives:
            return ProcessingReason.SKIPPED_EMPTY_BACKLOG, None
        if ctx.state.pending_internal_id is not None:
            return ProcessingReason.SKIPPED_IN_FLIGHT, None
        if not internal_interval_elapsed(ctx.state, now=ctx.now, min_interval=self._min_interval):
            return ProcessingReason.SKIPPED_INTERVAL, None
        if not internal_budget_available(ctx.state, now=ctx.now, daily_ceiling=self._daily_ceiling):
            return ProcessingReason.SKIPPED_NO_BUDGET, None
        return ProcessingReason.CHOSE_PROCESS, actives[0]  # live_thoughts is salience-desc


THOUGHT_PROCESSING_APPLY_ID = "thought-processing-apply"


class ThoughtProcessingApply:
    """Turn a completed processing pass's typed result into the thought's next state.

    The runner's injected ``apply`` (lm-705.6): it runs only inside the
    ``ASYNC_COMPLETION`` frame :func:`~lifemodel.core.internal_cognition.run_internal_completion`
    seeds, so it guards on an ``internal_result`` signal + a matching in-flight subject
    and no-ops otherwise (a subjectless noticing pass, a cleared/terminal subject, or a
    non-completion frame all fall through to ``[]``). Emits at most one
    ``TransitionRecord`` (the atomic committer applies it under the lock)."""

    id: str = THOUGHT_PROCESSING_APPLY_ID

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        subject_id = ctx.state.pending_internal_subject_id
        if subject_id is None:
            return []
        result = next(
            (
                read_internal_result(s)
                for s in ctx.signals
                if s.kind == KIND_INTERNAL_RESULT
                and s.payload.get("correlation_id") == ctx.state.pending_internal_id
            ),
            None,
        )
        if result is None:
            return []
        thought = self._live_subject(ctx, subject_id)
        if thought is None:
            self._log(ctx, ProcessingReason.NO_SUBJECT, subject_id)
            return []
        decision = decide_processing_transition(
            thought, parsed=result.parsed, raw=result.raw, now=ctx.now
        )
        self._log(ctx, decision.reason, subject_id)
        # The model's first-person reflection rides the span, never the thought (spec
        # §4.1 — no residue field) — this is the ONE place it is read at all.
        if (
            ctx.logger is not None
            and isinstance(result.parsed, dict)
            and "reflection" in result.parsed
        ):
            ctx.logger.span.set(reflection=str(result.parsed.get("reflection", ""))[:500])
        return [TransitionRecord(op=decision.transition)] if decision.transition is not None else []

    def _live_subject(self, ctx: TickContext, subject_id: str) -> Thought | None:
        for t in live_thoughts(ctx.objects):
            if t.id == subject_id and t.state == ThoughtState.ACTIVE.value:
                return t
        return None

    def _log(self, ctx: TickContext, reason: ProcessingReason, thought_id: str) -> None:
        if ctx.logger is not None:
            ctx.logger.span.set(processing_reason=reason.value, thought_id=thought_id)

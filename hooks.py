"""Afferent hooks — where Hermes events start an ExecutionFrame (spec §3/§4/§5).

The nervous flow is ephemeral: a Hermes event does not write to a durable log, it
**starts a frame** that folds the reading into the being's durable state and dies
(spec §2/§3). These two hooks are the afferent boundary:

``make_inbound_observer`` — on a genuine (non-internal, non-own-impulse,
non-slash-command) inbound message, starts an ``EVENT`` frame carrying a
``contact_observed`` signal. Any slash-prefixed message is a control command
operating the tool, not dialogue — it is filtered at the sensor band-pass (spec
§4, "as the ear does not hear ultrasound") and never becomes contact.

``make_post_llm_observer`` — when the being's async proactive turn finishes, starts
an ``ASYNC_COMPLETION`` frame carrying a ``proactive_outcome`` signal
(``sent``/``silent``). That frame commits the resolution IMMEDIATELY (its own
frame), not at the next heartbeat (spec §3/§5): aggregation resolves the pending
desire and writes ``action_pending``/backoff to ``AgentState``.

Each hook builds a FRESH :class:`~lifemodel.composition.LifeModel` per call (via the
injected builder) so the frame loads state fresh under the one process-wide
state-actor lock — no cached ``State`` can drift between frames.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

from .composition import LifeModel
from .core.correlate import open_correlated_span
from .core.desire_view import read_live_contact_desire
from .core.frame import FrameTrigger, run_frame
from .core.metrics import MetricRegistry
from .core.suppression import SuppressionReason, emit_suppression_span
from .core.taxonomy import contact_observed_signal, proactive_outcome_signal
from .core.tick_metrics import OBSERVER_ERRORS
from .core.wake_packet import DECLINE_MARKER, IMPULSE_LABEL_PREFIX
from .domain.egress import ProactiveOutcome
from .domain.objects import DesireState
from .log import SpanBoundLogger
from .ports.memory import MemoryPort
from .state.brain_health import BrainHealth

#: Observer names — carried on the ``component`` metric label + keyed into
#: :attr:`BrainHealth.last_observer_error` when a body raises (spec §4.3).
POST_LLM_OBSERVER = "post_llm_call"
INBOUND_OBSERVER = "pre_gateway_dispatch"

_LOG = logging.getLogger("lifemodel.hooks")


def _record_observer_failure(
    *,
    observer_name: str,
    exc: Exception,
    health: BrainHealth | None,
    metrics: MetricRegistry | None,
) -> None:
    """Plugin-owned handling for an observer body that raised (spec §4.3/MAJOR-4).

    ERROR + full traceback ALWAYS (never rely on Hermes' hook wrapper); when the
    live *health* / *metrics* are wired (they always are from ``register()``),
    record the LAST error for this observer on :class:`BrainHealth` and bump the
    failure metric. Never re-raises — an afferent hiccup must not crash the host's
    dispatch. *health* / *metrics* are optional only so off-host unit tests of the
    frame behavior can construct an observer without the full backbone; production
    always passes both, so live failures are fully observable.
    """
    detail = f"{type(exc).__name__}: {exc}"
    _LOG.error("observer_body_failed observer=%s error=%s", observer_name, detail, exc_info=True)
    if health is not None:
        health.record_observer_error(observer_name, detail)
    if metrics is not None:
        metrics.inc(OBSERVER_ERRORS, component=observer_name)


#: The two disjoint concepts that map a proactive turn to SILENT (spec §5). They
#: are matched DIFFERENTLY on purpose (lm-md6.5):
#:
#: * :data:`_SUBSTRING_DECLINE_MARKERS` — the BRACKETED sentinel the wake-packet
#:   INSTRUCTS the being to reply with to decline
#:   (:data:`~lifemodel.core.wake_packet.DECLINE_MARKER`, spliced in from that single
#:   source of truth so the affordance we advertise can never drift out of what we
#:   classify as SILENT). It is matched as a SUBSTRING and so is FAIL-CLOSED: if it
#:   appears ANYWHERE in the response, the turn is a decline and is NOT delivered — a
#:   marker wrapped in deliberation prose ("...I won't nudge them. [SILENT]") no longer
#:   leaks the being's private "I won't write" reasoning to the owner. The brackets +
#:   caps make a false positive in natural prose ~impossible, which is exactly why ONLY
#:   the bracketed token is safe to substring-match.
#: * :data:`_EXACT_NO_REPLY_MARKERS` — the BARE-WORD markers Hermes' gateway also
#:   honours. These match ONLY when the WHOLE (normalised) response equals one: a
#:   substring search for "SILENT"/"NO REPLY" would fire on genuine prose ("we sat in
#:   silent comfort", "no reply is needed") and deliver FALSE silence, so bare words
#:   stay exact-only — never substring-matched.
_SUBSTRING_DECLINE_MARKERS = frozenset({DECLINE_MARKER})
_EXACT_NO_REPLY_MARKERS = frozenset({"NO_REPLY", "NO REPLY", "SILENT"})


def _is_no_reply(text: str) -> bool:
    """True when *text* is the being choosing silence (spec §5).

    Fail-closed on the bracketed sentinel: if :data:`DECLINE_MARKER` appears ANYWHERE
    in *text* (case-insensitively), the turn is a decline — a marker wrapped in prose
    is still a decline, never delivered. The bare-word markers ("NO_REPLY", "NO REPLY",
    "SILENT") match only when the WHOLE response, after collapsing whitespace and
    uppercasing (mirrors the host's own ``_canonical_silence_candidate``), equals one —
    so ``" NO_REPLY "`` and ``"no   reply"`` match, but prose that merely *mentions*
    "silent"/"no reply" is delivered normally (it is not substring-matched).
    """
    upper = text.upper()
    if any(marker.upper() in upper for marker in _SUBSTRING_DECLINE_MARKERS):
        return True
    candidate = " ".join(upper.split())
    return candidate in _EXACT_NO_REPLY_MARKERS


def _is_pending_proactive_turn(pending_proactive_id: str | None, user_message: str) -> bool:
    """Best-effort correlation to the outstanding proactive turn.

    A proactive outcome is only ever pending (gate 1) for our own composed impulse
    text (gate 2, spec design doc §5 guard (c)). A genuine user turn that lands
    while a desire is pending fails gate 2 and is correctly ignored.
    """
    if pending_proactive_id is None:
        return False
    return user_message.strip().startswith(IMPULSE_LABEL_PREFIX)


def _log_outcome_detail(
    logger: SpanBoundLogger,
    *,
    correlation_id: str,
    outcome: ProactiveOutcome,
    assistant_response: str,
    extra: dict[str, Any],
) -> None:
    """DEBUG: full discovery detail for a resolved proactive outcome (bead lm-otq).

    Emitted through the origin-trace :class:`~lifemodel.log.SpanBoundLogger` (§4.4)
    so this DEBUG detail is durable in ``trace_events`` under the attempt's trace —
    the 5th-source collapse (§4.3), not a hindsight/DEBUG-log side channel. The full
    (untruncated) assistant response rides it, plus a safe string preview of every
    extra kwarg the host passed, keyed by field name.
    """
    logger.debug(
        "proactive_outcome_detail",
        correlation_id=correlation_id,
        outcome=outcome.value,
        assistant_response=assistant_response,
        extra_fields={k: str(v)[:800] for k, v in extra.items()},
    )


#: Generous per-message cap on logged reasoning text — untruncated in practice
#: (host reasoning chains are far shorter than this), just a sanity ceiling so
#: one pathological payload can't blow up the debug sink.
_REASONING_LOG_CAP = 4000


def _summarize_reasoning_message(message: Any) -> dict[str, Any]:
    """Best-effort, defensive summary of one ``conversation_history`` entry.

    Entries are *expected* to be dicts with ``role``/``finish_reason``/
    ``tool_calls``/``reasoning`` keys (spec discovery, bead lm-otq step 1), but
    the host payload is untrusted shape — any key may be missing, or the entry
    may not be a dict at all. Never raises; degrades to ``None``/``False``.
    """
    if not isinstance(message, dict):
        return {
            "role": None,
            "finish_reason": None,
            "has_tool_calls": False,
            "tool_call_names": [],
            "reasoning": None,
            "entry_type": type(message).__name__,
        }
    tool_calls = message.get("tool_calls")
    tool_call_names: list[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            name = fn.get("name") if isinstance(fn, dict) else call.get("name")
            if name is not None:
                tool_call_names.append(str(name))
    reasoning = message.get("reasoning")
    return {
        "role": message.get("role"),
        "finish_reason": message.get("finish_reason"),
        "has_tool_calls": bool(tool_calls),
        "tool_call_names": tool_call_names,
        "reasoning": str(reasoning)[:_REASONING_LOG_CAP] if reasoning is not None else None,
    }


def _log_proactive_reasoning(
    logger: SpanBoundLogger, *, correlation_id: str, conversation_history: Any
) -> None:
    """DEBUG: the full reasoning chain behind a resolved proactive turn (lm-otq step 2).

    ``conversation_history`` (a ``post_llm_call`` kwarg the host passes alongside
    ``user_message``/``assistant_response``) is a list of message dicts; assistant
    entries may carry a ``reasoning`` string — the being's own chain of thought for
    that turn. This is the ONE event that surfaces it, per message, untruncated
    (capped generously). Defensive: never raises on an unexpected shape. Emitted
    through the origin-trace :class:`~lifemodel.log.SpanBoundLogger` so the reasoning
    lands durably in ``trace_events`` under the attempt's trace (§4.3).
    """
    if not isinstance(conversation_history, list):
        logger.debug(
            "proactive_reasoning",
            correlation_id=correlation_id,
            available=False,
            reason=f"conversation_history is {type(conversation_history).__name__}, not a list",
        )
        return
    messages = [_summarize_reasoning_message(m) for m in conversation_history]
    logger.debug(
        "proactive_reasoning",
        correlation_id=correlation_id,
        available=True,
        message_count=len(messages),
        messages=messages,
    )


def _emit_async_outcome(
    lm: LifeModel,
    *,
    origin_traceparent: str | None,
    correlation_id: str,
    outcome: ProactiveOutcome,
    assistant_response: str,
    conversation_history: Any,
    extra: dict[str, Any],
) -> None:
    """Weave the async proactive outcome onto its ORIGIN trace (spec §4.4/§5 step 5).

    The being's own Hermes turn (``[SILENT]`` → SILENT, real text → SENT) is observed
    HERE, on the far side of a boundary that carries no in-band trace channel. So we
    raise the launch's ``origin_traceparent`` from the state anchor and ``child_of``
    it — the outcome/detail/reasoning (and, on a SILENT, the ``ACT_GATE_SILENT``
    suppression) land UNDER THE SAME ``trace_id`` as the launch, delivery, and
    resolving frame.

    **Miss policy (load-bearing, §4.4):** if the anchor is gone (``None``), NEVER
    attach the outcome to a fresh/foreign trace — emit an explicit
    ``orphan_async_outcome`` on its own trace, and leave every real trace untouched.

    Best-effort/fail-open (§4.2): a graph with no tracer/ring skips silently; the
    caller wraps this so a trace hiccup can never break the outcome control flow.
    """
    tracer = lm.tracer
    ring = lm.event_ring
    if tracer is None or ring is None:
        return
    now = lm.clock.now().isoformat()

    if origin_traceparent is None:
        orphan = open_correlated_span(
            tracer=tracer,
            writer=lm.trace_writer,
            ring=ring,
            origin_traceparent=None,  # a fresh orphan root — NOT a foreign trace
            component="hooks",
            started_at=now,
        )
        orphan.span.set(
            correlation_id=correlation_id,
            outcome=outcome.value,
            reason="async_correlation_missing",
        )
        orphan.logger.warning(
            "orphan_async_outcome", correlation_id=correlation_id, outcome=outcome.value
        )
        orphan.span.end(status="ok", ended_at=now)
        orphan.persist(ended_at=now)
        return

    bridge = open_correlated_span(
        tracer=tracer,
        writer=lm.trace_writer,
        ring=ring,
        origin_traceparent=origin_traceparent,
        component="hooks",
        started_at=now,
    )
    label = "silent" if outcome is ProactiveOutcome.SILENT else "delivered"
    bridge.span.set(correlation_id=correlation_id, outcome=label)
    # INFO: the always-visible "woke and chose silence" vs "woke and reached out"
    # (bead lm-j2w B3), keyed by the SAME correlation_id as the launch/delivery.
    bridge.logger.info("proactive_outcome", correlation_id=correlation_id, outcome=label)
    _log_outcome_detail(
        bridge.logger,
        correlation_id=correlation_id,
        outcome=outcome,
        assistant_response=assistant_response,
        extra=extra,
    )
    _log_proactive_reasoning(
        bridge.logger, correlation_id=correlation_id, conversation_history=conversation_history
    )
    if outcome is ProactiveOutcome.SILENT:
        # The 4th suppression reason (phase-2 carryover): the async act-gate returned
        # [SILENT] — a CONSCIOUS outcome, logged as a suppression under the origin
        # trace. ``emit_suppression_span`` sets ``reason`` + ends the span "suppressed".
        emit_suppression_span(
            bridge.logger,
            reason=SuppressionReason.ACT_GATE_SILENT,
            component="hooks",
            metrics=lm.metrics,
        )
    else:
        bridge.span.end(status="ok", ended_at=now)
    bridge.persist(ended_at=now)


def make_post_llm_observer(
    build_lm: Callable[[], LifeModel],
    *,
    health: BrainHealth | None = None,
    metrics: MetricRegistry | None = None,
) -> Callable[..., None]:
    """Return a ``post_llm_call`` handler that starts an ASYNC_COMPLETION frame (§3/§5).

    The whole body is plugin-owned fail-loud (spec §4.3/MAJOR-4): a throw anywhere
    in it (even in ``build_lm``) is logged ERROR + traceback, recorded on
    *health* + *metrics*, and swallowed — the host's dispatch is never crashed by an
    afferent hiccup, and the failure is observable rather than silent.
    """

    def _observer(
        *,
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: Any = None,
        **_ignored: Any,
    ) -> None:
        try:
            lm = build_lm()
            state = lm.state.load()
            if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
                return
            memory = lm.state if isinstance(lm.state, MemoryPort) else None
            desire = read_live_contact_desire(memory) if memory is not None else None
            if desire is None or desire.state != DesireState.ACTIVE:
                return
            outcome = (
                ProactiveOutcome.SILENT
                if _is_no_reply(assistant_response)
                else ProactiveOutcome.SENT
            )
            correlation_id = state.pending_proactive_id or ""
            # Weave the outcome onto the ORIGIN trace (§4.4). Best-effort: the async
            # trace is observability, NEVER the outcome control flow below — a trace
            # hiccup (or a lost origin anchor) must not stop the desire from resolving.
            with contextlib.suppress(Exception):  # advisory: must never break a turn
                _emit_async_outcome(
                    lm,
                    origin_traceparent=state.pending_proactive_origin_traceparent,
                    correlation_id=correlation_id,
                    outcome=outcome,
                    assistant_response=assistant_response,
                    conversation_history=conversation_history,
                    extra=_ignored,
                )
            now = lm.clock.now()
            assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
            # The async turn finished → its OWN frame commits the outcome immediately
            # (spec §3): aggregation resolves the pending desire + writes action_pending
            # / backoff to AgentState. Not deferred to the next heartbeat.
            run_frame(
                lm.coreloop,
                [
                    proactive_outcome_signal(
                        origin_id=f"outcome-{correlation_id}",
                        outcome=outcome,
                        timestamp=now.isoformat(),
                        correlation_id=correlation_id,
                    )
                ],
                trigger=FrameTrigger.ASYNC_COMPLETION,
            )
        except Exception as exc:  # plugin-owned observability — do not crash the caller
            _record_observer_failure(
                observer_name=POST_LLM_OBSERVER, exc=exc, health=health, metrics=metrics
            )

    return _observer


def _is_own_impulse(text: str) -> bool:
    """True when *text* is our own composed proactive impulse (spec §6)."""
    return text.strip().startswith(IMPULSE_LABEL_PREFIX)


def _is_control_command(text: str) -> bool:
    """True when *text* is a slash/control command, not conversational dialogue.

    ``pre_gateway_dispatch`` fires before the command router forks, so any
    ``/...`` message — ``/lifemodel force-wake``, ``/lifemodel debug``, ``/new``,
    ``/model``, ``/commands``, etc. — would otherwise look like a genuine inbound
    contact. Owner's decision: operating the tool via a slash command is not
    conversing with the being, so ANY slash-prefixed message is a control command
    and is filtered at the sensor band-pass (spec §4) — it must NOT count as contact.
    """
    return text.strip().startswith("/")


def make_inbound_observer(
    build_lm: Callable[[], LifeModel],
    *,
    health: BrainHealth | None = None,
    metrics: MetricRegistry | None = None,
) -> Callable[..., None]:
    """Return a ``pre_gateway_dispatch`` handler that starts an EVENT frame (§3/§4).

    Plugin-owned fail-loud (spec §4.3/MAJOR-4): a throw in the body (past the sensor
    band-pass) is logged ERROR + traceback, recorded on *health* + *metrics*, and
    swallowed — the host's dispatch is never crashed by an afferent hiccup.
    """

    def _observer(*, event: Any = None, **_ignored: Any) -> None:
        if event is None or getattr(event, "internal", False):
            return
        text = getattr(event, "text", "") or ""
        # Sensor band-pass (spec §4): our own impulse and control commands never
        # become contact — filtered here at the afferent boundary before any frame.
        if _is_own_impulse(text) or _is_control_command(text):
            return
        try:
            lm = build_lm()
            assert lm.coreloop is not None, "coreloop must be wired by build_lifemodel"
            now = lm.clock.now()
            origin = (
                getattr(event, "id", None)
                or getattr(event, "message_id", None)
                or f"contact-{now.isoformat()}"
            )
            # A genuine inbound → its OWN EVENT frame, processed at the moment of the
            # event (spec §3): the frame satiates u, stamps last_exchange_at, and
            # resolves any live desire → SATISFIED, committed immediately.
            run_frame(
                lm.coreloop,
                [
                    contact_observed_signal(
                        origin_id=str(origin),
                        actor="user",
                        label="two_way",
                        timestamp=now.isoformat(),
                    )
                ],
                trigger=FrameTrigger.EVENT,
            )
        except Exception as exc:  # plugin-owned observability — do not crash the caller
            _record_observer_failure(
                observer_name=INBOUND_OBSERVER, exc=exc, health=health, metrics=metrics
            )

    return _observer

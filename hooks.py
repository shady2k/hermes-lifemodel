"""Signal-publishing hooks — verdict + exchange (spec §7.1, Phase E3).

Before Phase E3 these hooks mutated ``State`` directly (calling
``core.decision.apply_verdict`` / ``observe_exchange``). Now they **publish
signals** to ``lm.bus`` — producers only enqueue (spec §7.1). The aggregation
layer inside ``coreloop.tick()`` consumes them on the next tick.

``make_post_llm_observer`` — on a correlated proactive turn (``pending_proactive_id``
set AND ``user_message`` starts with ``IMPULSE_LABEL_PREFIX``) whose desire is
still active, decides ``FULFILL`` (any text) vs ``REJECT`` (a silence marker),
runs ``lint_proactive`` on a FULFILL and logs a mechanical leak (advisory), then
publishes a ``verdict`` signal carrying the ``correlation_id``.

``make_inbound_observer`` — on a genuine (non-internal, non-own-impulse,
non-slash-command) inbound message, publishes an ``exchange`` signal. Any
slash-prefixed message is a control command operating the tool, not dialogue,
and is excluded (owner's decision, spec §7.1 / lm-ia3).

Neither mutates ``State``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from .composition import LifeModel
from .core.correlate import open_correlated_span
from .core.desire_view import read_live_contact_desire
from .core.output_lint import lint_proactive
from .core.suppression import SuppressionReason, emit_suppression_span
from .core.taxonomy import exchange_signal, verdict_signal
from .core.wake_packet import IMPULSE_LABEL_PREFIX
from .domain.egress import Verdict
from .domain.objects import DesireState
from .log import SpanBoundLogger
from .ports.memory import MemoryPort

#: The exact silence markers Hermes' own gateway treats as intentional silence.
_NO_REPLY_MARKERS = frozenset({"NO_REPLY", "NO REPLY", "[SILENT]", "SILENT"})


def _is_no_reply(text: str) -> bool:
    """True when *text* is exactly one silence marker (spec §5/§7).

    Case-insensitive and whitespace-collapsing (mirrors the host's own
    ``_canonical_silence_candidate``): ``"no_reply"``, ``" NO_REPLY "``, and
    ``"no   reply"`` all match, but prose that merely *mentions* a marker does
    not.
    """
    candidate = " ".join(text.strip().upper().split())
    return candidate in _NO_REPLY_MARKERS


def _is_pending_proactive_turn(pending_proactive_id: str | None, user_message: str) -> bool:
    """Best-effort correlation to the outstanding proactive turn.

    See the module docstring's "Correlation caveat" — this is the mechanism
    the SPIKE found, not a guess: a proactive verdict is only ever pending
    (gate 1) for our own composed impulse text (gate 2, spec design doc §5
    guard (c)). A genuine user turn that lands while a desire is pending fails
    gate 2 and is correctly ignored.
    """
    if pending_proactive_id is None:
        return False
    return user_message.strip().startswith(IMPULSE_LABEL_PREFIX)


def _log_verdict_detail(
    logger: SpanBoundLogger,
    *,
    correlation_id: str,
    verdict: Verdict,
    assistant_response: str,
    extra: dict[str, Any],
) -> None:
    """DEBUG: full discovery detail for a resolved proactive verdict (bead lm-otq).

    The host's ``post_llm_call`` hook may pass MORE than ``user_message`` /
    ``assistant_response`` — reasoning/thinking, model id, token counts, a full
    response object — currently swallowed by ``_observer``'s ``**_ignored``. This
    is the discovery event for the top-level fields: the full (untruncated)
    assistant response, plus a safe string preview of every extra kwarg the
    host passed, keyed by field name. ``conversation_history`` is captured
    separately by name in ``_observer`` (not part of ``extra``) and owned by
    :func:`_log_proactive_reasoning` instead — this event no longer dumps it.

    Emitted through the origin-trace :class:`~lifemodel.log.SpanBoundLogger`
    (§4.4) so this DEBUG detail is durable in ``trace_events`` under the attempt's
    trace — the 5th-source collapse (§4.3), not a hindsight/DEBUG-log side channel.
    """
    logger.debug(
        "proactive_verdict_detail",
        correlation_id=correlation_id,
        verdict=verdict.value,
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

    ``conversation_history`` (a ``post_llm_call`` kwarg the host passes
    alongside ``user_message``/``assistant_response``) is a list of message
    dicts; assistant entries may carry a ``reasoning`` string — the being's own
    chain of thought for that turn, including the impulse turn itself (the
    LAST assistant message). This is the ONE event that surfaces it, per
    message, untruncated (capped generously). Defensive: never raises on an
    unexpected shape. Emitted through the origin-trace
    :class:`~lifemodel.log.SpanBoundLogger` so the reasoning lands durably in
    ``trace_events`` under the attempt's trace (§4.3, 5th-source collapse).
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
    verdict: Verdict,
    assistant_response: str,
    conversation_history: Any,
    extra: dict[str, Any],
) -> None:
    """Weave the async proactive outcome onto its ORIGIN trace (spec §4.4/§5 step 5).

    This is the read-back the whole bridge exists for: the being's own Hermes turn
    (``[SILENT]`` → REJECT, real text → FULFILL) is observed HERE, in our
    ``post_llm`` hook, on the far side of a boundary that carries no in-band trace
    channel. So we raise the launch's ``origin_traceparent`` from the state anchor and
    ``child_of`` it — the outcome/verdict/reasoning (and, on a REJECT, the
    ``ACT_GATE_SILENT`` suppression, the 4th reason) land UNDER THE SAME ``trace_id``
    as the launch, delivery, and resolving tick.

    **Miss policy (load-bearing, §4.4):** if the anchor is gone (``None``), NEVER
    attach the outcome to a fresh/foreign trace — emit an explicit
    ``orphan_async_outcome`` on its own trace so the viewer can show "async
    correlation missing", and leave every real trace untouched.

    Best-effort/fail-open (§4.2): a graph with no tracer/ring (a bare test
    ``LifeModel``) skips silently; the caller wraps this so a trace hiccup can never
    break the verdict-signal control flow.
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
            verdict=verdict.value,
            reason="async_correlation_missing",
        )
        orphan.logger.warning(
            "orphan_async_outcome", correlation_id=correlation_id, verdict=verdict.value
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
    outcome = "silent" if verdict is Verdict.REJECT else "delivered"
    bridge.span.set(correlation_id=correlation_id, verdict=verdict.value, outcome=outcome)
    # INFO: the always-visible "woke and chose silence" vs "woke and reached out"
    # (bead lm-j2w B3), keyed by the SAME correlation_id as the launch/delivery.
    bridge.logger.info("proactive_outcome", correlation_id=correlation_id, outcome=outcome)
    _log_verdict_detail(
        bridge.logger,
        correlation_id=correlation_id,
        verdict=verdict,
        assistant_response=assistant_response,
        extra=extra,
    )
    _log_proactive_reasoning(
        bridge.logger, correlation_id=correlation_id, conversation_history=conversation_history
    )
    if verdict is Verdict.REJECT:
        # The 4th suppression reason (phase-2 carryover): the async act-gate returned
        # [SILENT] — a CONSCIOUS verdict, logged as a suppression under the origin
        # trace. ``emit_suppression_span`` sets ``reason`` + ends the span "suppressed".
        emit_suppression_span(
            bridge.logger, reason=SuppressionReason.ACT_GATE_SILENT, component="hooks"
        )
    else:
        lint = lint_proactive(assistant_response)
        if not lint.ok:
            # Advisory (spec §13): a delivered proactive message tripped the output-lint
            # (mechanical timer / filler) — feeds future prompt tuning, never blocks.
            bridge.logger.info("proactive_output_lint", reason=lint.reason)
        bridge.span.end(status="ok", ended_at=now)
    bridge.persist(ended_at=now)


def make_post_llm_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``post_llm_call`` handler that PUBLISHES a verdict signal (§7.1)."""

    def _observer(
        *,
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: Any = None,
        **_ignored: Any,
    ) -> None:
        state = lm.state.load()
        if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
            return
        memory = lm.state if isinstance(lm.state, MemoryPort) else None
        desire = read_live_contact_desire(memory) if memory is not None else None
        if desire is None or desire.state != DesireState.ACTIVE:
            return
        verdict = Verdict.REJECT if _is_no_reply(assistant_response) else Verdict.FULFILL
        correlation_id = state.pending_proactive_id or ""
        # Weave the outcome onto the ORIGIN trace (§4.4). Best-effort: the async trace
        # is observability, NEVER the verdict control flow below — a trace hiccup (or a
        # lost origin anchor) must not stop the desire from resolving.
        with contextlib.suppress(Exception):  # advisory: must never break a turn
            _emit_async_outcome(
                lm,
                origin_traceparent=state.pending_proactive_origin_traceparent,
                correlation_id=correlation_id,
                verdict=verdict,
                assistant_response=assistant_response,
                conversation_history=conversation_history,
                extra=_ignored,
            )
        now = lm.clock.now()
        lm.bus.publish(
            verdict_signal(
                origin_id=f"verdict-{state.pending_proactive_id}",
                verdict=verdict,
                timestamp=now.isoformat(),
                correlation_id=correlation_id,
            )
        )

    return _observer


def _is_own_impulse(text: str) -> bool:
    """True when *text* is our own composed proactive impulse (spec §6)."""
    return text.strip().startswith(IMPULSE_LABEL_PREFIX)


def _is_control_command(text: str) -> bool:
    """True when *text* is a slash/control command, not conversational dialogue.

    ``pre_gateway_dispatch`` fires before the command router forks, so any
    ``/...`` message — ``/lifemodel force-wake``, ``/lifemodel debug``,
    ``/new``, ``/model``, ``/commands``, etc. — would otherwise look like a
    genuine inbound exchange. Owner's decision: operating the tool via a
    slash command is not conversing with the being, so ANY slash-prefixed
    message is a control command and must NOT count as a two-way exchange
    (reverses the prior ``/lifemodel``-only scoping).
    """
    return text.strip().startswith("/")


def make_inbound_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``pre_gateway_dispatch`` handler that PUBLISHES an exchange signal (§7.1)."""

    def _observer(*, event: Any = None, **_ignored: Any) -> None:
        if event is None or getattr(event, "internal", False):
            return
        text = getattr(event, "text", "") or ""
        if _is_own_impulse(text) or _is_control_command(text):
            return
        now = lm.clock.now()
        origin = (
            getattr(event, "id", None)
            or getattr(event, "message_id", None)
            or f"exchange-{now.isoformat()}"
        )
        lm.bus.publish(
            exchange_signal(
                origin_id=str(origin),
                actor="user",
                label="two_way",
                timestamp=now.isoformat(),
            )
        )

    return _observer

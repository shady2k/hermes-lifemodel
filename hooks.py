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
from .core.desire_view import read_live_contact_desire
from .core.output_lint import lint_proactive
from .core.taxonomy import exchange_signal, verdict_signal
from .core.wake_packet import IMPULSE_LABEL_PREFIX
from .domain.egress import Verdict
from .domain.objects import DesireState
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


def _hooks_logger(lm: LifeModel) -> Any:
    """Return the logger this graph was built with, falling back to a fresh one.

    Prefers ``lm.logger`` — the SAME collaborator ``register(ctx)`` threads into
    ``build_lifemodel(..., logger=logger)`` — so hooks observability shares the
    plugin's one configured sink instead of constructing an ad-hoc logger. Bare
    test/script callers that never pass a logger (``lm.logger is None``) still
    get a working one via :func:`~lifemodel.log.get_logger`.
    """
    from .log import get_logger

    return lm.logger or get_logger("lifemodel.hooks")


def _log_lint(lm: LifeModel, reason: str) -> None:
    """Advisory: record that a delivered proactive message tripped the output-lint
    (mechanical timer / filler). Model A can't block the native send — this is
    observability feeding future prompt tuning (spec §13)."""
    with contextlib.suppress(Exception):  # advisory logging must never break a turn
        _hooks_logger(lm).info("proactive_output_lint", reason=reason)


def _log_verdict_detail(
    lm: LifeModel,
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
    DEBUG-gated (only visible when the effective log level is DEBUG, unlike the
    always-on ``proactive_outcome`` INFO log) and best-effort — advisory
    logging must never break a turn.
    """
    with contextlib.suppress(Exception):  # advisory logging must never break a turn
        _hooks_logger(lm).debug(
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
    lm: LifeModel, *, correlation_id: str, conversation_history: Any
) -> None:
    """DEBUG: the full reasoning chain behind a resolved proactive turn (lm-otq step 2).

    ``conversation_history`` (a ``post_llm_call`` kwarg the host passes
    alongside ``user_message``/``assistant_response``) is a list of message
    dicts; assistant entries may carry a ``reasoning`` string — the being's own
    chain of thought for that turn, including the impulse turn itself (the
    LAST assistant message). This is the ONE event that surfaces it, per
    message, untruncated (capped generously). Defensive: never raises on an
    unexpected shape. DEBUG-gated and best-effort like the sibling discovery
    log, ``proactive_verdict_detail`` — advisory logging must never break a
    turn or alter verdict/signal/outcome control flow.
    """
    with contextlib.suppress(Exception):  # advisory logging must never break a turn
        logger = _hooks_logger(lm)
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


def _log_outcome(lm: LifeModel, *, correlation_id: str, verdict: Verdict) -> None:
    """INFO: the resolved proactive outcome — "it woke and chose silence" vs "it
    woke and reached out" (bead lm-j2w B3, owner's core ask). INFO so it is
    ALWAYS visible (unlike the DEBUG-gated full prompt logged at launch in
    ``core/proactive.py``), keyed by the SAME ``correlation_id`` so the two
    events correlate end-to-end. Reuses the verdict this observer already
    computed (REJECT on a silence marker, FULFILL on real text) rather than
    re-deriving "silent vs delivered" a second way."""
    outcome = "silent" if verdict is Verdict.REJECT else "delivered"
    with contextlib.suppress(Exception):  # advisory logging must never break a turn
        _hooks_logger(lm).info("proactive_outcome", correlation_id=correlation_id, outcome=outcome)


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
        if verdict is Verdict.FULFILL:
            lint = lint_proactive(assistant_response)
            if not lint.ok:
                _log_lint(lm, lint.reason)
        _log_outcome(lm, correlation_id=state.pending_proactive_id or "", verdict=verdict)
        _log_verdict_detail(
            lm,
            correlation_id=state.pending_proactive_id or "",
            verdict=verdict,
            assistant_response=assistant_response,
            extra=_ignored,
        )
        _log_proactive_reasoning(
            lm,
            correlation_id=state.pending_proactive_id or "",
            conversation_history=conversation_history,
        )
        now = lm.clock.now()
        lm.bus.publish(
            verdict_signal(
                origin_id=f"verdict-{state.pending_proactive_id}",
                verdict=verdict,
                timestamp=now.isoformat(),
                correlation_id=state.pending_proactive_id or "",
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

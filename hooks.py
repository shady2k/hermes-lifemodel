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

``make_inbound_observer`` — on a genuine (non-internal, non-own-impulse) inbound
message, publishes an ``exchange`` signal.

Neither mutates ``State``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .composition import LifeModel
from .core.output_lint import lint_proactive
from .core.taxonomy import exchange_signal, verdict_signal
from .core.wake_packet import IMPULSE_LABEL_PREFIX
from .sim.aggregation import Verdict

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


def _log_lint(lm: LifeModel, reason: str) -> None:
    """Advisory: record that a delivered proactive message tripped the output-lint
    (mechanical timer / filler). Model A can't block the native send — this is
    observability feeding future prompt tuning (spec §13)."""
    try:
        from .log import get_logger

        get_logger("lifemodel.hooks").info("proactive_output_lint", reason=reason)
    except Exception:  # noqa: BLE001 - advisory logging must never break a turn
        pass


def make_post_llm_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``post_llm_call`` handler that PUBLISHES a verdict signal (§7.1)."""

    def _observer(*, user_message: str = "", assistant_response: str = "", **_ignored: Any) -> None:
        state = lm.state.load()
        if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
            return
        if state.desire_status != "active":
            return
        verdict = Verdict.REJECT if _is_no_reply(assistant_response) else Verdict.FULFILL
        if verdict is Verdict.FULFILL:
            lint = lint_proactive(assistant_response)
            if not lint.ok:
                _log_lint(lm, lint.reason)
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


def _is_own_command(text: str) -> bool:
    """True when *text* is the being's own ``/lifemodel`` control-plane command.

    ``pre_gateway_dispatch`` fires before the command router forks, so a
    ``/lifemodel force-wake`` (or even a read-only ``/lifemodel debug``) sent
    over the being's own channel would otherwise look like a genuine inbound
    exchange. Scoped to ``/lifemodel`` only: other slash commands (e.g.
    ``/new``, ``/model``) still mean the user is present and must keep
    counting as contact.
    """
    return text.strip().startswith("/lifemodel")


def make_inbound_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``pre_gateway_dispatch`` handler that PUBLISHES an exchange signal (§7.1)."""

    def _observer(*, event: Any = None, **_ignored: Any) -> None:
        if event is None or getattr(event, "internal", False):
            return
        text = getattr(event, "text", "") or ""
        if _is_own_impulse(text) or _is_own_command(text):
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

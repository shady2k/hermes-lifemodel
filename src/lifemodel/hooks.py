"""Verdict feedback via Hermes' ``post_llm_call`` hook (spec §5/§7, Task 5).

A wake only *launches* a proactive turn (``egress_service.run_proactive_tick``)
— cognition's actual answer only exists once the LLM's final output comes
back. This module observes that output via the ``post_llm_call`` lifecycle
hook and resolves the pending desire:

* the model answers ``NO_REPLY`` / ``NO REPLY`` / ``[SILENT]`` / ``SILENT``
  (case-insensitive, whitespace-collapsed) -> :data:`Verdict.REJECT` (growing
  backoff — the anti-drum guarantee, no message sent);
* any other text -> :data:`Verdict.FULFILL` (satiate, stamp contact).

SPIKE findings (read-only against ``~/.hermes/hermes-agent``, hermes-agent
0.17.0 — recorded here since the payload shape lives outside this repo):

* ``post_llm_call`` **is** in ``VALID_HOOKS`` (``hermes_cli/plugins.py``) and
  fires exactly once per turn, right after ``transform_llm_output``, from
  ``agent/turn_finalizer.py`` (~line 361)::

      invoke_hook(
          "post_llm_call",
          session_id=agent.session_id, task_id=effective_task_id,
          turn_id=turn_id, user_message=original_user_message,
          assistant_response=final_response, conversation_history=list(messages),
          model=agent.model, platform=getattr(agent, "platform", None) or "",
      )

  ``PluginManager.invoke_hook`` calls every registered callback as
  ``cb(**kwargs)`` (a **kwargs call, not a single payload object** — plugins.py
  ~line 1872) — so :func:`make_post_llm_observer` returns a handler with that
  keyword shape, not a one-argument payload handler.
* ``assistant_response`` is the raw final text (post any
  ``transform_llm_output``, *pre* delivery-suppression) — so a literal
  ``NO_REPLY`` is still observable here even though the gateway later hides it
  from the chat surface. ``gateway/response_filters.py`` already defines the
  exact same four canonical markers this module matches
  (``LIVE_GATEWAY_SILENT_MARKERS``) and a streaming-safe partial-marker buffer
  (``is_partial_silence_marker``) — so the Phase-2 worry "does NO_REPLY leak
  visibly under streaming" looks pre-addressed upstream; still worth a live
  confirmation, not re-derived here.
* **Correlation caveat — needs field verification.** The real payload carries
  no id we control that echoes ``state.pending_proactive_id``: Hermes does not
  thread a plugin-supplied id through the turn/hook pipeline, and upstreaming
  that is out of Phase-1 scope. The only signal available without a host
  change is the one the design spec's own guard table already anticipated
  (``docs/superpowers/specs/2026-07-04-lifemodel-proactive-egress-design.md``
  §5 guard (c)): plugin hooks see the synthetic impulse text verbatim as
  ``user_message``. So correlation here is two gates: (1) a proactive turn is
  actually outstanding (``state.pending_proactive_id is not None``) **and**
  (2) *this* turn's ``user_message`` is recognizably our own impulse (starts
  with :data:`~lifemodel.impulse.IMPULSE_LABEL_PREFIX`). Because
  ``core.decision.decide_reachout`` dedups to at most one live desire at a
  time, gate (1) already narrows this to "the" pending turn, and gate (2)
  filters out a genuine user turn that happens to land while one is pending.
  This has **not** been exercised against a real running Hermes turn (no live
  session in this environment) — treat it as the one field-verification item
  before relying on this in production.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .composition import LifeModel
from .core.decision import apply_verdict
from .impulse import IMPULSE_LABEL_PREFIX
from .sim.aggregation import Verdict

#: The exact silence markers Hermes' own gateway treats as intentional silence
#: (``gateway/response_filters.py::LIVE_GATEWAY_SILENT_MARKERS``, hermes-agent
#: 0.17.0). Kept as a local, stdlib-only constant — the plugin core stays
#: importable without the host (Global Constraints) — rather than imported, so
#: both sets must be kept in lockstep by hand if the host's ever changes.
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


def make_post_llm_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``post_llm_call`` handler that resolves the pending desire.

    Accepts the real Hermes kwargs shape (``invoke_hook`` calls
    ``cb(**kwargs)`` — see the SPIKE notes above), reading only
    ``user_message`` and ``assistant_response``; every other kwarg
    (``session_id``, ``task_id``, ``turn_id``, ``conversation_history``,
    ``model``, ``platform``, ``telemetry_schema_version``, ...) is accepted
    and ignored. A turn that does not correlate to the pending proactive
    desire (see :func:`_is_pending_proactive_turn`) is a no-op.
    """

    def _observer(
        *,
        user_message: str = "",
        assistant_response: str = "",
        **_ignored: Any,
    ) -> None:
        state = lm.state.load()
        if not _is_pending_proactive_turn(state.pending_proactive_id, user_message):
            return
        verdict = Verdict.REJECT if _is_no_reply(assistant_response) else Verdict.FULFILL
        apply_verdict(state, verdict, now=lm.clock.now())
        lm.state.commit(state)

    return _observer

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

Task 6 (inbound observation) SPIKE findings — read-only against the same host:

* ``pre_gateway_dispatch`` **is** in ``VALID_HOOKS`` (``hermes_cli/plugins.py``
  line 173) and is *preferred* over ``pre_llm_call`` per the plan: it fires
  **once per incoming** ``MessageEvent``, inside ``GatewayRunner._handle_message``
  (``gateway/run.py`` ~line 8650), via
  ``invoke_hook("pre_gateway_dispatch", event=event, gateway=self,
  session_store=self.session_store)`` — again a ``cb(**kwargs)`` call.
* **Load-bearing:** the host itself gates this invocation on
  ``if not is_internal:`` (``is_internal = bool(getattr(event, "internal",
  False))``), *before* any hook fires — and our own proactive impulse is
  injected via ``gateway_core.inject_proactive_turn`` -> ``_default_make_event``
  as ``MessageEvent(..., internal=True)``, dispatched through the very same
  ``adapter.handle_message`` -> ``_handle_message`` path (``adapter`` is wired
  with ``set_message_handler(self._handle_message)`` at ``gateway/run.py``
  ~line 6841). So the host *never* invokes ``pre_gateway_dispatch`` for our own
  nudge at all — disjointness from the post_llm_call verdict path is a host
  guarantee here, not something this module has to enforce alone. This module
  still checks ``event.internal`` and the
  :data:`~lifemodel.impulse.IMPULSE_LABEL_PREFIX` text defensively (belt and
  suspenders — cheap, and robust if that host guarantee ever changes).
* Return contract: ``None`` = normal dispatch, a dict with
  ``{"action": "skip"|"rewrite"|"allow"}`` influences flow. This observer only
  *observes* — it always returns ``None`` so it can never itself skip/rewrite a
  genuine inbound message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .composition import LifeModel
from .core.decision import apply_verdict, observe_exchange
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


def _is_own_impulse(text: str) -> bool:
    """True when *text* is our own composed proactive impulse (spec §6).

    Belt-and-suspenders alongside ``event.internal`` (see the module
    docstring's Task 6 SPIKE notes): the being must never treat its own nudge
    as user contact.
    """
    return text.strip().startswith(IMPULSE_LABEL_PREFIX)


def make_inbound_observer(lm: LifeModel) -> Callable[..., None]:
    """Return a ``pre_gateway_dispatch`` handler that observes genuine user contact.

    On a genuine inbound user message: satiates the drive, stamps
    ``last_exchange_at``, clears the reject record, and resolves any live
    desire (:func:`~lifemodel.core.decision.observe_exchange` with
    ``actor="user", label="two_way"``) — so silence resets on real contact
    (RC1: the being now *hears* the user).

    Accepts the real Hermes kwargs shape (``invoke_hook`` calls ``cb(**kwargs)``
    with ``event``, ``gateway``, ``session_store`` — see the SPIKE notes above),
    reading only ``event.text`` / ``event.internal``; every other kwarg is
    accepted and ignored. Two turns are ignored, never touching state: an
    internal/synthetic event (``event.internal``, or missing ``event``
    entirely — defensive), and our own composed impulse text (the
    :data:`~lifemodel.impulse.IMPULSE_LABEL_PREFIX` marker) — so the being
    never satiates its own urge with its own nudge, and never double-counts
    with the ``post_llm_call`` verdict path. Always returns ``None`` (normal
    dispatch) — this observer never skips or rewrites the message.
    """

    def _observer(*, event: Any = None, **_ignored: Any) -> None:
        if event is None or getattr(event, "internal", False):
            return
        text = getattr(event, "text", "") or ""
        if _is_own_impulse(text):
            return
        state = lm.state.load()
        observe_exchange(state, actor="user", label="two_way", now=lm.clock.now())
        lm.state.commit(state)

    return _observer

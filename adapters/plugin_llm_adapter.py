"""``PluginLlmPort`` — the real :class:`~lifemodel.core.llm_port.LlmPort` over
``ctx.llm.acomplete_structured`` (lm-705.6, Task 7).

``ctx.llm`` is the plugin's host-owned LLM facade
(``agent.plugin_llm.PluginLlm``, exposed as ``hermes_cli/plugins.py:351``'s
``PluginContext.llm`` property) — the ONE stable, trust-gated lane a plugin has
for its own model calls (``agent/plugin_llm.py``'s module docstring: "the
supported lane for that case"). This adapter is thin on purpose: it maps our
:class:`~lifemodel.core.llm_port.InternalCognitionRequest`/``Result`` onto that
facade's ``acomplete_structured(instructions=..., input=[...], json_schema=...,
timeout=...)`` shape (``agent/plugin_llm.py:823``) and back.

**Confirmed against the real source (not the aux-slot routing the design
sketched):** ``PluginLlm.acomplete_structured`` always calls the host's
``async_call_llm(task=None, ...)`` internally (``agent/plugin_llm.py``'s
``_invoke_async`` hard-codes ``task=None``) — there is no ``task=`` kwarg on the
public facade, so a plugin-registered ``ctx.register_auxiliary_task(key, ...)``
slot is **not** reachable through this call today; the aux-task config surface
that actually threads a ``task=`` key into routing is the lower-level
``agent.auxiliary_client.async_call_llm`` (what a Hermes-BUNDLED plugin like
``plugins/teams_pipeline`` calls directly — an unsupported-for-third-party-plugins
surface we deliberately do not reach into). So: this adapter goes through the
sanctioned ``ctx.llm`` facade (routes to the user's MAIN model, same as any other
plugin LLM call), and ``__init__.py`` still calls ``ctx.register_auxiliary_task``
so the ``auxiliary.lifemodel_internal`` config slot exists for a future host
build / a follow-up that wires ``model=``/``provider=`` overrides through the
trust-gated ``plugins.entries.lifemodel.llm.allow_model_override`` config (this
adapter passes ``purpose="lifemodel_internal"`` so the call is at least
attributable in host audit logs). Tracked as a known gap, not a blocker for the
seam itself (lm-705.6 ships no live emitter of ``LaunchInternalCognition`` yet).
"""

from __future__ import annotations

from typing import Any

from ..core.llm_port import InternalCognitionRequest, InternalCognitionResult

#: Default per-call timeout handed to ``ctx.llm.acomplete_structured`` — a cheap
#: internal pass has no business running long; mirrors
#: :data:`lifemodel.adapters.internal_runner.DEFAULT_TIMEOUT_SECONDS` (the
#: runner ALSO wraps the call in ``asyncio.wait_for`` at that same default, so
#: the two bounds agree rather than silently racing each other).
DEFAULT_TIMEOUT_SECONDS = 30.0


class PluginLlmPort:
    """Adapts ``ctx.llm`` (``agent.plugin_llm.PluginLlm``) to :class:`LlmPort`.

    ``ctx_llm`` is duck-typed (only ``acomplete_structured`` is called), so this
    module stays importable/constructible without the real Hermes ``agent``
    package on ``sys.path`` — a test injects a bare stand-in exposing just that
    one async method.
    """

    def __init__(self, ctx_llm: Any, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._ctx_llm = ctx_llm
        self._timeout = timeout

    async def complete_structured(self, req: InternalCognitionRequest) -> InternalCognitionResult:
        result = await self._ctx_llm.acomplete_structured(
            instructions=req.instructions,
            input=[{"type": "text", "text": req.input_text}],
            json_schema=req.json_schema,
            json_mode=req.json_schema is not None,
            timeout=self._timeout,
            purpose="lifemodel_internal",
        )
        parsed = result.parsed if isinstance(result.parsed, dict) else None
        return InternalCognitionResult(raw=str(result.text), parsed=parsed)

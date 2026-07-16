"""``LlmPort`` — the internal-cognition seam's Hermes-free LLM boundary (design §3.3).

The core stays Hermes-free (HLA §13, DI): this Protocol names no Hermes type, so
it unit-tests with :class:`~lifemodel.testing.llm.FakeLlmPort`. The real adapter
(:class:`~lifemodel.adapters.plugin_llm_adapter.PluginLlmPort`) wires it over
``ctx.llm.acomplete_structured`` at the composition boundary.

Distinct from the being's *delivered* cognition (``LaunchProactive`` → the
gateway's native turn machinery, read back via ``post_llm``): this port is the
NON-delivered, structural half of the seam — a caller awaits
:meth:`LlmPort.complete_structured` directly and applies the typed result itself
(:func:`~lifemodel.core.internal_cognition.run_internal_completion`); there is no
``post_llm`` outcome path for it, and it never touches the gateway egress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class InternalCognitionRequest:
    """One non-delivered aux call's inputs — a bounded, JSON-native value object.

    ``instructions`` is the system-level framing (what kind of judgment this is);
    ``input_text`` is the bounded segment/context the model actually judges.
    ``json_schema`` is optional — ``None`` for a plain-text pass (this bead's
    trivial internal pass), a JSON Schema dict for a typed-result pass (noticing's
    top-K seeds, lm-705.5).
    """

    instructions: str
    input_text: str
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class InternalCognitionResult:
    """One non-delivered aux call's output — ``raw`` text plus an optional parse.

    ``parsed`` is ``None`` when no ``json_schema`` was requested, or the model's
    response was not valid JSON against it (fail-soft — the caller decides what a
    missing parse means; this port never raises on a malformed *response*, only
    on a transport/provider failure).
    """

    raw: str
    parsed: dict[str, Any] | None = None


@runtime_checkable
class LlmPort(Protocol):
    """The Hermes-free boundary for one non-delivered internal-cognition call."""

    async def complete_structured(self, req: InternalCognitionRequest) -> InternalCognitionResult:
        """Run one bounded, non-delivered aux call. Never delivered to the human.

        May raise (timeout, provider failure, transport error) — the caller
        (:class:`~lifemodel.adapters.internal_runner.InternalCognitionRunner`) is
        responsible for turning an exception into a typed failure outcome; this
        method itself does not swallow anything.
        """
        ...

"""Construction/shape test for :class:`lifemodel.adapters.plugin_llm_adapter.PluginLlmPort`
(lm-705.6, Task 7).

Thin adapter over the real host call (``ctx.llm.acomplete_structured``) — not
unit-testable beyond its shape without the real Hermes ``agent`` package; the
seam's actual host wiring is exercised by the (skipped-off-host) integration
test, ``tests/hermes_internal_cognition_integration.py`` (Task 8). This test
proves: the request maps onto ``acomplete_structured``'s documented kwargs, and
the host's structured result maps back onto ``InternalCognitionResult``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from lifemodel.adapters.plugin_llm_adapter import PluginLlmPort
from lifemodel.core.llm_port import InternalCognitionRequest, LlmPort


@dataclass
class _FakeStructuredResult:
    text: str
    parsed: Any = None


class _FakeCtxLlm:
    """Duck-typed stand-in for ``agent.plugin_llm.PluginLlm`` — records the call."""

    def __init__(self, result: _FakeStructuredResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def acomplete_structured(self, **kwargs: Any) -> _FakeStructuredResult:
        self.calls.append(kwargs)
        return self._result


def test_plugin_llm_port_satisfies_the_protocol() -> None:
    port = PluginLlmPort(_FakeCtxLlm(_FakeStructuredResult(text="")))
    assert isinstance(port, LlmPort)


def test_complete_structured_maps_request_onto_acomplete_structured_kwargs() -> None:
    async def scenario() -> None:
        ctx_llm = _FakeCtxLlm(_FakeStructuredResult(text="hi", parsed={"a": 1}))
        port = PluginLlmPort(ctx_llm)
        req = InternalCognitionRequest(
            instructions="notice", input_text="the segment", json_schema={"type": "object"}
        )

        result = await port.complete_structured(req)

        assert len(ctx_llm.calls) == 1
        call = ctx_llm.calls[0]
        assert call["instructions"] == "notice"
        assert call["input"] == [{"type": "text", "text": "the segment"}]
        assert call["json_schema"] == {"type": "object"}
        assert call["json_mode"] is True
        assert call["purpose"] == "lifemodel_internal"
        assert result.raw == "hi"
        assert result.parsed == {"a": 1}

    asyncio.run(scenario())


def test_complete_structured_with_no_schema_requests_no_json_mode() -> None:
    async def scenario() -> None:
        ctx_llm = _FakeCtxLlm(_FakeStructuredResult(text="plain text"))
        port = PluginLlmPort(ctx_llm)
        req = InternalCognitionRequest(instructions="i", input_text="t")

        result = await port.complete_structured(req)

        call = ctx_llm.calls[0]
        assert call["json_schema"] is None
        assert call["json_mode"] is False
        assert result.raw == "plain text"
        assert result.parsed is None

    asyncio.run(scenario())


def test_complete_structured_ignores_a_non_dict_parsed_result() -> None:
    async def scenario() -> None:
        # A defensive shape guard: acomplete_structured's `parsed` is `Optional[Any]`
        # on the host side — if it ever came back as a bare list/scalar (not the
        # object our JSON schema requested), never hand it to a caller expecting
        # dict[str, Any] | None.
        ctx_llm = _FakeCtxLlm(_FakeStructuredResult(text="[1, 2]", parsed=[1, 2]))
        port = PluginLlmPort(ctx_llm)

        result = await port.complete_structured(
            InternalCognitionRequest(instructions="i", input_text="t")
        )

        assert result.parsed is None
        assert result.raw == "[1, 2]"

    asyncio.run(scenario())

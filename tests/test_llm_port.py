"""Tests for :mod:`lifemodel.core.llm_port` + :class:`lifemodel.testing.llm.FakeLlmPort`.

The core stays Hermes-free: the ``LlmPort`` Protocol names no Hermes type, and its
two carried dataclasses are plain JSON-native value objects. ``FakeLlmPort`` is the
test double every later task (the runner, the completion frame) drives instead of
a real host call.
"""

from __future__ import annotations

import asyncio

import pytest

from lifemodel.core.llm_port import InternalCognitionRequest, InternalCognitionResult, LlmPort
from lifemodel.testing.llm import FakeLlmPort


def test_internal_cognition_request_is_a_plain_value_object() -> None:
    req = InternalCognitionRequest(
        instructions="notice", input_text="the conversation", json_schema={"type": "object"}
    )
    assert req.instructions == "notice"
    assert req.input_text == "the conversation"
    assert req.json_schema == {"type": "object"}


def test_internal_cognition_request_json_schema_defaults_none() -> None:
    req = InternalCognitionRequest(instructions="i", input_text="t")
    assert req.json_schema is None


def test_internal_cognition_result_carries_raw_and_parsed() -> None:
    result = InternalCognitionResult(raw='{"a": 1}', parsed={"a": 1})
    assert result.raw == '{"a": 1}'
    assert result.parsed == {"a": 1}


def test_fake_llm_port_satisfies_the_protocol() -> None:
    fake = FakeLlmPort(InternalCognitionResult(raw="", parsed=None))
    assert isinstance(fake, LlmPort)


def test_fake_llm_port_returns_the_scripted_result() -> None:
    async def scenario() -> None:
        scripted = InternalCognitionResult(raw="hello", parsed={"gist": "hello"})
        fake = FakeLlmPort(scripted)
        req = InternalCognitionRequest(instructions="i", input_text="t")

        result = await fake.complete_structured(req)

        assert result is scripted
        assert fake.requests == [req]  # the request was recorded

    asyncio.run(scenario())


def test_fake_llm_port_raises_the_scripted_exception() -> None:
    async def scenario() -> None:
        fake = FakeLlmPort(RuntimeError("boom"))
        req = InternalCognitionRequest(instructions="i", input_text="t")

        with pytest.raises(RuntimeError, match="boom"):
            await fake.complete_structured(req)

        assert fake.requests == [req]  # still recorded before the raise

    asyncio.run(scenario())

"""Tests for the optional subject/instructions/schema fields on
:class:`~lifemodel.core.intents.LaunchInternalCognition` (lm-705.2).

These fields let an emitter fully specify its own aux LLM call so the adapter
can map intent -> :class:`~lifemodel.core.llm_port.InternalCognitionRequest`
generically, without knowing the pass type. All three are optional and
default so the lm-705.6 construction shape (``prompt``/``correlation_id``/
``origin_traceparent`` only) keeps working unchanged.
"""

from __future__ import annotations

from lifemodel.core.intents import LaunchInternalCognition

_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def test_defaults_are_back_compatible() -> None:
    # the lm-705.6 construction shape still works unchanged
    intent = LaunchInternalCognition(prompt="p", correlation_id="c", origin_traceparent=_TP)
    assert intent.subject_id is None
    assert intent.instructions == ""
    assert intent.json_schema is None


def test_emitter_can_fully_specify_its_call() -> None:
    schema = {"type": "object", "properties": {"outcome": {"type": "string"}}}
    intent = LaunchInternalCognition(
        prompt="the thought",
        correlation_id="process-t1@x",
        origin_traceparent=_TP,
        subject_id="thought:seed:abc",
        instructions="ruminate",
        json_schema=schema,
    )
    assert intent.subject_id == "thought:seed:abc"
    assert intent.instructions == "ruminate"
    assert intent.json_schema == schema

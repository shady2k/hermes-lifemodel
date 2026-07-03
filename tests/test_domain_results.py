"""Tests for the small result values — ``LayerResult`` and ``Decision``.

These carry the confidence a layer reports and the allow/suppress verdict the
act-gate returns. Imports no Hermes.
"""

from __future__ import annotations

from lifemodel.domain.act import Decision
from lifemodel.domain.layer import LayerResult


def test_layer_result_defaults() -> None:
    result = LayerResult(confidence=0.9)
    assert result.output is None
    assert result.escalate is False


def test_decision_allowed_and_suppressed_constructors() -> None:
    ok = Decision.allowed("author channel, within cooldown")
    assert ok.allow is True
    assert ok.reason

    no = Decision.suppressed("quiet hours")
    assert no.allow is False
    assert no.reason == "quiet hours"

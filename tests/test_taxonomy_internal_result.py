"""The ``internal_result`` signal — the internal-cognition seam's typed re-entry
(lm-705.6, design §3.3)."""

from __future__ import annotations

import pytest

from lifemodel.core.taxonomy import (
    KIND_INTERNAL_RESULT,
    internal_result_signal,
    read_internal_result,
)
from lifemodel.domain.signal import Signal


def test_internal_result_roundtrip_with_parsed():
    sig = internal_result_signal(
        origin_id="internal-result-c1",
        correlation_id="c1",
        raw='{"gist": "hi"}',
        parsed={"gist": "hi"},
        timestamp="2026-07-16T00:00:00+00:00",
    )
    assert sig.kind == KIND_INTERNAL_RESULT
    read = read_internal_result(sig)
    assert read.correlation_id == "c1"
    assert read.raw == '{"gist": "hi"}'
    assert read.parsed == {"gist": "hi"}


def test_internal_result_roundtrip_without_parsed():
    sig = internal_result_signal(
        origin_id="internal-result-c2",
        correlation_id="c2",
        raw="",
        parsed=None,
        timestamp=None,
    )
    read = read_internal_result(sig)
    assert read.correlation_id == "c2"
    assert read.raw == ""
    assert read.parsed is None


def test_read_internal_result_rejects_wrong_kind():
    with pytest.raises(ValueError):
        read_internal_result(Signal(origin_id="x", kind="not_it", payload={}, timestamp=None))


def test_read_internal_result_rejects_malformed_payload():
    with pytest.raises(ValueError):
        read_internal_result(
            Signal(origin_id="x", kind=KIND_INTERNAL_RESULT, payload={}, timestamp=None)
        )

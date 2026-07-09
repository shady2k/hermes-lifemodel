"""Tests for ``ActiveSpan`` — the mutable, decision-values span handle (spec §4.1).

Contract under test:
* it is SEPARATE from the frozen ``TraceContext`` (which is unchanged);
* ``set`` accumulates attributes; ``attrs`` is a read-only view of them;
* ``end`` records the terminal status (+ optional end instant);
* the live ``MutableActiveSpan`` and the ``FakeActiveSpan`` both satisfy the
  ``ActiveSpan`` protocol structurally.
"""

from __future__ import annotations

import pytest

from lifemodel.ports.tracer import ActiveSpan, MutableActiveSpan, TraceContext, start_span
from lifemodel.testing import FakeActiveSpan

_CTX = TraceContext(trace_id="a" * 32, span_id="b" * 16, parent_span_id="c" * 16)


def test_start_span_carries_context_and_metadata() -> None:
    span = start_span(_CTX, component="cognition", tick=7, started_at="2026-07-09T00:00:00+00:00")
    assert span.context is _CTX  # the frozen ids are carried, not copied/mutated
    assert span.component == "cognition"
    assert span.tick == 7
    assert span.started_at == "2026-07-09T00:00:00+00:00"
    assert span.status == "ok"  # not ended yet
    assert span.ended_at is None
    assert dict(span.attrs) == {}


def test_set_accumulates_and_merges_attrs() -> None:
    span = start_span(_CTX)
    span.set(u=0.5, gate="open")
    span.set(effective_pressure=1.25, gate="silent")  # later key wins
    assert dict(span.attrs) == {"u": 0.5, "gate": "silent", "effective_pressure": 1.25}


def test_set_returns_self_for_chaining() -> None:
    span = start_span(_CTX)
    assert span.set(a=1).set(b=2) is span
    assert dict(span.attrs) == {"a": 1, "b": 2}


def test_end_records_status_and_end_instant() -> None:
    span = start_span(_CTX)
    returned = span.end(status="suppressed", ended_at="2026-07-09T00:01:00+00:00")
    assert returned is span
    assert span.status == "suppressed"
    assert span.ended_at == "2026-07-09T00:01:00+00:00"


def test_end_defaults_to_ok_status() -> None:
    span = start_span(_CTX)
    span.end()
    assert span.status == "ok"


def test_trace_context_stays_frozen() -> None:
    # ActiveSpan is the mutable half; TraceContext must remain immutable.
    with pytest.raises((AttributeError, TypeError)):
        _CTX.trace_id = "z" * 32  # type: ignore[misc]


def test_mutable_and_fake_satisfy_the_active_span_protocol() -> None:
    assert isinstance(MutableActiveSpan(context=_CTX), ActiveSpan)
    assert isinstance(FakeActiveSpan(_CTX), ActiveSpan)


def test_fake_active_span_records_set_and_end() -> None:
    span = FakeActiveSpan(_CTX, component="aggregation", tick=3)
    span.set(verdict="fulfill").end(status="ok")
    assert dict(span.attrs) == {"verdict": "fulfill"}
    assert span.ended is True
    assert span.status == "ok"
    assert span.tick == 3

"""Tests for ``SpanLogger`` — the "no-log-without-span" fan-out logger (spec §4.1/§4.2).

Contract under test:
* every call SELF-stamps the bound span's ``trace_id``/``span_id``/``tick`` — a
  caller cannot forget, and a stray field cannot clobber the stamped ids;
* durable-first: the record is enqueued to the trace writer FIRST, and the ring
  + human ``agent.log`` projections happen ONLY on enqueue success;
* on a full queue (enqueue returns ``False``) NO projections are written;
* the human tail goes through stdlib ``logging.getLogger("lifemodel")``;
* ``FakeSpanLogger`` records events with the span's ids stamped in.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC
from typing import Any

import pytest

from lifemodel.events import EventRing
from lifemodel.log import SpanLogger
from lifemodel.ports.tracer import TraceContext
from lifemodel.testing import FakeActiveSpan, FakeSpanLogger

_CTX = TraceContext(trace_id="a" * 32, span_id="b" * 16)


class _StubWriter:
    """A trace-writer stand-in whose enqueue result is controllable."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[dict[str, Any]] = []

    def submit_event(
        self,
        *,
        record_id: int,
        trace_id: str,
        span_id: str | None,
        tick: int | None,
        event: str,
        ts: str,
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        self.calls.append(
            {
                "record_id": record_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "tick": tick,
                "event": event,
                "ts": ts,
                "fields": dict(fields) if fields else {},
            }
        )
        return self.ok


def _logger(writer: _StubWriter, ring: EventRing, *, tick: int = 5) -> SpanLogger:
    span = FakeActiveSpan(_CTX, tick=tick)
    return SpanLogger(span, writer=writer, ring=ring, now=lambda: _fixed_now())


def _fixed_now() -> Any:
    from datetime import datetime

    return datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)


def test_emit_stamps_span_ids_on_the_durable_record() -> None:
    writer = _StubWriter()
    ring = EventRing()
    _logger(writer, ring).info("tick", u=0.5)

    call = writer.calls[0]
    assert call["trace_id"] == _CTX.trace_id
    assert call["span_id"] == _CTX.span_id
    assert call["tick"] == 5
    assert call["event"] == "tick"
    assert call["fields"] == {"u": 0.5}
    assert isinstance(call["record_id"], int)


def test_emit_projects_onto_ring_with_ids_after_enqueue() -> None:
    writer = _StubWriter()
    ring = EventRing()
    _logger(writer, ring).info("tick", u=0.5)

    record = ring.read()[0]
    assert record["event"] == "tick"
    assert record["trace_id"] == _CTX.trace_id
    assert record["span_id"] == _CTX.span_id
    assert record["tick"] == 5
    assert record["u"] == 0.5
    assert isinstance(record["record_id"], int)
    assert record["record_id"] == writer.calls[0]["record_id"]  # same id both places


def test_stamped_ids_win_over_a_colliding_field() -> None:
    writer = _StubWriter()
    ring = EventRing()
    _logger(writer, ring).info("evil", trace_id="HACKED", span_id="HACKED")
    record = ring.read()[0]
    assert record["trace_id"] == _CTX.trace_id  # not "HACKED"
    assert record["span_id"] == _CTX.span_id


def test_durable_first_no_projection_when_queue_full() -> None:
    writer = _StubWriter(ok=False)  # enqueue always fails (queue full)
    ring = EventRing()
    with caplog_at("lifemodel") as records:
        _logger(writer, ring).info("tick", u=0.5)

    assert len(writer.calls) == 1  # the enqueue WAS attempted (durable-first)
    assert ring.read() == []  # ...but NO ring projection
    assert records == []  # ...and NO human agent.log line


def test_human_tail_goes_through_stdlib_logging(caplog: pytest.LogCaptureFixture) -> None:
    writer = _StubWriter()
    ring = EventRing()
    with caplog.at_level(logging.INFO, logger="lifemodel"):
        _logger(writer, ring).info("proactive_tick", u=0.9)
    assert "proactive_tick" in caplog.text
    assert _CTX.trace_id in caplog.text  # the summary references the full trace


def test_all_levels_emit_the_human_tail(caplog: pytest.LogCaptureFixture) -> None:
    writer = _StubWriter()
    ring = EventRing()
    logger = _logger(writer, ring)
    with caplog.at_level(logging.DEBUG, logger="lifemodel"):
        logger.debug("d_event")
        logger.info("i_event")
        logger.warning("w_event")
        logger.error("e_event")
        logger.critical("c_event")
    for expected in ("d_event", "i_event", "w_event", "e_event", "c_event"):
        assert expected in caplog.text
    # And every level reached the durable path too (sqlite is the complete trace).
    assert [c["event"] for c in writer.calls] == [
        "d_event",
        "i_event",
        "w_event",
        "e_event",
        "c_event",
    ]


def test_span_property_exposes_the_bound_span() -> None:
    writer = _StubWriter()
    ring = EventRing()
    span = FakeActiveSpan(_CTX, tick=1)
    logger = SpanLogger(span, writer=writer, ring=ring)
    assert logger.span is span


def test_fake_span_logger_records_events_with_ids() -> None:
    span = FakeActiveSpan(_CTX, tick=9)
    logger = FakeSpanLogger(span)
    logger.info("wake_decision", wake=True)
    logger.debug("detail", x=1)
    assert logger.events == [
        {
            "level": "info",
            "event": "wake_decision",
            "wake": True,
            "trace_id": _CTX.trace_id,
            "span_id": _CTX.span_id,
            "tick": 9,
        },
        {
            "level": "debug",
            "event": "detail",
            "x": 1,
            "trace_id": _CTX.trace_id,
            "span_id": _CTX.span_id,
            "tick": 9,
        },
    ]


# --- a tiny context manager to capture "lifemodel" records without caplog fixture ---


class caplog_at:
    """Capture records from *logger_name* into a list for the ``with`` block."""

    def __init__(self, logger_name: str) -> None:
        self._logger = logging.getLogger(logger_name)
        self._records: list[logging.LogRecord] = []
        self._handler = _ListHandler(self._records)
        self._prev_level = self._logger.level

    def __enter__(self) -> list[logging.LogRecord]:
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self._records

    def __exit__(self, *exc: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)


class _ListHandler(logging.Handler):
    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)

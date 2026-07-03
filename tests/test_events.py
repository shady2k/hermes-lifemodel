"""Tests for ``EventSink`` — the bounded, best-effort event ring (HLA §12/§13).

Contract under test:
* emit → read round-trips a record and preserves fields + order;
* the ring is bounded — many emits never grow the file past ``max_records``;
* emit is best-effort — a write failure (bad path) or an unserializable field
  is swallowed, never propagated to the caller;
* read is tolerant — a missing file yields ``[]`` and a torn/malformed line is
  skipped rather than raising.

Every test injects a ``tmp_path`` dir; no Hermes is imported.
"""

from __future__ import annotations

from pathlib import Path

from lifemodel.events import EVENTS_FILENAME, EventSink


def _sink(tmp_path: Path, **kw: int) -> EventSink:
    return EventSink(tmp_path / EVENTS_FILENAME, **kw)


def test_emit_then_read_roundtrips_fields_and_order(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.emit("tick", {"pressure": 2.5, "energy": 0.8})
    sink.emit("wake_decision", {"wake": True, "reason": "overdue"})

    records = sink.read()

    assert records == [
        {"event": "tick", "pressure": 2.5, "energy": 0.8},
        {"event": "wake_decision", "wake": True, "reason": "overdue"},
    ]


def test_emit_without_fields_records_bare_event(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.emit("dream_run")
    assert sink.read() == [{"event": "dream_run"}]


def test_ring_is_bounded_and_keeps_newest(tmp_path: Path) -> None:
    sink = _sink(tmp_path, max_records=10)

    for i in range(100):
        sink.emit("tick", {"seq": i})

    records = sink.read()
    # Bounded: never more than the cap, on disk and on read.
    assert len(records) == 10
    on_disk = (tmp_path / EVENTS_FILENAME).read_text(encoding="utf-8")
    assert on_disk.count("\n") == 10
    # And it kept the *newest* records (90..99), dropping the oldest.
    assert [r["seq"] for r in records] == list(range(90, 100))


def test_read_limit_returns_most_recent(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    for i in range(5):
        sink.emit("tick", {"seq": i})

    assert [r["seq"] for r in sink.read(limit=2)] == [3, 4]
    assert sink.read(limit=0) == []


def test_read_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _sink(tmp_path).read() == []


def test_read_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / EVENTS_FILENAME
    # A valid record, a non-JSON line, and a JSON non-object line.
    path.write_text('{"event": "tick", "seq": 1}\nnot json\n[1, 2, 3]\n', encoding="utf-8")

    records = EventSink(path).read()

    assert records == [{"event": "tick", "seq": 1}]


def test_emit_is_best_effort_on_unwritable_path(tmp_path: Path) -> None:
    # Parent is a regular file, so the sink can never create its dir/file.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    sink = EventSink(blocker / EVENTS_FILENAME)

    # Must not raise, and nothing is recorded.
    sink.emit("tick", {"seq": 1})

    assert sink.read() == []


def test_emit_is_best_effort_on_unserializable_field(tmp_path: Path) -> None:
    sink = _sink(tmp_path)

    # A non-finite float is not valid JSON (allow_nan=False) → swallowed, and a
    # later good emit still lands. The bad one leaves no torn line behind.
    sink.emit("tick", {"pressure": float("nan")})
    sink.emit("tick", {"pressure": 1.0})

    assert sink.read() == [{"event": "tick", "pressure": 1.0}]

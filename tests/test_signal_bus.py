"""Tests for ``FileSignalBus`` — the durable append-only signal log (HLA §2/§10).

Contract under test:
* publish then ``consume_unprocessed`` returns the unprocessed signals in order;
* a duplicate origin id is never returned twice (within a batch or across calls);
* consumed state persists across a fresh bus over the same directory (restart);
* a torn final log line (crash mid-append) is ignored, not misread.

Every test injects a ``tmp_path`` dir; no Hermes is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.signal_bus import SignalBus
from lifemodel.domain.signal import Signal


def _sig(origin_id: str, kind: str = "test") -> Signal:
    return Signal(origin_id=origin_id, kind=kind)


def test_is_a_signal_bus(tmp_path: Path) -> None:
    assert isinstance(FileSignalBus(tmp_path), SignalBus)


def test_publish_then_consume_returns_unprocessed_in_order(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    bus.publish(_sig("m2"))
    bus.publish(_sig("m3"))
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m1", "m2", "m3"]


def test_consume_is_idempotent_across_calls(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m1"]
    # Nothing new published — a second consume returns nothing.
    assert bus.consume_unprocessed() == []


def test_only_new_signals_are_returned_after_a_consume(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    bus.consume_unprocessed()
    bus.publish(_sig("m2"))
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m2"]


def test_duplicate_id_within_a_batch_is_returned_once(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1", kind="gateway-turn"))
    bus.publish(_sig("m1", kind="next-tick"))  # same origin id, HLA §10
    out = bus.consume_unprocessed()
    assert [s.origin_id for s in out] == ["m1"]


def test_duplicate_id_across_ticks_is_not_recounted(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    bus.consume_unprocessed()
    # A later producer re-publishes the same message id; it must not re-fire.
    bus.publish(_sig("m1"))
    assert bus.consume_unprocessed() == []


def test_consumed_state_persists_across_a_fresh_bus(tmp_path: Path) -> None:
    first = FileSignalBus(tmp_path)
    first.publish(_sig("m1"))
    first.publish(_sig("m2"))
    assert len(first.consume_unprocessed()) == 2

    # A brand-new bus over the same dir must see the same consumed ledger.
    second = FileSignalBus(tmp_path)
    assert second.consume_unprocessed() == []

    # ...but still deliver a signal published after the restart, exactly once.
    second.publish(_sig("m3"))
    assert [s.origin_id for s in second.consume_unprocessed()] == ["m3"]
    assert second.consume_unprocessed() == []


def test_published_signals_survive_a_restart_before_first_consume(tmp_path: Path) -> None:
    # Durability: signals published but not yet consumed are not lost on restart.
    FileSignalBus(tmp_path).publish(_sig("m1"))
    reopened = FileSignalBus(tmp_path)
    assert [s.origin_id for s in reopened.consume_unprocessed()] == ["m1"]


def test_consume_empty_bus_returns_empty(tmp_path: Path) -> None:
    assert FileSignalBus(tmp_path).consume_unprocessed() == []


def test_signal_payload_round_trips_through_the_log(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    original = Signal(
        origin_id="m1",
        kind="incoming",
        payload={"text": "hi", "n": 2},
        timestamp="2026-07-03T12:00:00Z",
        salience=3.0,
    )
    bus.publish(original)
    (restored,) = FileSignalBus(tmp_path).consume_unprocessed()
    assert restored == original


def test_torn_final_log_line_is_ignored(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    # Simulate a crash mid-append: a partial, unterminated JSON line.
    with open(tmp_path / "signals.log", "a", encoding="utf-8") as handle:
        handle.write('{"origin_id": "m2", "kind": "tor')  # no newline, truncated
    # The committed record is still readable; the torn tail is dropped.
    assert [s.origin_id for s in bus.consume_unprocessed()] == ["m1"]


def test_no_hermes_imported_by_signal_bus(tmp_path: Path) -> None:
    bus = FileSignalBus(tmp_path)
    bus.publish(_sig("m1"))
    bus.consume_unprocessed()
    assert "hermes_constants" not in sys.modules
    assert not any(m == "hermes" or m.startswith("hermes.") for m in sys.modules)

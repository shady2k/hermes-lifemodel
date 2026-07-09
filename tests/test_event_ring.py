"""Tests for ``EventRing`` — the thread-safe, in-memory freshness ring (spec §4.2).

Contract under test:
* ``append``/``emit`` round-trip records through ``read``, oldest→newest;
* the ring is bounded to ``max_records`` (newest kept);
* ``read(limit)`` returns the most-recent slice;
* records are copied in and out (no aliasing);
* concurrent appends from many threads never raise and stay bounded.
"""

from __future__ import annotations

import threading

from lifemodel.events import EventRing


def test_append_then_read_round_trips_in_order() -> None:
    ring = EventRing()
    ring.append({"event": "tick", "record_id": 1})
    ring.append({"event": "wake", "record_id": 2})
    assert ring.read() == [
        {"event": "tick", "record_id": 1},
        {"event": "wake", "record_id": 2},
    ]


def test_emit_mirrors_event_sink_shape() -> None:
    ring = EventRing()
    ring.emit("tick", {"pressure": 2.5})
    ring.emit("dream_run")
    assert ring.read() == [
        {"event": "tick", "pressure": 2.5},
        {"event": "dream_run"},
    ]


def test_ring_is_bounded_and_keeps_newest() -> None:
    ring = EventRing(max_records=10)
    for i in range(100):
        ring.append({"seq": i})
    records = ring.read()
    assert len(records) == 10
    assert [r["seq"] for r in records] == list(range(90, 100))


def test_read_limit_returns_most_recent() -> None:
    ring = EventRing()
    for i in range(5):
        ring.append({"seq": i})
    assert [r["seq"] for r in ring.read(limit=2)] == [3, 4]
    assert ring.read(limit=0) == []


def test_records_are_copied_not_aliased() -> None:
    ring = EventRing()
    source = {"event": "tick", "n": 1}
    ring.append(source)
    source["n"] = 999  # mutate after append
    assert ring.read()[0]["n"] == 1  # the ring kept its own copy
    ring.read()[0]["n"] = 42  # mutate what read returned
    assert ring.read()[0]["n"] == 1  # the ring is unaffected


def test_concurrent_appends_stay_bounded_and_never_raise() -> None:
    ring = EventRing(max_records=500)

    def worker(base: int) -> None:
        for i in range(200):
            ring.append({"seq": base + i})

    threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ring.read()) == 500  # bounded despite 1600 concurrent appends

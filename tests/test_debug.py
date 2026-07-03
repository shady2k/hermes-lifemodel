"""Tests for the debug dump — the owner's read-only inspection (HLA §12, NFR9).

Contract under test:
* the dump reflects a known ``State`` and known ``events.jsonl``, and shows
  ``n/a`` for event categories nothing has produced yet;
* it is **read-only** (HLA §9): the state store is never committed to and the
  signal-bus ledger is never mutated — proven with spies and byte-comparison;
* an empty / missing dir produces a clean dump without error;
* it never leaks state to the operator logs (NFR9): the debug path logs nothing.

No Hermes is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

from structlog.testing import capture_logs

from lifemodel.debug import render_debug_dump, render_dump_for_dir
from lifemodel.domain.signal import Signal
from lifemodel.events import EVENTS_FILENAME, EventSink
from lifemodel.state.model import SCHEMA_VERSION, State
from lifemodel.testing.fakes import FakeSignalBus, FakeStateStore


class SpyStateStore(FakeStateStore):
    """A state store that records every ``commit`` so a test can assert none."""

    def __init__(self, initial: State | None = None) -> None:
        super().__init__(initial)
        self.commits: list[State] = []

    def commit(self, state: State) -> None:
        self.commits.append(state)
        super().commit(state)


def _events(tmp_path: Path, *records: dict[str, object]) -> EventSink:
    path = tmp_path / EVENTS_FILENAME
    if records:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return EventSink(path)


def _line(dump: str, label: str) -> str:
    """The single dump line containing *label* (robust to column spacing)."""
    matches = [line for line in dump.splitlines() if label in line]
    assert len(matches) == 1, f"expected exactly one line with {label!r}, got {matches}"
    return matches[0]


def test_dump_shows_state_bus_and_events_with_na_for_absent(tmp_path: Path) -> None:
    state = FakeStateStore(
        State(
            pressure=2.5,
            energy=0.8,
            last_tick_at="2026-07-03T12:00:00Z",
            processed_signal_ids=["a", "b", "c"],
        )
    )
    bus = FakeSignalBus()
    bus.publish(Signal(origin_id="msg-9", kind="incoming"))
    # Only a tick event was produced; wake/act/dream must show n/a.
    events = _events(tmp_path, {"event": "tick", "pressure": 2.5, "energy": 0.8})

    dump = render_debug_dump(state=state, bus=bus, events=events)

    # --- state values present, each on its own line ---
    assert str(SCHEMA_VERSION) in _line(dump, "schema_version:")
    assert "2.5" in _line(dump, "pressure:")
    assert "0.8" in _line(dump, "energy:")
    assert "2026-07-03T12:00:00Z" in _line(dump, "last_tick_at:")
    assert "n/a" in _line(dump, "last_contact_at:")  # None → n/a
    assert "3" in _line(dump, "processed_signal_ids:")
    # --- bus summary ---
    assert "1" in _line(dump, "unprocessed:")
    assert "incoming(msg-9)" in _line(dump, "recent:")
    # --- events: tick populated, the rest n/a ---
    assert "pressure=2.5" in _line(dump, "last tick:")
    assert "n/a" in _line(dump, "last wake_decision:")
    assert "n/a" in _line(dump, "last act_gate:")
    assert "n/a" in _line(dump, "last dream_run:")
    # --- lock status is n/a in Phase 1 ---
    assert "n/a" in _line(dump, "lock status:")


def test_dump_is_read_only_never_commits_state(tmp_path: Path) -> None:
    state = SpyStateStore(State(pressure=1.0))
    bus = FakeSignalBus()

    render_debug_dump(state=state, bus=bus, events=_events(tmp_path))

    assert state.commits == []  # the debug path never writes state


def test_render_dump_for_dir_leaves_files_byte_identical(tmp_path: Path) -> None:
    # Seed a real state.json + a consumed ledger, then prove the dump mutates
    # neither (read-only, HLA §9).
    from lifemodel.adapters.signal_bus import FileSignalBus
    from lifemodel.state.json_store import JsonStateStore

    JsonStateStore(tmp_path).commit(State(pressure=3.0, energy=0.5))
    bus = FileSignalBus(tmp_path)
    bus.publish(Signal(origin_id="s1", kind="incoming"))
    bus.consume_unprocessed()  # writes signals.consumed
    bus.publish(Signal(origin_id="s2", kind="overdue"))  # one still unprocessed

    tracked = ("state.json", "signals.log", "signals.consumed")
    before = {name: (tmp_path / name).read_bytes() for name in tracked}

    dump = render_dump_for_dir(tmp_path)

    after = {name: (tmp_path / name).read_bytes() for name in tracked}
    assert after == before  # nothing on disk changed
    assert "1" in _line(dump, "unprocessed:")  # s2 still pending after the peek
    assert "3.0" in _line(dump, "pressure:")  # the committed state is reflected


def test_dump_on_empty_dir_is_clean(tmp_path: Path) -> None:
    # No state.json, no events.jsonl, no bus files: a clean default dump.
    dump = render_dump_for_dir(tmp_path)

    assert "unreadable" not in dump
    assert str(SCHEMA_VERSION) in _line(dump, "schema_version:")
    assert "0.0" in _line(dump, "pressure:")  # documented default State
    assert "0" in _line(dump, "unprocessed:")
    assert "n/a" in _line(dump, "last tick:")
    # Read-only: inspecting an empty dir must not create any files.
    assert not any(tmp_path.iterdir())


def test_dump_survives_a_corrupt_state_file(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{ not json", encoding="utf-8")

    dump = render_dump_for_dir(tmp_path)

    # A debug tool must report the breakage, not crash on it.
    assert "unreadable" in dump
    # Other sections still render.
    assert "0" in _line(dump, "unprocessed:")


def test_debug_path_emits_nothing_to_operator_logs(tmp_path: Path) -> None:
    # Privacy (NFR9): the dump is returned to the owner, never logged. Seed
    # state so there is soul-ish content that must NOT reach the logs.
    from lifemodel.state.json_store import JsonStateStore

    JsonStateStore(tmp_path).commit(State(pressure=9.9))

    with capture_logs() as logs:
        render_dump_for_dir(tmp_path)

    assert logs == []

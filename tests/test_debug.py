"""Tests for the structured debug dump — read-only introspection (lm-zmf, HLA §12/NFR9).

Contract under test (spec §4/§7):
* every section is present with its expected labels (META, TEMPERAMENT, DRIVE,
  DESIRE LIFECYCLE, TIMING, WAKE READINESS, AUTONOMOUS LOOP);
* the WAKE READINESS branch is honest about runtime ``in_flight`` — ``n/a`` when
  ``u < θ`` (cannot matter), ``UNKNOWN`` + a ⚠ caveat when ``u ≥ θ``;
* the gate verdict and "would launch outreach" are shown separately (the dedup
  case: gate=URGE but launch=no);
* an unreadable store yields an ``<unreadable: ...>`` banner, the rest still renders;
* it is **read-only** (HLA §9): the dump never writes a byte; it logs nothing.

No Hermes is imported.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from lifemodel.core.introspect import PersonalityReadings, compute_readings
from lifemodel.debug import render_debug_dump, render_dump_for_dir
from lifemodel.domain.signal import Signal
from lifemodel.events import EVENTS_FILENAME, EventSink
from lifemodel.state.model import SCHEMA_VERSION, State
from lifemodel.testing.fakes import FakeSignalBus


def at(mins: float) -> datetime:
    return datetime(2026, 7, 5, 0, 0, tzinfo=UTC) + timedelta(minutes=mins)


def _readings(state: State, *, now: datetime) -> PersonalityReadings:
    return compute_readings(state, now=now)


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


def _rung(dump: str, name: str) -> str:
    """The single wake-gate ladder rung line for *name* (indented under the ladder)."""
    matches = [line for line in dump.splitlines() if line.startswith(f"    {name}")]
    assert len(matches) == 1, f"expected one ladder rung {name!r}, got {matches}"
    return matches[0]


# --- section presence + values ---------------------------------------------------------------


def test_all_sections_present_with_expected_labels(tmp_path: Path) -> None:
    state = State(
        u=2.5,
        energy=0.8,
        tick_count=668,
        last_tick_at=at(0).isoformat(),
    )
    bus = FakeSignalBus()
    bus.publish(Signal(origin_id="msg-9", kind="incoming"))

    dump = render_debug_dump(
        readings=_readings(state, now=at(10)), bus=bus, events=_events(tmp_path)
    )

    # every section header
    for header in (
        "META",
        "TEMPERAMENT",
        "DRIVE",
        "DESIRE LIFECYCLE",
        "TIMING",
        "WAKE READINESS",
        "AUTONOMOUS LOOP",
    ):
        assert header in dump
    # META values
    assert str(SCHEMA_VERSION) in _line(dump, "schema_version:")
    assert "668" in _line(dump, "tick_count:")
    # TEMPERAMENT imports the real constants (no restated formulas)
    assert "wake threshold θ:" in dump
    assert "silence window w:" in dump
    assert "pending-verdict timeout:" in dump
    # DRIVE surfaces the persisted urge + the tracked-not-gating duration field
    assert "2.5" in _line(dump, "u:")
    assert "duration_over_theta:" in dump
    assert "tracked; not currently gating" in dump
    assert "energy:" in dump and "placeholder" in dump
    # DESIRE LIFECYCLE
    assert "desire_status:" in dump
    assert "pending:" in dump
    assert "decline_count:" in dump
    # TIMING
    assert "last_exchange:" in dump
    assert "last_contact:" in dump
    assert "last_tick:" in dump
    # WAKE READINESS shows verdict + launch distinctly
    assert "gate verdict:" in dump
    assert "would launch outreach:" in dump
    assert "assuming in_flight=false" in dump
    # AUTONOMOUS LOOP
    assert "1" in _line(dump, "signal bus unprocessed:")
    assert "incoming(msg-9)" in _line(dump, "recent:")
    assert "lock status:" in dump


def test_absent_events_render_na_in_loop_summary(tmp_path: Path) -> None:
    state = State(last_tick_at=at(0).isoformat())
    # Only a tick event; wake/act/dream must show n/a in the compact summary.
    events = _events(tmp_path, {"event": "tick", "deferred": "service_alive"})

    dump = render_debug_dump(
        readings=_readings(state, now=at(1)), bus=FakeSignalBus(), events=events
    )

    assert "deferred=service_alive" in _line(dump, "last tick outcome:")
    summary = _line(dump, "events:")
    assert "wake_decision n/a" in summary
    assert "act_gate n/a" in summary
    assert "dream_run n/a" in summary


# --- the in_flight honesty branch (must-fix 1 & 2) --------------------------------------------


def test_below_threshold_shows_in_flight_na_and_no_warning(tmp_path: Path) -> None:
    # u stays well under θ: in_flight cannot matter, so no runtime caveat.
    state = State(last_tick_at=at(0).isoformat())  # u=0
    dump = render_debug_dump(
        readings=_readings(state, now=at(10)), bus=FakeSignalBus(), events=_events(tmp_path)
    )
    assert "n/a" in _rung(dump, "in_flight")
    assert "⚠" not in dump


def test_over_threshold_shows_in_flight_unknown_with_warning(tmp_path: Path) -> None:
    # u over θ with no exchange/decline → gate=URGE; in_flight is runtime-only.
    state = State(u=50.0, last_tick_at=at(0).isoformat())
    dump = render_debug_dump(
        readings=_readings(state, now=at(10)), bus=FakeSignalBus(), events=_events(tmp_path)
    )
    assert "UNKNOWN" in _rung(dump, "in_flight")
    assert "⚠" in dump
    assert "no_wake_in_flight" in dump  # the caveat names the real runtime verdict


def test_gate_verdict_and_would_launch_shown_separately_for_dedup(tmp_path: Path) -> None:
    # Gate clears (URGE) but a desire is already live → no launch (Aggregator dedup).
    state = State(u=50.0, desire_status="active", last_tick_at=at(0).isoformat())
    dump = render_debug_dump(
        readings=_readings(state, now=at(10)), bus=FakeSignalBus(), events=_events(tmp_path)
    )
    assert "URGE" in _line(dump, "gate verdict:")
    launch = _line(dump, "would launch outreach:")
    assert launch.startswith("  would launch outreach: no")
    assert "Aggregator dedup" in launch


def test_clean_urge_launches_and_ladder_reached(tmp_path: Path) -> None:
    state = State(last_tick_at=at(0).isoformat())
    dump = render_debug_dump(
        readings=_readings(state, now=at(240)), bus=FakeSignalBus(), events=_events(tmp_path)
    )
    assert _line(dump, "would launch outreach:").startswith("  would launch outreach: yes")
    assert "reached" in _rung(dump, "urge")


def test_stale_pending_recovery_surfaced(tmp_path: Path) -> None:
    from lifemodel.core.decision import PENDING_TIMEOUT_MIN

    state = State(
        u=99.0,
        desire_status="active",
        pending_proactive_id="p1",
        pending_proactive_since=at(0).isoformat(),
        last_tick_at=at(0).isoformat(),
    )
    dump = render_debug_dump(
        readings=_readings(state, now=at(PENDING_TIMEOUT_MIN)),
        bus=FakeSignalBus(),
        events=_events(tmp_path),
    )
    stale = _line(dump, "stale-pending recovery:")
    assert "fires" in stale and "REJECT" in stale


# --- read-only + degradation + privacy -------------------------------------------------------


def test_render_dump_for_dir_leaves_files_byte_identical(tmp_path: Path) -> None:
    # Seed a real state.json + a consumed ledger, then prove the dump mutates
    # neither (read-only, HLA §9) — including the live decision run on a copy.
    from lifemodel.adapters.signal_bus import FileSignalBus
    from lifemodel.state.json_store import JsonStateStore

    JsonStateStore(tmp_path).commit(State(u=3.0, energy=0.5))
    bus = FileSignalBus(tmp_path)
    bus.publish(Signal(origin_id="s1", kind="incoming"))
    bus.consume_unprocessed()  # writes signals.consumed
    bus.publish(Signal(origin_id="s2", kind="overdue"))  # one still unprocessed

    tracked = ("state.json", "signals.log", "signals.consumed")
    before = {name: (tmp_path / name).read_bytes() for name in tracked}

    dump = render_dump_for_dir(tmp_path)

    after = {name: (tmp_path / name).read_bytes() for name in tracked}
    assert after == before  # nothing on disk changed — the copy never reached disk
    assert "1" in _line(dump, "signal bus unprocessed:")  # s2 still pending after the peek
    assert "(300% of θ)" in _line(dump, "u:")  # the committed u=3.0 is reflected


def test_render_dump_for_dir_routes_through_the_composition_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # debug must build its object graph via the SINGLE composition root
    # (build_lifemodel), not a divergent second wiring — so a later change to
    # build_lifemodel's defaults is reflected by the debug dump automatically.
    from lifemodel.composition import build_lifemodel as real_build

    seen: list[Path] = []

    def spy(*, base_dir: Path) -> object:
        seen.append(base_dir)
        return real_build(base_dir=base_dir)

    monkeypatch.setattr("lifemodel.debug.build_lifemodel", spy)
    dump = render_dump_for_dir(tmp_path)

    assert seen == [tmp_path]  # built exactly once, through the composition root
    assert str(SCHEMA_VERSION) in _line(dump, "schema_version:")  # and still renders


def test_dump_on_empty_dir_is_clean(tmp_path: Path) -> None:
    # No state.json, no events.jsonl, no bus files: a clean default dump.
    dump = render_dump_for_dir(tmp_path)

    assert "unreadable" not in dump
    assert str(SCHEMA_VERSION) in _line(dump, "schema_version:")
    assert "(0% of θ)" in _line(dump, "u:")  # documented default State (u=0)
    assert "0" in _line(dump, "signal bus unprocessed:")
    assert "n/a" in _line(dump, "last tick outcome:")
    # Read-only: inspecting an empty dir must not create any files.
    assert not any(tmp_path.iterdir())


def test_dump_survives_a_corrupt_state_file(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{ not json", encoding="utf-8")

    dump = render_dump_for_dir(tmp_path)

    # A debug tool must report the breakage, not crash on it.
    assert "unreadable" in dump
    # The constant TEMPERAMENT block still renders (it needs no state).
    assert "TEMPERAMENT" in dump
    # Other sections still render.
    assert "0" in _line(dump, "signal bus unprocessed:")
    assert "n/a" in _line(dump, "lock status:")


def test_debug_path_emits_nothing_to_operator_logs(tmp_path: Path) -> None:
    # Privacy (NFR9): the dump is returned to the owner, never logged. Seed
    # state so there is soul-ish content that must NOT reach the logs.
    from lifemodel.state.json_store import JsonStateStore

    JsonStateStore(tmp_path).commit(State(u=9.9))

    with capture_logs() as logs:
        render_dump_for_dir(tmp_path)

    assert logs == []

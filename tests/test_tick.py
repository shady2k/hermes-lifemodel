"""Unit tests for the heartbeat tick entrypoint (roadmap 1.1; Task 4: watchdog).

These drive the tick with injected fakes (``FakeClock`` / ``FakeStateStore``) —
no Hermes, no LLM. Per the wire-desire-model plan (Task 4), the in-process
egress service is the sole decision brain — this cron tick never wakes, however
mature the persisted urge is. What remains: it (1) defers entirely, writing
nothing, while the in-process service's liveness stamp is fresh, (2) otherwise
advances ``tick_count``/``last_tick_at`` and commits (pure bookkeeping, zero
LLM), (3) always emits the ``{"wakeAgent": false}`` gate line, and (4) fails
closed on any crash so Hermes never wakes on a tick error.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import lifemodel.tick as tick_mod
from lifemodel.composition import build_lifemodel
from lifemodel.domain.wake import WakeDecision, WakePacket
from lifemodel.events import EVENT_TICK, EVENT_TICK_FAILED, EVENTS_FILENAME, EventSink
from lifemodel.state.json_store import JsonStateStore
from lifemodel.state.model import State
from lifemodel.testing.fakes import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore
from lifemodel.tick import main, run_tick, wake_gate_line

_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


class _RecordingLogger:
    """Minimal :class:`EventLogger` that records the events it is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.calls.append((event, dict(fields)))


NULL_LOGGER = _RecordingLogger()


def _build(**overrides: Any) -> Any:
    """Assemble a graph over fakes; callers override individual collaborators."""
    params: dict[str, Any] = {
        "base_dir": Path("/unused-for-fakes"),
        "state": FakeStateStore(),
        "bus": FakeSignalBus(),
        "clock": FakeClock(_T0),
        "delivery": FakeDelivery(),
        "neurons": (),
    }
    params.update(overrides)
    return build_lifemodel(**params)


def make_lm_high_u() -> Any:
    """``u`` well past ``THETA``, past the active-silence window, no reject — the
    state that WOULD wake if this were the egress path's ``decide_reachout``.
    Proves the cron watchdog stays silent regardless of how mature the urge is —
    the in-process service is the only brain that ever decides to wake."""
    return _build(
        state=FakeStateStore(
            State(u=50.0, last_exchange_at=(_T0 - timedelta(minutes=20)).isoformat())
        )
    )


def test_run_tick_never_wakes_even_with_a_mature_urge() -> None:
    # Task 4 acceptance: cron is a silent watchdog — it never wakes, whatever
    # state it finds (that call belongs solely to the in-process service).
    lm = make_lm_high_u()

    decision = run_tick(lm, logger=NULL_LOGGER)

    assert decision.wake is False
    assert lm.state.load().last_tick_at is not None  # still ticks bookkeeping


def test_cron_gate_line_is_always_stay_asleep() -> None:
    assert wake_gate_line(run_tick(make_lm_high_u(), logger=NULL_LOGGER)) == '{"wakeAgent": false}'


def test_run_tick_advances_state_and_commits() -> None:
    store = FakeStateStore()
    lm = _build(state=store, clock=FakeClock(_T0))

    decision = run_tick(lm, logger=_RecordingLogger())

    persisted = store.load()
    assert persisted.tick_count == 1
    assert persisted.last_tick_at == _T0.isoformat()
    assert decision.wake is False


def test_run_tick_persists_between_ticks() -> None:
    store = FakeStateStore()
    clock = FakeClock(_T0)
    lm = _build(state=store, clock=clock)

    run_tick(lm, logger=_RecordingLogger())
    first = store.load()
    clock.advance(timedelta(minutes=1))
    run_tick(lm, logger=_RecordingLogger())
    second = store.load()

    assert (first.tick_count, second.tick_count) == (1, 2)
    assert second.last_tick_at == (_T0 + timedelta(minutes=1)).isoformat()
    assert second.last_tick_at != first.last_tick_at


def test_run_tick_emits_tick_event_with_bookkeeping() -> None:
    logger = _RecordingLogger()
    lm = _build(clock=FakeClock(_T0))

    run_tick(lm, logger=logger)

    ticks = [fields for event, fields in logger.calls if event == EVENT_TICK]
    assert len(ticks) == 1
    assert ticks[0]["tick_count"] == 1
    assert ticks[0]["last_tick_at"] == _T0.isoformat()
    assert ticks[0]["wake"] is False


def test_run_tick_never_wakes_and_never_delivers() -> None:
    # The walking skeleton stays zero-cost every tick: no wake, so nothing is
    # delivered and no LLM path is entered.
    delivery = FakeDelivery()
    lm = _build(delivery=delivery)

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert delivery.sent == []


def test_run_tick_defers_and_writes_nothing_when_service_is_alive() -> None:
    # Liveness watchdog (spec §6): while the in-process service's stamp is
    # fresh, it owns state exclusively — the cron tick must not touch
    # tick_count/last_tick_at (no two brains racing the same commit).
    store = FakeStateStore(State(egress_service_alive_at=_T0.isoformat(), tick_count=5))
    lm = _build(state=store, clock=FakeClock(_T0))

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert store.load().tick_count == 5  # untouched while the service is alive


def test_run_tick_falls_back_to_bookkeeping_once_the_stamp_is_stale() -> None:
    from lifemodel.tick import SERVICE_LIVENESS_MAX_AGE

    stale_stamp = (_T0 - SERVICE_LIVENESS_MAX_AGE - timedelta(seconds=1)).isoformat()
    store = FakeStateStore(State(egress_service_alive_at=stale_stamp, tick_count=5))
    lm = _build(state=store, clock=FakeClock(_T0))

    decision = run_tick(lm, logger=_RecordingLogger())

    assert decision.wake is False
    assert store.load().tick_count == 6  # stale stamp -> cron takes the fallback


def test_wake_gate_line_stays_asleep_is_wakeagent_false() -> None:
    line = wake_gate_line(WakeDecision.stay_asleep())
    assert json.loads(line) == {"wakeAgent": False}


def test_wake_gate_line_wake_carries_packet_and_true_flag() -> None:
    # wake_gate_line itself stays a general renderer (unit-tested on its own),
    # even though this entrypoint's run_tick never feeds it a waking decision.
    packet = WakePacket(reason="overdue", pressure_kind="connection", pressure=2.0)
    gate = json.loads(wake_gate_line(WakeDecision.wake_with(packet)))
    assert gate["wakeAgent"] is True
    assert gate["reason"] == "overdue"
    assert gate["pressure_kind"] == "connection"


def test_main_prints_wake_gate_and_persists_across_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Drive the real entrypoint against a throwaway home: the wake gate the
    # scheduler parses (the last non-empty stdout line) is {"wakeAgent": false},
    # and state persists across two invocations.
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    sdir = tmp_path / "workspace" / "lifemodel"
    state = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    assert state["tick_count"] == 2

    records = [
        json.loads(line)
        for line in (sdir / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    tick_events = [r for r in records if r.get("event") == EVENT_TICK]
    assert [r["tick_count"] for r in tick_events] == [1, 2]
    assert all(r["wake"] is False for r in tick_events)


def test_main_stdout_last_line_is_readable_by_the_scheduler_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Belt-and-suspenders: reproduce the scheduler's own _parse_wake_gate logic
    # over main()'s stdout and confirm it decides "do not wake".
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    main()
    out = capsys.readouterr().out
    gate = _last_gate_line(out)
    assert gate.get("wakeAgent", True) is False  # scheduler: False => skip agent


def _last_gate_line(stdout: str) -> dict[str, Any]:
    """Mirror cron/scheduler.py:_parse_wake_gate — parse the last non-empty line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, "tick produced no stdout"
    parsed = json.loads(lines[-1])
    assert isinstance(parsed, dict)
    return parsed


def test_sink_reads_back_the_last_tick(tmp_path: Path) -> None:
    # The tick event lands in the queryable EventSink so /lifemodel debug can
    # answer "last tick" (HLA §12) — exercised via the real EventTee wiring.
    from lifemodel.logging import EventTee, get_logger

    sink = EventSink(tmp_path / EVENTS_FILENAME)
    logger = EventTee(get_logger("lifemodel.tick.test"), sink)
    lm = _build(clock=FakeClock(_T0))

    run_tick(lm, logger=logger)

    last = sink.read()[-1]
    assert last["event"] == EVENT_TICK
    assert last["tick_count"] == 1


def _events_in(sdir: Path) -> list[dict[str, Any]]:
    """Read the on-disk event sink under *sdir* (the profile state dir)."""
    text = (sdir / EVENTS_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def test_main_fails_closed_when_the_tick_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # FINDING 1 (fail CLOSED): Hermes wakes the agent + delivers a "Script Error"
    # message when a cron --script exits non-zero, so ANY unhandled exception in
    # the tick must still print {"wakeAgent": false} and exit 0 — a crash must
    # never wake or deliver. Here run_tick raises; main() must stay silent.
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> WakeDecision:
        raise RuntimeError("simulated tick failure")

    monkeypatch.setattr(tick_mod, "run_tick", _boom)

    assert main() == 0  # exit 0 → Hermes reads the gate, not a crash
    gate = _last_gate_line(capsys.readouterr().out)
    assert gate == {"wakeAgent": False}
    assert gate.get("wakeAgent", True) is False  # scheduler rule: False => skip agent

    records = _events_in(tmp_path / "workspace" / "lifemodel")
    assert any(r.get("event") == EVENT_TICK_FAILED for r in records)  # error recorded
    assert not any(r.get("event") == EVENT_TICK for r in records)  # tick never completed


def test_main_commit_failure_stays_silent_and_state_unwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A commit() failure (disk full / unwritable state dir) during the ordinary
    # bookkeeping tick must still fail closed: {"wakeAgent": false}, exit 0, and
    # the on-disk state is left exactly as seeded (the failing commit never
    # wrote) — a clean tick retries later rather than wedging on a crash.
    monkeypatch.setattr(tick_mod, "_hermes_home", lambda: tmp_path)
    sdir = tmp_path / "workspace" / "lifemodel"

    seed = JsonStateStore(sdir)
    seed.commit(State(tick_count=7))

    def _fail_commit(self: Any, state: State) -> None:
        raise OSError("simulated unwritable state dir")

    monkeypatch.setattr(JsonStateStore, "commit", _fail_commit)

    assert main() == 0
    assert _last_gate_line(capsys.readouterr().out) == {"wakeAgent": False}

    # On-disk state is unchanged (the failing commit never wrote).
    persisted = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    assert persisted["tick_count"] == 7

    records = _events_in(sdir)
    assert any(r.get("event") == EVENT_TICK_FAILED for r in records)

# tests/test_debug.py
from __future__ import annotations

from lifemodel.debug import render_dump_for_dir
from lifemodel.state.json_store import JsonStateStore
from lifemodel.state.model import State


def test_dump_renders_the_sections(tmp_path) -> None:
    JsonStateStore(tmp_path).commit(
        State(u=2.0, energy=0.6, fatigue=0.2, last_tick_at="2026-07-06T00:00:00+00:00")
    )
    out = render_dump_for_dir(tmp_path)
    for section in ("PHYSIOLOGY", "DRIVE", "DESIRE", "GATES", "BACKSTOP", "HEALTH"):
        assert section in out
    assert "effective" in out.lower()


def test_health_shows_brain_liveness_and_drops_service_alive(tmp_path) -> None:
    # last_tick far in the past (relative to real now) -> brain reads STALE
    JsonStateStore(tmp_path).commit(State(last_tick_at="2026-07-06T00:00:00+00:00"))
    out = render_dump_for_dir(tmp_path)
    assert "HEALTH" in out
    assert "brain" in out.lower()
    assert "STALE" in out  # a stale last_tick surfaces as an explicit warning
    assert "service_alive" not in out  # the old two-brain liveness line is gone


def test_dump_survives_a_corrupt_store(tmp_path) -> None:
    (tmp_path / "state.json").write_text("{ not json", encoding="utf-8")
    out = render_dump_for_dir(tmp_path)
    assert "unreadable" in out.lower()  # graceful banner, no crash

"""The ``/lifemodel status`` brain-liveness block (spec §4.4, bead lm-fib.9.3).

Renders the owner-facing liveness section from BOTH sources the spec names:
the process-local :class:`~lifemodel.state.brain_health.BrainHealth` (state /
loop_alive / death_count / last errors) AND the DURABLE primary liveness
(``last_tick_at`` / ``ticks_total`` in AgentState) plus the durable BOOT record
— so a re-raise+restart still shows ``boot_failed: <reason>`` in a fresh process
where the in-memory health is ``never_started``. Deterministic: the pure
renderer is fed values; the ``*_lines`` reader is exercised over a real dir with
an injected ``now``. Rendering NEVER raises — a flaky health read degrades to a
clear ``unknown`` line, logged.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lifemodel.state.brain_health import (
    STALE_AFTER_SECONDS,
    BrainHealth,
    BrainHealthSnapshot,
    get_brain_health,
)
from lifemodel.state.brain_liveness import brain_liveness_lines, render_brain_liveness

_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _snap(
    state: str,
    *,
    boot_error: str | None = None,
    last_loop_death: str | None = None,
    death_count: int = 0,
    last_observer_error: dict[str, str] | None = None,
    connected_at: str | None = None,
) -> BrainHealthSnapshot:
    return BrainHealthSnapshot(
        state=state,  # type: ignore[arg-type]
        boot_error=boot_error,
        last_loop_death=last_loop_death,
        death_count=death_count,
        last_observer_error=last_observer_error or {},
        connected_at=connected_at,
    )


def _text(lines: list[str]) -> str:
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# render_brain_liveness — the pure renderer (fed values)
# --------------------------------------------------------------------------- #


def test_connected_and_fresh_renders_healthy() -> None:
    lines = render_brain_liveness(
        _snap("connected", connected_at=_iso(_NOW - timedelta(seconds=30))),
        last_tick_at=_iso(_NOW - timedelta(seconds=30)),
        ticks_total=42,
        boot_record=None,
        now=_NOW,
    )
    text = _text(lines)
    assert "connected" in text
    assert "loop_alive:** yes" in text
    assert "ticks_total:** 42" in text
    assert _iso(_NOW - timedelta(seconds=30)) in text
    # A fresh tick renders NO staleness warning.
    assert "stale" not in text


def test_connected_but_stale_renders_a_staleness_warning() -> None:
    lines = render_brain_liveness(
        _snap("connected", connected_at=_iso(_NOW - timedelta(hours=1))),
        last_tick_at=_iso(_NOW - timedelta(hours=1)),
        ticks_total=7,
        boot_record=None,
        now=_NOW,
    )
    text = _text(lines)
    assert "connected" in text
    # A wedged brain (last tick an hour ago) shows a visible staleness warning.
    assert "stale" in text
    assert "3600" in text


def test_loop_dead_is_visible() -> None:
    lines = render_brain_liveness(
        _snap(
            "loop_dead",
            last_loop_death="proactive loop died: RuntimeError('boom')\nTraceback...",
            death_count=2,
            connected_at=_iso(_NOW - timedelta(minutes=5)),
        ),
        last_tick_at=_iso(_NOW - timedelta(minutes=5)),
        ticks_total=99,
        boot_record=None,
        now=_NOW,
    )
    text = _text(lines)
    assert "loop_dead" in text
    assert "loop_alive:** no" in text
    assert "death_count:** 2" in text
    assert "boom" in text
    # The multi-line traceback is trimmed to its first line in the block.
    assert "Traceback" not in text


def test_boot_failed_from_the_durable_record_in_a_fresh_process() -> None:
    # The in-memory health is never_started (fresh process after a re-raise+restart),
    # but the DURABLE boot record still explains WHY — the block must surface it.
    boot_record = {
        "state": "boot_failed",
        "boot_error": "register_being_platform: ModuleNotFoundError: No module named 'lifemodel'",
    }
    lines = render_brain_liveness(
        _snap("never_started"),
        last_tick_at=None,
        ticks_total=0,
        boot_record=boot_record,
        now=_NOW,
    )
    text = _text(lines)
    assert "boot_failed" in text
    assert "loop_alive:** no" in text
    assert "ModuleNotFoundError" in text


def test_in_memory_boot_failed_takes_precedence_over_a_record() -> None:
    lines = render_brain_liveness(
        _snap("boot_failed", boot_error="register_command: RuntimeError: nope"),
        last_tick_at=None,
        ticks_total=0,
        boot_record={"state": "boot_failed", "boot_error": "an older reason"},
        now=_NOW,
    )
    text = _text(lines)
    assert "boot_failed" in text
    assert "nope" in text


def test_observer_errors_are_shown() -> None:
    lines = render_brain_liveness(
        _snap(
            "connected",
            connected_at=_iso(_NOW),
            last_observer_error={"post_llm_call": "KeyError: 'x'"},
        ),
        last_tick_at=_iso(_NOW),
        ticks_total=1,
        boot_record=None,
        now=_NOW,
    )
    text = _text(lines)
    assert "observer_errors" in text
    assert "post_llm_call" in text


def test_absent_health_read_degrades_to_a_clear_unknown_line() -> None:
    # snapshot is None (a flaky/absent health read) → a clear "unknown" line, never
    # a crash. The durable last_tick_at is still surfaced if we have it.
    lines = render_brain_liveness(
        None,
        last_tick_at=None,
        ticks_total=None,
        boot_record=None,
        now=_NOW,
    )
    text = _text(lines)
    assert "unknown" in text
    assert "ticks_total:** ?" in text
    assert "never" in text


def test_unknown_health_still_surfaces_a_durable_boot_failure() -> None:
    # Even when the in-memory read failed, a durable boot record still explains it.
    lines = render_brain_liveness(
        None,
        last_tick_at=None,
        ticks_total=None,
        boot_record={"state": "boot_failed", "boot_error": "ImportError: boom"},
        now=_NOW,
    )
    text = _text(lines)
    assert "boot_failed" in text
    assert "ImportError: boom" in text


# --------------------------------------------------------------------------- #
# brain_liveness_lines — the for_dir reader (real dir, injected now)
# --------------------------------------------------------------------------- #


def test_for_dir_reads_durable_liveness_and_default_health(tmp_path: Path) -> None:
    # A never-connected being (fresh dir): never_started, 0 ticks, no crash.
    lines = brain_liveness_lines(tmp_path, now=_NOW)
    text = _text(lines)
    assert "never_started" in text
    assert "ticks_total:** 0" in text


def test_for_dir_loop_dead_is_visible(tmp_path: Path) -> None:
    h = get_brain_health(tmp_path)
    h.mark_connected(at=_iso(_NOW - timedelta(minutes=1)))
    h.record_loop_death("proactive loop died: RuntimeError('boom')", "tb")
    text = _text(brain_liveness_lines(tmp_path, now=_NOW))
    assert "loop_dead" in text
    assert "boom" in text
    assert "death_count:** 1" in text


def test_for_dir_boot_failed_visible_in_a_fresh_process(tmp_path: Path) -> None:
    # Simulate a prior process that boot-failed and persisted brain_boot.json; THIS
    # (fresh) process's in-memory singleton is never_started — the durable record is
    # the only surviving explanation, and the block must still show it.
    BrainHealth(tmp_path).mark_boot_failed(
        "register_being_platform: ModuleNotFoundError: No module named 'lifemodel'"
    )
    assert get_brain_health(tmp_path).state == "never_started"  # fresh in-memory truth
    text = _text(brain_liveness_lines(tmp_path, now=_NOW))
    assert "boot_failed" in text
    assert "ModuleNotFoundError" in text


def test_for_dir_never_raises_on_a_flaky_health_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import lifemodel.state.brain_liveness as bl

    class _Exploding:
        def snapshot(self) -> BrainHealthSnapshot:
            raise RuntimeError("health store on fire")

    monkeypatch.setattr(bl, "get_brain_health", lambda _base: _Exploding())
    with caplog.at_level(logging.WARNING, logger="lifemodel.brain_health"):
        text = _text(brain_liveness_lines(tmp_path, now=_NOW))
    # Degraded to a clear unknown line, and the failure is logged (not swallowed).
    assert "unknown" in text
    assert any("brain_liveness" in r.getMessage() for r in caplog.records)


def test_for_dir_never_raises_on_a_flaky_state_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lifemodel.composition as comp

    def _boom(**_kw: object) -> object:
        raise RuntimeError("state store locked")

    monkeypatch.setattr(comp, "build_lifemodel", _boom)
    text = _text(brain_liveness_lines(tmp_path, now=_NOW))
    # The durable read degraded (ticks unknown) but the block still rendered.
    assert "ticks_total:** ?" in text

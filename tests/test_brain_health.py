"""``BrainHealth`` — the process-local single source of brain liveness (spec §4.2).

The backbone the fail-loud wiring writes and ``check_fn`` / ``/lifemodel status``
read: one small, thread-safe, per-base_dir singleton whose ``state`` is the truth
about whether the being's brain booted, connected, is ticking, or died. A SMALL
durable ``brain_boot.json`` record survives a gateway restart so the owner can
still read *why* a boot failed after the process is revived (spec §4.2/§4.3).

These are deterministic unit tests — no Hermes, no gateway, injected ``now`` /
``last_tick_at`` — mirroring the ``get_metric_registry`` / ``acquire_trace_writer``
singleton style already in the tree.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from lifemodel.state.brain_health import (
    DEFAULT_TICK_INTERVAL_SECONDS,
    STALE_AFTER_SECONDS,
    BrainHealth,
    brain_boot_path,
    get_brain_health,
    read_boot_record,
    tick_staleness,
)

_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
_STALE = 300.0  # a few 60s intervals


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# --------------------------------------------------------------------------- #
# Construction + transitions
# --------------------------------------------------------------------------- #


def test_fresh_health_is_never_started(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    assert h.state == "never_started"
    assert h.boot_error is None
    assert h.last_loop_death is None
    assert h.death_count == 0
    assert h.last_observer_error == {}


def test_connecting_then_connected(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connecting()
    assert h.state == "connecting"
    h.mark_connected(at=_iso(_NOW))
    assert h.state == "connected"


def test_record_loop_death_sets_state_and_counts(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    h.record_loop_death(
        "proactive loop died: RuntimeError('boom')", "Traceback...\nRuntimeError: boom"
    )
    assert h.state == "loop_dead"
    assert h.last_loop_death is not None
    assert "boom" in h.last_loop_death
    assert h.death_count == 1
    # A second death bumps the count.
    h.record_loop_death("proactive loop died: ValueError('again')", None)
    assert h.death_count == 2


def test_clean_reconnect_clears_loop_dead(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    h.record_loop_death("died", "tb")
    assert h.state == "loop_dead"
    # A subsequent clean reconnect flips state back to connected (spec §4.3).
    h.mark_connected(at=_iso(_NOW + timedelta(seconds=1)))
    assert h.state == "connected"
    # death_count is cumulative history, not reset by the recovery.
    assert h.death_count == 1


def test_record_observer_error_is_keyed_by_name(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.record_observer_error("post_llm_call", "KeyError: 'x'")
    h.record_observer_error("pre_gateway_dispatch", "ValueError: y")
    assert h.last_observer_error == {
        "post_llm_call": "KeyError: 'x'",
        "pre_gateway_dispatch": "ValueError: y",
    }
    # A later error for the same observer overwrites (keeps the LAST).
    h.record_observer_error("post_llm_call", "TypeError: z")
    assert h.last_observer_error["post_llm_call"] == "TypeError: z"


# --------------------------------------------------------------------------- #
# Durable boot-health record (survives a gateway restart, spec §4.2/§4.3)
# --------------------------------------------------------------------------- #


def test_mark_boot_failed_persists_a_durable_record(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_boot_failed("register_being_platform: ModuleNotFoundError: No module named 'lifemodel'")
    assert h.state == "boot_failed"
    assert h.boot_error is not None and "ModuleNotFoundError" in h.boot_error
    # The durable record is written so a revived process can report WHY.
    assert brain_boot_path(tmp_path).exists()
    record = read_boot_record(tmp_path)
    assert record is not None
    assert record["state"] == "boot_failed"
    assert "ModuleNotFoundError" in record["boot_error"]


def test_boot_ok_clears_the_durable_record(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_boot_failed("boom")
    assert brain_boot_path(tmp_path).exists()
    # A clean boot this process wipes a stale failure record (a fixed deploy).
    h.mark_boot_ok()
    assert not brain_boot_path(tmp_path).exists()
    assert h.boot_error is None


def test_connected_clears_the_durable_boot_record(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_boot_failed("boom")
    assert brain_boot_path(tmp_path).exists()
    h.mark_connected(at=_iso(_NOW))
    assert not brain_boot_path(tmp_path).exists()


def test_read_boot_record_absent_or_malformed_is_none(tmp_path: Path) -> None:
    assert read_boot_record(tmp_path) is None
    brain_boot_path(tmp_path).write_text("not json{", encoding="utf-8")
    assert read_boot_record(tmp_path) is None


# --------------------------------------------------------------------------- #
# Singleton per base_dir (shared between register() and the /lifemodel handler)
# --------------------------------------------------------------------------- #


def test_get_brain_health_is_a_singleton_per_base_dir(tmp_path: Path) -> None:
    a = get_brain_health(tmp_path)
    b = get_brain_health(tmp_path)
    assert a is b
    other = get_brain_health(tmp_path / "elsewhere")
    assert other is not a


def test_singleton_shares_state_across_readers(tmp_path: Path) -> None:
    # The same gateway process hosts the being adapter AND the /lifemodel handler,
    # so a write through one handle is visible through another (spec §4.2).
    writer = get_brain_health(tmp_path)
    writer.mark_connecting()
    reader = get_brain_health(tmp_path)
    assert reader.state == "connecting"


# --------------------------------------------------------------------------- #
# check() — the enablement-safe liveness predicate feeding check_fn
# --------------------------------------------------------------------------- #


def test_check_boot_failed_is_unhealthy_with_reason(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_boot_failed("register_being_platform: ImportError: x")
    ok, reason = h.check(last_tick_at=None, now=_NOW, stale_after_seconds=_STALE)
    assert ok is False
    assert "boot_failed" in reason and "ImportError" in reason


def test_check_loop_dead_is_unhealthy_with_reason(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    h.record_loop_death("proactive loop died: RuntimeError('boom')", "tb")
    ok, reason = h.check(last_tick_at=_iso(_NOW), now=_NOW, stale_after_seconds=_STALE)
    assert ok is False
    assert "loop_dead" in reason


def test_check_connected_and_fresh_is_healthy(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    ok, reason = h.check(
        last_tick_at=_iso(_NOW), now=_NOW + timedelta(seconds=30), stale_after_seconds=_STALE
    )
    assert ok is True


def test_check_connected_but_stale_tick_is_unhealthy(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    # Connected long ago, last tick long ago → wedged brain.
    h.mark_connected(at=_iso(_NOW - timedelta(hours=1)))
    ok, reason = h.check(
        last_tick_at=_iso(_NOW - timedelta(hours=1)),
        now=_NOW,
        stale_after_seconds=_STALE,
    )
    assert ok is False
    assert "stale" in reason


def test_check_just_connected_without_a_tick_is_healthy_within_grace(tmp_path: Path) -> None:
    # Right after connect the loop hasn't ticked yet; connected_at anchors the
    # grace so we don't flag a false-stale in the first interval.
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    ok, _ = h.check(last_tick_at=None, now=_NOW + timedelta(seconds=30), stale_after_seconds=_STALE)
    assert ok is True


def test_check_never_started_is_ENABLEMENT_SAFE_healthy(tmp_path: Path) -> None:
    # DELIBERATE DEVIATION FROM THE LITERAL SPEC (loudly noted, spec §4.2 text says
    # "state != connected → False"). check_fn is Hermes' *enablement/instantiation*
    # gate: it is evaluated at gateway config-load and in ``_create_adapter``,
    # BEFORE ``connect()`` ever runs — so the state is necessarily ``never_started``
    # then. Returning False here makes Hermes never enable / never instantiate the
    # platform → ``connect()`` never runs → the brain never starts → the being is
    # PERMANENTLY dead at every cold boot. That is the exact silent-death incident
    # this whole spec exists to kill, self-inflicted. So the pre-connect transient
    # states are enablement-safe (healthy), while the reason still names the state
    # so /lifemodel status can show the truth.
    h = BrainHealth(tmp_path)
    ok, reason = h.check(last_tick_at=None, now=_NOW, stale_after_seconds=_STALE)
    assert ok is True
    assert "never_started" in reason


def test_check_connecting_is_enablement_safe_healthy(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connecting()
    ok, reason = h.check(last_tick_at=None, now=_NOW, stale_after_seconds=_STALE)
    assert ok is True
    assert "connecting" in reason


# --------------------------------------------------------------------------- #
# snapshot() — the thread-safe read of every display field (Slice 3, /status)
# --------------------------------------------------------------------------- #


def test_snapshot_captures_every_display_field(tmp_path: Path) -> None:
    h = BrainHealth(tmp_path)
    h.mark_connected(at=_iso(_NOW))
    h.record_loop_death("proactive loop died: RuntimeError('boom')", "tb")
    h.record_observer_error("post_llm_call", "KeyError: x")
    snap = h.snapshot()
    assert snap.state == "loop_dead"
    assert snap.death_count == 1
    assert snap.last_loop_death is not None and "boom" in snap.last_loop_death
    assert snap.last_observer_error == {"post_llm_call": "KeyError: x"}
    assert snap.connected_at == _iso(_NOW)


def test_snapshot_is_an_immutable_copy(tmp_path: Path) -> None:
    # A snapshot must not change under a concurrent writer — the observer map is
    # copied, not aliased, so a later mutation is invisible to an earlier read.
    h = BrainHealth(tmp_path)
    h.record_observer_error("post_llm_call", "first")
    snap = h.snapshot()
    h.record_observer_error("post_llm_call", "second")
    assert snap.last_observer_error == {"post_llm_call": "first"}


# --------------------------------------------------------------------------- #
# tick_staleness() — the shared staleness helper check() and /status both use
# --------------------------------------------------------------------------- #


def test_tick_staleness_no_anchor_is_not_stale(tmp_path: Path) -> None:
    # Unknown (no connect stamp, no tick) → not stale (age is None).
    age, stale = tick_staleness(None, None, now=_NOW, stale_after_seconds=_STALE)
    assert age is None
    assert stale is False


def test_tick_staleness_fresh_tick_is_not_stale(tmp_path: Path) -> None:
    age, stale = tick_staleness(
        _iso(_NOW - timedelta(seconds=30)),
        _iso(_NOW - timedelta(seconds=30)),
        now=_NOW,
        stale_after_seconds=_STALE,
    )
    assert age == 30.0
    assert stale is False


def test_tick_staleness_old_tick_is_stale(tmp_path: Path) -> None:
    age, stale = tick_staleness(
        _iso(_NOW - timedelta(hours=1)),
        _iso(_NOW - timedelta(hours=1)),
        now=_NOW,
        stale_after_seconds=_STALE,
    )
    assert age == 3600.0
    assert stale is True


def test_tick_staleness_uses_the_freshest_anchor(tmp_path: Path) -> None:
    # A recent tick beats an old connect stamp (and vice-versa) — the freshest wins.
    age, stale = tick_staleness(
        _iso(_NOW - timedelta(hours=1)),  # connected long ago
        _iso(_NOW - timedelta(seconds=10)),  # but ticked just now
        now=_NOW,
        stale_after_seconds=_STALE,
    )
    assert age == 10.0
    assert stale is False


def test_shared_staleness_threshold_constants(tmp_path: Path) -> None:
    # The threshold lives HERE (Hermes-free) so both the adapter's check_fn and
    # /lifemodel status read ONE value — "a few intervals" (spec §4.2).
    assert DEFAULT_TICK_INTERVAL_SECONDS == 60.0
    assert STALE_AFTER_SECONDS == 300.0

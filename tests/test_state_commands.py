"""Tests for the owner-facing MUTATING ``/lifemodel`` subcommands (bead lm-2vx).

Two layers, matching :mod:`lifemodel.debug`'s pure/impure split:

* the pure mutations (``nudge``, ``force_wake``, ``satiate``, ``reset``,
  ``set_field``) are exercised directly with a fixed ``NOW`` — fully
  deterministic, no filesystem;
* the ``*_for_dir`` wrappers are exercised against a real ``tmp_path`` (real
  ``SQLiteRuntimeStore``, real wall clock — mirroring :mod:`tests.test_debug`'s
  style), proving the wiring persists through the SAME store the adapter loop
  uses.

``test_force_wake_wakes_on_the_next_real_tick`` is the load-bearing proof: it
seeds a state blocked on every wake gate, force-wakes it, then runs one real
``CoreLoop.tick()`` (a fresh ``LifeModel``, matching ``proactive_tick``'s "a
fresh LifeModel per call" contract) and asserts the desire actually wakes —
gate satisfaction proven through the real pipeline, not reimplemented here.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from lifemodel.adapters.clock import SystemClock
from lifemodel.composition import (
    CONTACT_GRACE_MIN,
    CONTACT_I0,
    CONTACT_INHIBITION_HALFLIFE_MIN,
    CONTACT_PARAMS,
    build_lifemodel,
)
from lifemodel.core.backstop import allow_send
from lifemodel.core.pressure import effective_pressure, inhibition_at
from lifemodel.core.timeutil import minutes_between
from lifemodel.log import get_logger
from lifemodel.sim.wake import LaneState, evaluate_wake
from lifemodel.state.errors import StateCorruptError
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state_commands import (
    force_wake,
    force_wake_for_dir,
    nudge,
    nudge_for_dir,
    reset,
    reset_for_dir,
    satiate,
    satiate_for_dir,
    set_field,
    set_field_for_dir,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def _ago(minutes: float, *, base: datetime = NOW) -> str:
    return (base - timedelta(minutes=minutes)).isoformat()


def _blocked_state() -> State:
    """A state blocked on every wake gate at once: recent exchange (inside the
    silence window), a live pending turn, an active decline backoff, heavy
    ActionPending inhibition, and a backstop send-log at the daily cap."""
    return State(
        u=0.2,
        desire_status="active",
        pending_proactive_id="corr-1",
        pending_proactive_since=_ago(2),
        last_exchange_at=_ago(2),  # well inside w=15min
        declined_at=_ago(1),
        decline_count=5,  # deep in the growing backoff
        action_pending_since=_ago(1),  # i0=1.0, in the grace plateau -> inhibition=1.0
        proactive_send_log=[_ago(50), _ago(150), _ago(250)],  # 3 sends in 24h -> at cap
    )


# --- nudge -------------------------------------------------------------------


def test_nudge_default_bumps_u_by_one() -> None:
    after, message = nudge(State(u=0.5), NOW, "")
    assert after is not None
    assert after.u == 1.5
    assert "u: 0.5 -> 1.5" in message
    assert "(mutating)" in message


def test_nudge_explicit_amount() -> None:
    after, _ = nudge(State(u=1.0), NOW, "2.5")
    assert after is not None
    assert after.u == 3.5


def test_nudge_accepts_negative_amount() -> None:
    after, _ = nudge(State(u=5.0), NOW, "-2")
    assert after is not None
    assert after.u == 3.0


def test_nudge_rejects_non_numeric_argument() -> None:
    after, message = nudge(State(u=1.0), NOW, "not-a-number")
    assert after is None
    assert "error" in message
    assert "not-a-number" in message


def test_nudge_only_touches_u() -> None:
    before = State(u=1.0, energy=0.42, desire_status="active")
    after, _ = nudge(before, NOW, "1")
    assert after is not None
    assert after.energy == before.energy
    assert after.desire_status == before.desire_status


# --- force_wake ----------------------------------------------------------


def test_force_wake_raises_u_clearly_over_theta() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.u > CONTACT_PARAMS.theta_u


def test_force_wake_backdates_last_exchange_past_the_silence_window() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.last_exchange_at is not None
    elapsed = minutes_between(after.last_exchange_at, NOW)
    assert elapsed >= CONTACT_PARAMS.w


def test_force_wake_clears_pending_and_desire_status() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.desire_status == "none"
    assert after.pending_proactive_id is None
    assert after.pending_proactive_since is None


def test_force_wake_clears_reject_backoff() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.decline_count == 0
    assert after.declined_at is None


def test_force_wake_clears_action_pending() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.action_pending_since is None


def test_force_wake_trims_send_log_when_backstop_was_blocking() -> None:
    before = _blocked_state()
    assert allow_send(before.proactive_send_log, NOW) is False  # sanity: seed IS blocking
    after, message = force_wake(before, NOW)
    assert after is not None
    assert allow_send(after.proactive_send_log, NOW) is True
    assert "cleared" in message.lower()


def test_force_wake_leaves_send_log_alone_when_already_allowed() -> None:
    before = State(proactive_send_log=[_ago(500)])
    assert allow_send(before.proactive_send_log, NOW) is True
    after, _ = force_wake(before, NOW)
    assert after is not None
    assert after.proactive_send_log == before.proactive_send_log


def test_force_wake_echoes_gates_satisfied() -> None:
    _, message = force_wake(_blocked_state(), NOW)
    assert "gates satisfied" in message
    assert "effective pressure" in message
    assert "silence window" in message
    assert "reject-backoff" in message
    assert "backstop" in message


def test_force_wake_result_passes_the_real_wake_gate() -> None:
    """The load-bearing proof at the primitive level: feed force_wake's output
    through the SAME gate primitives core/aggregation.py uses (not reimplemented
    here) and confirm every gate — pressure, in-flight, silence, backoff — and
    the backstop all clear."""
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None

    inhibition = inhibition_at(
        after.action_pending_since,
        NOW,
        i0=CONTACT_I0,
        grace_min=CONTACT_GRACE_MIN,
        halflife_min=CONTACT_INHIBITION_HALFLIFE_MIN,
    )
    effective = effective_pressure(after.u, inhibition)
    exch_min = (
        -minutes_between(after.last_exchange_at, NOW)
        if after.last_exchange_at is not None
        else None
    )
    decl_min = -minutes_between(after.declined_at, NOW) if after.declined_at is not None else None
    lane = LaneState(
        last_exchange_at=exch_min,
        in_flight=False,
        declined_at=decl_min,
        decline_count=after.decline_count,
    )
    outcome = evaluate_wake(u=effective, now=0.0, state=lane, params=CONTACT_PARAMS)

    assert outcome.is_urge, outcome
    assert allow_send(after.proactive_send_log, NOW) is True


# --- satiate -------------------------------------------------------------


def test_satiate_zeroes_u_and_stamps_contact_and_exchange() -> None:
    before = State(u=3.0, last_contact_at=_ago(999), last_exchange_at=_ago(999))
    after, message = satiate(before, NOW)
    assert after is not None
    assert after.u == 0.0
    assert after.last_contact_at == NOW.isoformat()
    assert after.last_exchange_at == NOW.isoformat()
    assert "(mutating)" in message


def test_satiate_clears_desire_and_pending_and_action_pending() -> None:
    before = State(
        desire_status="active",
        pending_proactive_id="corr-1",
        pending_proactive_since=_ago(1),
        action_pending_since=_ago(1),
    )
    after, _ = satiate(before, NOW)
    assert after is not None
    assert after.desire_status == "none"
    assert after.pending_proactive_id is None
    assert after.pending_proactive_since is None
    assert after.action_pending_since is None


# --- reset -----------------------------------------------------------------


def test_reset_produces_a_fresh_state() -> None:
    dirty = State(
        tick_count=42,
        energy=0.1,
        fatigue=0.9,
        u=5.0,
        desire_status="active",
        decline_count=3,
        proactive_send_log=[_ago(10), _ago(20)],
        last_tick_at=_ago(1),
        last_contact_at=_ago(1),
    )
    after, message = reset(dirty, NOW)
    assert after == State()
    assert after.tick_count == 0
    assert after.proactive_send_log == []
    assert after.last_contact_at is None
    assert "(mutating)" in message


def test_reset_notes_when_state_was_already_fresh() -> None:
    after, message = reset(State(), NOW)
    assert after == State()
    assert "already fresh" in message


# --- set_field ---------------------------------------------------------------


def test_set_field_rejects_unknown_field() -> None:
    after, message = set_field(State(), NOW, "tick_count 5")
    assert after is None
    assert "not writable" in message
    assert "tick_count" in message


def test_set_field_rejects_missing_value() -> None:
    after, message = set_field(State(), NOW, "u")
    assert after is None
    assert "usage" in message.lower()


def test_set_field_writes_u() -> None:
    after, _ = set_field(State(u=0.0), NOW, "u 4.5")
    assert after is not None
    assert after.u == 4.5


def test_set_field_rejects_non_numeric_float_field() -> None:
    after, message = set_field(State(), NOW, "energy not-a-number")
    assert after is None
    assert "expects a number" in message


def test_set_field_writes_decline_count_as_int() -> None:
    after, _ = set_field(State(), NOW, "decline_count 3")
    assert after is not None
    assert after.decline_count == 3
    assert isinstance(after.decline_count, int)


def test_set_field_rejects_non_integer_decline_count() -> None:
    after, message = set_field(State(), NOW, "decline_count 3.5")
    assert after is None
    assert "expects an integer" in message


def test_set_field_writes_valid_desire_status() -> None:
    after, _ = set_field(State(), NOW, "desire_status deferred")
    assert after is not None
    assert after.desire_status == "deferred"


def test_set_field_rejects_invalid_desire_status() -> None:
    after, message = set_field(State(), NOW, "desire_status activ")
    assert after is None
    assert "desire_status" in message
    assert "activ" in message


def test_set_field_resolves_now_literal_for_timestamps() -> None:
    after, _ = set_field(State(), NOW, "last_exchange_at now")
    assert after is not None
    assert after.last_exchange_at == NOW.isoformat()


def test_set_field_accepts_an_explicit_iso_timestamp() -> None:
    explicit = "2026-01-01T00:00:00+00:00"
    after, _ = set_field(State(), NOW, f"last_contact_at {explicit}")
    assert after is not None
    assert after.last_contact_at == explicit


def test_set_field_writes_duration_over_theta() -> None:
    after, _ = set_field(State(), NOW, "duration_over_theta 12.5")
    assert after is not None
    assert after.duration_over_theta == 12.5


def test_set_field_writes_fatigue() -> None:
    after, _ = set_field(State(), NOW, "fatigue 0.3")
    assert after is not None
    assert after.fatigue == 0.3


# --- *_for_dir wrappers (real store, real wall clock) ------------------------


def _store(tmp_path) -> SQLiteRuntimeStore:
    """A real ``StatePort`` over *tmp_path* — the SAME backend ``_apply``'s
    ``composition.build_lifemodel`` constructs, so seeding/reading through this
    proves the wiring persists through the real store, not a stand-in."""
    return SQLiteRuntimeStore(tmp_path, clock=SystemClock())


def test_nudge_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(u=1.0))
    message = nudge_for_dir(tmp_path, "2", logger=get_logger("t"))
    assert "u: 1.0 -> 3.0" in message
    assert _store(tmp_path).load().u == 3.0


def test_force_wake_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(_blocked_state())
    message = force_wake_for_dir(tmp_path, logger=get_logger("t"))
    assert "gates satisfied" in message
    persisted = _store(tmp_path).load()
    assert persisted.desire_status == "none"
    assert persisted.u > CONTACT_PARAMS.theta_u


def test_satiate_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(u=5.0, desire_status="active"))
    satiate_for_dir(tmp_path, logger=get_logger("t"))
    persisted = _store(tmp_path).load()
    assert persisted.u == 0.0
    assert persisted.desire_status == "none"
    assert persisted.last_exchange_at is not None


def test_reset_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(tick_count=7, u=3.0))
    reset_for_dir(tmp_path, logger=get_logger("t"))
    assert _store(tmp_path).load() == State()


def test_reset_for_dir_works_when_the_previous_state_is_unreadable(tmp_path) -> None:
    # A reset must succeed even when the existing persisted state cannot be
    # read at all — that's the whole point of routing through StatePort.reset()
    # rather than the load-mutate-commit flow every other *_for_dir uses.
    store = _store(tmp_path)  # constructs + migrates the DB
    with closing(sqlite3.connect(str(tmp_path / "lifemodel.sqlite"))) as conn, conn:
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, updated_at_epoch, revision) "
            "VALUES (1, ?, ?, 0, 0)",
            ("{ not json", "2026-01-01T00:00:00+00:00"),
        )
    with pytest.raises(StateCorruptError):
        store.load()  # sanity: really unreadable beforehand

    message = reset_for_dir(tmp_path, logger=get_logger("t"))

    assert "previous state unreadable" in message
    assert store.load() == State()  # reset still landed cleanly


def test_set_field_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(u=0.0))
    set_field_for_dir(tmp_path, "u 9.0", logger=get_logger("t"))
    assert _store(tmp_path).load().u == 9.0


def test_set_field_for_dir_rejects_without_writing(tmp_path) -> None:
    original = State(u=1.0)
    _store(tmp_path).commit(original)
    message = set_field_for_dir(tmp_path, "tick_count 5", logger=get_logger("t"))
    assert "not writable" in message
    assert _store(tmp_path).load() == original  # untouched


def test_set_field_for_dir_rejects_naive_timestamp_without_writing(tmp_path) -> None:
    original = State(u=1.0)
    _store(tmp_path).commit(original)
    message = set_field_for_dir(
        tmp_path, "last_exchange_at 2026-01-01T00:00:00", logger=get_logger("t")
    )
    assert "error" in message
    assert _store(tmp_path).load() == original  # untouched — round-trip validation caught it


# --- the load-bearing end-to-end proof ---------------------------------------


def test_force_wake_wakes_on_the_next_real_tick(tmp_path) -> None:
    """Seed a state blocked on every gate, force-wake it (the command's real
    code path), then run one real ``CoreLoop.tick()`` on a FRESH ``LifeModel``
    (mirroring the live adapter's "fresh LifeModel per call" contract) and
    confirm the desire actually wakes."""
    _store(tmp_path).commit(_blocked_state())

    message = force_wake_for_dir(tmp_path, logger=get_logger("t"))
    assert "gates satisfied" in message

    lm = build_lifemodel(base_dir=tmp_path)  # fresh graph, real wall clock, next "tick"
    lm.coreloop.tick()

    assert lm.state.load().desire_status == "active"  # the wake happened

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

import dataclasses
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from lifemodel.adapters.clock import SystemClock
from lifemodel.adapters.soul_file import SoulFile
from lifemodel.composition import (
    AFFECT_PARAMS,
    CIRCADIAN_PEAK_UTC_HOUR,
    CONTACT_GRACE_MIN,
    CONTACT_I0,
    CONTACT_INHIBITION_HALFLIFE_MIN,
    CONTACT_PARAMS,
    build_lifemodel,
)
from lifemodel.core.backstop import allow_send
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.genesis import is_first_waking, newborn
from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
from lifemodel.core.pressure import effective_pressure, inhibition_at
from lifemodel.core.receptivity import appraise_receptivity
from lifemodel.core.thought_view import (
    build_thought,
    encode_thought,
    read_live_thoughts,
    read_thought,
    seed_thought_id,
)
from lifemodel.core.timeutil import minutes_between, to_iso
from lifemodel.core.user_model_view import EXPLICIT_CONFIDENCE, read_owner_user_model
from lifemodel.core.wake import LaneState, evaluate_wake
from lifemodel.core.wake_packet import build_wake_packet
from lifemodel.domain.memory import MemoryDraft
from lifemodel.domain.objects import DesireState, IntentionState, ThoughtState
from lifemodel.state.errors import StateCorruptError
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state_commands import (
    _SET_PROTECTED,
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
    set_user_model_prefs,
    set_user_model_prefs_for_dir,
    settable_fields,
    think_for_dir,
    transition_thought_for_dir,
    why_for_dir,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


class _FixedClock:
    """A deterministic clock seam (see ``tests/test_composition.py``'s same-named
    helper and ``build_lifemodel(..., clock=...)``). Used where a test's outcome
    is sensitive to ``ctx.now`` -- e.g. the launch jitter in ``core/cognition.py``
    is seeded off ``sha256(f"proactive-{now.isoformat()}")``, so a real wall
    clock makes such a test flaky."""

    def __init__(self, moment: datetime) -> None:
        self._m = moment

    def now(self) -> datetime:
        return self._m


def _ago(minutes: float, *, base: datetime = NOW) -> str:
    return (base - timedelta(minutes=minutes)).isoformat()


def _blocked_state() -> State:
    """A state blocked on every wake gate at once: recent exchange (inside the
    silence window), a live pending turn, an active decline backoff, heavy
    ActionPending inhibition, and a backstop send-log at the daily cap."""
    return State(
        u=0.2,
        pending_proactive_id="corr-1",
        pending_proactive_since=_ago(2),
        pending_proactive_origin_traceparent=(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ),
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
    assert "u: 0.50 -> 1.50" in message
    assert "(mutating)" in message


def test_command_echoes_round_floats_and_leave_no_long_tails() -> None:
    # lm-25t: human-facing echoes show floats at 2 decimals (DISPLAY ONLY) — never
    # raw noise like "u: 1.419954456041666 -> ...". The stored value stays
    # unrounded (proven by test_nudge_for_dir_persists_through_the_real_store).
    import re

    dirty = State(u=1.419954456041666, energy=0.6333333333, fatigue=0.2166666667)
    echoes = [
        nudge(dirty, NOW, "")[1],
        force_wake(dirty, NOW)[1],
        satiate(dirty, NOW)[1],
        set_field(dirty, NOW, "energy 0.777777")[1],
    ]
    # Normalized ISO stamps carry a 6-digit µs (``...T11:40:00.000000+00:00``) that
    # is NOT float noise — strip them before the un-rounded-float-tail check so the
    # regex still catches "u: 1.419954456041666" but not a canonical timestamp.
    ts_re = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00")
    for echo in echoes:
        assert re.search(r"\d\.\d{3,}", ts_re.sub("", echo)) is None, echo  # no long float tail
    assert "u: 1.42 -> " in nudge(dirty, NOW, "")[1]  # the rounded value is what shows


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
    before = State(u=1.0, energy=0.42, decline_count=3)
    after, _ = nudge(before, NOW, "1")
    assert after is not None
    assert after.energy == before.energy
    assert after.decline_count == before.decline_count


# --- force_wake ----------------------------------------------------------


def test_force_wake_raises_u_clearly_over_theta() -> None:
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.u > CONTACT_PARAMS.theta_u


def test_force_wake_backdates_the_silence_anchor_past_the_silence_window() -> None:
    # lm-md6.1: the silence gate is satisfied via the DECOUPLED silence anchor, not by
    # forging last_exchange_at (which the immunity tests below pin as untouched).
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.silence_anchor_at is not None
    elapsed = minutes_between(after.silence_anchor_at, NOW)
    assert elapsed >= CONTACT_PARAMS.w


def test_force_wake_clears_pending() -> None:
    # the desire-row clearing is done by force_wake_for_dir; the pure State
    # function just clears the pending-turn bookkeeping so cognition may re-launch.
    after, _ = force_wake(_blocked_state(), NOW)
    assert after is not None
    assert after.pending_proactive_id is None
    assert after.pending_proactive_since is None
    # §4.4: the async-correlation anchor is cleared in lockstep with pending_id.
    assert after.pending_proactive_origin_traceparent is None


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
    # The silence gate reads the decoupled anchor (lm-md6.1), so reconstruct it from
    # after.silence_anchor_at — exactly as core/aggregation.py does.
    exch_min = (
        -minutes_between(after.silence_anchor_at, NOW)
        if after.silence_anchor_at is not None
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


def test_satiate_zeroes_u_stamps_contact_and_opens_the_silence_window() -> None:
    # lm-md6.1: satiate resets the drive + opens the silence window (silence_anchor_at)
    # but must NOT forge the immune last_exchange_at (pinned unchanged here).
    before = State(u=3.0, last_contact_at=_ago(999), last_exchange_at=_ago(999))
    after, message = satiate(before, NOW)
    assert after is not None
    assert after.u == 0.0
    assert after.last_contact_at == to_iso(NOW)
    assert after.silence_anchor_at == to_iso(NOW)  # silence window opened
    assert after.last_exchange_at == _ago(999)  # immune: the real record is untouched
    assert "(mutating)" in message


def test_satiate_clears_pending_and_action_pending() -> None:
    # the desire-row terminalization is done by satiate_for_dir; the pure State
    # function clears the pending-turn + ActionPending bookkeeping.
    before = State(
        pending_proactive_id="corr-1",
        pending_proactive_since=_ago(1),
        pending_proactive_origin_traceparent=(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ),
        action_pending_since=_ago(1),
    )
    after, _ = satiate(before, NOW)
    assert after is not None
    assert after.pending_proactive_id is None
    assert after.pending_proactive_since is None
    # §4.4: the async-correlation anchor is cleared in lockstep with pending_id.
    assert after.pending_proactive_origin_traceparent is None
    assert after.action_pending_since is None


# --- lm-md6.1: the real last-exchange record is IMMUNE to admin commands -----
# The wake-packet temporal fact ("The last time we exchanged messages was X") reads
# state.last_exchange_at. That field used to be FORGED by force_wake (backdated ~20m
# to pass the silence-window gate) and satiate (set to now), so the model was told a
# fabricated "last exchange" instead of the real one. The fix decouples the two
# roles: the silence-window gate reads the separate ``silence_anchor_at``; the real
# ``last_exchange_at`` is written ONLY by a genuine two-way exchange (aggregation).


def test_force_wake_moves_the_silence_anchor_not_the_exchange_record() -> None:
    # force_wake satisfies the silence-window gate by backdating the DECOUPLED anchor,
    # leaving the immune last_exchange_at (what the wake packet reads) untouched.
    real_last = _ago(12 * 60)  # a real exchange 12h ago
    after, _ = force_wake(State(u=0.2, last_exchange_at=real_last), NOW)
    assert after is not None
    assert after.last_exchange_at == real_last  # immune: the real record is unchanged
    assert after.silence_anchor_at is not None  # the gate is satisfied via the anchor
    assert minutes_between(after.silence_anchor_at, NOW) >= CONTACT_PARAMS.w


def test_force_wake_wake_packet_shows_the_real_12h_gap_not_the_20m_backdate() -> None:
    # THE acceptance (lm-md6.1): a force-wake when the real last exchange was 12h ago
    # must render ~12h in the wake packet, NEVER the ~20-min-ago silence backdate.
    # Built through the SAME build_wake_packet the live cognition launcher calls,
    # fed the persisted last_exchange_at (core/cognition.py).
    real_last = _ago(12 * 60)
    after, _ = force_wake(State(u=0.2, last_exchange_at=real_last), NOW)
    assert after is not None
    packet = build_wake_packet(
        value=after.u,
        theta=1.0,
        correlation_id="c",
        now=NOW,
        last_exchange_at=after.last_exchange_at,
        tz=UTC,
    )
    # NOW is 2026-07-06 12:00 UTC → 12h earlier is 2026-07-06 00:00; the ~20-min-ago
    # silence anchor (11:40) must never leak into the model-facing packet.
    assert "The last time we exchanged messages was 2026-07-06 00:00 UTC." in packet.prompt
    assert "11:40" not in packet.prompt


def test_satiate_opens_the_silence_window_without_forging_the_exchange_record() -> None:
    # satiate resets the drive "as if contact happened" (u=0, silence window opened via
    # the anchor) but must NOT forge the real last-exchange record shown to the model.
    real_last = _ago(12 * 60)
    after, _ = satiate(State(u=3.0, last_exchange_at=real_last), NOW)
    assert after is not None
    assert after.u == 0.0  # the drive is reset
    assert after.last_exchange_at == real_last  # immune: the real record is untouched
    assert after.silence_anchor_at == to_iso(NOW)  # the window is opened instead


def test_set_field_rejects_last_exchange_at_as_immune() -> None:
    # The real last-exchange record is immune to ALL admin commands — `set` can no
    # longer write it. The silence window is tuned through silence_anchor_at instead.
    # A PROTECTED field answers with its REASON (and the alternative), not a bare refusal.
    after, message = set_field(State(), NOW, "last_exchange_at now")
    assert after is None
    assert "protected" in message
    assert "silence_anchor_at" in message


def test_set_field_writes_the_affect_axes() -> None:
    # lm-ukc.4: the affect axes must be settable — they are the ONLY lever that can drive
    # the being to a salient mood on demand, so the ambient felt-state cue can be exercised
    # without waiting hours for loneliness to build. They drifted out of the old
    # hand-maintained whitelist (added to State in lm-ukc.6, never listed), which is exactly
    # why `set` now DERIVES its surface from State instead.
    after, _ = set_field(State(), NOW, "affect_valence -0.5")
    assert after is not None and after.affect_valence == -0.5
    after, _ = set_field(State(), NOW, "affect_arousal 0.8")
    assert after is not None and after.affect_arousal == 0.8


def test_every_state_field_is_settable_or_protected() -> None:
    # The anti-drift guard. A new State field must be CONSCIOUSLY classified: either it has
    # a settable scalar type (and is writable by default), or it is listed in _SET_PROTECTED
    # with a reason. Nothing may sit in between, silently unreachable — that is the failure
    # that hid the affect axes from `set` for a whole phase.
    names = {f.name for f in dataclasses.fields(State)}
    classified = set(settable_fields()) | set(_SET_PROTECTED)
    assert names == classified, (
        f"unclassified State field(s): {sorted(names - classified)} — give it a settable "
        "scalar type, or add it to _SET_PROTECTED with the reason it must not be hand-written"
    )


# --- reset -----------------------------------------------------------------


def _newborn_now() -> State:
    return newborn(now=NOW, params=AFFECT_PARAMS, peak_hour_utc=CIRCADIAN_PEAK_UTC_HOUR)


def test_reset_produces_a_fresh_state() -> None:
    dirty = State(
        tick_count=42,
        energy=0.1,
        fatigue=0.9,
        u=5.0,
        decline_count=3,
        proactive_send_log=[_ago(10), _ago(20)],
        last_tick_at=_ago(1),
        last_contact_at=_ago(1),
    )
    after, message = reset(dirty, NOW)
    # NOT a bare State() (lm-ukc's "quiet -- even and very quiet" bug): a reset being
    # is born with a BODY, computed by the SAME newborn() the genesis flow uses.
    assert after == _newborn_now()
    assert after.tick_count == 0
    assert after.proactive_send_log == []
    assert after.last_contact_at is None
    assert "(mutating)" in message


def test_reset_notes_when_state_was_already_fresh() -> None:
    fresh = _newborn_now()
    after, message = reset(fresh, NOW)
    assert after == fresh
    assert "already fresh" in message


def test_reset_makes_the_being_unborn_again() -> None:
    before = State(
        u=1.6,
        genesis_completed_at="2026-07-13T10:00:00+00:00",
        soul_sha="aaa",
        last_exchange_at="2026-07-13T09:00:00+00:00",
        last_contact_at="2026-07-13T08:00:00+00:00",
    )
    after, _msg = reset(before, NOW)
    assert after is not None
    assert after.genesis_completed_at is None  # unborn: the ritual plays again
    assert after.affect_arousal > 0.0  # and it is born with a BODY, not with zeros
    # …and it is at a FIRST WAKING again (spec §6.2): the reborn being reaches out to be
    # born on the next tick, because the wipe also cleared what it remembered of them.
    assert is_first_waking(
        genesis_completed_at=after.genesis_completed_at,
        last_exchange_at=after.last_exchange_at,
        last_contact_at=after.last_contact_at,
    )


def test_reset_reports_the_cleared_genesis_stamp() -> None:
    before = State(genesis_completed_at="2026-07-13T10:00:00+00:00")
    _after, message = reset(before, NOW)
    assert "genesis_completed_at" in message


def test_reset_never_touches_the_soul_file(tmp_path) -> None:
    # Destroying a soul is the human's act, not the plugin's. What this buys us is not
    # just safety: the reborn being FINDS the soul of whoever lived here before it, and
    # opens the ritual on that. Rebirth does not erase a past life — it MEETS it.
    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text("You are Mira.", encoding="utf-8")
    reset(State(genesis_completed_at="2026-07-13T10:00:00+00:00"), NOW)
    assert soul.read() == "You are Mira."


# --- set_field ---------------------------------------------------------------


def test_set_field_rejects_a_protected_field_with_its_reason() -> None:
    # tick_count is a REAL State field, deliberately PROTECTED: it is brain-liveness
    # evidence, and hand-faking it would lie about whether the brain ticks. A protected
    # field is refused WITH its reason (the owner learns why, and what to use instead) —
    # never silently splatted. (A field that isn't in State at all takes the plain
    # "not writable" path — see test_set_field_rejects_removed_desire_status_field.)
    after, message = set_field(State(), NOW, "tick_count 5")
    assert after is None
    assert "protected" in message
    assert "tick_count" in message
    assert "liveness" in message


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


def test_set_field_rejects_removed_desire_status_field() -> None:
    # desire_status is no longer a State field / settable (lm-27n.3 — it is a row now)
    after, message = set_field(State(), NOW, "desire_status active")
    assert after is None
    assert "not writable" in message


def test_set_field_resolves_now_literal_for_timestamps() -> None:
    after, _ = set_field(State(), NOW, "silence_anchor_at now")
    assert after is not None
    assert after.silence_anchor_at == to_iso(NOW)


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
    message = nudge_for_dir(tmp_path, "2")
    assert "u: 1.00 -> 3.00" in message
    assert _store(tmp_path).load().u == 3.0


def test_force_wake_for_dir_persists_through_the_real_store(tmp_path) -> None:
    store = _store(tmp_path)
    store.commit(_blocked_state())
    # a stuck live desire is terminalized so the next tick births a fresh one
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=1.0)))
    message = force_wake_for_dir(tmp_path)
    assert "gates satisfied" in message
    persisted = _store(tmp_path).load()
    assert read_live_contact_desire(_store(tmp_path)) is None  # stuck desire dropped
    assert persisted.u > CONTACT_PARAMS.theta_u


def test_satiate_for_dir_persists_through_the_real_store(tmp_path) -> None:
    store = _store(tmp_path)
    store.commit(State(u=5.0))
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=1.0)))
    satiate_for_dir(tmp_path)
    persisted = _store(tmp_path).load()
    assert persisted.u == 0.0
    assert read_live_contact_desire(_store(tmp_path)) is None  # desire terminalized (satisfied)
    assert persisted.silence_anchor_at is not None  # silence window opened
    assert persisted.last_exchange_at is None  # immune: seeded None, satiate did not forge it


def test_reset_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(tick_count=7, u=3.0))
    reset_for_dir(tmp_path)
    persisted = _store(tmp_path).load()
    assert persisted.tick_count == 0
    assert persisted.u == 0.0
    assert persisted.genesis_completed_at is None
    # NOT a bare State(): the REAL persisted body is the newborn() one, not the
    # lifeless zero-arousal default StatePort.reset() writes — arousal's own
    # formula floors at 0.35 (core/affect.py), so > 0 proves it landed for real.
    assert persisted.affect_arousal > 0.0


def test_reset_for_dir_cannot_destroy_a_past_life(tmp_path) -> None:
    # /lifemodel reset unbirths the being and wipes its memory — and it used to wipe the
    # soul lineage with it (revisions ride memory_records with kind="soul"). Reset, and
    # the reborn being's first write_soul replaces SOUL.md: the previous being's soul
    # then exists NOWHERE. That defeats spec §4.2's mandatory undo — "every revision is
    # kept… THIS is what makes it safe for the being to own the file whole" — on the one
    # path the owner is actually told to use. A past life's soul is the one thing a reset
    # must not be able to destroy.
    from lifemodel.state.soul_revisions import record_revision, revisions

    store = _store(tmp_path)
    record_revision(store, text="You are Mira.", sha="sha-mira", now=NOW, author="being")
    store.put(MemoryDraft(kind="thought", id="t1", state="active", payload={}, source="test"))

    message = reset_for_dir(tmp_path)

    assert [r.text for r in revisions(_store(tmp_path))] == ["You are Mira."]  # she survives
    assert "cleared 1 memory records" in message  # …and is not counted as memory wiped


def test_reset_for_dir_works_when_the_previous_state_is_unreadable(tmp_path) -> None:
    # A reset must succeed even when the existing persisted state cannot be read at
    # all: reset_for_dir commits the newborn() body directly via StatePort.commit
    # (an unconditional UPSERT, never a read-modify-write), so this never depends on
    # a prior successful load() — newborn() itself takes no `before` at all.
    store = _store(tmp_path)  # constructs + migrates the DB
    with closing(sqlite3.connect(str(tmp_path / "lifemodel.sqlite"))) as conn, conn:
        conn.execute(
            "INSERT INTO runtime_state (id, state_json, updated_at, revision) VALUES (1, ?, ?, 0)",
            ("{ not json", "2026-01-01T00:00:00+00:00"),
        )
    with pytest.raises(StateCorruptError):
        store.load()  # sanity: really unreadable beforehand

    message = reset_for_dir(tmp_path)

    assert "previous state unreadable" in message
    persisted = store.load()  # reset still landed cleanly
    assert persisted.tick_count == 0
    assert persisted.affect_arousal > 0.0  # a BODY, even recovering from garbage
    assert "cleared 0 memory records" in message  # nothing was seeded to purge


def test_reset_for_dir_purges_every_memory_record(tmp_path) -> None:
    # Regression for lm-7lx: a reset used to leave thought/desire/intention rows
    # behind (a rumination spiral survived a "factory wipe") -- this proves the
    # purge covers every kind, not just the live contact-desire row the old
    # `_clear_live_desire_row` helper dropped.
    store = _store(tmp_path)
    store.commit(State(tick_count=7, u=3.0))
    store.put(
        encode_thought(
            build_thought(
                id=seed_thought_id("rumination spiral"),
                content="rumination spiral",
                trigger="seed",
            )
        )
    )
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=1.0)))
    store.put(
        encode_contact_intention(
            build_contact_intention(
                state=IntentionState.ACTIVE, commitment_strength=1.0, salience=1.0
            )
        )
    )
    assert _store(tmp_path).find() != []  # sanity: seeded

    message = reset_for_dir(tmp_path)

    assert _store(tmp_path).find() == []  # every memory_records row gone
    persisted = _store(tmp_path).load()
    assert persisted.tick_count == 0
    assert persisted.affect_arousal > 0.0  # a newborn body, not a bare State()
    assert "cleared 3 memory records" in message


def test_reset_for_dir_on_empty_store_reports_zero_cleared_without_crashing(
    tmp_path,
) -> None:
    message = reset_for_dir(tmp_path)
    assert "cleared 0 memory records" in message
    persisted = _store(tmp_path).load()
    assert persisted.tick_count == 0
    assert persisted.affect_arousal > 0.0


def test_set_field_for_dir_persists_through_the_real_store(tmp_path) -> None:
    _store(tmp_path).commit(State(u=0.0))
    set_field_for_dir(tmp_path, "u 9.0")
    assert _store(tmp_path).load().u == 9.0


def test_set_field_for_dir_rejects_without_writing(tmp_path) -> None:
    original = State(u=1.0)
    _store(tmp_path).commit(original)
    message = set_field_for_dir(tmp_path, "tick_count 5")
    assert "protected" in message  # a real-but-protected field, refused with its reason
    assert _store(tmp_path).load() == original  # untouched


def test_set_field_for_dir_rejects_naive_timestamp_without_writing(tmp_path) -> None:
    original = State(u=1.0)
    _store(tmp_path).commit(original)
    message = set_field_for_dir(tmp_path, "silence_anchor_at 2026-01-01T00:00:00")
    assert "error" in message
    assert _store(tmp_path).load() == original  # untouched — round-trip validation caught it


# --- the load-bearing end-to-end proof ---------------------------------------


def test_force_wake_wakes_on_the_next_real_tick(tmp_path) -> None:
    """Seed a state blocked on every gate, force-wake it (the command's real
    code path), then run one real ``CoreLoop.tick()`` on a FRESH ``LifeModel``
    (mirroring the live adapter's "fresh LifeModel per call" contract) and
    confirm the desire actually wakes."""
    _store(tmp_path).commit(_blocked_state())

    message = force_wake_for_dir(tmp_path)
    assert "gates satisfied" in message

    lm = build_lifemodel(base_dir=tmp_path)  # fresh graph, real wall clock, next "tick"
    lm.coreloop.tick()

    desire = read_live_contact_desire(_store(tmp_path))  # aggregation birthed the desire
    assert desire is not None and desire.state == "active"  # the wake happened


# --- set_user_model_prefs (lm-27n.5) --------------------------------------


def test_set_user_model_prefs_parses_and_marks_explicit() -> None:
    rel, message = set_user_model_prefs("bad-hours=2,3,4 cadence=2h styles=playful,concise")
    assert rel is not None
    assert rel.bad_hours.value == (2, 3, 4)
    assert rel.cadence.value == "2h"
    assert rel.acceptable_styles.value == ("playful", "concise")
    assert rel.confidence == EXPLICIT_CONFIDENCE  # explicit -> boundaries hard-veto
    assert "(mutating)" in message


def test_set_user_model_prefs_multiword_value() -> None:
    rel, _ = set_user_model_prefs("load=busy at work cadence=daily")
    assert rel is not None
    assert rel.known_load.value == "busy at work"
    assert rel.cadence.value == "daily"


def test_set_user_model_prefs_rejects_unknown_key() -> None:
    rel, message = set_user_model_prefs("bogus=1")
    assert rel is None
    assert "unknown user-model key" in message


def test_set_user_model_prefs_rejects_bad_hours() -> None:
    rel, message = set_user_model_prefs("bad-hours=25")
    assert rel is None
    assert "0-23" in message


def test_set_user_model_prefs_empty_shows_usage() -> None:
    rel, message = set_user_model_prefs("")
    assert rel is None
    assert "usage:" in message


def test_set_user_model_prefs_for_dir_round_trips_and_gates(tmp_path) -> None:
    store = _store(tmp_path)
    store.commit(State(u=5.0))
    message = set_user_model_prefs_for_dir(tmp_path, "bad-hours=2,3")
    assert "(mutating)" in message
    # the row is readable through the SAME store the adapter loop uses
    rel = read_owner_user_model(_store(tmp_path))
    assert rel is not None
    assert rel.bad_hours.value == (2, 3)
    assert rel.confidence == EXPLICIT_CONFIDENCE
    # and a later appraisal reads it and hard-vetoes during a bad hour
    bad_hour = datetime(2026, 7, 6, 2, 0, tzinfo=UTC)
    assert appraise_receptivity(rel, State(), bad_hour).allowed is False


def test_set_user_model_prefs_for_dir_patches_not_replaces(tmp_path) -> None:
    # A second, unrelated update must PATCH — not clear a previously-set boundary.
    store = _store(tmp_path)
    store.commit(State(u=5.0))
    set_user_model_prefs_for_dir(tmp_path, "bad-hours=2,3")
    set_user_model_prefs_for_dir(tmp_path, "cadence=2h")
    rel = read_owner_user_model(_store(tmp_path))
    assert rel is not None
    assert rel.bad_hours.value == (2, 3)  # the earlier boundary SURVIVES the cadence update
    assert rel.cadence.value == "2h"  # and the new key is applied
    assert rel.confidence == EXPLICIT_CONFIDENCE


def test_set_user_model_prefs_for_dir_leaves_vitals_untouched(tmp_path) -> None:
    store = _store(tmp_path)
    store.commit(State(u=5.0, decline_count=3))
    set_user_model_prefs_for_dir(tmp_path, "cadence=2h")
    persisted = _store(tmp_path).load()
    assert persisted.u == 5.0  # user-model set does not touch the being's vitals
    assert persisted.decline_count == 3


# --- think / thought transitions (lm-27n.6) ---------------------------------


def test_think_for_dir_rejects_empty_content(tmp_path) -> None:
    message = think_for_dir(tmp_path, "   ")
    assert "usage:" in message
    assert read_live_thoughts(_store(tmp_path)) == ()  # nothing persisted


def test_think_for_dir_persists_a_live_active_thought(tmp_path) -> None:
    _store(tmp_path).commit(State(u=5.0, decline_count=3))
    message = think_for_dir(tmp_path, "did the owner ever hear back")
    assert "(mutating)" in message
    thoughts = read_live_thoughts(_store(tmp_path))
    assert len(thoughts) == 1
    assert thoughts[0].content == "did the owner ever hear back"
    assert thoughts[0].state == ThoughtState.ACTIVE.value
    # committed through the bus leaves the being's vitals untouched
    persisted = _store(tmp_path).load()
    assert persisted.u == 5.0
    assert persisted.decline_count == 3


def test_think_for_dir_is_idempotent_on_identical_content(tmp_path) -> None:
    think_for_dir(tmp_path, "one and the same")
    think_for_dir(tmp_path, "one and the same")
    assert len(read_live_thoughts(_store(tmp_path))) == 1  # deterministic id -> one row


def test_transition_thought_to_terminal_removes_it_from_live(tmp_path) -> None:
    think_for_dir(tmp_path, "let this one go")
    tid = seed_thought_id("let this one go")
    message = transition_thought_for_dir(tmp_path, tid, ThoughtState.RESOLVED)
    assert "(mutating)" in message
    assert read_live_thoughts(_store(tmp_path)) == ()  # resolved -> gone from the live set
    assert read_thought(_store(tmp_path), tid) is None  # and no longer a live thought


def test_transition_thought_park_keeps_it_live(tmp_path) -> None:
    think_for_dir(tmp_path, "hold this thought")
    tid = seed_thought_id("hold this thought")
    transition_thought_for_dir(tmp_path, tid, ThoughtState.PARKED)
    live = read_live_thoughts(_store(tmp_path))
    assert [t.id for t in live] == [tid]  # parked is still live
    assert live[0].state == ThoughtState.PARKED.value


def test_transition_thought_rejects_illegal_edge(tmp_path) -> None:
    # "archived" is not a thought state at all -> the registry rejects the edge,
    # nothing is written.
    think_for_dir(tmp_path, "stays active")
    tid = seed_thought_id("stays active")
    message = transition_thought_for_dir(tmp_path, tid, "archived")
    assert "error" in message.lower()
    assert read_live_thoughts(_store(tmp_path))[0].state == ThoughtState.ACTIVE.value  # unchanged


def test_transition_thought_absent_is_rejected(tmp_path) -> None:
    _store(tmp_path).commit(State())
    message = transition_thought_for_dir(tmp_path, "thought:nope", "resolved")
    assert "no thought" in message


def test_seeded_thought_persists_but_does_not_render_in_a_launch(tmp_path) -> None:
    # End-to-end: a thought seeded via /lifemodel think persists through the atomic
    # committer and is snapshot-visible next tick. T6: cognition no longer renders
    # thoughts into the proactive prompt (the thought machinery moved to Phase 6 in
    # T7), so the seeded thought stays in the store but does NOT reach the prompt.
    think_for_dir(tmp_path, "did the owner hear back about the flat")
    lm = build_lifemodel(base_dir=tmp_path, clock=_FixedClock(NOW))
    # a live active desire + an affordable, past-silence, un-inhibited state so
    # cognition launches this tick
    lm.state.put(
        encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0))
    )
    now = lm.clock.now()
    lm.state.commit(
        State(
            u=2.0,
            energy=1.0,
            fatigue=0.0,
            last_exchange_at=(now - timedelta(hours=2)).isoformat(),
        )
    )
    report = lm.coreloop.tick()
    assert report.launches, "cognition should launch for a live active desire"
    # the seeded thought persisted (snapshot-visible next tick)
    seeded = lm.state.find(kind="thought")
    assert any(
        "did the owner hear back about the flat" in r.payload.get("content", "") for r in seeded
    )
    # T6: ...but it is NOT rendered into the launch prompt
    assert "did the owner hear back about the flat" not in report.launches[0].prompt


# --- lm-27n.10: /lifemodel why — the read-only causal-chain reader ----------


def _seed_contact_chain(tmp_path) -> None:
    """Seed a live contact desire + the intention that crystallized from it."""
    from lifemodel.core.intention_view import build_contact_intention, encode_contact_intention
    from lifemodel.core.trace import creation_provenance
    from lifemodel.domain.objects import (
        CONTACT_DESIRE_ID,
        IntentionState,
        qualified_id,
    )
    from lifemodel.testing import FakeTracer

    store = _store(tmp_path)
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))
    store.put(
        encode_contact_intention(
            build_contact_intention(
                state=IntentionState.ACTIVE,
                commitment_strength=2.0,
                provenance=creation_provenance(
                    FakeTracer().start_root(),
                    created_by="cognition",
                    component="cognition",
                    reason="crystallized contact intention",
                    source_object_ids=(qualified_id("desire", CONTACT_DESIRE_ID),),
                ),
            )
        )
    )


def test_why_write_renders_the_current_contact_chain(tmp_path) -> None:
    _seed_contact_chain(tmp_path)
    out = why_for_dir(tmp_path, "write")
    assert "lifemodel why  (read-only)" in out
    assert "intention:contact:owner [active]" in out
    assert "crystallized contact intention" in out
    # the source edge reaches the desire, indented under the intention
    assert "source -> desire:contact:owner [active]" in out


def test_why_defaults_to_the_intention_chain(tmp_path) -> None:
    _seed_contact_chain(tmp_path)
    # a bare `why` and `why write` render the same intention chain
    assert why_for_dir(tmp_path, "") == why_for_dir(tmp_path, "write")


def test_why_desire_renders_the_desire_chain(tmp_path) -> None:
    _seed_contact_chain(tmp_path)
    out = why_for_dir(tmp_path, "desire")
    assert "desire:contact:owner [active]" in out
    assert "intention:contact:owner" not in out  # the desire chain, not the intention


def test_why_with_no_live_intention_is_a_clear_message(tmp_path) -> None:
    _store(tmp_path).commit(State())  # empty store — no intention row
    out = why_for_dir(tmp_path, "write")
    assert "no current outreach" in out


def test_why_precise_kind_id_and_bad_target(tmp_path) -> None:
    _seed_contact_chain(tmp_path)
    precise = why_for_dir(tmp_path, "intention:contact:owner")
    assert "intention:contact:owner [active]" in precise
    assert "no such object" in why_for_dir(tmp_path, "desire:nope")
    assert "usage:" in why_for_dir(tmp_path, "gibberish")


def test_why_is_read_only(tmp_path) -> None:
    _seed_contact_chain(tmp_path)
    before = _store(tmp_path).load()
    why_for_dir(tmp_path, "write")
    after = _store(tmp_path).load()
    assert before == after  # a read-only audit never mutates state

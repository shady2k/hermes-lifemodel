"""Acceptance scenarios for the ephemeral-core rethink (spec §12, SLICE 1).

Each test maps to one numbered regression scenario in the design doc's "Критерий
готовности" (§12). All are deterministic and 0-LLM: they drive the REAL spine
(``ContactSensor → SolitudeDrive → ContactAggregation → CognitionLauncher``) over
the real SQLite store through fake ports, exercising ExecutionFrames via
``run_frame`` / ``proactive_tick`` and the afferent hooks.

Scenario (6) (external-event idempotency ring) is owned by SLICE 3 (lm-fib.8.5).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from lifemodel.composition import build_lifemodel
from lifemodel.core.contact_sensor import CONTACT_SENSOR_ID
from lifemodel.core.desire_view import (
    build_contact_desire,
    encode_contact_desire,
    read_live_contact_desire,
)
from lifemodel.core.frame import FrameTrigger, run_frame
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.proactive import proactive_tick
from lifemodel.core.taxonomy import contact_observed_signal, proactive_outcome_signal
from lifemodel.core.timeutil import to_iso
from lifemodel.domain.egress import ProactiveOutcome, ReachOutcome
from lifemodel.domain.objects import DesireState
from lifemodel.hooks import make_inbound_observer
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State
from lifemodel.testing import FakeClock
from lifemodel.testing.harness import RecordingEgress

_NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
_TARGET: dict[str, str | None] = {"platform": "test", "chat_id": "1", "thread_id": None}


def _build(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _seed_active_desire(lm, salience: float = 3.0) -> None:
    lm.state.put(
        encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=salience))
    )


# --- (1) a real inbound contact satiates u + last_exchange_at + resolves desire ---


def test_scenario_1_inbound_contact_satiates_and_resolves_desire(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    lm.state.commit(State(u=2.0, last_tick_at=to_iso(_NOW)))
    _seed_active_desire(lm)

    run_frame(
        lm.coreloop,
        [contact_observed_signal(origin_id="m-1", actor="user", label="two_way", timestamp=None)],
        trigger=FrameTrigger.EVENT,
    )

    final = lm.state.load()
    assert final.u < 2.0  # the genuine two_way contact satiated the drive
    assert final.last_exchange_at is not None  # exchange record stamped
    assert read_live_contact_desire(lm.state) is None  # the live desire resolved...
    assert lm.state.get("desire", "contact:owner").state == "satisfied"  # ...to SATISFIED


# --- (1b) a real inbound while a proactive turn is IN FLIGHT clears pending ---
# Regression (contact-dominates-proactive, spec §7.3): when a genuine contact
# dominates a same-frame/in-flight proactive attempt it must clear the pending-
# proactive anchor in lockstep with terminalizing the desire. Otherwise a stranded
# pending_proactive_id deadlocks proactive cognition forever — the ASYNC_COMPLETION
# frame that would clear it never runs (the desire is already gone), and the
# launcher HOLDs every future launch while pending is set.


def test_inbound_while_proactive_in_flight_clears_pending_and_unblocks_cognition(
    tmp_path: Path,
) -> None:
    lm = _build(tmp_path)
    # A proactive turn is IN FLIGHT (pending set) with a live ACTIVE desire and a
    # high drive, when a real reply lands the very same moment.
    lm.state.commit(
        State(
            u=3.0,
            pending_proactive_id="p-inflight",
            pending_proactive_since="2026-07-06T11:55:00+00:00",
            pending_proactive_origin_traceparent=_ORIGIN_TP,
            last_tick_at=to_iso(_NOW),
        )
    )
    _seed_active_desire(lm)

    run_frame(
        lm.coreloop,
        [contact_observed_signal(origin_id="m-1", actor="user", label="two_way", timestamp=None)],
        trigger=FrameTrigger.EVENT,
    )

    after = lm.state.load()
    # (a) the pending-proactive attempt is cleared in lockstep — no stranded anchor.
    assert after.pending_proactive_id is None
    assert after.pending_proactive_since is None
    assert after.pending_proactive_origin_traceparent is None
    # (b) the live desire terminalized to SATISFIED (the reply resolved the pull)...
    assert lm.state.get("desire", "contact:owner").state == "satisfied"
    # (c) ...and the exchange record was stamped this frame.
    assert after.last_exchange_at == to_iso(_NOW)

    # A subsequent over-threshold tick is NOT gated: pending is clear, so cognition
    # can launch again. (On the buggy code pending stays "p-inflight" and the
    # launcher HOLDs forever — proactive cognition deadlocked.)
    lm.state.commit(replace(after, u=3.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00"))
    _seed_active_desire(lm)  # a fresh live desire to reach for
    egress = RecordingEgress(ReachOutcome.DELIVERED)

    outcome = proactive_tick(lm, egress, _TARGET)

    assert outcome is ReachOutcome.DELIVERED  # cognition launched again — not gated
    assert len(egress.calls) == 1
    assert lm.state.load().pending_proactive_id is not None  # a NEW turn is in flight


# --- (2) a /... control command is NOT contact (sensor band-pass) ------------


def test_scenario_2_control_command_is_not_contact(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    lm.state.commit(State(u=2.0, last_tick_at=to_iso(_NOW)))

    make_inbound_observer(lambda: lm, health=BrainHealth(tmp_path), metrics=MetricRegistry())(
        event=SimpleNamespace(text="/lifemodel force-wake", internal=False, id="m-2")
    )

    final = lm.state.load()
    assert final.u == 2.0  # unchanged — the command never became contact
    assert final.last_exchange_at is None  # no exchange recorded


# --- (3) u>=theta + receptivity → aggregation LAUNCHes cognition -------------


def test_scenario_3_over_threshold_launches_cognition(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    # high drive, energy to afford a launch, 1 min elapsed, a live active desire
    lm.state.commit(State(u=3.0, energy=1.0, last_tick_at="2026-07-06T11:59:00+00:00"))
    _seed_active_desire(lm)
    egress = RecordingEgress(ReachOutcome.DELIVERED)

    outcome = proactive_tick(lm, egress, _TARGET)  # a heartbeat frame

    assert outcome is ReachOutcome.DELIVERED  # the launch was delivered
    assert len(egress.calls) == 1  # cognition launched → reached the egress
    assert lm.state.load().pending_proactive_id is not None  # a turn is now in flight


# --- (4) cognition in-flight → a repeat frame does NOT double-launch ---------


def test_scenario_4_in_flight_does_not_double_launch(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    # a turn already in flight (pending set) with the drive still high
    lm.state.commit(
        State(
            u=3.0,
            energy=1.0,
            pending_proactive_id="p-inflight",
            pending_proactive_since="2026-07-06T11:59:00+00:00",
            pending_proactive_origin_traceparent=_ORIGIN_TP,
            last_tick_at="2026-07-06T11:59:00+00:00",
        )
    )
    _seed_active_desire(lm)
    egress = RecordingEgress(ReachOutcome.DELIVERED)

    outcome = proactive_tick(lm, egress, _TARGET)  # a repeat heartbeat frame

    assert outcome is None  # no delivery — cognition is gated off while in flight
    assert egress.calls == []  # no second launch reached the egress
    assert lm.state.load().pending_proactive_id == "p-inflight"  # still the SAME turn


# --- (5) proactive_outcome sent → action_pending/backoff; silent → decline ---


def test_scenario_5a_sent_sets_action_pending_and_clears_pending(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    lm.state.commit(
        State(
            u=1.5,
            pending_proactive_id="p-sent",
            pending_proactive_since="2026-07-06T11:55:00+00:00",
            pending_proactive_origin_traceparent=_ORIGIN_TP,
            last_tick_at=to_iso(_NOW),
        )
    )
    _seed_active_desire(lm)

    run_frame(
        lm.coreloop,
        [
            proactive_outcome_signal(
                origin_id="o1",
                outcome=ProactiveOutcome.SENT,
                timestamp=None,
                correlation_id="p-sent",
            )
        ],
        trigger=FrameTrigger.ASYNC_COMPLETION,
    )

    final = lm.state.load()
    assert lm.state.get("desire", "contact:owner").state == "satisfied"
    assert final.action_pending_since is not None  # send → ActionPending inhibition window
    assert final.proactive_send_log  # the global backstop counter recorded the send
    assert final.pending_proactive_id is None  # pending cleaned


def test_scenario_5b_silent_applies_decline_backoff_and_clears_pending(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    lm.state.commit(
        State(
            u=1.5,
            pending_proactive_id="p-silent",
            pending_proactive_since="2026-07-06T11:55:00+00:00",
            pending_proactive_origin_traceparent=_ORIGIN_TP,
            last_tick_at=to_iso(_NOW),
        )
    )
    _seed_active_desire(lm)

    run_frame(
        lm.coreloop,
        [
            proactive_outcome_signal(
                origin_id="o1",
                outcome=ProactiveOutcome.SILENT,
                timestamp=None,
                correlation_id="p-silent",
            )
        ],
        trigger=FrameTrigger.ASYNC_COMPLETION,
    )

    final = lm.state.load()
    assert lm.state.get("desire", "contact:owner").state == "dropped"
    assert final.decline_count >= 1  # decline backoff applied
    assert final.declined_at is not None
    assert final.action_pending_since is None  # silence is not a send → no inhibition window
    assert final.pending_proactive_id is None  # pending cleaned


# --- (6) a duplicate external event (same origin_id) is deduped by the ring ---


def test_scenario_6_duplicate_origin_id_satiates_u_only_once(tmp_path: Path) -> None:
    clock = FakeClock(_NOW)
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.state.commit(State(u=5.0, last_tick_at=to_iso(_NOW)))
    _seed_active_desire(lm)

    dup = contact_observed_signal(origin_id="m-dup", actor="user", label="two_way", timestamp=None)
    run_frame(lm.coreloop, [dup], trigger=FrameTrigger.EVENT)
    after_first = lm.state.load()
    assert after_first.u < 5.0  # the genuine two_way contact satiated the drive once
    assert after_first.last_exchange_at == to_iso(_NOW)  # exchange record stamped
    assert lm.state.get("desire", "contact:owner").state == "satisfied"  # desire resolved

    # A retry of the SAME Hermes event id (clock pinned → dt=0, so any SECOND
    # satiation would show as a further drop in u). The ring must drop it: u,
    # last_exchange_at, and the resolved desire all stay put.
    run_frame(lm.coreloop, [dup], trigger=FrameTrigger.EVENT)
    after_dup = lm.state.load()
    assert after_dup.u == after_first.u  # satiated EXACTLY once (the retry was deduped)
    assert after_dup.last_exchange_at == to_iso(_NOW)  # last_exchange_at stamped once
    assert lm.state.get("desire", "contact:owner").state == "satisfied"  # resolved once
    assert "m-dup" in after_dup.processed_external_event_ids  # remembered in the ring


def test_scenario_6_different_origin_id_after_first_still_satiates(tmp_path: Path) -> None:
    clock = FakeClock(_NOW)
    lm = build_lifemodel(base_dir=tmp_path, clock=clock)
    lm.state.commit(State(u=5.0, last_tick_at=to_iso(_NOW)))

    run_frame(
        lm.coreloop,
        [contact_observed_signal(origin_id="m-1", actor="user", label="two_way", timestamp=None)],
        trigger=FrameTrigger.EVENT,
    )
    assert lm.state.load().last_exchange_at == to_iso(_NOW)

    # A DIFFERENT inbound 30 min later: the ring dedups by id, not blanket-suppress,
    # so this genuine new contact satiates normally and re-stamps last_exchange_at.
    later = _NOW + timedelta(minutes=30)
    clock.set(later)
    run_frame(
        lm.coreloop,
        [contact_observed_signal(origin_id="m-2", actor="user", label="two_way", timestamp=None)],
        trigger=FrameTrigger.EVENT,
    )
    final = lm.state.load()
    assert final.last_exchange_at == to_iso(later)  # the new id satiated normally
    assert set(final.processed_external_event_ids) == {"m-1", "m-2"}  # both remembered


def test_scenario_6_contact_sensor_failure_does_not_record_id(tmp_path: Path, monkeypatch) -> None:
    # Regression (spec §8): the idempotency ring must not record a fresh external id
    # until the load-bearing contact consumer (ContactSensor → the u-satiation path)
    # has actually processed it. If ContactSensor throws (per-component isolation),
    # u is never satiated — so recording the id would durably lose the inbound: every
    # retry would be deduped away against an id whose effect never landed.
    from lifemodel.core.contact_sensor import ContactSensor

    lm = _build(tmp_path)
    lm.state.commit(State(u=5.0, last_tick_at=to_iso(_NOW)))

    def _boom(self, ctx):  # ContactSensor faults this frame
        raise RuntimeError("sensor down")

    monkeypatch.setattr(ContactSensor, "step", _boom)
    dup = contact_observed_signal(origin_id="m-boom", actor="user", label="two_way", timestamp=None)
    report = run_frame(lm.coreloop, [dup], trigger=FrameTrigger.EVENT)

    assert "contact" in report.failed  # the sensor faulted this frame
    after_fail = lm.state.load()
    assert after_fail.u == 5.0  # never satiated — the sensor didn't run, so no drop
    # The id must NOT be durably recorded — otherwise the retry is deduped forever.
    assert "m-boom" not in after_fail.processed_external_event_ids

    # The SAME id, retried with a healthy sensor, is re-processed (not deduped) and
    # satiates u exactly once — proof the first frame left the ring clean.
    monkeypatch.undo()
    run_frame(lm.coreloop, [dup], trigger=FrameTrigger.EVENT)
    after_retry = lm.state.load()
    assert after_retry.u < 5.0  # the retry re-fired and satiated the drive
    assert "m-boom" in after_retry.processed_external_event_ids  # now durably recorded


def test_scenario_6_circuit_broken_contact_sensor_does_not_record_id(tmp_path: Path) -> None:
    # Regression (spec §8): a CIRCUIT-BROKEN ContactSensor is SKIPPED (coreloop:
    # ``if component.id in self._broken: continue``), so it lands in NEITHER ``ran``
    # NOR ``failed``. The inbound is never processed (u unsatiated) — the same durable
    # loss as the exception path, but reached via the broken/skip path. So the record
    # gate must key on POSITIVE success (``in ran``), not absence-of-failure: the id
    # must NOT be recorded, so a retry after the sensor recovers re-fires.
    lm = _build(tmp_path)
    lm.state.commit(State(u=5.0, last_tick_at=to_iso(_NOW)))
    lm.coreloop._broken.add(CONTACT_SENSOR_ID)  # simulate the tripped breaker

    dup = contact_observed_signal(
        origin_id="m-broken", actor="user", label="two_way", timestamp=None
    )
    report = run_frame(lm.coreloop, [dup], trigger=FrameTrigger.EVENT)

    assert CONTACT_SENSOR_ID in report.skipped_broken  # skipped, not run
    assert CONTACT_SENSOR_ID not in report.ran  # ...and NOT in ran
    after = lm.state.load()
    assert after.u == 5.0  # never satiated — the sensor didn't run
    # The id must NOT be durably recorded — otherwise the retry (after recovery) is
    # deduped away against an id whose effect never landed.
    assert "m-broken" not in after.processed_external_event_ids


def test_scenario_6_ring_is_durable_across_restart(tmp_path: Path) -> None:
    # Unlike the ephemeral bus, the ring is durable (spec §8): a duplicate that
    # arrives AFTER a restart is still deduped.
    lm1 = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    lm1.state.commit(State(u=5.0, last_tick_at=to_iso(_NOW)))
    dup = contact_observed_signal(
        origin_id="m-restart", actor="user", label="two_way", timestamp=None
    )
    run_frame(lm1.coreloop, [dup], trigger=FrameTrigger.EVENT)
    after_first = lm1.state.load()
    assert after_first.u < 5.0  # satiated once
    assert "m-restart" in after_first.processed_external_event_ids  # persisted to sqlite

    # "Restart": a brand-new graph over the SAME durable store, clock pinned to _NOW
    # so a second satiation (dt=0) would show as a further drop in u.
    lm2 = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    run_frame(lm2.coreloop, [dup], trigger=FrameTrigger.EVENT)
    after_restart = lm2.state.load()
    assert after_restart.u == after_first.u  # the durable ring deduped the post-restart retry
    assert after_restart.last_exchange_at == to_iso(_NOW)  # not re-stamped


# --- (7) an async-completion frame commits the outcome IMMEDIATELY -----------


def test_scenario_7_async_completion_commits_immediately_no_heartbeat(tmp_path: Path) -> None:
    lm = _build(tmp_path)
    lm.state.commit(
        State(
            u=1.5,
            pending_proactive_id="p-async",
            pending_proactive_since="2026-07-06T11:55:00+00:00",
            pending_proactive_origin_traceparent=_ORIGIN_TP,
            last_tick_at=to_iso(_NOW),
        )
    )
    _seed_active_desire(lm)

    # The async cognition turn finishes → its OWN frame. NO heartbeat/proactive_tick is
    # run afterwards; the outcome must already be committed (spec §3 — a completion frame
    # commits in the moment, not deferred to the next heartbeat).
    report = run_frame(
        lm.coreloop,
        [
            proactive_outcome_signal(
                origin_id="o1",
                outcome=ProactiveOutcome.SENT,
                timestamp=None,
                correlation_id="p-async",
            )
        ],
        trigger=FrameTrigger.ASYNC_COMPLETION,
    )

    assert report.trigger is FrameTrigger.ASYNC_COMPLETION
    assert report.committed  # the frame committed its own outcome
    committed = lm.state.load()  # read straight back — no intervening heartbeat
    assert committed.pending_proactive_id is None  # resolved in the completion frame itself
    assert committed.action_pending_since is not None


# --- (8) restart → the in-memory bus is empty, durable state intact ----------


def test_scenario_8_restart_has_empty_bus_and_intact_durable_state(tmp_path: Path) -> None:
    # Drive one event frame (mutating durable state), then "restart" by building a
    # fresh graph over the SAME base_dir. There is no durable signal log to replay
    # ("lost consciousness → don't replay stale impulses"); AgentState + Memory persist.
    lm1 = _build(tmp_path)
    lm1.state.commit(State(u=2.0, decline_count=2, last_tick_at=to_iso(_NOW)))
    _seed_active_desire(lm1)
    run_frame(
        lm1.coreloop,
        [contact_observed_signal(origin_id="m-1", actor="user", label="two_way", timestamp=None)],
        trigger=FrameTrigger.EVENT,
    )
    before = lm1.state.load()
    tick_before = before.tick_count

    # No durable-bus artifacts were ever written — the flow is ephemeral.
    assert not (tmp_path / "signals.log").exists()
    assert not (tmp_path / "signals.consumed").exists()

    # "Restart": a brand-new graph (fresh coreloop, fresh state-actor, fresh empty
    # in-memory SignalFrame) over the SAME durable store.
    lm2 = build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))
    reloaded = lm2.state.load()

    # Durable state (AgentState) survived the restart, byte-for-byte on the fields
    # the frame committed.
    assert reloaded.u == before.u
    assert reloaded.last_exchange_at == before.last_exchange_at
    assert reloaded.decline_count == before.decline_count
    assert reloaded.tick_count == tick_before
    # Memory (the resolved desire row) survived too.
    assert lm2.state.get("desire", "contact:owner").state == "satisfied"

    # Behaviour continues: a fresh heartbeat frame runs and bumps the durable tick
    # count — nothing stale was replayed from a bus (there is none).
    egress = RecordingEgress(ReachOutcome.DELIVERED)
    proactive_tick(lm2, egress, _TARGET)
    assert lm2.state.load().tick_count == tick_before + 1

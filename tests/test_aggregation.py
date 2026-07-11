# tests/test_aggregation.py
#
# The contact-desire behaviour contract, migrated to the typed desire row
# (lm-27n.3). The lifecycle is no longer a ``State.desire_status`` flag: it lives
# in the ``kind='desire'`` singleton ``contact:owner``, read from the start-of-tick
# ``ctx.objects`` snapshot and mutated by ONE PutRecord (birth) / TransitionRecord
# (advance) the layer emits. Every assertion below preserves the ORIGINAL behaviour
# the flag pinned; only the representation changed (flag -> row):
#   * old ``desire_status == "active"`` on an urge  -> a PutRecord births active;
#   * old ``desire_status == "none"`` (suppressed)  -> NO desire intent;
#   * old ``desire_status`` unchanged on a dedup     -> NO desire intent;
#   * old ``-> none`` on FULFILL/REJECT/exchange     -> a TransitionRecord to a
#     terminal state (satisfied/dropped/satisfied);
#   * old ``-> deferred`` on DEFER                    -> TransitionRecord to deferred.
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.component import TickContext
from lifemodel.core.intents import Intent, PutRecord, TransitionRecord, UpdateState
from lifemodel.core.taxonomy import (
    contact_observed_signal,
    contact_pressure_signal,
    in_flight_signal,
    proactive_outcome_signal,
)
from lifemodel.domain.egress import ProactiveOutcome
from lifemodel.domain.objects import DesireSpring
from lifemodel.ports.tracer import TraceContext
from lifemodel.sim.wake import GateParams
from lifemodel.state.model import State
from lifemodel.testing import (
    FakeActiveSpan,
    FakeSpanLogger,
    FakeTracer,
    contact_desire_objects,
    contact_desire_record,
    contact_intention_record,
)

PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)

# ctx.trace is non-optional (spec §4.1) — a literal span's ids for the fixtures that
# do not assert on the born object's trace.
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)

# a live active-desire snapshot (what the old ``desire_status="active"`` meant)
ACTIVE = contact_desire_objects("active")
# a held deferred-desire snapshot (what the old ``desire_status="deferred"`` meant —
# reachable via a backstop-blocked proactive launch)
DEFERRED = contact_desire_objects("deferred")


def _agg() -> ContactAggregation:
    return ContactAggregation(params=PARAMS, theta=1.0, beta=1.0, u_max=100.0)


def _ctx(state: State, now: datetime, signals=(), *, objects=(), tmp_path) -> TickContext:
    return TickContext(
        state=state,
        now=now,
        signals=tuple(signals),
        objects=tuple(objects),
        trace=_TRACE,
    )


def _changes(intents) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def _desire_intent(intents) -> Intent | None:
    return next((i for i in intents if isinstance(i, PutRecord | TransitionRecord)), None)


def _created_active(intents) -> bool:
    """A fresh desire was born active this tick (old flag none -> active)."""
    di = _desire_intent(intents)
    return isinstance(di, PutRecord) and di.op.draft.state == "active"


def _transition(intents) -> tuple[str, str] | None:
    """The (from_state, to_state) of the tick's lone desire transition, if any."""
    di = _desire_intent(intents)
    return (di.op.from_state, di.op.to_state) if isinstance(di, TransitionRecord) else None


CORR = "proactive-2026-07-06T03:55:00+00:00"


def _live_pending_state(**over) -> State:
    """State with a proactive turn in flight, matching CORR (pair with ACTIVE)."""
    base = dict(
        u=1.5,
        pending_proactive_id=CORR,
        pending_proactive_since="2026-07-06T03:55:00+00:00",
        pending_proactive_origin_traceparent=(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ),
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    base.update(over)
    return State(**base)


def test_urge_over_threshold_creates_active_desire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _created_active(intents)  # no live desire -> births one active


def test_below_threshold_stays_none(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=0.5, delta=0.0, timestamp=None)  # < theta
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # no wake -> no desire


def test_second_urge_is_deduped_no_refire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], objects=ACTIVE, tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # a desire is already live -> dedup, no new row


def test_deferred_desire_is_held_not_recreated(tmp_path) -> None:
    # A DEFERRED desire (old ``desire_status="deferred"``, e.g. after a backstop
    # block) is still LIVE: a fresh high-pressure urge must be deduped and the
    # desire held — NOT a new active row created over the held one. This regresses
    # the snapshot-visibility bug where deferred rows were filtered out.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], objects=DEFERRED, tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # held: no new PutRecord, no transition


def test_deferred_desire_cleared_by_exchange(tmp_path) -> None:
    # A real exchange terminalizes a held deferred desire, exactly like an active
    # one (old ``on_exchange`` cleared active OR deferred to none).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], objects=DEFERRED, tmp_path=tmp_path))
    assert _transition(intents) == ("deferred", "satisfied")


def test_silence_window_suppresses_wake(tmp_path) -> None:
    # exchange 5 min ago (< w=15) -> SILENCE_WINDOW, no wake even with high u
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        last_exchange_at="2026-07-06T03:55:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None


def test_silence_anchor_overrides_last_exchange_for_the_gate(tmp_path) -> None:
    # lm-md6.1 (force_wake's mechanism): a REAL exchange 2 min ago would normally
    # suppress (inside w=15), but the decoupled silence_anchor_at backdated 20 min past
    # the window opens the gate — WITHOUT touching the immune last_exchange_at, which
    # the wake packet reads. Proven through the real aggregation gate.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        last_exchange_at="2026-07-06T03:58:00+00:00",  # 2 min ago — would suppress
        silence_anchor_at="2026-07-06T03:40:00+00:00",  # 20 min ago — clears the window
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _created_active(intents)  # the anchor opened the gate → a desire is born
    # the immune exchange record is passed through untouched (never re-anchored here)
    assert _changes(intents)["last_exchange_at"] == "2026-07-06T03:58:00+00:00"


def test_silence_anchor_now_suppresses_wake_despite_an_old_exchange(tmp_path) -> None:
    # lm-md6.1 (satiate's mechanism): silence_anchor_at=now opens the window and
    # suppresses a wake even though the real last exchange is 20 min old (past w) — and
    # the old exchange record is left intact for the model to read.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        last_exchange_at="2026-07-06T03:40:00+00:00",  # 20 min ago — would wake
        silence_anchor_at="2026-07-06T04:00:00+00:00",  # now — inside the window
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # anchor holds the wake
    assert _changes(intents)["last_exchange_at"] == "2026-07-06T03:40:00+00:00"  # untouched


def test_exchange_clears_the_silence_anchor(tmp_path) -> None:
    # A genuine exchange re-anchors the silence gate on itself: it stamps
    # last_exchange_at=now AND clears any admin silence override, so the window measures
    # from the real exchange again (lm-md6.1).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        silence_anchor_at="2026-07-06T03:40:00+00:00",  # a stale admin override
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path)))
    assert changes["last_exchange_at"] == now.isoformat()
    assert changes["silence_anchor_at"] is None  # override cleared by the real exchange


def test_in_flight_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    busy = in_flight_signal(origin_id="f1", value=True, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, busy], tmp_path=tmp_path))
    assert _desire_intent(intents) is None


def test_decline_backoff_suppresses_then_allows(tmp_path) -> None:
    # declined 10 min ago, decline_count=1 -> backoff r0=30 min active -> no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        decline_count=1,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # inside backoff


def test_duration_over_theta_accumulates(tmp_path) -> None:
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5 min
    state = State(u=2.0, duration_over_theta=10.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)  # >= theta
    changes = _changes(_agg().step(_ctx(state, now, [c], objects=ACTIVE, tmp_path=tmp_path)))
    assert abs(changes["duration_over_theta"] - 15.0) < 1e-9


def test_aggregation_does_not_write_u_on_normal_tick(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert "u" not in changes  # neuron owns u; aggregation only writes it on FULFILL (Task 4)


def test_exchange_clears_desire_and_resets_clocks(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        decline_count=2,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "satisfied")  # exchange terminalizes the desire
    assert changes["decline_count"] == 0
    assert changes["declined_at"] is None
    assert changes["last_exchange_at"] == now.isoformat()


def test_exchange_this_tick_suppresses_wake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # fresh exchange -> SILENCE_WINDOW, no wake


def test_internal_impulse_is_not_an_exchange(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    own = contact_observed_signal(
        origin_id="e1", actor="proactive_internal", label="two_way", timestamp=None
    )
    intents = _agg().step(_ctx(state, now, [c, own], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert changes["last_exchange_at"] is None  # own nudge did not reset the clock
    assert _desire_intent(intents) is None  # desire not cleared by own nudge (dedup)


def test_fulfill_starts_action_pending_and_clears_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(duration_over_theta=99.0)
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "satisfied")
    assert changes["action_pending_since"] == now.isoformat()  # send -> ActionPending
    assert "u" not in changes  # not satiated (send != contact)
    assert changes["last_contact_at"] == now.isoformat()
    assert changes["pending_proactive_id"] is None  # turn resolved
    # §4.4: the async anchor is cleared in lockstep with pending_id at resolution.
    assert changes["pending_proactive_origin_traceparent"] is None


def test_readback_send_does_not_satiate_u(tmp_path) -> None:
    # spec §4.5 read-back invariant: a delivered proactive message (FULFILL verdict)
    # starts the ActionPending inhibition + clears pending + logs the send, but does
    # NOT reduce u — send ≠ contact. Only a genuine inbound exchange satiates u
    # (SolitudeDrive). u is never in the FULFILL changes.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert "u" not in changes  # the send left the drive deficit untouched
    assert _transition(intents) == ("active", "satisfied")
    assert changes["pending_proactive_since"] is None


def test_reject_records_backoff_and_clears_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(decline_count=1)
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "dropped")
    assert changes["decline_count"] == 2
    assert changes["declined_at"] == now.isoformat()
    assert changes["pending_proactive_id"] is None
    # §4.4: the async anchor is cleared in lockstep with pending_id at resolution.
    assert changes["pending_proactive_origin_traceparent"] is None


def test_stale_outcome_drops_desire_and_clears_pending_without_backoff(tmp_path) -> None:
    # STALE/FAILED: the attempt ended with nothing to reinforce (spec §5/§6) — the
    # desire is dropped and pending cleared, but no decline backoff and no
    # ActionPending window (no send happened).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.STALE, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "dropped")
    assert changes["pending_proactive_id"] is None  # turn resolved
    assert changes["action_pending_since"] is None  # no send → no inhibition window
    assert changes["decline_count"] == 0  # STALE is not a decline
    assert "u" not in changes  # not satiated


def test_exchange_clears_action_pending(tmp_path) -> None:
    # a real reply resolves the pull: clears ActionPending (neuron satiates u separately)
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=1.0,
        action_pending_since="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=1.0, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], objects=ACTIVE, tmp_path=tmp_path))
    assert _changes(intents)["action_pending_since"] is None  # contact resolved the pull
    assert _transition(intents) == ("active", "satisfied")


def test_reject_does_not_set_action_pending(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path)))
    assert changes["action_pending_since"] is None  # REJECT never inhibits
    assert changes["decline_count"] == 1  # existing backoff bookkeeping intact


def test_reject_then_backoff_blocks_immediate_rewake(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(u=5.0)
    c = contact_pressure_signal(origin_id="c1", value=5.0, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    assert _transition(intents) == ("active", "dropped")  # rejected...
    assert not _created_active(intents)  # ...and backoff vetoes a same-tick re-wake


def test_stale_verdict_wrong_correlation_is_dropped(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1",
        outcome=ProactiveOutcome.SENT,
        timestamp=None,
        correlation_id="proactive-OTHER",
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # verdict dropped -> desire untouched (dedup)
    assert _changes(intents)["action_pending_since"] is None


def test_exchange_dominates_same_tick_verdict(tmp_path) -> None:
    # a real reply this tick clears the desire; the (now-stale) fulfill is ignored
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, ex, v], objects=ACTIVE, tmp_path=tmp_path))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "satisfied")  # exchange terminalized it
    assert changes["action_pending_since"] is None  # fulfill was dropped (desire resolved)
    assert changes["last_exchange_at"] == now.isoformat()


# --- Phase C1: effective pressure gates ---


def test_action_pending_grace_suppresses_wake_despite_high_latent(tmp_path) -> None:
    # latent u=3 (>= theta) but a send 10 min ago (within 45-min grace) -> effective ~0 -> no wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        action_pending_since="2026-07-06T03:50:00+00:00",  # 10 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # inhibited during grace


def test_pressure_recovers_after_grace_and_decay(tmp_path) -> None:
    # send ~3h ago: grace(45m)+ ~2 half-lives -> inhibition ~0.06 -> effective ~ u -> wake
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        action_pending_since="2026-07-06T01:00:00+00:00",  # 180 min ago
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _created_active(intents)  # ignored long enough -> loneliness returns


def test_duration_over_theta_uses_latent_not_effective(tmp_path) -> None:
    # even fully inhibited (effective 0), latent u>=theta so duration keeps accruing
    now = datetime(2026, 7, 6, 0, 5, tzinfo=UTC)  # dt=5
    state = State(
        u=2.0,
        duration_over_theta=10.0,
        action_pending_since="2026-07-06T00:04:00+00:00",  # in grace -> inhibition 1
        last_tick_at="2026-07-06T00:00:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    changes = _changes(intents)
    assert (
        abs(changes["duration_over_theta"] - 15.0) < 1e-9
    )  # latent-based, accrues under inhibition
    assert _desire_intent(intents) is None  # but no wake (effective suppressed)


def test_fulfill_records_a_send(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(proactive_send_log=["2026-07-06T02:00:00+00:00"])
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path)))
    log = changes["proactive_send_log"]
    assert log[-1] == now.isoformat()  # this send recorded
    assert len(log) == 2  # appended to the prior one


def test_negative_dt_does_not_shrink_duration(tmp_path) -> None:
    state = State(u=2.0, duration_over_theta=30.0, last_tick_at="2026-07-06T12:10:00+00:00")
    now = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)  # before last_tick
    c = contact_pressure_signal(origin_id="c1", value=2.0, delta=0.0, timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c], objects=ACTIVE, tmp_path=tmp_path)))
    assert changes["duration_over_theta"] == 30.0  # unchanged (dt clamped to 0), not reduced


def test_reject_does_not_record_a_send(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(proactive_send_log=["2026-07-06T02:00:00+00:00"])
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path)))
    assert changes["proactive_send_log"] == ["2026-07-06T02:00:00+00:00"]  # unchanged


# --- lm-8o3.1 Task 7: unanswered pure-longing outreach counter --------------
#
# ``State.unanswered_outbound_count`` tracks consecutive FULFILLED pure-longing
# (drive-sprung, no thought backing) proactive sends with no genuine reply in
# between. A top-down (thought/mixed-sprung) send is a materially new reason,
# not a repeat longing bid, so it must NOT bump the counter. Any genuine inbound
# exchange resets the counter — the same site that resets ``decline_count``.


def test_fulfill_pure_longing_increments_unanswered_outbound_count(tmp_path) -> None:
    # ACTIVE is spring=DRIVE by default (contact_desire_record's default) — a
    # pure-longing, bottom-up desire with no crystallized-thought backing.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(unanswered_outbound_count=2)
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=ACTIVE, tmp_path=tmp_path)))
    assert changes["unanswered_outbound_count"] == 3  # bumped by this longing FULFILL


def test_fulfill_top_down_send_does_not_increment_unanswered_outbound_count(tmp_path) -> None:
    # A thought-crystallized (top-down) send is a materially new reason, not a
    # repeat longing bid -> the counter must hold, not bump.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(unanswered_outbound_count=2)
    objects = contact_desire_objects("active", spring=DesireSpring.THOUGHT)
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=objects, tmp_path=tmp_path)))
    assert changes["unanswered_outbound_count"] == 2  # unchanged — top-down, not longing


def test_fulfill_mixed_spring_send_does_not_increment_unanswered_outbound_count(tmp_path) -> None:
    # MIXED still carries a source thought (a genuine reason co-fired with the
    # drive) -> also NOT a pure-longing repeat.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state(unanswered_outbound_count=2)
    objects = contact_desire_objects(
        "active", spring=DesireSpring.MIXED, source_thought_ids=("t-serve",)
    )
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    changes = _changes(_agg().step(_ctx(state, now, [c, v], objects=objects, tmp_path=tmp_path)))
    assert changes["unanswered_outbound_count"] == 2  # unchanged — thought-backed, not longing


def test_exchange_resets_unanswered_outbound_count(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        unanswered_outbound_count=4,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    changes = _changes(_agg().step(_ctx(state, now, [c, ex], objects=ACTIVE, tmp_path=tmp_path)))
    assert changes["unanswered_outbound_count"] == 0  # a genuine reply resets the longing bid


# --- lm-8o3.1 Task 8: the unanswered-outbound HOLD gate (T3, simplified) -----
#
# After one FULFILLED pure-longing send with no reply since
# (``unanswered_outbound_count >= 1``), a SECOND drive-only bid must HOLD — no new
# desire born — until a genuine exchange resets the counter. T3 kept the
# anti-repeat concern and shed the baroque top-down-override machinery
# (aggregation is drive-only now).


def test_repeat_pure_longing_bid_holds_when_unanswered_outbound_pending(tmp_path) -> None:
    # unanswered_outbound_count=1 (one unreplied longing send already out),
    # drive-only urge -> HOLD: no desire created.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, unanswered_outbound_count=1, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # held — no second pure-longing bid


def test_pure_longing_bid_unheld_when_no_outbound_is_unanswered(tmp_path) -> None:
    # Baseline, unchanged: unanswered_outbound_count == 0 -> drive-urge alone
    # still creates the desire.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, unanswered_outbound_count=0, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    intents = _agg().step(_ctx(state, now, [c], tmp_path=tmp_path))
    assert _created_active(intents)


# --- lm-27n.4: atomic Desire<->Intention lifecycle interlock ---
#
# On resolution, aggregation transitions BOTH the desire AND the live intention
# (the decision record) in the SAME tick commit — never one without the other
# (split-brain guard). Both TransitionRecords ride out in one intent batch.


def _kind_transition(intents, kind: str) -> tuple[str, str] | None:
    """The (from_state, to_state) of the tick's transition for *kind*, if any."""
    for i in intents:
        if isinstance(i, TransitionRecord) and i.op.kind == kind:
            return (i.op.from_state, i.op.to_state)
    return None


def _live_pair(desire_state: str = "active", intention_state: str = "active"):
    """A snapshot holding BOTH a live desire and a live intention (the state a
    launched, un-resolved contact frame is in)."""
    return (
        contact_desire_record(desire_state),
        contact_intention_record(intention_state),
    )


def test_fulfill_completes_intention_and_satisfies_desire_atomically(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=_live_pair(), tmp_path=tmp_path))
    # both transitions in ONE batch — the pair resolves together
    assert _kind_transition(intents, "desire") == ("active", "satisfied")
    assert _kind_transition(intents, "intention") == ("active", "completed")


def test_reject_drops_both_intention_and_desire_atomically(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, v], objects=_live_pair(), tmp_path=tmp_path))
    assert _kind_transition(intents, "desire") == ("active", "dropped")
    assert _kind_transition(intents, "intention") == ("active", "dropped")


def test_exchange_completes_intention_and_satisfies_desire(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], objects=_live_pair(), tmp_path=tmp_path))
    assert _kind_transition(intents, "desire") == ("active", "satisfied")
    assert _kind_transition(intents, "intention") == ("active", "completed")


def test_exchange_dominates_verdict_for_both_desire_and_intention(tmp_path) -> None:
    # A real reply this tick clears the pair; the same-tick fulfill is ignored —
    # exchange dominates for the intention exactly as it does for the desire.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = _live_pending_state()
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    v = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(state, now, [c, ex, v], objects=_live_pair(), tmp_path=tmp_path))
    assert _kind_transition(intents, "desire") == ("active", "satisfied")
    assert _kind_transition(intents, "intention") == ("active", "completed")


def test_exchange_completes_a_deferred_intention(tmp_path) -> None:
    # After a backstop block the pair is held deferred; a real exchange terminalizes
    # BOTH — the intention transitions from its actual ``deferred`` state.
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(
        _ctx(state, now, [c, ex], objects=_live_pair("deferred", "deferred"), tmp_path=tmp_path)
    )
    assert _kind_transition(intents, "desire") == ("deferred", "satisfied")
    assert _kind_transition(intents, "intention") == ("deferred", "completed")


def test_desire_resolves_without_a_never_crystallized_intention(tmp_path) -> None:
    # A desire can resolve before it ever crystallized (exchange terminalizes a
    # never-launched desire). No intention row exists -> only the desire transitions;
    # this is NOT split-brain (there is nothing to split).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(state, now, [c, ex], objects=ACTIVE, tmp_path=tmp_path))
    assert _kind_transition(intents, "desire") == ("active", "satisfied")
    assert _kind_transition(intents, "intention") is None  # nothing to transition


# --- T3: drive-only aggregation (receptivity + top-down spring cut) ---------
#
# Aggregation is drive-only now: the receptivity-appraisal gate and the top-down
# thought-proposal spring are cut (appropriateness is the async act-gate's job;
# thoughts return in a later phase). ``spring`` is always DRIVE. The pure-longing
# anti-repeat CONCERN is kept (see the HOLD-gate tests above), shed of its machinery.


def _created_desire(intents):
    di = _desire_intent(intents)
    return di.op.draft if isinstance(di, PutRecord) else None


def test_drive_only_urge_springs_a_drive_desire(tmp_path) -> None:
    # A drive urge -> spring=DRIVE, no source thoughts (the only spring now).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    draft = _created_desire(_agg().step(_ctx(state, now, [c], tmp_path=tmp_path)))
    assert draft is not None
    assert draft.payload["spring"] == "drive"
    assert draft.payload["source_thought_ids"] == []


# --- T3: a quiet tick emits a suppression span naming the silent gate (§5) ----
#
# On a non-wake, aggregation emits a suppression span whose ``reason`` names the
# gate that held fire (silence is a logged decision, not the absence of a record).
# A creation or resolution is NOT silent; an URGE held by the anti-repeat gate is
# ``repeat_pure_longing``.


def _run_with_logger(state, now, signals, *, tmp_path):  # type: ignore[no-untyped-def]
    # The live tick hands aggregation a span-bound logger over its child span; the
    # FakeSpanLogger records the emitted events (with the span's ids stamped) so we
    # can read back the suppression reason (spec §4.1/§5).
    trace = FakeTracer().start_root()
    span = FakeActiveSpan(trace, component="aggregation", tick=state.tick_count + 1)
    logger = FakeSpanLogger(span)
    _agg().step(
        TickContext(
            state=state,
            now=now,
            signals=tuple(signals),
            objects=(),
            trace=trace,
            logger=logger,
        )
    )
    return logger.events


def _supp_reason(events):  # type: ignore[no-untyped-def]
    for record in events:
        if record["event"] == "suppression":
            return record["reason"]
    return None


def test_below_threshold_emits_below_threshold_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=0.5, delta=0.0, timestamp=None)  # < theta
    calls = _run_with_logger(state, now, [c], tmp_path=tmp_path)
    assert _supp_reason(calls) == "below_threshold"


def test_silence_window_emits_silence_window_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        last_exchange_at="2026-07-06T03:55:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    assert _supp_reason(_run_with_logger(state, now, [c], tmp_path=tmp_path)) == "silence_window"


def test_decline_backoff_emits_decline_backoff_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(
        u=3.0,
        decline_count=1,
        declined_at="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    assert _supp_reason(_run_with_logger(state, now, [c], tmp_path=tmp_path)) == "decline_backoff"


def test_in_flight_emits_in_flight_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=3.0, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=3.0, delta=0.0, timestamp=None)
    busy = in_flight_signal(origin_id="f1", value=True, timestamp=None)
    assert _supp_reason(_run_with_logger(state, now, [c, busy], tmp_path=tmp_path)) == "in_flight"


def test_repeat_pure_longing_hold_emits_repeat_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, unanswered_outbound_count=1, last_tick_at="2026-07-06T03:59:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    assert (
        _supp_reason(_run_with_logger(state, now, [c], tmp_path=tmp_path)) == "repeat_pure_longing"
    )


def test_urged_creation_emits_no_suppression(tmp_path) -> None:
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)  # >= theta
    assert _supp_reason(_run_with_logger(state, now, [c], tmp_path=tmp_path)) is None


# --- lm-27n.11: a born desire carries the tick's execution trace in its provenance ---


def _decode_desire_provenance(draft):
    from lifemodel.domain.memory import MemoryRecord
    from lifemodel.domain.objects import default_registry

    record = MemoryRecord(
        kind=draft.kind,
        id=draft.id,
        state=draft.state,
        payload=draft.payload,
        source=draft.source,
        recipient_id=draft.recipient_id,
        salience=draft.salience,
        confidence=draft.confidence,
        expires_at=draft.expires_at,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
        revision=0,
        schema_version=draft.schema_version,
    )
    return default_registry().decode(record).provenance


def _traced_ctx(state, now, signals, *, tmp_path, trace):
    return TickContext(
        state=state,
        now=now,
        signals=tuple(signals),
        objects=(),
        trace=trace,
    )


def test_born_desire_carries_the_tick_trace(tmp_path) -> None:
    from lifemodel.testing import FakeTracer

    trace = FakeTracer().start_root()
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    c = contact_pressure_signal(origin_id="c1", value=1.5, delta=0.0, timestamp=None)
    draft = _created_desire(
        _agg().step(_traced_ctx(state, now, [c], tmp_path=tmp_path, trace=trace))
    )
    assert draft is not None
    prov = _decode_desire_provenance(draft)
    assert prov is not None
    assert prov.trace_id == trace.trace_id  # logs and durable provenance JOIN on this
    assert prov.creation_span_id == trace.span_id
    assert prov.component == "aggregation"


# --- lm-fib.8.2: priority-class backpressure in the AGGREGATION gate (spec §7) ---
#
# The gate classifies each frame signal into must_process vs best_effort and
# coalesces the best_effort class to a bounded count BEFORE reducing. must_process
# (contact_observed / proactive_outcome / in_flight / the drive's contact_pressure)
# is never shed; best_effort sensor noise is folded so it can't each drive a step.


def _noise(i: int):
    from lifemodel.domain.signal import Signal

    return Signal(origin_id=f"n{i}", kind="sensor_noise")


def test_contact_observed_survives_a_best_effort_flood(tmp_path) -> None:
    # 1 real contact_observed + 200 best_effort sensor-noise signals: the
    # contact_observed is ALWAYS processed — it resets the exchange clocks and
    # terminalizes the live desire — no matter how full the frame is (spec §7).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    observed = contact_observed_signal(origin_id="e", actor="user", label="two_way", timestamp=None)
    signals = [observed, *(_noise(i) for i in range(200))]
    intents = _agg().step(_ctx(state, now, signals, objects=ACTIVE, tmp_path=tmp_path))
    assert _changes(intents)["last_exchange_at"] == now.isoformat()  # contact processed
    assert _transition(intents) == ("active", "satisfied")  # live desire resolved


def test_best_effort_flood_is_coalesced_to_a_bounded_count(tmp_path) -> None:
    # The 200 best_effort signals are coalesced to the bounded cap: the gate emits
    # the intake shed/coalesced counters so the backpressure is observable (spec §7).
    from lifemodel.core.intake import MAX_BEST_EFFORT
    from lifemodel.core.metrics import MetricRegistry
    from lifemodel.core.tick_metrics import SIGNALS_INTAKE, register_universal_metrics
    from lifemodel.domain.signal import Signal

    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=1.5, last_tick_at="2026-07-06T03:59:00+00:00")
    reg = MetricRegistry()
    register_universal_metrics(reg)
    observed = contact_observed_signal(origin_id="e", actor="user", label="two_way", timestamp=None)
    signals = (observed, *(Signal(origin_id=f"n{i}", kind="sensor_noise") for i in range(200)))
    ctx = TickContext(
        state=state, now=now, signals=signals, objects=ACTIVE, trace=_TRACE, metrics=reg
    )
    _agg().step(ctx)
    intake = reg.get(SIGNALS_INTAKE)
    assert intake is not None
    assert intake.value(outcome="shed_sensor") == float(200 - MAX_BEST_EFFORT)  # type: ignore[attr-defined]
    assert intake.value(outcome="coalesced") == float(MAX_BEST_EFFORT)  # type: ignore[attr-defined]


def test_only_best_effort_noise_no_spurious_launch(tmp_path) -> None:
    # A frame of ONLY sensor noise (no must_process, u below theta) behaves sanely:
    # coalesced, no crash, NO desire born / no cognition launch (spec §7).
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
    state = State(u=0.0, last_tick_at="2026-07-06T03:59:00+00:00")
    intents = _agg().step(_ctx(state, now, [_noise(i) for i in range(200)], tmp_path=tmp_path))
    assert _desire_intent(intents) is None  # no spurious desire / launch

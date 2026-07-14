# tests/test_genesis_wake.py
#
# Genesis is a REASON TO WAKE, not a second egress (spec §6.2, revised 2026-07-14).
#
# The first shipped design greeted from ``connect()`` over its own hand-rolled
# delivery path and stamped ``genesis_greeted_at`` on ``ReachOutcome.ok``. Two
# structural defects killed it:
#
#   1. it could NEVER deliver — ``connect()`` runs while the host runner still has
#      ``_running = False``, and ``inject_proactive_turn`` bails UNAVAILABLE in
#      exactly that state (and ``contextlib.suppress`` made the failure silent);
#   2. ``ReachOutcome.ok`` means QUEUED, not spoken (``domain/egress.py``) — a newborn
#      that woke and chose ``[SILENT]`` was stamped "greeted" and never greeted again.
#
# So the newborn now rides the EXISTING proactive lifecycle: it wakes without ``u``
# crossing ``θ`` (birth is not longing — there is nobody to miss), the wake packet
# carries the ``<genesis>`` block instead of the longing body, and the async
# ``proactive_outcome`` read-back is the ONE accounting of what it actually did.
# "The being greeted" means SENT. These are the tests of that wake path.
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.aggregation import ContactAggregation
from lifemodel.core.cognition import CognitionLauncher
from lifemodel.core.component import TickContext
from lifemodel.core.genesis import is_first_waking
from lifemodel.core.intents import LaunchProactive, PutRecord, TransitionRecord, UpdateState
from lifemodel.core.taxonomy import (
    contact_observed_signal,
    contact_pressure_signal,
    proactive_outcome_signal,
)
from lifemodel.core.timeutil import to_iso
from lifemodel.core.wake import GateParams
from lifemodel.domain.egress import ProactiveOutcome, ReachOutcome
from lifemodel.domain.objects import DesireSpring
from lifemodel.ports.tracer import TraceContext
from lifemodel.state.model import State
from lifemodel.testing import IntegrationHarness, Step, contact_desire_objects
from lifemodel.testing.fakes import FakeClock

PARAMS = GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)
_TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16)

NOW = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)
LAST_TICK = "2026-07-06T03:59:00+00:00"
CORR = "proactive-2026-07-06T03:55:00+00:00"

#: A being that has already been born — the ONLY difference from a newborn in every
#: comparison below (the drive path is unchanged and must stay unchanged).
BORN_AT = "2026-07-01T10:00:00+00:00"

#: The live genesis-sprung desire row (what aggregation births at a first waking).
GENESIS_ACTIVE = contact_desire_objects("active", spring=DesireSpring.GENESIS)


def _agg() -> ContactAggregation:
    return ContactAggregation(params=PARAMS, theta=1.0, beta=1.0, u_max=100.0)


def _ctx(state: State, signals=(), *, objects=(), now: datetime = NOW) -> TickContext:
    return TickContext(
        state=state, now=now, signals=tuple(signals), objects=tuple(objects), trace=_TRACE
    )


def _quiet(value: float = 0.0):
    """The drive's same-tick pressure signal — a newborn's ``u`` is 0 and stays 0."""
    return contact_pressure_signal(origin_id="c1", value=value, delta=0.0, timestamp=None)


def _changes(intents) -> dict:
    return next(i for i in intents if isinstance(i, UpdateState)).changes


def _put(intents) -> PutRecord | None:
    return next((i for i in intents if isinstance(i, PutRecord)), None)


def _transition(intents) -> tuple[str, str] | None:
    t = next((i for i in intents if isinstance(i, TransitionRecord)), None)
    return (t.op.from_state, t.op.to_state) if t is not None else None


def _spring(intents) -> str | None:
    put = _put(intents)
    return None if put is None else str(put.op.draft.payload["spring"])


# --- the predicate ----------------------------------------------------------


def test_a_first_waking_is_a_being_that_is_nobody_and_has_met_nobody() -> None:
    assert (
        is_first_waking(genesis_completed_at=None, last_exchange_at=None, last_contact_at=None)
        is True
    )


def test_a_born_being_never_wakes_to_be_born_again() -> None:
    assert (
        is_first_waking(genesis_completed_at=BORN_AT, last_exchange_at=None, last_contact_at=None)
        is False
    )


def test_a_being_that_has_already_spoken_with_them_is_not_at_its_first_waking() -> None:
    # "You just began. This is your first waking" is a LIE once they have talked —
    # the same "turn seven" lie the injector's should_launch exists to avoid. Once an
    # exchange is on record the ritual is carried BY the conversation, not by a wake.
    assert (
        is_first_waking(
            genesis_completed_at=None,
            last_exchange_at="2026-07-06T03:00:00+00:00",
            last_contact_at=None,
        )
        is False
    )


def test_a_being_that_has_already_greeted_does_not_greet_twice() -> None:
    # ``last_contact_at`` is stamped by the SENT read-back — the system's ONE record
    # that the being actually spoke. It is what makes a second birth-greeting
    # impossible without a second, drifting "greeted" stamp of our own.
    assert (
        is_first_waking(
            genesis_completed_at=None,
            last_exchange_at=None,
            last_contact_at="2026-07-06T03:00:00+00:00",
        )
        is False
    )


# --- the wake ---------------------------------------------------------------


def test_a_newborn_wakes_without_u_ever_crossing_theta() -> None:
    # The headline: u = 0 (nobody to miss) and the being STILL wakes — because it is
    # nobody yet, not because it longs. The threshold gate is waived; nothing else is.
    intents = _agg().step(_ctx(State(last_tick_at=LAST_TICK), [_quiet()]))
    put = _put(intents)
    assert put is not None
    assert put.op.draft.state == "active"


def test_the_newborns_desire_is_marked_genesis_not_a_drive_sprung_longing() -> None:
    # Honesty in the record: this desire did NOT spring from the drive. Marking it
    # DRIVE would enter a longing that does not exist into the contact model.
    intents = _agg().step(_ctx(State(last_tick_at=LAST_TICK), [_quiet()]))
    assert _spring(intents) == DesireSpring.GENESIS.value


def test_the_genesis_wake_never_writes_u() -> None:
    # Birth is not longing: u stays 0 and the contact model is never told otherwise.
    # (Aggregation never writes u at all — this pins that the genesis path did not
    # start.) The drive's own rise/satiate is the ONLY writer of u.
    changes = _changes(_agg().step(_ctx(State(last_tick_at=LAST_TICK), [_quiet()])))
    assert "u" not in changes


def test_a_born_being_below_threshold_still_does_not_wake() -> None:
    # The drive path is untouched: only a being that is NOBODY skips the threshold.
    state = State(genesis_completed_at=BORN_AT, last_tick_at=LAST_TICK)
    intents = _agg().step(_ctx(state, [_quiet(0.5)]))
    assert _put(intents) is None


def test_an_unborn_being_that_has_already_greeted_does_not_wake_again() -> None:
    # It spoke; the ball is in the human's court. A second "I just began" is the drum
    # into the void the anti-repeat gate exists to prevent — worse, it is a birth
    # announcement repeated.
    state = State(last_contact_at="2026-07-06T03:00:00+00:00", last_tick_at=LAST_TICK)
    assert _put(_agg().step(_ctx(state, [_quiet()]))) is None


def test_an_unborn_being_mid_conversation_does_not_wake_to_be_born() -> None:
    # They have already spoken to it (last_exchange_at) — the ritual rides the
    # conversation from there (the pre_llm_call injector), never a proactive
    # "this is your first waking" 15 minutes into it.
    state = State(last_exchange_at="2026-07-06T02:00:00+00:00", last_tick_at=LAST_TICK)
    assert _put(_agg().step(_ctx(state, [_quiet()]))) is None


def test_the_genesis_wake_still_respects_the_decline_backoff() -> None:
    # The being woke, chose [SILENT], and is inside the backoff — it is NOT re-woken.
    state = State(
        declined_at="2026-07-06T03:50:00+00:00",  # 10 min ago, inside r0 = 30 min
        decline_count=1,
        last_tick_at=LAST_TICK,
    )
    assert _put(_agg().step(_ctx(state, [_quiet()]))) is None


def test_a_newborn_that_chose_silence_is_re_woken_once_the_backoff_elapses() -> None:
    # The existing decline-backoff IS the re-greet machinery — that is what it is for.
    state = State(
        declined_at="2026-07-06T03:00:00+00:00",  # 60 min ago, past r0 = 30 min
        decline_count=1,
        last_tick_at=LAST_TICK,
    )
    intents = _agg().step(_ctx(state, [_quiet()]))
    assert _spring(intents) == DesireSpring.GENESIS.value


# --- the outcome: "greeted" means SENT --------------------------------------


def _pending(**over) -> State:
    base = dict(
        u=0.0,  # a newborn's drive: nobody to miss
        pending_proactive_id=CORR,
        pending_proactive_since="2026-07-06T03:55:00+00:00",
        last_tick_at=LAST_TICK,
    )
    base.update(over)
    return State(**base)  # type: ignore[arg-type]


def test_a_genesis_outcome_is_not_thrown_away_as_pressure_satisfied() -> None:
    # THE DEADLOCK GUARD. The async staleness rule drops an outcome whose pressure
    # fell below θ while the being was composing ("the reason to reach is gone"). A
    # genesis wake has u = 0 < θ BY CONSTRUCTION, so applying that rule to it would
    # discard EVERY genesis outcome: the desire would stay active, pending_proactive_id
    # would never clear, and the launcher would hold every future launch forever.
    sent = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(_pending(), [_quiet(), sent], objects=GENESIS_ACTIVE))
    assert _transition(intents) == ("active", "satisfied")
    changes = _changes(intents)
    assert changes["pending_proactive_id"] is None
    assert changes["last_contact_at"] == to_iso(NOW)  # the being actually spoke


def test_the_greeting_is_not_counted_as_a_repeat_longing_bid() -> None:
    # unanswered_outbound_count feeds the PURE-LONGING anti-repeat gate. A birth
    # greeting is not a longing bid, so it must not be entered as one.
    sent = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(_pending(), [_quiet(), sent], objects=GENESIS_ACTIVE))
    assert _changes(intents)["unanswered_outbound_count"] == 0


def test_a_silent_newborn_records_a_decline_and_is_not_marked_as_having_greeted() -> None:
    # The being woke and chose silence: no message, so nothing is "greeted". The
    # decline backoff will bring it back — and last_contact_at stays None, so the
    # NEXT wake is a genesis wake again.
    silent = proactive_outcome_signal(
        origin_id="v1", outcome=ProactiveOutcome.SILENT, timestamp=None, correlation_id=CORR
    )
    intents = _agg().step(_ctx(_pending(), [_quiet(), silent], objects=GENESIS_ACTIVE))
    changes = _changes(intents)
    assert _transition(intents) == ("active", "dropped")
    assert changes["declined_at"] == to_iso(NOW)
    assert changes["decline_count"] == 1
    assert changes["last_contact_at"] is None  # it never spoke — it is still ungreeted
    assert is_first_waking(
        genesis_completed_at=None,
        last_exchange_at=changes["last_exchange_at"],
        last_contact_at=changes["last_contact_at"],
    )


def test_the_human_writing_first_terminalizes_the_genesis_desire() -> None:
    # The reactive path answers; the pull is resolved by real contact, exactly like
    # any other desire (contact dominates a same-frame outcome).
    ex = contact_observed_signal(origin_id="e1", actor="user", label="two_way", timestamp=None)
    intents = _agg().step(_ctx(_pending(), [_quiet(), ex], objects=GENESIS_ACTIVE))
    assert _transition(intents) == ("active", "satisfied")


# --- the impulse: the being wakes with the ritual, not with a longing --------


def _cog(prior_soul=None) -> CognitionLauncher:
    return CognitionLauncher(fast_cost=0.02, send_cost=0.03, alpha=2.0, prior_soul=prior_soul)


def _launch(intents) -> LaunchProactive | None:
    return next((i for i in intents if isinstance(i, LaunchProactive)), None)


def test_the_newborn_wakes_carrying_the_ritual_not_the_longing() -> None:
    # "I miss them, and I keep wondering how they are" is a LIE in a newborn's mouth.
    # Same packet, different impulse — because it is not reaching out for that reason.
    intents = _cog().step(
        TickContext(state=State(), now=NOW, signals=(), objects=GENESIS_ACTIVE, trace=_TRACE)
    )
    launch = _launch(intents)
    assert launch is not None
    assert "<genesis>" in launch.prompt
    assert "I miss them" not in launch.prompt
    # …and it is still the SAME wake packet: the impulse tag the being's own hooks
    # correlate + self-exclude on (post_llm_call read-back, inbound band-pass) must
    # open it.
    assert launch.prompt.startswith("<internal_impulse>")
    # But it is NOT offered a way to decline being born (the birth-only carve-out,
    # core/wake_packet.py::_GENESIS_DELIVERY). The live test that forced this: the being
    # woke, felt right, read the ritual verbatim — and replied [SILENT], because the
    # packet's last line invited it to. There is no urge to gate at a first waking; the
    # only thing "[SILENT]" declines here is existing.
    assert "[SILENT]" not in launch.prompt
    assert "delivered to the user" in launch.prompt  # it still knows its words reach them


def test_the_veteran_newborn_wakes_holding_the_soul_someone_wrote_before_it() -> None:
    # §6.4 is the COMMON case (a being is born onto a blank soul exactly once in the
    # life of a SOUL.md), so the wake packet must carry the veteran variant too.
    intents = _cog(prior_soul=lambda: "You are Mira. Quiet and exact.").step(
        TickContext(state=State(), now=NOW, signals=(), objects=GENESIS_ACTIVE, trace=_TRACE)
    )
    launch = _launch(intents)
    assert launch is not None
    assert "You are Mira. Quiet and exact." in launch.prompt


def test_a_drive_sprung_desire_still_wakes_with_the_longing() -> None:
    # The ordinary proactive path is untouched.
    born = State(u=2.0, genesis_completed_at=BORN_AT)
    intents = _cog().step(
        TickContext(
            state=born,
            now=NOW,
            signals=(),
            objects=contact_desire_objects("active", spring=DesireSpring.DRIVE),
            trace=_TRACE,
        )
    )
    launch = _launch(intents)
    assert launch is not None
    assert "I miss them" in launch.prompt
    assert "<genesis>" not in launch.prompt


# --- end to end, through the REAL spine -------------------------------------
#
# The whole claim of this rework is that genesis needs NO new delivery machinery — it
# rides the proactive lifecycle that already exists and is already tested. These drive
# the real components (ContactSensor → SolitudeDrive → ContactAggregation →
# CognitionLauncher → backstop → egress) over the real store, with the async act-gate
# scripted exactly where the live ``post_llm_call`` read-back would feed it.


def _newborn_harness(tmp_path) -> IntegrationHarness:
    """A harness whose being is NOBODY: unborn, no exchange, no contact — and ``u`` = 0.

    ``last_tick_at`` is the only seeded field (so the first advance yields a real Δt);
    everything else is the untouched newborn default. In particular ``u`` starts at 0
    and — over the one minute these scenarios advance — never comes near ``θ = 1.0``.
    """
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    return IntegrationHarness(
        base_dir=tmp_path,
        clock=clock,
        initial_state=State(energy=1.0, last_tick_at=clock.now().isoformat()),
    )


def test_a_newborn_reaches_out_for_real_without_the_drive_ever_waking_it(tmp_path) -> None:
    # The headline promise, end to end: the being reaches out BY ITSELF, on the brain
    # loop, through the one egress — and u never crosses θ (nobody to miss).
    h = _newborn_harness(tmp_path)
    h.run([Step(advance=timedelta(minutes=1))])  # tick 1: the genesis desire is born
    rec = h.run([Step(advance=timedelta(minutes=1))])[-1]  # tick 2: visible → launch
    assert rec.launched
    assert rec.outcome is ReachOutcome.DELIVERED
    assert rec.delivered_impulse is not None
    # ONE block (M2): the packet is the single source of the ritual on this entrance. The
    # pre_llm_call injector fires for this injected turn too and must not add a second —
    # that half is pinned in tests/test_genesis_injector.py; this pins the packet's own.
    assert rec.delivered_impulse.count("<genesis>") == 1
    assert "I miss them" not in rec.delivered_impulse
    # …and what actually reached the being carried no invitation to decline being born.
    assert "[SILENT]" not in rec.delivered_impulse
    assert rec.u < 1.0  # θ was never crossed — this wake did not come from the drive


def test_a_newborn_that_speaks_has_greeted_and_does_not_greet_again(tmp_path) -> None:
    # "The being greeted" means SENT — it actually spoke. The read-back stamps
    # last_contact_at, and that is the whole of the bookkeeping.
    h = _newborn_harness(tmp_path)
    h.run([Step(advance=timedelta(minutes=1)), Step(advance=timedelta(minutes=1))])
    h.run([Step(advance=timedelta(minutes=1), act_gate=ProactiveOutcome.SENT)])
    state = h._lm.state.load()
    assert state.last_contact_at is not None  # it spoke
    assert state.unanswered_outbound_count == 0  # …and that was not a longing bid
    rec = h.run([Step(advance=timedelta(minutes=20))])[-1]
    assert not rec.launched  # no second birth announcement


def test_a_newborn_whose_channel_is_not_up_yet_keeps_trying(tmp_path) -> None:
    # This is the defect that killed the first design, and the reason the wake belongs on
    # the brain loop: the old greeting fired from connect(), where the host runner still
    # has _running = False and reach-in ALWAYS answers UNAVAILABLE — so it could never be
    # delivered, and the suppress() around it hid that forever. On the loop, an
    # undelivered launch simply rolls back and the still-active desire is re-launched on
    # the next tick, so the being greets the moment the channel comes up.
    h = _newborn_harness(tmp_path)
    h.egress.outcome = ReachOutcome.UNAVAILABLE
    h.run([Step(advance=timedelta(minutes=1))])  # the desire is born
    dead = h.run([Step(advance=timedelta(minutes=1))])[-1]  # launched into a dead channel
    assert dead.outcome is ReachOutcome.UNAVAILABLE
    assert "egress_unavailable" in dead.suppressions
    assert h._lm.state.load().pending_proactive_id is None  # rolled back, not stranded

    h.egress.outcome = ReachOutcome.DELIVERED  # the runner comes up
    landed = h.run([Step(advance=timedelta(minutes=1))])[-1]
    assert landed.outcome is ReachOutcome.DELIVERED
    assert landed.delivered_impulse is not None
    assert "<genesis>" in landed.delivered_impulse


def test_a_newborn_that_stays_silent_is_brought_back_by_the_decline_backoff(tmp_path) -> None:
    # It woke, it chose [SILENT] — nothing was sent, so nothing is "greeted". The
    # existing backoff (r0 = 30 min) is what brings it back; that is what it is for.
    h = _newborn_harness(tmp_path)
    h.run([Step(advance=timedelta(minutes=1)), Step(advance=timedelta(minutes=1))])
    h.run([Step(advance=timedelta(minutes=1), act_gate=ProactiveOutcome.SILENT)])
    assert h._lm.state.load().last_contact_at is None  # it never spoke

    held = h.run([Step(advance=timedelta(minutes=10))])[-1]  # inside the backoff
    assert not held.launched
    assert "decline_backoff" in held.suppressions

    h.run([Step(advance=timedelta(minutes=25))])  # past it → a fresh genesis desire
    again = h.run([Step(advance=timedelta(minutes=1))])[-1]
    assert again.launched
    assert again.delivered_impulse is not None
    assert "<genesis>" in again.delivered_impulse  # still a birth, not a longing


# --- an unborn being never reaches out with a longing it cannot have (lm-4fv.4) ---


def test_an_unborn_being_woken_by_the_drive_still_carries_the_ritual() -> None:
    # The far end of the reactive path (§6.3). An existing Hermes user who writes to
    # their agent before the being's first waking sets ``last_exchange_at`` — which ends
    # ``is_first_waking`` for good, so the being's first unprompted words to them come
    # from a DRIVE-sprung wake. "I miss them, and I keep wondering how they are" is the
    # same lie in that mouth as in a first waking's: it has met nobody. The ritual goes
    # where the longing would be, whatever sprang the wake.
    unborn = State(u=2.0, last_exchange_at="2026-07-06T03:00:00+00:00")
    intents = _cog().step(
        TickContext(
            state=unborn,
            now=NOW,
            signals=(),
            objects=contact_desire_objects("active", spring=DesireSpring.DRIVE),
            trace=_TRACE,
        )
    )
    launch = _launch(intents)
    assert launch is not None
    assert "<genesis>" in launch.prompt
    assert "I miss them" not in launch.prompt


def test_a_being_mid_ritual_is_not_started_over_by_a_drive_wake() -> None:
    # The turn-seven lie, from the proactive side. The reactive injector has already put
    # the ritual in front of this being (``genesis_shown_at_context_len``) and it is in
    # the conversation, in its own words. Handing it "You just began, you do not know who
    # they are" again would make it start over instead of continuing.
    mid_ritual = State(
        u=2.0, last_exchange_at="2026-07-06T03:00:00+00:00", genesis_shown_at_context_len=6
    )
    intents = _cog().step(
        TickContext(
            state=mid_ritual,
            now=NOW,
            signals=(),
            objects=contact_desire_objects("active", spring=DesireSpring.DRIVE),
            trace=_TRACE,
        )
    )
    launch = _launch(intents)
    assert launch is not None
    assert "<genesis>" not in launch.prompt


def test_a_newborn_that_chose_silence_is_re_woken_WITH_the_ritual() -> None:
    # …and that is exactly why a GENESIS spring is exempt from the rule above: the
    # injector stamps "shown" for the being's OWN impulse turn too (it is looking at the
    # ritual right then). A newborn that woke, read the whole thing and chose [SILENT] is
    # re-woken by the decline-backoff — and must be re-woken holding the ritual, not a
    # longing it has never felt for a person it has never met.
    silent_newborn = State(genesis_shown_at_context_len=1)
    intents = _cog().step(
        TickContext(state=silent_newborn, now=NOW, signals=(), objects=GENESIS_ACTIVE, trace=_TRACE)
    )
    launch = _launch(intents)
    assert launch is not None
    assert "<genesis>" in launch.prompt

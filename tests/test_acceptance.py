"""Acceptance scenarios on REAL components (spec §10) — the phase finale.

Every scenario runs through :class:`lifemodel.testing.IntegrationHarness`, which
drives the actual ``ContactSensor → SolitudeDrive → ContactAggregation →
CognitionLauncher`` spine (not a third model — T8) over the real SQLite store
through fake ports, scripting the async act-gate via the verdict read-back path.
Each asserts BOTH the outcome AND a suppression-span reason, so any silent tick is
explained — silence is a logged choice, not a bug (spec §5/§10).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.adapters.clock import SystemClock
from lifemodel.core.desire_view import read_live_contact_desire
from lifemodel.core.wake_packet import build_wake_packet
from lifemodel.domain.egress import ProactiveOutcome, ReachOutcome
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state_commands import force_wake_for_dir
from lifemodel.testing import IntegrationHarness, Step
from lifemodel.testing.fakes import FakeClock

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


# --- Positive: the [SILENT] cure (silence → threshold → message) -------------


def test_silence_to_threshold_to_a_delivered_message(tmp_path) -> None:
    h = IntegrationHarness(base_dir=tmp_path)
    # silence accumulates, then crosses the threshold → a DRIVE desire is born
    h.run([Step(advance=timedelta(minutes=100)), Step(advance=timedelta(minutes=200))])
    desire = read_live_contact_desire(h._memory())
    assert desire is not None and desire.state == "active"
    assert str(desire.spring) == "drive"  # drive-only (T3): the only spring

    # next tick the desire is snapshot-visible → CognitionLauncher launches
    launch = h.run([Step(advance=timedelta(minutes=1))])[-1]
    assert launch.outcome is ReachOutcome.DELIVERED
    assert launch.launched
    assert launch.delivered_impulse
    # the [SILENT] cure (T6): the wake-packet is the owner-approved felt impulse —
    # it names the longing (the feeling) and its cause as a sufficient reason.
    assert "miss them" in launch.delivered_impulse.lower()
    assert "reason enough" in launch.delivered_impulse.lower()
    u_at_launch = launch.u

    # scripted act-gate: the being actually sent a message → FULFILL read-back
    fulfill = h.run([Step(advance=timedelta(minutes=1), act_gate=ProactiveOutcome.SENT)])[-1]
    final = h._lm.state.load()
    assert final.action_pending_since is not None  # ActionPending inhibition started
    assert fulfill.u >= u_at_launch  # send ≠ contact: the message did NOT satiate u
    assert read_live_contact_desire(h._memory()) is None  # desire resolved (satisfied)


# --- Negatives: observability + unobtrusiveness ------------------------------


def test_below_threshold_emits_no_launch_and_a_below_threshold_span(tmp_path) -> None:
    h = IntegrationHarness(base_dir=tmp_path)
    rec = h.run([Step(advance=timedelta(minutes=100))])[-1]  # u ≈ 0.42 < θ
    assert rec.outcome is None  # quiet — no egress outcome (no false 'busy')
    assert not rec.launched
    assert rec.suppressions == ("below_threshold",)  # silence is a logged decision


def test_silent_act_gate_rejects_applies_backoff_without_satiating(tmp_path) -> None:
    h = IntegrationHarness(base_dir=tmp_path)
    h.run([Step(advance=timedelta(minutes=100)), Step(advance=timedelta(minutes=200))])
    h.run([Step(advance=timedelta(minutes=1))])  # launch + deliver
    u_before = h._lm.state.load().u
    # scripted act-gate: the being chose silence ([SILENT]) → REJECT read-back
    h.run([Step(advance=timedelta(minutes=1), act_gate=ProactiveOutcome.SILENT)])
    final = h._lm.state.load()
    assert read_live_contact_desire(h._memory()) is None  # desire dropped (terminal)
    assert final.decline_count >= 1  # decline backoff applied
    assert final.pending_proactive_id is None  # the turn resolved
    assert final.u >= u_before  # silence is not contact: u NOT satiated by [SILENT]


def test_repeat_pure_longing_holds_a_second_bid_after_a_send(tmp_path) -> None:
    h = IntegrationHarness(base_dir=tmp_path)
    # full positive flow → a send (FULFILL) leaves an unanswered pure-longing bid
    h.run([Step(advance=timedelta(minutes=100)), Step(advance=timedelta(minutes=200))])
    h.run([Step(advance=timedelta(minutes=1))])  # launch + deliver
    h.run([Step(advance=timedelta(minutes=1), act_gate=ProactiveOutcome.SENT)])  # send
    # wait past ActionPending decay so the urge returns, with no reply since → a
    # second pure-longing bid must HOLD (unobtrusiveness), not relaunch.
    rec = h.run([Step(advance=timedelta(minutes=200))])[-1]
    assert rec.outcome is None  # no second launch
    assert not rec.launched
    assert "repeat_pure_longing" in rec.suppressions


def test_backstop_rate_limited_holds_a_launch_with_a_recent_send(tmp_path) -> None:
    # Seed a recent send so the fail-closed backstop (min-interval 60 min) blocks the
    # launch. The desire is born naturally (u ≥ θ) and cognition launches, but the
    # egress backstop holds it — logged as backstop_rate_limited.
    clock = FakeClock(_T0)
    initial = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        proactive_send_log=[clock.now().isoformat()],  # a send "now" → within 60 min
        last_tick_at=clock.now().isoformat(),
    )
    h = IntegrationHarness(base_dir=tmp_path, clock=clock, initial_state=initial)
    h.run([Step(advance=timedelta(minutes=1))])  # u ≥ θ → desire born (not visible yet)
    rec = h.run([Step(advance=timedelta(minutes=1))])[-1]  # visible → launch → backstop holds
    assert rec.outcome is None  # quiet — the backstop held, no delivery
    assert not rec.launched
    assert "backstop_rate_limited" in rec.suppressions
    assert rec.desire_state == "deferred"  # held, not sent


def test_inbound_exchange_satiates_the_drive_and_prevents_wake(tmp_path) -> None:
    h = IntegrationHarness(base_dir=tmp_path)
    h.run([Step(advance=timedelta(minutes=100))])  # u ≈ 0.42 (below θ)
    u_before = h._lm.state.load().u
    # a genuine inbound two-way exchange satiates the drive (SolitudeDrive)
    rec = h.run([Step(advance=timedelta(minutes=1), exchange=("user", "two_way"))])[-1]
    final = h._lm.state.load()
    assert final.u < u_before  # u satiated by the real exchange
    assert rec.outcome is None and not rec.launched  # no wake (u below θ)


def test_silence_window_suppresses_a_wake_right_after_an_exchange(tmp_path) -> None:
    clock = FakeClock(_T0)
    initial = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        last_exchange_at=clock.now().isoformat(),
        last_tick_at=clock.now().isoformat(),
    )
    h = IntegrationHarness(base_dir=tmp_path, clock=clock, initial_state=initial)
    rec = h.run([Step(advance=timedelta(minutes=10))])[-1]  # 10 min < 15 min window
    assert rec.outcome is None and not rec.launched
    assert rec.suppressions == ("silence_window",)


def test_force_wake_keeps_the_real_last_exchange_in_the_wake_packet(tmp_path) -> None:
    # THE lm-md6.1 acceptance, end-to-end through the REAL store + command wiring the
    # live being uses (force_wake_for_dir over SQLiteRuntimeStore). The real last
    # exchange is seeded at a FIXED absolute instant far in the past so the render is
    # deterministic under the real wall clock. force_wake used to overwrite it with a
    # ~20-min-ago backdate; now it moves only the decoupled silence anchor, so the
    # wake packet the model reads still carries the genuine last-exchange time.
    real_last = "2020-01-01T00:00:00+00:00"  # a genuine, long-ago exchange
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(u=0.2, last_exchange_at=real_last))

    message = force_wake_for_dir(tmp_path)
    assert "gates satisfied" in message

    persisted = SQLiteRuntimeStore(tmp_path, clock=SystemClock()).load()
    assert persisted.last_exchange_at == real_last  # immune: the real record survives
    assert persisted.silence_anchor_at is not None  # the gate was satisfied via the anchor

    packet = build_wake_packet(
        value=persisted.u,
        theta=1.0,
        correlation_id="c",
        now=datetime(2026, 1, 1, tzinfo=UTC),
        last_exchange_at=persisted.last_exchange_at,
        tz=UTC,
    )
    assert "The last time we exchanged messages was 2020-01-01 00:00 UTC." in packet.prompt


def test_decline_backoff_suppresses_a_wake_right_after_a_rejection(tmp_path) -> None:
    clock = FakeClock(_T0)
    initial = State(
        u=2.0,
        energy=1.0,
        fatigue=0.0,
        decline_count=1,
        declined_at=clock.now().isoformat(),
        last_tick_at=clock.now().isoformat(),
    )
    h = IntegrationHarness(base_dir=tmp_path, clock=clock, initial_state=initial)
    rec = h.run([Step(advance=timedelta(minutes=10))])[-1]  # 10 min < r0 = 30 min backoff
    assert rec.outcome is None and not rec.launched
    assert rec.suppressions == ("decline_backoff",)

"""The simulation harness — the tick-stepper that certifies the model (spec §9).

Pure Python, no Hermes. The harness ties the drive (``u`` dynamics), the
wake-decision gates, and the aggregation desire-lifecycle together, drives them
over a trace of lane events + a scripted cognition (the ``verdicts`` FIFO), and
records per-tick observations. This file exercises the *mechanics* of the
stepper; ``test_sim_scenarios.py`` asserts the §9 invariants over the §9
scenarios.

Time is in minutes, tick ``Δt = 1``.
"""

from __future__ import annotations

from lifemodel.sim.aggregation import DesireStatus, Verdict
from lifemodel.sim.harness import SimConfig, SimEvent, run


def test_pure_silence_rises_and_eventually_wakes() -> None:
    # α=0.25, θ=1.0, dt=1 → u crosses θ exactly at t=4 (first rise is at t=1). No
    # prior exchange, so the active-silence window never gates. Default rejects.
    # (0.25 is exact in binary float, so the crossing tick is unambiguous.)
    cfg = SimConfig(alpha=0.25, theta_u=1.0, horizon=6.0)

    result = run(cfg, events=[])

    assert result.wake_times() == [4.0]
    first = result.wakes[0]
    assert first.verdict is Verdict.REJECT
    assert first.u >= cfg.theta_u


def test_no_wake_before_threshold() -> None:
    cfg = SimConfig(alpha=0.25, theta_u=1.0, horizon=3.0)  # u peaks at 0.75 < θ

    result = run(cfg, events=[])

    assert result.wake_times() == []


def test_user_message_satiates_and_clears_desire_and_reject_record() -> None:
    # Start already above threshold with a stale reject on the books; a genuine
    # two-way user message drops u by β·1 and wipes the desire + reject record.
    cfg = SimConfig(
        alpha=0.0, theta_u=1.0, beta=1.0, horizon=2.0, u0=2.0, decline_count0=3, declined_at0=0.0
    )

    result = run(cfg, events=[SimEvent(time=1.0, actor="user", label="two_way")])

    rec = result.at(1.0)
    assert rec.u == 1.0  # 2.0 − β·1.0
    assert rec.status is DesireStatus.NONE
    assert rec.decline_count == 0
    assert rec.last_exchange_at == 1.0


def test_no_wake_within_active_silence_window_after_an_exchange() -> None:
    # A user message at t=0 with u already high: the window W=15 forbids any wake
    # in (0,15) even though u ≥ θ the whole time (the anti-drum window gate).
    cfg = SimConfig(alpha=0.0, theta_u=1.0, w=15.0, horizon=20.0, u0=5.0)

    result = run(cfg, events=[SimEvent(time=0.0, actor="user", label="two_way")])

    # u0=5, minus β on the t=0 message = 4.0, still ≥ θ throughout.
    assert all(t >= 15.0 for t in result.wake_times())
    assert result.wake_times()  # but it *does* eventually wake once W passes


def test_fulfill_satiates_resets_duration_and_updates_clock() -> None:
    cfg = SimConfig(alpha=0.25, theta_u=1.0, beta=1.0, horizon=6.0)

    result = run(cfg, events=[], verdicts=[Verdict.FULFILL])

    wake = result.wakes[0]
    assert wake.verdict is Verdict.FULFILL
    assert wake.u == 0.0  # satiated by a full q=1 exchange
    assert wake.duration_over_theta == 0.0  # contact resets the deprivation clock
    assert wake.last_exchange_at == wake.time  # a delivered outreach updates the clock


def test_defer_holds_the_desire_and_does_not_reset_u() -> None:
    cfg = SimConfig(alpha=0.25, theta_u=1.0, horizon=6.0)

    result = run(cfg, events=[], verdicts=[Verdict.DEFER])

    wake = result.wakes[0]
    assert wake.verdict is Verdict.DEFER
    assert wake.status is DesireStatus.DEFERRED
    assert wake.u >= cfg.theta_u  # not reset — no contact happened


def test_deferred_desire_releases_on_observed_presence() -> None:
    # Wake while away → defer (held). When the user becomes available, the held
    # intention is re-presented (deferred_intention_releases) and fulfilled.
    cfg = SimConfig(alpha=0.25, theta_u=1.0, horizon=10.0)

    result = run(
        cfg,
        events=[SimEvent(time=6.0, user_available=True)],
        verdicts=[Verdict.DEFER, Verdict.FULFILL],
    )

    assert len(result.wakes) == 2
    release = result.wakes[1]
    assert release.time >= 6.0
    assert release.verdict is Verdict.FULFILL
    assert release.outcome == "WAKE_RELEASE"


def test_internal_impulse_neither_satiates_nor_updates_the_clock() -> None:
    # A proactive_internal row is the being's own nudge: q=0, never an exchange.
    # u keeps rising through it and the conversation clock stays untouched.
    cfg = SimConfig(alpha=0.125, theta_u=1.0, horizon=8.0)

    result = run(cfg, events=[SimEvent(time=5.0, actor="proactive_internal", label="monologue")])

    rec = result.at(5.0)
    assert rec.u == 0.625  # 5 ticks × 0.125 — unaffected by the internal impulse
    assert rec.last_exchange_at is None  # clock never advanced


def test_reject_backs_off_and_grows_no_fixed_period_drum() -> None:
    # First wake at t=10 → reject (R0=30). u stays pinned above θ, but the next
    # wake is suppressed until the backoff expires — and the SECOND backoff (R0·k)
    # is longer than the first, so the gaps grow instead of drumming.
    cfg = SimConfig(alpha=0.5, theta_u=1.0, r0=30.0, k=2.0, horizon=200.0, u_max=5.0)

    result = run(cfg, events=[], verdicts=[Verdict.REJECT, Verdict.REJECT, Verdict.REJECT])

    times = result.wake_times()
    assert len(times) >= 3
    gap1 = times[1] - times[0]
    gap2 = times[2] - times[1]
    assert gap2 > gap1  # growing, not a fixed drum


def test_restart_resumes_state_without_a_spurious_wake() -> None:
    # Snapshot the drive/lane/desire mid-run, then start a fresh run seeded with
    # it: the reload tick must not manufacture a wake (scenario 7).
    cfg1 = SimConfig(alpha=0.1, theta_u=1.0, horizon=6.0)
    first = run(cfg1, events=[])
    snap = first.records[-1]

    cfg2 = SimConfig(
        alpha=0.1,
        theta_u=1.0,
        horizon=2.0,
        u0=snap.u,
        duration_over_theta0=snap.duration_over_theta,
        last_exchange_at0=snap.last_exchange_at,
    )
    resumed = run(cfg2, events=[])

    # u was 0.6 < θ at snapshot, so the resumed run has no wake at the reload tick.
    assert resumed.wake_times() == []
    assert resumed.records[0].u > 0.0  # state genuinely carried over

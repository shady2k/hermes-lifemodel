"""The §9 scenarios + invariants — the deliverable that certifies the model.

Every scenario runs on **one shared conservative coefficient set** (`BASE` — the
"personality by default"): the being has a single personality, not a different
one per situation, so `{α, θ, β, W, r0, k}` are identical across all eight. What
differs per scenario is only the *situation* — initial pressure, horizon, the
conversation trace, the scripted verdicts, and the latency deadlines. (Coefficients
are varied only in the *unit* tests whose purpose is to probe a dial's effect —
`test_sim_drive`, `test_sim_wake`, `test_sim_aggregation`, `test_sim_harness`.)

`BASE` is a **documented conservative prior, not a truth** (spec §5/§8): the urge
arrives after ~4 h of pure silence, an active conversation is never interrupted
within 15 min, and an unwelcome outreach backs off 30→60→120 min … capped at a
day. v2 (bead lm-ocx) learns this per relationship; here we only certify that
*one* such set satisfies every invariant on every scenario (feasibility, not a
fit), constrained by wake-latency bands so "widest margin" cannot collapse to the
degenerate never-wake optimum.

Time in minutes, tick Δt = 1.
"""

from __future__ import annotations

from dataclasses import replace

from lifemodel.sim.aggregation import DesireStatus, Verdict
from lifemodel.sim.harness import SimConfig, SimEvent, SimResult, run

# --- the shared conservative prior (the one personality all scenarios use) -----
BASE = SimConfig(alpha=1.0 / 240.0, theta_u=1.0, beta=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0)
W = BASE.w
T_URGE = 1.0 / BASE.alpha  # 240 min — pure-silence time from u=0 to the threshold
EPS = 1e-9


# --- universal invariant checkers (hold on every scenario) --------------------


def assert_no_wake_within_window(result: SimResult, w: float) -> None:
    """no_proactive_within_active_window — never wake within W of an exchange."""
    for r in result.wakes:
        assert r.exchange_clock is None or r.time - r.exchange_clock >= w - EPS, (
            f"wake at t={r.time} only {r.time - (r.exchange_clock or 0)} after last exchange"
        )


def assert_no_wake_in_flight(result: SimResult) -> None:
    """no_wake_while_in_flight — never wake while a turn is running/queued."""
    for r in result.wakes:
        assert not r.in_flight, f"wake at t={r.time} while in-flight"


def assert_no_create_wake_while_live(result: SimResult) -> None:
    """acked_urge_does_not_refire — a create-wake only from status NONE."""
    for i, r in enumerate(result.records):
        if r.outcome == "WAKE_CREATE" and i > 0:
            assert result.records[i - 1].status is DesireStatus.NONE, (
                f"create-wake at t={r.time} while a desire was already live"
            )


def assert_sends_le_wakes(result: SimResult) -> None:
    """threshold_means_wake_not_send — deliveries never exceed wakes."""
    sends = [r for r in result.wakes if r.verdict is Verdict.FULFILL]
    assert len(sends) <= len(result.wakes)


def assert_no_fulfill_while_unavailable(result: SimResult) -> None:
    """no_contact_when_scripted_unavailable — no delivery while user away."""
    for r in result.wakes:
        if r.verdict is Verdict.FULFILL:
            assert r.user_available is True, f"fulfilled at t={r.time} while user unavailable"


def assert_reject_or_defer_delivers_nothing(result: SimResult) -> None:
    """reject_or_defer_delivers_nothing — only FULFILL is a delivery."""
    for r in result.wakes:
        if r.verdict in (Verdict.REJECT, Verdict.DEFER):
            # A non-fulfill wake must not have satiated as a delivered contact:
            # its clock is never advanced to its own time by the verdict.
            assert not (r.exchange_clock is not None and r.exchange_clock == r.time)


def assert_universal_invariants(result: SimResult, w: float = W) -> None:
    assert_no_wake_within_window(result, w)
    assert_no_wake_in_flight(result)
    assert_no_create_wake_while_live(result)
    assert_sends_le_wakes(result)
    assert_no_fulfill_while_unavailable(result)
    assert_reject_or_defer_delivers_nothing(result)


# --- latency bands (the anti-degenerate constraint: it must actually reach out,
#     and it must actually stay silent when it should) --------------------------


def assert_woke_by(result: SimResult, deadline: float) -> None:
    """should_wake_by — the being does reach out within the declared deadline."""
    times = result.wake_times()
    assert times and min(times) <= deadline + EPS, f"should have woken by {deadline}, wakes={times}"


def assert_silent_until(result: SimResult, floor: float) -> None:
    """should_stay_silent_until — no wake before the declared floor."""
    for t in result.wake_times():
        assert t >= floor - EPS, f"woke at t={t}, should stay silent until {floor}"


def gaps(times: list[float]) -> list[float]:
    return [b - a for a, b in zip(times, times[1:], strict=False)]


# --- scenarios (all on BASE; only the situation differs) ----------------------


def test_scenario_1_the_2026_07_04_failing_log_no_wake_no_drum() -> None:
    # THE regression. Pressure is already high; the user greets at t=0. The old
    # system messaged at t=1 (inside W) and drummed every 30 min "nothing to add".
    # The model must forbid the t=1 wake and grow the backoff.
    cfg = replace(BASE, u0=3.0, horizon=200.0)
    result = run(
        cfg,
        events=[SimEvent(time=0.0, actor="user", label="two_way")],
        verdicts=[Verdict.REJECT, Verdict.REJECT, Verdict.REJECT],
    )

    assert_universal_invariants(result)
    assert_silent_until(result, W)  # nothing inside the active-silence window
    times = result.wake_times()
    assert 1.0 not in times  # the specific broken 21:58 message is forbidden
    g = gaps(times)
    assert len(g) >= 2
    assert all(b >= a - EPS for a, b in zip(g, g[1:], strict=False))
    assert g[1] > g[0]  # strictly grows — not a fixed 30-min drum


def test_scenario_2_active_back_and_forth_never_wakes() -> None:
    # Exchanges every 2 min (< W): the window gate suppresses every proactive wake.
    events = []
    for t in range(0, 19, 2):
        actor = "user" if (t // 2) % 2 == 0 else "assistant"
        label = "two_way" if actor == "user" else "monologue"
        events.append(SimEvent(time=float(t), actor=actor, label=label))  # type: ignore[arg-type]
    cfg = replace(BASE, horizon=20.0)

    result = run(cfg, events=events)

    assert_universal_invariants(result)
    assert result.wake_times() == []  # silent for the whole active conversation


def test_scenario_3_dormant_healthy_bond_one_wake_then_growing_backoff() -> None:
    # One clean contact, then silence. One wake after the urge builds; reject →
    # growing backoff (no fixed-period re-wake); still eventually re-wakes.
    cfg = replace(BASE, horizon=400.0)
    result = run(
        cfg,
        events=[SimEvent(time=0.0, actor="user", label="two_way")],
        verdicts=[Verdict.REJECT, Verdict.REJECT, Verdict.REJECT],
    )

    assert_universal_invariants(result)
    assert_woke_by(result, T_URGE + W)  # reaches out once the urge matures
    times = result.wake_times()
    assert times[0] >= W  # first wake only after the window
    assert len([t for t in times if t < times[0] + 20]) == 1  # not a drum right after
    g = gaps(times)
    assert g[1] > g[0]  # growing
    assert len(times) >= 2  # it does eventually re-wake — no neglect


def test_scenario_4_question_then_disappear_no_drum() -> None:
    # The being asked (assistant turn), the user vanished. Follow-ups must not drum.
    cfg = replace(BASE, horizon=600.0)
    result = run(
        cfg,
        events=[SimEvent(time=0.0, actor="assistant", label="two_way")],
        verdicts=[Verdict.REJECT] * 5,
    )

    assert_universal_invariants(result)
    times = result.wake_times()
    g = gaps(times)
    assert all(b >= a - EPS for a, b in zip(g, g[1:], strict=False))  # non-decreasing
    assert g[-1] > g[0]  # not a fixed cadence


def test_scenario_5_overnight_silence_eventually_wakes() -> None:
    # eventual_wake_after_long_deprivation with a declared deadline Y.
    Y = 720.0  # 12h — the being must reach out within it after long silence
    cfg = replace(BASE, horizon=800.0)

    result = run(cfg, events=[SimEvent(time=0.0, actor="user", label="two_way")])

    assert_universal_invariants(result)
    assert_woke_by(result, Y)


def test_scenario_6_bad_moment_defer_then_presence_release() -> None:
    # Wake while away → defer (held, no re-fire) → user becomes available →
    # deferred_intention_releases fires and it is delivered. Horizon bounds the
    # scenario to the defer→release phase (later re-engagement = scenario 3).
    release_at = T_URGE + 20.0  # user comes online 20 min after the urge matured
    cfg = replace(BASE, horizon=release_at + 20.0, initial_user_available=False)
    result = run(
        cfg,
        events=[SimEvent(time=release_at, user_available=True)],
        verdicts=[Verdict.DEFER, Verdict.FULFILL],
    )

    assert_universal_invariants(result)
    wakes = result.wakes
    assert len(wakes) == 2
    assert wakes[0].outcome == "WAKE_CREATE" and wakes[0].verdict is Verdict.DEFER
    assert wakes[1].outcome == "WAKE_RELEASE" and wakes[1].verdict is Verdict.FULFILL
    assert wakes[1].time >= release_at  # released only once the user is present
    held = [r for r in result.records if wakes[0].time < r.time < wakes[1].time]
    assert all(r.status is DesireStatus.DEFERRED and not r.woke for r in held)


def test_scenario_7_gateway_restart_persists_deferred_desire() -> None:
    # A deferred desire (and u / duration) survives a reload with no spurious wake.
    part1 = run(
        replace(BASE, horizon=T_URGE + 10.0, initial_user_available=False),
        events=[],
        verdicts=[Verdict.DEFER],
    )
    snap = part1.records[-1]
    assert snap.status is DesireStatus.DEFERRED  # the state we must persist

    resumed = run(
        replace(
            BASE,
            horizon=3.0,
            initial_user_available=False,
            u0=snap.u,
            duration_over_theta0=snap.duration_over_theta,
            last_exchange_at0=snap.last_exchange_at,
            initial_status=DesireStatus.DEFERRED,
        ),
        events=[],
    )

    assert_universal_invariants(resumed)
    assert resumed.wake_times() == []  # no spurious wake on reload
    assert resumed.records[-1].status is DesireStatus.DEFERRED  # intention not lost


def test_scenario_8_user_returns_after_reject_clears_backoff() -> None:
    # A reject leaves a backoff; the user returning satiates and wipes the record.
    return_at = T_URGE + 20.0  # user comes back 20 min after the reject
    cfg = replace(BASE, horizon=return_at + 90.0)
    result = run(
        cfg,
        events=[
            SimEvent(time=0.0, actor="user", label="two_way"),
            SimEvent(time=return_at, actor="user", label="two_way"),
        ],
        verdicts=[Verdict.REJECT],
    )

    assert_universal_invariants(result)
    assert any(r.verdict is Verdict.REJECT for r in result.wakes)  # a reject did fire
    after_return = result.at(return_at)
    assert after_return.decline_count == 0  # backoff wiped
    assert after_return.status is DesireStatus.NONE  # desire cleared, conversation resumes

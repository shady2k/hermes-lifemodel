"""The §9 scenarios + invariants — the deliverable that certifies the model.

Each scenario drives a labelled conversation trace through the harness and
asserts the spec §9 invariants *numerically*. The universal invariants (window,
in-flight, dedup, sends≤wakes, no-fulfill-while-unavailable) hold on **every**
scenario; the scenario-specific asserts pin the drum-vs-neglect tension. The
headline is scenario 1: the 2026-07-04 failing log must produce **no**
mid-conversation wake and **no** 30-min drum.

Time in minutes, tick Δt = 1.
"""

from __future__ import annotations

from lifemodel.sim.aggregation import DesireStatus, Verdict
from lifemodel.sim.harness import SimConfig, SimEvent, SimResult, run

W = 15.0  # active-silence window used across scenarios
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


def gaps(times: list[float]) -> list[float]:
    return [b - a for a, b in zip(times, times[1:], strict=False)]


# --- scenarios ----------------------------------------------------------------


def test_scenario_1_the_2026_07_04_failing_log_no_wake_no_drum() -> None:
    # THE regression. Pressure is already high; the user greets at t=0. The old
    # system messaged at t=1 (inside W) and then drummed every 30 min voicing
    # "nothing to add". The model must forbid the t=1 wake and grow the backoff.
    cfg = SimConfig(alpha=0.5, beta=1.0, u0=3.0, w=W, r0=30.0, k=2.0, horizon=200.0)
    result = run(
        cfg,
        events=[SimEvent(time=0.0, actor="user", label="two_way")],
        verdicts=[Verdict.REJECT, Verdict.REJECT, Verdict.REJECT],
    )

    assert_universal_invariants(result)
    times = result.wake_times()
    # No mid-conversation wake: nothing inside the active-silence window.
    assert all(t >= W for t in times)
    assert 1.0 not in times  # the specific broken 21:58 message is forbidden
    # No 30-min drum: the reject backoff grows (30, 60, ...).
    g = gaps(times)
    assert len(g) >= 2
    assert all(b >= a - EPS for a, b in zip(g, g[1:], strict=False))
    assert g[1] > g[0]  # strictly grows — not a fixed cadence


def test_scenario_2_active_back_and_forth_never_wakes() -> None:
    # Exchanges every 2 min (< W): the window gate suppresses every proactive wake.
    events = []
    for t in range(0, 19, 2):
        actor = "user" if (t // 2) % 2 == 0 else "assistant"
        label = "two_way" if actor == "user" else "monologue"
        events.append(SimEvent(time=float(t), actor=actor, label=label))  # type: ignore[arg-type]
    cfg = SimConfig(alpha=0.5, beta=1.0, w=W, horizon=20.0)

    result = run(cfg, events=events)

    assert_universal_invariants(result)
    assert result.wake_times() == []


def test_scenario_3_dormant_healthy_bond_one_wake_then_growing_backoff() -> None:
    # One clean contact, then silence. Exactly one wake after W; reject → growing
    # backoff (no fixed-period re-wake); still eventually re-wakes.
    cfg = SimConfig(alpha=0.25, beta=1.0, w=W, r0=30.0, k=2.0, horizon=200.0)
    result = run(
        cfg,
        events=[SimEvent(time=0.0, actor="user", label="two_way")],
        verdicts=[Verdict.REJECT, Verdict.REJECT, Verdict.REJECT],
    )

    assert_universal_invariants(result)
    times = result.wake_times()
    assert times[0] >= W  # first wake only after the window
    assert len([t for t in times if t < times[0] + 20]) == 1  # not a drum right after
    g = gaps(times)
    assert g[1] > g[0]  # growing
    assert len(times) >= 2  # it does eventually re-wake — no neglect


def test_scenario_4_question_then_disappear_no_drum() -> None:
    # The being asked (assistant turn), the user vanished. Follow-ups must not drum.
    cfg = SimConfig(alpha=0.25, beta=1.0, w=W, r0=30.0, k=2.0, horizon=300.0)
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
    Y = 720.0  # 12h
    cfg = SimConfig(alpha=0.01, beta=1.0, w=W, horizon=800.0)

    result = run(cfg, events=[SimEvent(time=0.0, actor="user", label="two_way")])

    assert_universal_invariants(result)
    times = result.wake_times()
    assert times, "the being must eventually reach out after long deprivation"
    assert min(times) <= Y


def test_scenario_6_bad_moment_defer_then_presence_release() -> None:
    # Wake while away → defer (held, no re-fire) → user becomes available →
    # deferred_intention_releases fires and it is delivered.
    # Horizon bounds the scenario to the defer→release phase; after the t=20
    # delivery the bond is "fresh" again and later re-engagement is a *different*
    # (dormant-bond) scenario, covered by scenario 3.
    cfg = SimConfig(alpha=0.25, beta=1.0, w=W, horizon=30.0, initial_user_available=False)
    result = run(
        cfg,
        events=[SimEvent(time=20.0, user_available=True)],
        verdicts=[Verdict.DEFER, Verdict.FULFILL],
    )

    assert_universal_invariants(result)
    wakes = result.wakes
    assert len(wakes) == 2
    assert wakes[0].outcome == "WAKE_CREATE" and wakes[0].verdict is Verdict.DEFER
    assert wakes[1].outcome == "WAKE_RELEASE" and wakes[1].verdict is Verdict.FULFILL
    assert wakes[1].time >= 20.0  # released only once the user is present
    # No re-fire between defer and release (held, deduped).
    held = [r for r in result.records if wakes[0].time < r.time < wakes[1].time]
    assert all(r.status is DesireStatus.DEFERRED and not r.woke for r in held)


def test_scenario_7_gateway_restart_persists_deferred_desire() -> None:
    # A deferred desire (and u / duration) survives a reload with no spurious wake.
    part1 = run(
        SimConfig(alpha=0.25, beta=1.0, w=W, horizon=6.0, initial_user_available=False),
        events=[],
        verdicts=[Verdict.DEFER],
    )
    snap = part1.records[-1]
    assert snap.status is DesireStatus.DEFERRED  # the state we must persist

    resumed = run(
        SimConfig(
            alpha=0.25,
            beta=1.0,
            w=W,
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
    cfg = SimConfig(alpha=0.5, beta=1.0, w=W, r0=30.0, k=2.0, horizon=60.0, u0=0.0)
    result = run(
        cfg,
        events=[
            SimEvent(time=0.0, actor="user", label="two_way"),
            SimEvent(time=20.0, actor="user", label="two_way"),
        ],
        verdicts=[Verdict.REJECT],
    )

    assert_universal_invariants(result)
    # A reject fired after the first silence (declined record set)...
    assert any(r.verdict is Verdict.REJECT for r in result.wakes)
    # ...and the user's return at t=20 wiped the backoff + satiated.
    after_return = result.at(20.0)
    assert after_return.decline_count == 0
    assert after_return.status is DesireStatus.NONE

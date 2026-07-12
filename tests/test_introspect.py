# tests/test_introspect.py
from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.core.introspect import DebugConfig, Readings, compute_readings
from lifemodel.core.wake import GateParams
from lifemodel.state.model import State

CFG = DebugConfig(
    params=GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0),
    theta=1.0,
    i0=1.0,
    grace_min=45.0,
    halflife_min=60.0,
    peak_hour_utc=13.0,
    max_per_day=3,
    min_interval_min=60.0,
    alpha=1.0 / 240.0,
    u_max=100.0,
)
NOW = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)


def test_reads_physiology_and_drive() -> None:
    state = State(u=2.0, energy=0.7, fatigue=0.3, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert isinstance(r, Readings)
    assert r.energy == 0.7
    assert r.fatigue == 0.3
    assert 0.0 <= r.circadian <= 1.0
    assert r.u > 2.0  # risen by 1 min * alpha from persisted value
    assert r.inhibition == 0.0  # no ActionPending
    assert r.effective > 0.0  # u*(1-0)
    assert r.would_wake is True  # effective >= theta, no gates


def test_action_pending_suppresses_effective_and_wake() -> None:
    state = State(
        u=3.0,
        action_pending_since="2026-07-06T03:50:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)  # 10 min ago -> in grace -> inhibition 1
    assert r.inhibition == 1.0
    assert r.action_pending_phase == "grace"
    assert r.effective == 0.0
    assert r.would_wake is False
    assert r.wake_reason == "no_wake_below_threshold"  # effective 0 < theta


def test_backstop_readings() -> None:
    log = ["2026-07-06T03:30:00+00:00", "2026-07-06T02:00:00+00:00"]  # 2 today, last 30m ago
    state = State(u=2.0, proactive_send_log=log, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.sends_today == 2
    assert r.sends_cap == 3
    assert r.send_allowed is False  # last send 30 min ago < 60 min interval


def test_silence_window_and_backoff() -> None:
    state = State(
        u=2.0,
        last_exchange_at="2026-07-06T03:55:00+00:00",  # 5 min ago, w=15 -> 10 left
        declined_at="2026-07-06T03:40:00+00:00",
        decline_count=1,  # 20 min ago, r0=30 -> 10 left
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert abs((r.silence_window_remaining_min or 0) - 10.0) < 1e-6
    assert abs((r.backoff_remaining_min or 0) - 10.0) < 1e-6
    assert r.would_wake is False  # silence window blocks


def test_drive_is_risen_as_of_now_for_display() -> None:
    # persisted u=0, but 240 min elapsed at alpha=1/240 -> should display ~1.0
    cfg = DebugConfig(
        params=GateParams(theta_u=1.0, w=15.0, r0=30.0, k=2.0, r_max=1440.0),
        theta=1.0,
        i0=1.0,
        grace_min=45.0,
        halflife_min=60.0,
        peak_hour_utc=13.0,
        max_per_day=3,
        min_interval_min=60.0,
        alpha=1.0 / 240.0,
        u_max=100.0,
    )
    state = State(u=0.0, last_tick_at="2026-07-06T00:00:00+00:00")
    now = datetime(2026, 7, 6, 4, 0, tzinfo=UTC)  # 240 min later
    r = compute_readings(state, now=now, cfg=cfg)
    assert abs(r.u - 1.0) < 1e-6  # risen as of now
    assert r.would_wake is True  # and the debug reflects the imminent wake


# --- receptivity readings (lm-27n.5) ----------------------------------------


def test_receptivity_defaults_are_behavior_neutral() -> None:
    # No user-model passed -> permissive default -> allowed / multiplier 1.0.
    state = State(u=2.0, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.receptivity_allowed is True
    assert r.receptivity_multiplier == 1.0
    assert r.receptivity_hard_reasons == ()


def test_receptivity_surfaces_an_explicit_hard_veto() -> None:
    from lifemodel.core.user_model_view import EXPLICIT_CONFIDENCE, build_owner_user_model

    # NOW is hour 04 UTC; explicit bad-hours=(4,) -> hard veto surfaced for the audit.
    um = build_owner_user_model(bad_hours=(4,), confidence=EXPLICIT_CONFIDENCE)
    state = State(u=2.0, last_tick_at="2026-07-06T03:59:00+00:00")
    r = compute_readings(state, now=NOW, cfg=CFG, user_model=um)
    assert r.receptivity_allowed is False
    assert r.receptivity_hard_reasons  # non-empty: the "why silent" audit


# --- lm-27n.10: the compact "why did I write" contact-chain summary ----------


def test_contact_chain_summary_none_is_no_outreach() -> None:
    from lifemodel.core.introspect import contact_chain_summary

    assert contact_chain_summary(None) == "no current outreach"


def test_contact_chain_summary_follows_the_primary_lineage() -> None:
    from lifemodel.core.introspect import contact_chain_summary
    from lifemodel.core.why_graph import WhyEdge, WhyNode

    def _node(kind, oid, edges=()):
        return WhyNode(
            kind=kind,
            id=oid,
            state="active",
            reason=None,
            component=None,
            trace_id=None,
            creation_span_id=None,
            created_at="",
            updated_at="",
            edges=edges,
        )

    desire = _node("desire", "contact:owner")
    intention = _node("intention", "contact:owner", (WhyEdge(label="source", node=desire),))
    assert (
        contact_chain_summary(intention)
        == "intention:contact:owner <- desire:contact:owner (source)"
    )


def test_contact_chain_summary_marks_a_cycle() -> None:
    from lifemodel.core.introspect import contact_chain_summary
    from lifemodel.core.why_graph import WhyEdge, WhyNode

    node = WhyNode(
        kind="thought",
        id="thought:a",
        state="active",
        reason=None,
        component=None,
        trace_id=None,
        creation_span_id=None,
        created_at="",
        updated_at="",
        edges=(WhyEdge(label="parent_thought", cycle=True),),
    )
    assert contact_chain_summary(node) == "thought:a <- [cycle] (parent_thought)"


def test_reads_core_affect_current_and_recomputed_target() -> None:
    # lm-ukc.6: the debug readings surface the being's felt state — the CURRENT eased
    # axes come straight from stored state, while the this-tick TARGET is recomputed
    # from the snapshot (like drive u is), with contributors ranked so the reader sees
    # what tugs valence/arousal hardest.
    state = State(
        u=6.0,  # ~a day of silence → loneliness dominates the valence target
        affect_valence=-0.12,
        affect_arousal=0.40,
        affect_updated_at="2026-07-06T03:59:00+00:00",
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.affect_valence == -0.12  # stored eased value, NOT recomputed
    assert r.affect_arousal == 0.40
    assert r.affect_updated_at == "2026-07-06T03:59:00+00:00"
    assert r.affect_target_valence < 0.0  # loneliness pulls the target negative
    assert r.affect_valence_contributions[0][0] == "u"  # ranked: loneliness leads
    assert r.affect_arousal_contributions  # arousal always carries a baseline term


def test_reads_the_felt_word_from_current_axes() -> None:
    # lm-ukc.3: the dump shows the being's mood in a word — from the CURRENT eased axes
    # (what it feels now), not the recomputed target. Deep unpleasant + calm → "lonely".
    state = State(
        affect_valence=-0.6,
        affect_arousal=0.30,
        last_tick_at="2026-07-06T03:59:00+00:00",
    )
    r = compute_readings(state, now=NOW, cfg=CFG)
    assert r.affect_word == "lonely"

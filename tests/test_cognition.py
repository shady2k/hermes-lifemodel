from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.adapters.signal_bus import FileSignalBus
from lifemodel.core.cognition import Cognition
from lifemodel.core.component import TickContext
from lifemodel.core.intents import LaunchProactive, PutRecord, UpdateState
from lifemodel.core.relationship_view import EXPLICIT_CONFIDENCE
from lifemodel.state.model import State
from lifemodel.testing import (
    contact_desire_objects,
    contact_desire_record,
    owner_relationship_record,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)

# a live active-desire snapshot (what the old ``desire_status="active"`` meant)
ACTIVE = contact_desire_objects("active")


def _intention_put(intents):
    """The PutRecord birthing the contact intention this tick, if any."""
    return next(
        (i for i in intents if isinstance(i, PutRecord) and i.op.draft.kind == "intention"),
        None,
    )


def _cog() -> Cognition:
    return Cognition(fast_cost=0.02, send_cost=0.03, alpha=2.0)


def _ctx(state: State, *, objects=(), tmp_path) -> TickContext:
    return TickContext(
        state=state, now=NOW, bus=FileSignalBus(tmp_path), signals=(), objects=tuple(objects)
    )


def _launch(intents):
    return next((i for i in intents if isinstance(i, LaunchProactive)), None)


def _update(intents):
    return next((i for i in intents if isinstance(i, UpdateState)), None)


def test_no_active_desire_does_nothing(tmp_path) -> None:
    # no live desire in the snapshot -> the old desire_status="none"
    intents = _cog().step(_ctx(State(u=2.0), tmp_path=tmp_path))
    assert list(intents) == []


def test_active_desire_launches_proactive_turn(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    launch = _launch(intents)
    assert launch is not None
    assert launch.correlation_id == f"proactive-{NOW.isoformat()}"
    assert launch.prompt  # carries the wake-packet prompt
    upd = _update(intents)
    assert upd.changes["pending_proactive_id"] == launch.correlation_id
    assert upd.changes["pending_proactive_since"] == NOW.isoformat()
    assert upd.changes["energy"] < 1.0  # reserved


def test_pending_turn_is_not_relaunched(tmp_path) -> None:
    state = State(u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is None  # idempotent — a turn is already in flight


def test_insufficient_energy_holds_no_launch(tmp_path) -> None:
    # estimate = (0.02+0.03)*(1+2*1.0)=0.15 at max fatigue; energy 0.05 can't afford
    state = State(u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is None  # emergent shutoff — hold
    assert _update(intents) is None  # energy untouched, desire stays active


def test_deferred_desire_does_not_launch(tmp_path) -> None:
    # only an ACTIVE desire launches; a deferred one is held (cognition never re-wakes it)
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(
        _ctx(state, objects=contact_desire_objects("deferred"), tmp_path=tmp_path)
    )
    assert _launch(intents) is None


def test_launch_carries_the_reserved_energy(tmp_path) -> None:
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    # estimate = (0.02+0.03)*(1+2*0) = 0.05
    assert abs(launch.reserved_energy - 0.05) < 1e-9


def test_prompt_has_no_raw_numbers(tmp_path) -> None:
    import re

    state = State(u=3.2, energy=1.0)
    launch = _launch(_cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path)))
    assert not re.search(r"\d", launch.prompt)


# --- lm-27n.4: 0-LLM crystallization of the Bratman decision record ---


def test_launch_crystallizes_an_active_intention(tmp_path) -> None:
    # A launch now ALSO births the singleton intention, directly ``active`` so it
    # is visible in the next tick's snapshot and owns the gate.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    desire = contact_desire_objects("active", salience=2.5, source_drive=2.0)
    intents = _cog().step(_ctx(state, objects=desire, tmp_path=tmp_path))
    put = _intention_put(intents)
    assert put is not None
    draft = put.op.draft
    assert draft.kind == "intention"
    assert draft.id == "contact:owner"
    assert draft.state == "active"  # born committed, not pending
    # Rubicon fields recorded for auditability (0-LLM, deterministic).
    payload = draft.payload
    assert payload["commitment_strength"] == 2.5  # = the desire's effective pressure
    assert payload["goal"]
    assert payload["plan"]
    assert payload["implementation_trigger"]
    assert payload["reconsideration_triggers"]  # recorded, not yet acted on
    assert payload["rationale"]


def test_crystallize_and_launch_fire_together(tmp_path) -> None:
    # Behavior-neutral parity: the intention is created EXACTLY when a launch
    # happens — same tick, same gate. Never one without the other.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _launch(intents) is not None
    assert _intention_put(intents) is not None


def test_no_active_desire_crystallizes_nothing(tmp_path) -> None:
    # Parity: no live desire -> no launch AND no intention (old none gate).
    intents = _cog().step(_ctx(State(u=2.0), tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_pending_turn_crystallizes_nothing(tmp_path) -> None:
    # Parity: a turn in flight -> no launch AND no intention (no double-crystallize).
    state = State(u=2.0, pending_proactive_id="proactive-earlier")
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_insufficient_energy_crystallizes_nothing(tmp_path) -> None:
    # Parity: unaffordable -> hold; no launch AND no intention (emergent shutoff).
    state = State(u=2.0, energy=0.05, fatigue=1.0)
    intents = _cog().step(_ctx(state, objects=ACTIVE, tmp_path=tmp_path))
    assert _intention_put(intents) is None


def test_deferred_desire_crystallizes_nothing(tmp_path) -> None:
    # Parity: only an ACTIVE desire launches; a deferred one crystallizes nothing.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    intents = _cog().step(
        _ctx(state, objects=contact_desire_objects("deferred"), tmp_path=tmp_path)
    )
    assert _intention_put(intents) is None


# --- lm-27n.5: receptivity re-check before launch ---


def test_default_relationship_launches_identically(tmp_path) -> None:
    # Parity: a permissive relationship in the snapshot launches EXACTLY as the
    # no-relationship path.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    objects = (contact_desire_record("active"), owner_relationship_record())
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is not None
    assert _intention_put(intents) is not None


def test_explicit_quiet_hours_holds_the_launch(tmp_path) -> None:
    # NOW is hour 12 UTC; an explicit bad-hours=(12,) that started AFTER the desire
    # was born -> cognition re-checks and HOLDS (no launch, no intention). The
    # live desire persists for a later admissible tick.
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    rel = owner_relationship_record(bad_hours=(12,), confidence=EXPLICIT_CONFIDENCE)
    objects = (contact_desire_record("active"), rel)
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is None  # held
    assert _intention_put(intents) is None
    assert list(intents) == []  # nothing committed -> the desire survives


def test_launch_records_appraisal_constraints_on_the_intention(tmp_path) -> None:
    # An allowed launch with style/topic norms records them on the intention (audit).
    state = State(u=2.0, energy=1.0, fatigue=0.0)
    rel = owner_relationship_record(
        acceptable_styles=("playful", "concise"),
        topic_sensitivity=("work",),
        confidence=EXPLICIT_CONFIDENCE,
    )
    objects = (contact_desire_record("active"), rel)
    intents = _cog().step(_ctx(state, objects=objects, tmp_path=tmp_path))
    assert _launch(intents) is not None  # styles/topics are constraints, not vetoes
    put = _intention_put(intents)
    assert put is not None
    constraints = put.op.draft.payload["constraints"]
    assert any(c == "style: playful|concise" for c in constraints)
    assert any(c == "avoid topic: work" for c in constraints)

"""Real-code sim (lm-705.21, final task): a commitment surfaces into a live turn, the being
closes it with the tool and it leaves the active set, a freshly created one shows up next
turn, and a deferred one never surfaces — all over the actual on-disk store, no mocks, so a
green scenario honestly predicts live behaviour. Mirrors tests/test_belief_harness.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.composition import build_lifemodel
from lifemodel.core.commitment_view import commitment_from_live_fields, encode_commitment
from lifemodel.hooks import make_commitment_injector, make_commitment_tool
from lifemodel.state.model import State
from lifemodel.testing import FakeClock

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _lm(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _seed_active(store, content, *, trigger_value):
    c = commitment_from_live_fields(
        fields={
            "content": content,
            "basis": "self_assumed",
            "trigger_kind": "condition",
            "trigger_value": trigger_value,
        }
    )
    store.put(encode_commitment(c))
    return c.id


def test_commitment_shapes_the_turn_then_the_being_closes_it(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    cid = _seed_active(
        lm.state, "reflect the spending question back", trigger_value="he asks to spend on himself"
    )
    injector = make_commitment_injector(lambda: _lm(tmp_path))
    tool = make_commitment_tool(lambda: _lm(tmp_path))

    # turn 1: it surfaces into the reply
    first = injector(session_id="s", user_message="can I buy myself the good headphones?")
    assert first is not None and cid in first["context"]

    # the being closes it with the tool, in its own turn
    closed = json.loads(tool({"action": "discharge", "id": cid, "outcome": "honoured"}))
    assert closed["status"] == "ok"

    # turn 2: it no longer surfaces (left the active set) — and with nothing else, None
    assert injector(session_id="s", user_message="thanks") is None

    # the being creates a new one mid-turn → it surfaces on the following turn
    new_id = json.loads(
        tool(
            {
                "action": "create",
                "content": "ask how the move went",
                "basis": "follow_up",
                "trigger_kind": "event",
                "trigger_value": "next time we talk",
            }
        )
    )["id"]
    third = injector(session_id="s", user_message="hi again")
    assert third is not None and new_id in third["context"]


def test_deferred_commitment_never_surfaces(tmp_path: Path):
    lm = _lm(tmp_path)
    lm.state.commit(State())
    cid = _seed_active(
        lm.state, "come back to the promotion topic", trigger_value="he mentions work"
    )
    tool = make_commitment_tool(lambda: _lm(tmp_path))
    json.loads(tool({"action": "defer", "id": cid}))

    injector = make_commitment_injector(lambda: _lm(tmp_path))
    assert injector(session_id="s", user_message="hi") is None

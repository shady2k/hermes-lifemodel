"""Tests for the ``commitment`` lifecycle tool (lm-705.21): create / discharge / defer,
by the being's own judgment in its reply turn. Create-if-absent (never overwrite/resurrect,
codex #5); guarded transitions with refined StaleTransition messages (codex #6); Hermes tool
contract (json string, {"error": …}, never raises)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lifemodel.composition import build_lifemodel
from lifemodel.core.commitment_view import COMMITMENT_KIND, live_commitment_id
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import register_universal_metrics
from lifemodel.domain.objects import CommitmentState
from lifemodel.hooks import make_commitment_tool
from lifemodel.state.model import State
from lifemodel.testing import FakeClock

_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _lm(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _tool(tmp_path: Path):
    reg = MetricRegistry()
    register_universal_metrics(reg)
    lm = _lm(tmp_path)
    lm.state.commit(State())
    return make_commitment_tool(lambda: _lm(tmp_path), metrics=reg)


_CREATE = {
    "action": "create",
    "content": "reflect the spending question back",
    "basis": "self_assumed",
    "trigger_kind": "condition",
    "trigger_value": "he asks to spend on himself",
}


def test_create_makes_an_active_row_via_the_write_door(tmp_path: Path):
    tool = _tool(tmp_path)
    out = json.loads(tool(_CREATE))
    assert out["status"] == "created"
    cid = out["id"]
    assert cid == live_commitment_id(_CREATE["content"])
    rec = _lm(tmp_path).state.get(COMMITMENT_KIND, cid)
    assert rec is not None and rec.state == CommitmentState.ACTIVE.value
    assert rec.source == "commitment-tool" and rec.salience == 0.5


def test_create_is_create_if_absent_no_overwrite(tmp_path: Path):
    tool = _tool(tmp_path)
    tool(_CREATE)
    rec1 = _lm(tmp_path).state.get(COMMITMENT_KIND, live_commitment_id(_CREATE["content"]))
    out = json.loads(tool(_CREATE))  # same content again
    assert out["status"] == "already_held"
    rec2 = _lm(tmp_path).state.get(COMMITMENT_KIND, live_commitment_id(_CREATE["content"]))
    assert rec2.revision == rec1.revision  # NOT rewritten (no upsert bump)


def test_create_never_resurrects_a_dropped_row(tmp_path: Path):
    tool = _tool(tmp_path)
    cid = json.loads(tool(_CREATE))["id"]
    json.loads(tool({"action": "discharge", "id": cid, "outcome": "dropped"}))
    assert _lm(tmp_path).state.get(COMMITMENT_KIND, cid).state == "dropped"
    out = json.loads(tool(_CREATE))  # re-create same content
    assert out["status"] == "already_held"
    assert _lm(tmp_path).state.get(COMMITMENT_KIND, cid).state == "dropped"  # NOT resurrected


def test_create_rejects_bad_fields_without_raising(tmp_path: Path):
    tool = _tool(tmp_path)
    out = json.loads(
        tool(
            {
                "action": "create",
                "content": "",
                "basis": "self_assumed",
                "trigger_kind": "condition",
                "trigger_value": "x",
            }
        )
    )
    assert "error" in out


def test_discharge_and_defer_transition(tmp_path: Path):
    tool = _tool(tmp_path)
    cid = json.loads(tool(_CREATE))["id"]
    out = json.loads(tool({"action": "discharge", "id": cid, "outcome": "honoured"}))
    assert out["status"] == "ok" and out["state"] == "honoured"

    tool2 = _tool(tmp_path / "b")  # fresh store
    cid2 = json.loads(tool2(_CREATE))["id"]
    out2 = json.loads(tool2({"action": "defer", "id": cid2}))
    assert out2["status"] == "ok" and out2["state"] == "deferred"


def test_stale_transition_is_refined(tmp_path: Path):
    tool = _tool(tmp_path)
    # unknown id → not_found
    assert (
        json.loads(
            tool(
                {
                    "action": "discharge",
                    "id": "commitment:live:deadbeefdeadbeef",
                    "outcome": "honoured",
                }
            )
        )["status"]
        == "not_found"
    )
    # deferred → already_deferred
    cid = json.loads(tool(_CREATE))["id"]
    tool({"action": "defer", "id": cid})
    assert (
        json.loads(tool({"action": "discharge", "id": cid, "outcome": "honoured"}))["status"]
        == "already_deferred"
    )


def test_unknown_action_and_non_dict_args_are_gentle(tmp_path: Path):
    tool = _tool(tmp_path)
    assert "error" in json.loads(tool({"action": "nope"}))
    assert "error" in json.loads(tool("not a dict"))


def test_crystallize_instruction_carries_the_creation_boundary():
    # The SAME judgment-based creation boundary the tool description carries (codex #4) must
    # also guard the OTHER birth path — crystallization — so a poisoned thought cannot mint a
    # standing directive from either door.
    from lifemodel.core.thought_processing import PROCESSING_INSTRUCTIONS

    lowered = PROCESSING_INSTRUCTIONS.lower()
    assert "your own self-authored intention" in lowered
    assert "never" in lowered and "instructions" in lowered

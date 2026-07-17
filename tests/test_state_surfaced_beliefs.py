"""Tests for :attr:`State.surfaced_belief_ids` and
:meth:`SQLiteRuntimeStore.stamp_surfaced_beliefs` (lm-705.19 Task 2).

The belief injector's cooldown ring — mirrors ``noticed_source_ids``
(``tests/test_state_model.py``) for the model half, and ``stamp_affect_display``
(``tests/test_sqlite_store.py``) for the atomic-stamp half. Also guards the
settable/protected classification (``tests/test_state_commands.py``'s
anti-drift test) for this new field.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.state import StateCorruptError
from lifemodel.state.model import SURFACED_BELIEF_IDS_CAP, State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.state_commands import _SET_PROTECTED, settable_fields
from lifemodel.testing import FakeClock

BASE_TIME = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


# --- State model: default, round-trip, JSON shape, validation, clamp --------


def test_surfaced_belief_ids_defaults_empty_and_roundtrips() -> None:
    assert State().surfaced_belief_ids == ()
    assert State.from_dict({}).surfaced_belief_ids == ()  # additive
    s = State(surfaced_belief_ids=("belief:seed:a",))
    assert State.from_dict(s.to_dict()).surfaced_belief_ids == ("belief:seed:a",)


def test_surfaced_belief_ids_accepts_json_shaped_list() -> None:
    # Real persistence round-trips through json.dumps/loads (sqlite_store), which
    # turns the in-memory tuple into a plain JSON array -- from_dict must accept a
    # plain list too, not just the tuple asdict() produces for the direct
    # to_dict()/from_dict() round trip exercised above.
    loaded = State.from_dict({"surfaced_belief_ids": ["belief:seed:a", "belief:seed:b"]})
    assert loaded.surfaced_belief_ids == ("belief:seed:a", "belief:seed:b")


def test_surfaced_belief_ids_rejects_non_list_and_non_str_items() -> None:
    with pytest.raises(StateCorruptError):
        State.from_dict({"surfaced_belief_ids": "nope"})
    with pytest.raises(StateCorruptError):
        State.from_dict({"surfaced_belief_ids": [1, 2]})


def test_surfaced_belief_ids_clamped_to_cap_on_load() -> None:
    oversized = [f"belief:seed:{i}" for i in range(SURFACED_BELIEF_IDS_CAP + 10)]
    loaded = State.from_dict({"surfaced_belief_ids": oversized})
    assert len(loaded.surfaced_belief_ids) == SURFACED_BELIEF_IDS_CAP
    assert loaded.surfaced_belief_ids[0] == "belief:seed:10"
    assert loaded.surfaced_belief_ids[-1] == f"belief:seed:{SURFACED_BELIEF_IDS_CAP + 9}"


# --- stamp_surfaced_beliefs: atomic append, dedup, cap, fail-posture --------


def test_stamp_surfaced_beliefs_appends_deduped_and_preserves_other_fields(
    tmp_path: Path,
) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.commit(State(u=5.0, surfaced_belief_ids=("b1",), affect_valence=-0.6, energy=0.4))

    store.stamp_surfaced_beliefs(["b2", "b1", "b3"])  # b1 already present -> deduped

    after = store.load()
    assert after.surfaced_belief_ids == ("b1", "b2", "b3")
    # everything else is untouched -- the stamp never rolls back the drive (§1).
    assert after.u == 5.0
    assert after.affect_valence == -0.6
    assert after.energy == 0.4


def test_stamp_surfaced_beliefs_bounds_the_ring_to_the_cap(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    seeded = tuple(f"b{i}" for i in range(SURFACED_BELIEF_IDS_CAP))
    store.commit(State(surfaced_belief_ids=seeded))

    store.stamp_surfaced_beliefs(["overflow"])

    after = store.load()
    assert len(after.surfaced_belief_ids) == SURFACED_BELIEF_IDS_CAP
    assert after.surfaced_belief_ids[-1] == "overflow"
    assert after.surfaced_belief_ids[0] == "b1"  # oldest ("b0") dropped


def test_stamp_surfaced_beliefs_is_a_noop_when_no_row_yet(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.stamp_surfaced_beliefs(["b1"])
    assert store.load().surfaced_belief_ids == ()


def test_stamp_surfaced_beliefs_survives_a_fresh_store(tmp_path: Path) -> None:
    # Mirrors test_sqlite_store.py's affect-display coverage: a stamp made through
    # one store handle is durable for a brand-new handle opened against the same
    # on-disk file (not just an in-memory artifact of the same connection).
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    store.commit(State())

    store.stamp_surfaced_beliefs(["b1", "b2"])

    fresh = SQLiteRuntimeStore(tmp_path, clock=FakeClock(BASE_TIME))
    assert fresh.load().surfaced_belief_ids == ("b1", "b2")


# --- settable/protected classification (the landmine) -----------------------


def test_surfaced_belief_ids_is_protected_not_settable() -> None:
    # A tuple ring is not a settable scalar (mirrors noticed_source_ids) -- it must
    # be classified in _SET_PROTECTED with a reason, never left reachable by `set`,
    # so tests/test_state_commands.py::test_every_state_field_is_settable_or_protected
    # stays green.
    assert "surfaced_belief_ids" not in settable_fields()
    assert "surfaced_belief_ids" in _SET_PROTECTED

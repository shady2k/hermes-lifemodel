"""Tests for the reactive felt-state boundary in :mod:`lifemodel.hooks` (lm-ukc.4/.4.1).

Two adapter seams over the pure gate/composers in ``core.felt_display``:

* ``make_felt_state_injector`` — the ``pre_llm_call`` hook. Reads committed state,
  runs the ambient gate, and on LIGHT returns ``{"context": <block>}`` (ephemeral,
  one turn) after stamping the last-shown word/time; else ``None``. Fail-soft.
* ``make_check_in_tool`` — the ``check_in`` LLM tool handler. Returns a felt,
  first-person JSON self-read; NEVER raises (errors as ``{"error": …}``); NEVER
  leaks a raw axis number (the spec §4b first-class guarantee).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifemodel.composition import build_lifemodel
from lifemodel.core.desire_view import build_contact_desire, encode_contact_desire
from lifemodel.core.metrics import MetricRegistry
from lifemodel.core.tick_metrics import (
    CHECK_IN_TOTAL,
    FELT_DISPLAY_TOTAL,
    OBSERVER_ERRORS,
    register_universal_metrics,
)
from lifemodel.domain.objects import DesireState
from lifemodel.hooks import make_check_in_tool, make_felt_state_injector
from lifemodel.ports.memory import MemoryPort
from lifemodel.state.brain_health import BrainHealth
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import FakeClock

_NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)


def _registry() -> MetricRegistry:
    reg = MetricRegistry()
    register_universal_metrics(reg)
    return reg


def _boom() -> object:
    raise RuntimeError("build blew up")


def _lm(tmp_path: Path):
    return build_lifemodel(base_dir=tmp_path, clock=FakeClock(_NOW))


def _warmed_salient() -> State:
    # word "lonely", texture "sore and settled"; salient + warmed.
    return State(
        affect_valence=-0.6,
        affect_arousal=0.3,
        affect_updated_at="2026-07-12T11:30:00+00:00",
    )


def _seed_desire(store: MemoryPort) -> None:
    store.put(encode_contact_desire(build_contact_desire(state=DesireState.ACTIVE, salience=2.0)))


# --------------------------------------------------------------------------- #
# make_felt_state_injector
# --------------------------------------------------------------------------- #


def test_injector_injects_light_cue_and_stamps_state(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(_warmed_salient())
    reg = _registry()
    injector = make_felt_state_injector(lambda: _lm(tmp_path), metrics=reg)

    result = injector(user_message="how are you?", conversation_history=[])

    assert isinstance(result, dict)
    assert result["context"].startswith("<felt-state>")
    assert "sore and settled" in result["context"]
    # stamped the last-shown word + time so the cooldown/change gate can throttle.
    after = lm.state.load()
    assert after.affect_display_last_word == "lonely"
    assert after.affect_display_last_at is not None
    assert reg.get(FELT_DISPLAY_TOTAL).value(outcome="light") == 1.0


def test_injector_silent_on_cold_start_and_leaves_state_untouched(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())  # cold start
    reg = _registry()
    injector = make_felt_state_injector(lambda: _lm(tmp_path), metrics=reg)

    assert injector(user_message="how are you?", conversation_history=[]) is None
    after = lm.state.load()
    assert after.affect_display_last_word is None
    assert after.affect_display_last_at is None
    assert reg.get(FELT_DISPLAY_TOTAL).value(outcome="not_warmed") == 1.0


def test_injector_silent_and_counted_on_task_context(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(_warmed_salient())
    reg = _registry()
    injector = make_felt_state_injector(lambda: _lm(tmp_path), metrics=reg)

    result = injector(user_message="```py\nx = 1\n```", conversation_history=[])
    assert result is None
    assert reg.get(FELT_DISPLAY_TOTAL).value(outcome="task") == 1.0
    # not shown → not stamped
    assert lm.state.load().affect_display_last_word is None


def test_injector_body_throw_is_loud_and_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = BrainHealth(tmp_path)
    reg = _registry()
    injector = make_felt_state_injector(_boom, health=h, metrics=reg)

    with caplog.at_level(logging.DEBUG):
        assert injector(user_message="hi", conversation_history=[]) is None

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert h.last_observer_error.get("pre_llm_call") is not None
    assert reg.get(OBSERVER_ERRORS).value(component="pre_llm_call") == 1.0


class _RacingStore:
    """Delegates to the real store but, right AFTER the injector reads state for its
    decision, simulates the ~60s tick committing a fresh drive value on the SAME db —
    the exact interleave the atomic display stamp must survive (review finding #1)."""

    def __init__(self, real: SQLiteRuntimeStore, on_load) -> None:
        self._real = real
        self._on_load = on_load
        self._raced = False

    def load(self) -> State:
        state = self._real.load()
        if not self._raced:  # fire the "tick" exactly once, between read and stamp
            self._raced = True
            self._on_load()
        return state

    def __getattr__(self, name: str):  # delegate stamp_affect_display, isinstance shims, etc.
        return getattr(self._real, name)


def test_injector_stamp_does_not_clobber_concurrent_drive_change(tmp_path: Path) -> None:
    # THE finding-1 guarantee: the display stamp is a field-level atomic merge, so a
    # tick that advances `u` between the injector's read and its stamp is NOT rolled
    # back — the display path never writes the drive/wake state (one-way invariant §1).
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(_NOW))
    store.commit(State(**{**vars(_warmed_salient()), "u": 5.0}))

    def _tick_advances_u() -> None:
        other = SQLiteRuntimeStore(tmp_path, clock=FakeClock(_NOW))
        other.commit(dataclasses.replace(other.load(), u=6.0))

    racing = _RacingStore(store, _tick_advances_u)
    lm = SimpleNamespace(state=racing, clock=FakeClock(_NOW))
    injector = make_felt_state_injector(lambda: lm)  # type: ignore[arg-type]

    assert injector(user_message="how are you?", conversation_history=[]) is not None  # LIGHT
    after = store.load()
    assert after.u == 6.0, "the tick's drive advance must survive the display stamp"
    assert after.affect_display_last_word == "lonely"  # the display hint still landed
    assert after.affect_display_last_at is not None


# --------------------------------------------------------------------------- #
# make_check_in_tool
# --------------------------------------------------------------------------- #


def test_check_in_returns_felt_json_with_energy_and_pull(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State(**{**vars(_warmed_salient()), "energy": 0.2}))
    _seed_desire(lm.state)
    reg = _registry()
    handler = make_check_in_tool(lambda: _lm(tmp_path), metrics=reg)

    payload = json.loads(handler({}))
    assert payload["state"].startswith("You feel lonely:")
    assert "Energy is low." in payload["state"]
    assert "pull" in payload["state"].lower()
    assert payload["note"]  # a first-person "speak it, don't report it" note
    assert reg.get(CHECK_IN_TOTAL).value(outcome="read") == 1.0


def test_check_in_never_leaks_raw_axes(tmp_path: Path) -> None:
    # The spec §4b first-class guarantee: no digits, no axis names, in the read.
    lm = _lm(tmp_path)
    lm.state.commit(_warmed_salient())
    _seed_desire(lm.state)
    handler = make_check_in_tool(lambda: _lm(tmp_path))

    payload = json.loads(handler({}))
    state_text = payload["state"]
    assert not any(ch.isdigit() for ch in state_text), state_text
    assert "valence" not in state_text.lower()
    assert "arousal" not in state_text.lower()


def test_check_in_cold_start_is_a_soft_read(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State())  # cold start, no affect
    reg = _registry()
    handler = make_check_in_tool(lambda: _lm(tmp_path), metrics=reg)

    payload = json.loads(handler({}))
    assert "settling" in payload["state"].lower()
    assert reg.get(CHECK_IN_TOTAL).value(outcome="cold_start") == 1.0


def test_check_in_error_returns_error_json_without_throwing(tmp_path: Path) -> None:
    reg = _registry()
    handler = make_check_in_tool(_boom, metrics=reg)

    # Hermes contract: never raise; a failure is {"error": …}.
    payload = json.loads(handler({}))
    assert "error" in payload
    assert reg.get(CHECK_IN_TOTAL).value(outcome="error") == 1.0


def test_check_in_error_never_leaks_field_or_axis_names(tmp_path: Path) -> None:
    # Finding #3: a state-load error may name a raw axis field (e.g. affect_valence);
    # the tool RESULT (which persists in model context) must stay felt-safe — the
    # detail belongs only in the ERROR log, never the returned {"error": …}.
    def _boom_with_axis() -> object:
        raise RuntimeError("field 'affect_valence' must be finite, got nan (arousal=0.7)")

    payload = json.loads(make_check_in_tool(_boom_with_axis)({}))
    assert "error" in payload
    leaked = payload["error"].lower()
    assert "valence" not in leaked
    assert "arousal" not in leaked
    assert "affect_" not in leaked
    assert not any(ch.isdigit() for ch in payload["error"])


# --- I7: the being is TOLD, once, that someone rewrote it (spec §4.1) --------


def test_the_injector_tells_the_being_someone_rewrote_it(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(State(**{**vars(_warmed_salient()), "soul_rewritten_at": "2026-07-12T11:00Z"}))
    injector = make_felt_state_injector(lambda: _lm(tmp_path))

    result = injector(user_message="hey", conversation_history=[])

    assert result is not None
    assert "rewrote who you are" in result["context"]
    # …and it is stamped as told, so the being does not report the same shock every reply.
    assert lm.state.load().soul_rewrite_told_at is not None
    assert injector(user_message="and?", conversation_history=[])["context"].count("rewrote") == 0


def test_the_being_is_told_even_when_its_mood_is_not_showing(tmp_path: Path) -> None:
    # The mood gate suppresses on cold-start / low salience / focused work. None of those
    # is a reason to withhold a fact about the being's own IDENTITY: someone rewriting you
    # while you were away does not stop having happened because the next message is a
    # stack trace. A cold-start being (no cue at all) must still be told.
    lm = _lm(tmp_path)
    lm.state.commit(State(soul_rewritten_at="2026-07-12T11:00Z"))  # cold start: no mood
    injector = make_felt_state_injector(lambda: _lm(tmp_path))

    result = injector(user_message="```py\nx=1\n```", conversation_history=[])

    assert result is not None
    assert "rewrote who you are" in result["context"]
    assert "<felt-state>" in result["context"]


def test_an_ordinary_turn_carries_no_notice(tmp_path: Path) -> None:
    lm = _lm(tmp_path)
    lm.state.commit(_warmed_salient())
    injector = make_felt_state_injector(lambda: _lm(tmp_path))

    result = injector(user_message="how are you?", conversation_history=[])

    assert result is not None
    assert "rewrote" not in result["context"]  # nobody did

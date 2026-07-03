"""Tests for the JSON file adapter — the safety-critical state store.

Every test injects a ``tmp_path`` base_dir; no real profile home is ever
touched, and no Hermes module is imported. The store's contract:

* atomic writes (tmp + ``os.replace``) so a partial ``state.json`` is never
  observable and a mid-write crash leaves the previous good file intact;
* a ``schema_version`` header, with a typed error on an unsupported version;
* graceful defaults on a missing file, typed errors on corrupt data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from lifemodel.state import (
    SCHEMA_VERSION,
    JsonStateStore,
    State,
    StateCorruptError,
    StateError,
    StatePort,
    StateSchemaError,
)

STATE_FILE = "state.json"


def _state_json(base: Path) -> Path:
    return base / STATE_FILE


def _tmp_files(base: Path) -> list[Path]:
    """Any lingering atomic-write temp files in the base dir."""
    return list(base.glob(".state-*"))


def test_store_satisfies_the_state_port(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    assert isinstance(store, StatePort)


def test_load_missing_file_returns_default_state(tmp_path: Path) -> None:
    assert JsonStateStore(tmp_path).load() == State()


def test_load_when_base_dir_absent_returns_default_state(tmp_path: Path) -> None:
    # The composition root injects a *computed* state dir that paths.py does not
    # create; load() must not require it to exist yet.
    absent = tmp_path / "does-not-exist-yet"
    assert JsonStateStore(absent).load() == State()
    assert not absent.exists()  # load() is read-only; it must not create the dir


def test_round_trip_commit_then_load_is_equal(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    state = State(
        pressure=2.75,
        energy=0.5,
        last_tick_at="2026-07-03T12:00:00Z",
        last_contact_at="2026-07-03T11:30:00Z",
        processed_signal_ids=["turn-1", "turn-2"],
    )
    store.commit(state)
    assert store.load() == state


def test_commit_creates_base_dir_and_human_readable_json(tmp_path: Path) -> None:
    base = tmp_path / "profile-home" / "lifemodel"
    JsonStateStore(base).commit(State(pressure=1.0))

    path = _state_json(base)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    # Human-readable: pretty-printed with the schema_version header first.
    assert "\n" in text
    data = json.loads(text)
    assert data["schema_version"] == SCHEMA_VERSION
    assert next(iter(data)) == "schema_version"


def test_commit_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    store.commit(State(pressure=1.0))
    assert _tmp_files(tmp_path) == []
    assert _state_json(tmp_path).exists()


def test_commit_emits_state_commit_event(tmp_path: Path) -> None:
    with capture_logs() as logs:
        JsonStateStore(tmp_path).commit(State())
    events = [e for e in logs if e.get("event") == "state_commit"]
    assert len(events) == 1
    assert events[0]["schema_version"] == SCHEMA_VERSION


def test_mid_write_failure_leaves_previous_state_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonStateStore(tmp_path)
    store.commit(State(pressure=1.0))  # a known-good baseline
    good_bytes = _state_json(tmp_path).read_bytes()

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("simulated disk failure during rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated disk failure"):
        store.commit(State(pressure=999.0))

    # The previous good file is byte-for-byte intact...
    assert _state_json(tmp_path).read_bytes() == good_bytes
    # ...the partial write never became observable...
    assert store.load().pressure == 1.0
    # ...and the temp file was cleaned up, not left lingering.
    assert _tmp_files(tmp_path) == []


def test_commit_survives_unavailable_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Directory fsync is best-effort (not portable everywhere); commit must
    # still succeed if the base dir cannot be opened for fsync.
    store = JsonStateStore(tmp_path)
    real_open = os.open

    def selective_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        # mkstemp opens the temp *file*; only the base *dir* fsync open fails.
        if str(path) == str(tmp_path):
            raise OSError("cannot open dir for fsync")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", selective_open)
    store.commit(State(pressure=1.0))
    assert store.load().pressure == 1.0
    assert _tmp_files(tmp_path) == []


# --- Finding 1: non-finite floats must never poison persisted state ---


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field_name", ["pressure", "energy"])
def test_commit_rejects_non_finite_float(tmp_path: Path, field_name: str, bad: float) -> None:
    # Fail-closed: a non-finite value raises a typed StateError *before* any
    # file is touched — no state.json is written, no temp file lingers.
    store = JsonStateStore(tmp_path)
    with pytest.raises(StateError):
        store.commit(State(**{field_name: bad}))
    assert not _state_json(tmp_path).exists()
    assert _tmp_files(tmp_path) == []


def test_failed_non_finite_commit_leaves_previous_state_intact(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    store.commit(State(pressure=1.0))  # known-good baseline
    good_bytes = _state_json(tmp_path).read_bytes()

    with pytest.raises(StateError):
        store.commit(State(pressure=float("nan")))

    assert _state_json(tmp_path).read_bytes() == good_bytes
    assert store.load().pressure == 1.0
    assert _tmp_files(tmp_path) == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_finite_tokens(tmp_path: Path, token: str) -> None:
    # json.loads accepts these non-standard tokens by default; the store must
    # reject the resulting non-finite floats as corrupt.
    _state_json(tmp_path).write_text(
        f'{{"schema_version": {SCHEMA_VERSION}, "pressure": {token}}}',
        encoding="utf-8",
    )
    with pytest.raises(StateCorruptError):
        JsonStateStore(tmp_path).load()


def test_normal_finite_floats_still_round_trip(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    state = State(pressure=3.5, energy=0.0)
    store.commit(state)
    assert store.load() == state


# --- Finding 2: invalid UTF-8 must honor the typed-error contract ---


def test_load_invalid_utf8_raises_corrupt(tmp_path: Path) -> None:
    _state_json(tmp_path).write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")
    with pytest.raises(StateCorruptError):
        JsonStateStore(tmp_path).load()


def test_load_rejects_newer_schema_version(tmp_path: Path) -> None:
    _state_json(tmp_path).write_text(
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "pressure": 0.0}),
        encoding="utf-8",
    )
    with pytest.raises(StateSchemaError):
        JsonStateStore(tmp_path).load()


def test_load_rejects_unknown_older_schema_version(tmp_path: Path) -> None:
    # No migrations in Phase 1: any non-matching version fails loud, not silent.
    _state_json(tmp_path).write_text(
        json.dumps({"schema_version": 0, "pressure": 0.0}), encoding="utf-8"
    )
    with pytest.raises(StateSchemaError):
        JsonStateStore(tmp_path).load()


def test_load_unparseable_file_raises_corrupt(tmp_path: Path) -> None:
    _state_json(tmp_path).write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(StateCorruptError):
        JsonStateStore(tmp_path).load()


def test_load_non_object_json_raises_corrupt(tmp_path: Path) -> None:
    _state_json(tmp_path).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(StateCorruptError):
        JsonStateStore(tmp_path).load()


def test_load_missing_schema_version_raises_corrupt(tmp_path: Path) -> None:
    _state_json(tmp_path).write_text(json.dumps({"pressure": 0.0}), encoding="utf-8")
    with pytest.raises(StateCorruptError):
        JsonStateStore(tmp_path).load()


def test_store_does_not_import_hermes(tmp_path: Path) -> None:
    store = JsonStateStore(tmp_path)
    store.commit(State(pressure=1.0))
    store.load()
    assert "hermes_constants" not in sys.modules
    assert not any(m == "hermes" or m.startswith("hermes.") for m in sys.modules)

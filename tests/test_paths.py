"""Unit tests for the pure per-profile path helpers (no Hermes import)."""

from __future__ import annotations

from pathlib import Path

from lifemodel.paths import STATE_DIR_NAME, state_dir


def test_state_dir_is_under_profile_home() -> None:
    home = Path("/var/hermes/profiles/nika")
    assert state_dir(home) == home / "workspace" / STATE_DIR_NAME


def test_state_dir_name_is_lifemodel() -> None:
    assert STATE_DIR_NAME == "lifemodel"


def test_state_dir_resolves_without_touching_disk(tmp_path: Path) -> None:
    result = state_dir(tmp_path)
    assert result == tmp_path / "workspace" / "lifemodel"
    # Resolving the path must not create it (creation is deferred to task 0.2).
    assert not result.exists()

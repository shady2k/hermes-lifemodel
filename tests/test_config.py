"""Tests for the plugin's tiny persisted JSON config (bead lm-j2w B2).

Two layers: the generic tolerant ``read_config``/``write_config`` pair, and the
focused log-level helpers built on top of it (``read_log_level``/
``write_log_level``), plus the ``loglevel`` command handler
(``set_log_level_for_dir``) that wires them to ``/lifemodel loglevel``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import lifemodel.log as lm_logging
from lifemodel.config import (
    CONFIG_FILENAME,
    read_config,
    read_log_level,
    set_log_level_for_dir,
    write_config,
    write_log_level,
)
from lifemodel.log import LOG_LEVEL_NAMES

# --- read_config / write_config ----------------------------------------------


def test_read_config_on_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert read_config(tmp_path) == {}


def test_read_config_on_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("", encoding="utf-8")
    assert read_config(tmp_path) == {}


def test_read_config_on_malformed_json_returns_empty_dict_without_raising(
    tmp_path: Path,
) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("{not valid json", encoding="utf-8")
    assert read_config(tmp_path) == {}


def test_read_config_on_non_dict_json_returns_empty_dict(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    assert read_config(tmp_path) == {}


def test_read_config_on_missing_directory_returns_empty_dict(tmp_path: Path) -> None:
    assert read_config(tmp_path / "does" / "not" / "exist") == {}


def test_read_config_on_invalid_utf8_bytes_returns_empty_dict_without_raising(
    tmp_path: Path,
) -> None:
    # A hand-edited/corrupted config.json with invalid UTF-8 bytes must never
    # take the plugin down at load — same tolerance as malformed JSON.
    (tmp_path / CONFIG_FILENAME).write_bytes(b"\xff\xfe{not even close to utf-8")
    assert read_config(tmp_path) == {}


def test_write_config_then_read_config_round_trips(tmp_path: Path) -> None:
    write_config(tmp_path, {"a": 1, "b": "two"})
    assert read_config(tmp_path) == {"a": 1, "b": "two"}


def test_write_config_creates_missing_parent_directories(tmp_path: Path) -> None:
    base_dir = tmp_path / "nested" / "workspace"
    write_config(base_dir, {"x": 1})
    assert read_config(base_dir) == {"x": 1}


def test_write_config_is_atomic_no_leftover_tmp_file(tmp_path: Path) -> None:
    write_config(tmp_path, {"a": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name != CONFIG_FILENAME]
    assert leftovers == []
    assert (tmp_path / CONFIG_FILENAME).exists()


def test_write_config_overwrites_existing_file(tmp_path: Path) -> None:
    write_config(tmp_path, {"a": 1})
    write_config(tmp_path, {"a": 2})
    assert read_config(tmp_path) == {"a": 2}


# --- read_log_level / write_log_level -----------------------------------------


def test_read_log_level_defaults_to_info_with_no_config(tmp_path: Path) -> None:
    assert read_log_level(tmp_path) == "info"


def test_write_log_level_then_read_log_level_round_trips(tmp_path: Path) -> None:
    write_log_level(tmp_path, "debug")
    assert read_log_level(tmp_path) == "debug"


def test_write_log_level_preserves_other_config_keys(tmp_path: Path) -> None:
    write_config(tmp_path, {"unrelated": "keep-me"})
    write_log_level(tmp_path, "warning")
    config = read_config(tmp_path)
    assert config["unrelated"] == "keep-me"
    assert config["log_level"] == "warning"


def test_write_log_level_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="debug"):
        write_log_level(tmp_path, "loud")
    # nothing was persisted
    assert read_config(tmp_path) == {}


def test_write_log_level_is_case_insensitive(tmp_path: Path) -> None:
    write_log_level(tmp_path, "DEBUG")
    assert read_log_level(tmp_path) == "debug"


def test_read_log_level_on_invalid_persisted_name_falls_back_to_default(
    tmp_path: Path,
) -> None:
    # A hand-edited config.json with a plausible-but-wrong level name (e.g. a
    # "warn"/"warning" typo) must degrade to the default rather than handing
    # back a string parse_log_level() would reject — read_log_level() must be
    # safe-by-construction for every downstream caller, not just the command
    # layer that has its own validation error path.
    write_config(tmp_path, {"log_level": "warn"})
    assert read_log_level(tmp_path) == "info"


# --- set_log_level_for_dir (the `/lifemodel loglevel` handler) ---------------


def test_set_log_level_for_dir_no_arg_returns_current_level(tmp_path: Path) -> None:
    write_log_level(tmp_path, "warning")
    message = set_log_level_for_dir(tmp_path, "")
    assert "warning" in message


def test_set_log_level_for_dir_no_arg_defaults_to_info_with_no_config(
    tmp_path: Path,
) -> None:
    message = set_log_level_for_dir(tmp_path, "")
    assert "info" in message


def test_set_log_level_for_dir_valid_arg_persists_and_echoes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guard against leaking the real (non-monkeypatched) log.configure() call
    # this triggers into later tests: monkeypatch reverts _effective_level to
    # its pre-test value on teardown regardless of what happens in between.
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    message = set_log_level_for_dir(tmp_path, "debug")
    assert read_log_level(tmp_path) == "debug"
    assert "info" in message  # old
    assert "debug" in message  # new
    assert "->" in message


def test_set_log_level_for_dir_valid_arg_applies_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    set_log_level_for_dir(tmp_path, "debug")
    assert lm_logging._effective_level == logging.DEBUG


def test_set_log_level_for_dir_invalid_arg_returns_usage_listing_valid_names(
    tmp_path: Path,
) -> None:
    message = set_log_level_for_dir(tmp_path, "loud")
    for name in LOG_LEVEL_NAMES:
        assert name in message


def test_set_log_level_for_dir_invalid_arg_does_not_change_persisted_level(
    tmp_path: Path,
) -> None:
    write_log_level(tmp_path, "warning")
    set_log_level_for_dir(tmp_path, "loud")
    assert read_log_level(tmp_path) == "warning"


def test_set_log_level_for_dir_invalid_arg_does_not_raise(tmp_path: Path) -> None:
    # The command boundary prefers a clean usage message over a raised
    # exception for a plainly-invalid argument.
    set_log_level_for_dir(tmp_path, "loud")  # must not raise


def test_config_file_is_valid_json_on_disk(tmp_path: Path) -> None:
    write_log_level(tmp_path, "error")
    raw = (tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert json.loads(raw) == {"log_level": "error"}

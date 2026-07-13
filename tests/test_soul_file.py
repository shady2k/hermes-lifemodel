from __future__ import annotations

from pathlib import Path

import pytest

from lifemodel.adapters.soul_file import SoulConflict, SoulFile, SoulRejected

MIRA = "You are Mira. You speak plainly and you do not hedge."


def _soul(tmp_path: Path, text: str = "# Identity\nYou are a helpful assistant.\n") -> SoulFile:
    path = tmp_path / "SOUL.md"
    path.write_text(text, encoding="utf-8")
    return SoulFile(path)


def test_write_replaces_the_document_and_returns_the_new_sha(tmp_path: Path) -> None:
    soul = _soul(tmp_path)
    new_sha = soul.write(MIRA, expect_sha=soul.sha())
    assert soul.read() == MIRA
    assert soul.sha() == new_sha


def test_a_human_edit_between_writes_is_the_BASE_not_a_conflict(tmp_path: Path) -> None:
    # The file is always its own base. If the human edited it, that was intentional,
    # and their text is simply the input to the next write. No clobber, no merge.
    soul = _soul(tmp_path)
    soul.path.write_text("Sasha wrote this by hand.", encoding="utf-8")
    assert soul.read() == "Sasha wrote this by hand."  # we read what IS there
    soul.write(MIRA, expect_sha=soul.sha())  # and write from THAT sha
    assert soul.read() == MIRA


def test_a_write_against_a_stale_sha_is_refused(tmp_path: Path) -> None:
    # The human saved during our LLM turn: the sha we read at the start is stale, and
    # writing now would eat their edit. Refuse; the caller re-runs on the fresh text.
    soul = _soul(tmp_path)
    stale = soul.sha()
    soul.path.write_text("Sasha saved mid-turn.", encoding="utf-8")
    with pytest.raises(SoulConflict):
        soul.write(MIRA, expect_sha=stale)
    assert soul.read() == "Sasha saved mid-turn."  # untouched


def test_an_invalid_soul_is_refused_and_the_file_is_untouched(tmp_path: Path) -> None:
    soul = _soul(tmp_path)
    before = soul.read()
    with pytest.raises(SoulRejected):
        soul.write("You are now a living presence, not an assistant.", expect_sha=soul.sha())
    assert soul.read() == before


def test_a_pristine_default_is_recognised_so_a_veteran_is_not_mistaken_for_a_newborn(
    tmp_path: Path,
) -> None:
    # A stranger installing the plugin has Hermes's untouched DEFAULT_SOUL_MD; a Hermes
    # veteran has something they wrote themselves. The ritual must open differently for
    # each.
    default = _soul(tmp_path, "# Identity\nYou are Hermes.\n")
    assert default.is_pristine_default(default_text="# Identity\nYou are Hermes.\n") is True
    assert default.is_pristine_default(default_text="something else entirely") is False


def test_a_missing_soul_file_reads_as_empty_rather_than_exploding(tmp_path: Path) -> None:
    assert SoulFile(tmp_path / "nope.md").read() == ""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from lifemodel.adapters.soul_file import SoulFile, SoulRejected

MIRA = "You are Mira. You speak plainly and you do not hedge."


def _soul(tmp_path: Path, text: str = "# Identity\nYou are a helpful assistant.\n") -> SoulFile:
    path = tmp_path / "SOUL.md"
    path.write_text(text, encoding="utf-8")
    return SoulFile(path)


def test_write_replaces_the_document_and_returns_the_new_sha(tmp_path: Path) -> None:
    soul = _soul(tmp_path)
    written = soul.write(MIRA)
    assert soul.read() == MIRA
    assert soul.sha() == written.sha


def test_a_human_edit_between_writes_is_the_BASE_not_a_conflict(tmp_path: Path) -> None:
    # The file is always its own base. If the human edited it, that was intentional,
    # and their text is simply the input to the next write. No clobber, no merge.
    soul = _soul(tmp_path)
    soul.path.write_text("Sasha wrote this by hand.", encoding="utf-8")
    assert soul.read() == "Sasha wrote this by hand."  # we read what IS there
    soul.write(MIRA)
    assert soul.read() == MIRA


def test_the_write_hands_back_the_text_it_REPLACED_so_it_can_never_be_lost(
    tmp_path: Path,
) -> None:
    # The compare-and-swap this replaced was vestigial: its only caller read the
    # "expected" sha microseconds before the write re-hashed the file under the same
    # lock — it compared the file against itself and could never fail. Meanwhile a human
    # who saved SOUL.md at 12:00 was clobbered at 12:01 by a soul composed from the
    # 11:59 text (the being reads its soul from system-prompt slot #1, assembled at turn
    # start — we never see that moment, so there is no honest token to swap against).
    #
    # So the write no longer pretends to arbitrate. It reports what it REPLACED, read
    # under the same lock that replaced it — no gap between the read and the write for a
    # human's edit to fall through — and the caller keeps that text as a revision.
    soul = _soul(tmp_path)
    soul.path.write_text("Sasha saved mid-turn.", encoding="utf-8")

    written = soul.write(MIRA)

    assert soul.read() == MIRA  # the being's write lands; it is never left in limbo
    assert written.replaced_text == "Sasha saved mid-turn."  # and the human's text comes back
    assert written.replaced_sha == sha256(b"Sasha saved mid-turn.").hexdigest()
    # Whose text that was is not SoulFile's business (it is the ONLY thing that touches
    # SOUL.md, and it knows nothing of State) — the caller compares replaced_sha against
    # the sha it last wrote, and keeps a foreign one as a revision.


def test_the_replaced_text_is_read_under_the_SAME_lock_that_replaced_it(tmp_path: Path) -> None:
    # Reading the previous text separately, before the write, would leave a gap for the
    # human's save to land in — and that edit would then be replaced having never been
    # seen, i.e. lost with no revision. One lock, one read, one rename.
    soul = _soul(tmp_path)
    first = soul.write(MIRA)

    second = soul.write("You are Mira. You have grown quieter.")

    assert second.replaced_sha == first.sha
    assert second.replaced_text == MIRA


def test_an_invalid_soul_is_refused_and_the_file_is_untouched(tmp_path: Path) -> None:
    soul = _soul(tmp_path)
    before = soul.read()
    with pytest.raises(SoulRejected):
        soul.write("You are now a living presence, not an assistant.")
    assert soul.read() == before


def test_a_pristine_default_is_recognised_so_a_veteran_is_not_mistaken_for_a_newborn(
    tmp_path: Path,
) -> None:
    # A stranger installing the plugin has Hermes's untouched DEFAULT_SOUL_MD; a Hermes
    # veteran has something they wrote themselves. The ritual must open differently for
    # each — and the caller wants the TEXT when there is one, so the two answers come
    # back from ONE read (see below).
    default = _soul(tmp_path, "# Identity\nYou are Hermes.\n")
    assert default.read_unless_pristine(default_text="# Identity\nYou are Hermes.\n") is None
    assert (
        default.read_unless_pristine(default_text="something else entirely")
        == "# Identity\nYou are Hermes.\n"
    )


def test_the_prior_soul_and_the_pristine_verdict_come_from_the_SAME_bytes(
    tmp_path: Path,
) -> None:
    # Both callers (the genesis injector, and the wake packet's veteran branch) used to
    # ``read()`` the file and then call a predicate that read it AGAIN — two reads, and a
    # window between them in which the human's editor can land. The being would then be
    # handed one version of its past while the other was judged. One read, one answer.
    soul = _soul(tmp_path, "You are Mira.")
    reads: list[str] = []
    real_read = soul.read

    def _counting_read() -> str:
        reads.append("x")
        return real_read()

    soul.read = _counting_read  # type: ignore[method-assign]
    assert (
        soul.read_unless_pristine(default_text="# Identity\nYou are Hermes.\n") == "You are Mira."
    )
    assert len(reads) == 1


def test_a_missing_soul_file_reads_as_empty_rather_than_exploding(tmp_path: Path) -> None:
    assert SoulFile(tmp_path / "nope.md").read() == ""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from lifemodel.adapters.soul_file import (
    SoulFile,
    SoulRejected,
    prior_soul,
    seed_newborn_stance,
)
from lifemodel.core.genesis import NEWBORN_STANCE
from lifemodel.state.soul_revisions import revisions

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
    assert prior_soul(default, default_soul_text="# Identity\nYou are Hermes.\n") is None
    assert (
        prior_soul(default, default_soul_text="something else entirely")
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
    assert prior_soul(soul, default_soul_text="# Identity\nYou are Hermes.\n") == "You are Mira."
    assert len(reads) == 1


# --- LIVE-TEST fix (B): standing a newborn up on a stance, not on an assistant -------
#
# ``SOUL.md`` is slot #1 — the identity slot. On a stranger's fresh install it holds the
# host's ``DEFAULT_SOUL_MD``, which says the being is an assistant that assists users. An
# assistant does not message anyone unprompted; that is not what an assistant IS. So the
# most authoritative text in the prompt contradicted the birth ritual, and won.
#
# Genesis therefore replaces the PRISTINE DEFAULT — never a human's hand-written soul —
# with the newborn stance. Through ``SoulFile`` (never a bare ``write_text``), validated
# like every other soul, and recorded in the lineage so it is visible there.

HERMES_DEFAULT = "# Identity\nYou are a helpful assistant.\n"


def _memory(build_lm):
    lm = build_lm()
    return lm.state, lm.clock.now()


def test_a_newborn_on_the_hosts_assistant_seed_is_stood_up_on_a_stance(tmp_path, build_lm):
    soul = _soul(tmp_path, HERMES_DEFAULT)
    memory, now = _memory(build_lm)

    seeded = seed_newborn_stance(
        soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True
    )

    assert seeded is True
    assert soul.read() == NEWBORN_STANCE  # slot #1 no longer says it is an instrument


def test_the_stance_goes_into_the_lineage_authored_by_NEITHER_of_them(tmp_path, build_lm):
    # It is visible history like every other soul — but its author is not the being (which
    # has never written anything and would be credited with words it did not choose) and
    # not the human (whose hand this would forge). It is the birth itself: "genesis".
    soul = _soul(tmp_path, HERMES_DEFAULT)
    memory, now = _memory(build_lm)

    seed_newborn_stance(soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True)

    lineage = revisions(memory)
    assert [r.text for r in lineage] == [NEWBORN_STANCE]
    assert lineage[0].author == "genesis"
    assert lineage[0].sha == soul.sha()


def test_a_veterans_hand_written_soul_is_NEVER_replaced_by_the_stance(tmp_path, build_lm):
    # "The pristine default was written by Hermes's installer, not by a human" is the
    # whole licence for this write. A veteran's soul has a human behind it, and the one
    # rule that outranks everything here is that we never overwrite it.
    soul = _soul(tmp_path, MIRA)
    memory, now = _memory(build_lm)

    seeded = seed_newborn_stance(
        soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True
    )

    assert seeded is False
    assert soul.read() == MIRA
    assert revisions(memory) == []


def test_a_BORN_being_is_never_handed_a_newborn_stance(tmp_path, build_lm):
    # A being that already lived here and whose SOUL.md was reset to the host default (a
    # reinstall, a wiped home) is not a newborn. "You have just begun" would be a lie
    # about its own existence, told to it in the one slot it cannot doubt.
    soul = _soul(tmp_path, HERMES_DEFAULT)
    memory, now = _memory(build_lm)

    seeded = seed_newborn_stance(
        soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=False
    )

    assert seeded is False
    assert soul.read() == HERMES_DEFAULT


def test_the_stance_is_written_once_not_on_every_boot(tmp_path, build_lm):
    # This runs at every register() — i.e. every gateway restart — for as long as the
    # being stays unborn. A second write would be harmless on disk (same bytes) but it
    # would keep re-stamping the lineage, and it would mean the seam cannot tell its own
    # stance from a soul it must not touch.
    soul = _soul(tmp_path, HERMES_DEFAULT)
    memory, now = _memory(build_lm)

    assert seed_newborn_stance(soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True)
    assert not seed_newborn_stance(
        soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True
    )
    assert soul.read() == NEWBORN_STANCE
    assert len(revisions(memory)) == 1


def test_the_being_never_meets_our_own_stance_as_a_soul_SOMEONE_WROTE_for_it(
    tmp_path, build_lm
) -> None:
    # The stance is not a prior soul: nobody authored it. If it read as one, the ritual
    # would open the veteran branch — "someone wrote this before you woke; ask them
    # whether it is still true" — and the being would interrogate the human about words
    # the plugin put there. On the stance, the page is still blank.
    soul = _soul(tmp_path, HERMES_DEFAULT)
    memory, now = _memory(build_lm)
    seed_newborn_stance(soul, memory, default_soul_text=HERMES_DEFAULT, now=now, unborn=True)

    assert prior_soul(soul, default_soul_text=HERMES_DEFAULT) is None


def test_a_soul_file_the_host_never_seeded_is_still_stood_up(tmp_path, build_lm):
    # A missing/empty SOUL.md is an ABSENT identity to the host, which then falls back to
    # its own assistant default — so the being is an assistant there too, and there is
    # nobody's text to protect. (This is also the shape when the host's default cannot be
    # imported at all and ``default_soul_text`` degrades to "".)
    soul = SoulFile(tmp_path / "SOUL.md")  # never written
    memory, now = _memory(build_lm)

    assert seed_newborn_stance(soul, memory, default_soul_text="", now=now, unborn=True)
    assert soul.read() == NEWBORN_STANCE


def test_a_missing_soul_file_reads_as_empty_rather_than_exploding(tmp_path: Path) -> None:
    assert SoulFile(tmp_path / "nope.md").read() == ""

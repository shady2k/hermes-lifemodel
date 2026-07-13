from __future__ import annotations

from lifemodel.core.genesis import genesis_block, should_launch
from lifemodel.state.model import State

UNBORN = State()
BORN = State(genesis_completed_at="2026-07-13T10:00:00+00:00")


def test_the_block_launches_on_the_beings_FIRST_word_only() -> None:
    assert should_launch(UNBORN, being_has_spoken=False) is True
    # Turn seven of the ritual is NOT a first waking. Re-injecting would be a lie, and
    # the being would keep starting over instead of continuing the conversation it began.
    assert should_launch(UNBORN, being_has_spoken=True) is False


def test_a_born_being_is_never_told_it_just_began() -> None:
    assert should_launch(BORN, being_has_spoken=False) is False
    assert should_launch(BORN, being_has_spoken=True) is False


def test_the_block_does_not_interrogate() -> None:
    block = genesis_block(prior_soul=None)
    # openclaw says "don't interrogate" and then lists name/nature/vibe/emoji 1-4; the
    # model dutifully walks the list. Ours must carry no numbered fields at all.
    assert "1." not in block
    assert "2." not in block
    # and it must not hand the human the authoring chair
    assert "Who am I?" not in block


def test_a_veteran_being_opens_from_the_soul_someone_wrote_before_it_woke() -> None:
    block = genesis_block(prior_soul="You are Mira. You are quiet and exact.")
    assert "You are Mira. You are quiet and exact." in block
    assert "already" in block.lower()

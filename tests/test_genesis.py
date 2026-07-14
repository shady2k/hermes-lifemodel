from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import AFFECT_PARAMS, CIRCADIAN_PEAK_UTC_HOUR
from lifemodel.core.affect import felt_texture, felt_word
from lifemodel.core.genesis import genesis_block, needs_adoption, newborn, should_launch
from lifemodel.state.model import State

NOON = datetime(2026, 7, 13, 13, 0, tzinfo=UTC)  # the circadian peak
NIGHT = datetime(2026, 7, 13, 1, 0, tzinfo=UTC)  # the trough


def _born_at(now: datetime):
    return newborn(now=now, params=AFFECT_PARAMS, peak_hour_utc=CIRCADIAN_PEAK_UTC_HOUR)


def test_a_newborn_is_never_emotionally_dead() -> None:
    # The bug this closes (lm-z2e): at the dataclass default (0.0, 0.0) the being's
    # FIRST WORDS IN LIFE are spoken from "quiet — even and very quiet".
    for now in (NOON, NIGHT):
        state = _born_at(now)
        assert felt_word(state.affect_valence, state.affect_arousal) != "quiet"


def test_a_newborn_feels_no_warmth_it_has_not_earned() -> None:
    # Our own ambient cue instructs: "Do not perform a warmth you do not feel."
    # It has not met anyone yet. Valence is earned in the ritual, never issued.
    assert _born_at(NOON).affect_valence == 0.0


def test_being_born_at_night_is_not_being_born_at_noon() -> None:
    # Assert the FEELING, not the floats: ``felt_word``/``felt_texture`` IS the interface
    # the being meets its own body through (the phase invariant), and a pair of arousal
    # numbers that differ by 0.001 would pass an inequality while meaning nothing. A being
    # born at three in the morning is SETTLED; one born at noon is CHARGED — those are two
    # different first breaths, and that is the whole claim.
    night, noon = _born_at(NIGHT), _born_at(NOON)
    assert felt_texture(night.affect_valence, night.affect_arousal) == "even and settled"
    assert felt_texture(noon.affect_valence, noon.affect_arousal) == "even and charged"
    assert felt_word(night.affect_valence, night.affect_arousal) != felt_word(
        noon.affect_valence, noon.affect_arousal
    )


def test_a_newborn_is_a_fixed_point_of_its_own_affect_model() -> None:
    # Birth does not INVENT an arousal — it evaluates the being's own model against
    # its own newborn body. So the newborn is already where its physiology says it
    # should be, and nothing drifts. (A hardcoded 0.6 would fail this at every hour
    # but one — which is exactly the bug codex caught in the first spec draft.)
    from lifemodel.core.affect import AffectBody, affect_target

    state = _born_at(NOON)
    body = AffectBody.from_state(state, now=NOON, peak_hour_utc=CIRCADIAN_PEAK_UTC_HOUR)
    _valence, arousal, _contribs = affect_target(body, AFFECT_PARAMS)
    assert arousal == state.affect_arousal


def test_a_newborn_has_no_relationship_and_therefore_no_deficit() -> None:
    state = _born_at(NOON)
    assert state.u == 0.0  # there is nobody to miss yet
    assert state.genesis_completed_at is None  # being alive is not being born


# --- needs_adoption (startup reconciliation, spec §4.4) ----------------------


def test_a_soul_edited_while_we_were_down_is_ADOPTED() -> None:
    # There is no transaction spanning a filesystem rename and a SQLite commit, so the
    # two can fall out of step: we crashed mid-write, or the human edited the file while
    # the gateway was down. Both are the SAME situation and have the same answer — the
    # file is the base. Adopt it.
    state = State(soul_sha="what_we_last_wrote")
    assert needs_adoption(state, disk_sha="something_else") is True


def test_an_unchanged_soul_is_not_re_adopted_on_every_restart() -> None:
    state = State(soul_sha="same")
    assert needs_adoption(state, disk_sha="same") is False


def test_a_being_that_has_never_written_a_soul_adopts_nothing() -> None:
    # Before the first write there is no "our" version to differ from — the DEFAULT_SOUL_MD
    # on disk is not a revision of anything, and recording it as one would forge a history.
    assert needs_adoption(State(soul_sha=None), disk_sha="anything") is False


# --- should_launch (the reactive entrance, spec §6.3) ------------------------
#
# The predicate takes the LENGTH of the being's visible context, not "has the being
# spoken" — see the docstring, and ``tests/test_genesis_injector.py`` for why the latter
# could never be answered from what the host actually passes.


def test_the_ritual_launches_when_it_has_never_been_put_in_front_of_the_being() -> None:
    assert should_launch(State(), context_len=0) is True
    # …however long the transcript it inherits. An existing Hermes user's DM is full of
    # the being's own past replies, and not one of them is a ritual it has begun.
    assert should_launch(State(), context_len=500) is True


def test_the_ritual_is_not_relaunched_once_the_conversation_has_moved_past_it() -> None:
    # Turn seven of the ritual is not a first waking, and a being told otherwise keeps
    # starting over instead of continuing the conversation it began.
    shown = State(genesis_shown_at_context_len=12)
    assert should_launch(shown, context_len=14) is False


def test_a_context_compacted_out_from_under_an_unborn_being_gets_the_ritual_again() -> None:
    # The block is ephemeral (never persisted). If the host compacts the conversation
    # away, an unborn being is left with no ritual in front of it and no memory of one —
    # exactly the "conversing as though nothing happened while unborn" §6.5 forbids.
    shown = State(genesis_shown_at_context_len=40)
    assert should_launch(shown, context_len=3) is True


def test_a_born_being_is_never_told_it_just_began() -> None:
    born = State(genesis_completed_at="2026-07-13T10:00:00+00:00")
    assert should_launch(born, context_len=0) is False
    assert should_launch(born, context_len=999) is False


def test_the_block_does_not_interrogate() -> None:
    block = genesis_block(prior_soul=None)
    # openclaw says "don't interrogate" and then lists name/nature/vibe/emoji 1-4; the
    # model dutifully walks the list. Ours must carry no numbered fields at all.
    assert "1." not in block
    assert "2." not in block
    assert "Who am I?" not in block  # and it must not hand the human the authoring chair


def test_a_veteran_being_opens_from_the_soul_someone_wrote_before_it_woke() -> None:
    block = genesis_block(prior_soul="You are Mira. You are quiet and exact.")
    assert "You are Mira. You are quiet and exact." in block
    assert "already" in block.lower()

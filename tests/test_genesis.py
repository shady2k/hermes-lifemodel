from __future__ import annotations

from datetime import UTC, datetime

from lifemodel.composition import AFFECT_PARAMS, CIRCADIAN_PEAK_UTC_HOUR
from lifemodel.core.affect import felt_word
from lifemodel.core.genesis import newborn

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
    assert _born_at(NIGHT).affect_arousal < _born_at(NOON).affect_arousal


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

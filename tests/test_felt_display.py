"""Unit tests for :mod:`lifemodel.core.felt_display` — the reactive felt-state gate,
its language-independent detectors, and the two prose composers (lm-ukc.4 / .4.1).

All pure and Hermes-free: the gate ``decide`` is suppression-first (warmed →
salient → not-task → changed|cooldown), the detectors read robust BEHAVIORAL
signals only (zero language detection), and neither composer ever emits a raw
axis number — the first-class "feeling, not sensor" guarantee (spec §4b).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifemodel.core.desire_view import build_contact_desire
from lifemodel.core.felt_display import (
    DEFAULT_FELT_DISPLAY_PARAMS,
    Decision,
    FeltDisplayParams,
    RecentMessage,
    TurnSignals,
    compose_light_cue,
    compose_self_read,
    cooldown_elapsed,
    decide,
    felt_changed,
    is_salient,
    is_task_context,
    warmed,
)
from lifemodel.domain.objects import DesireState
from lifemodel.state.model import State

_NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)


def _warmed_state(**kw: object) -> State:
    """A warmed being (affect off the cold-start origin, stamp present)."""
    base: dict[str, object] = {
        "affect_valence": -0.6,
        "affect_arousal": 0.3,
        "affect_updated_at": "2026-07-12T11:30:00+00:00",
    }
    base.update(kw)
    return State(**base)  # type: ignore[arg-type]


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# --- params ----------------------------------------------------------------


def test_params_are_frozen_with_calibratable_defaults() -> None:
    p = FeltDisplayParams()
    assert p is not DEFAULT_FELT_DISPLAY_PARAMS or p == DEFAULT_FELT_DISPLAY_PARAMS
    assert p.salience_threshold > 0
    assert p.cooldown_min > 0
    assert p.long_paste_chars > 0
    assert p.task_window > 0
    # frozen dataclass — no mutation
    import dataclasses

    assert dataclasses.is_dataclass(p)


# --- warmed ----------------------------------------------------------------


def test_cold_start_is_not_warmed() -> None:
    # A brand-new being (no affect derived yet) never surfaces a cue.
    assert warmed(State(), DEFAULT_FELT_DISPLAY_PARAMS) is False


def test_affect_still_pinned_near_origin_is_not_warmed() -> None:
    # Stamp present but affect barely moved off 0/0 → still cold.
    s = State(
        affect_valence=0.02, affect_arousal=0.03, affect_updated_at="2026-07-12T11:59:00+00:00"
    )
    assert warmed(s, DEFAULT_FELT_DISPLAY_PARAMS) is False


def test_affect_off_the_origin_is_warmed() -> None:
    assert warmed(_warmed_state(), DEFAULT_FELT_DISPLAY_PARAMS) is True


# --- is_salient ------------------------------------------------------------


def test_mild_content_is_below_salience() -> None:
    # A soft pleasant tint (low magnitude) reads as empty retrieval, not salient.
    assert is_salient(0.2, 0.35, DEFAULT_FELT_DISPLAY_PARAMS) is False


def test_deep_valence_is_salient() -> None:
    assert is_salient(-0.6, 0.30, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_high_arousal_at_neutral_valence_is_salient() -> None:
    # restless: near-zero valence but strongly keyed-up — salient via the arousal arm.
    assert is_salient(0.0, 0.75, DEFAULT_FELT_DISPLAY_PARAMS) is True


# --- is_task_context (robust behavioral signals only) ----------------------


def test_task_context_recent_tool_calls() -> None:
    turn = TurnSignals(
        user_message="ok thanks",
        recent_messages=(RecentMessage(role="assistant", text="", has_tool_calls=True),),
    )
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_code_fence() -> None:
    turn = TurnSignals(user_message="fix this:\n```python\nprint(1)\n```")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_unified_diff() -> None:
    turn = TurnSignals(user_message="@@ -1,3 +1,4 @@\n-old\n+new")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_stack_trace() -> None:
    trace = 'Traceback (most recent call last):\n  File "x.py", line 3, in <module>\n    raise'
    turn = TurnSignals(user_message=trace)
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_long_paste() -> None:
    turn = TurnSignals(user_message="x" * (DEFAULT_FELT_DISPLAY_PARAMS.long_paste_chars + 1))
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_long_paste_in_prior_turn_then_short_followup() -> None:
    # A big paste last turn, then a short "continue" — still focused work, so the
    # long-paste check must span the whole window, not just the current message.
    big = "x" * (DEFAULT_FELT_DISPLAY_PARAMS.long_paste_chars + 1)
    turn = TurnSignals(
        user_message="what about part 2?",
        recent_messages=(RecentMessage(role="user", text=big, has_tool_calls=False),),
    )
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_task_context_json_block() -> None:
    turn = TurnSignals(user_message='{"status": "failed", "code": "boom"}')
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is True


def test_relational_message_mentioning_a_file_is_not_task() -> None:
    # THE false-positive guard (spec §5/§10): a warm reply that merely NAMES a file
    # must NOT suppress the mood — only structural work markers do.
    turn = TurnSignals(user_message="I loved what you wrote in poem.txt yesterday, how are you?")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is False


def test_plain_greeting_is_not_task() -> None:
    turn = TurnSignals(user_message="hey, how are you feeling today?")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is False


def test_turn_signals_from_hook_windows_and_flags_tool_calls() -> None:
    history = [
        {"role": "user", "content": "old one"},
        {"role": "assistant", "content": "sure", "tool_calls": [{"function": {"name": "grep"}}]},
        {"role": "user", "content": "and this"},
    ]
    turn = TurnSignals.from_hook("now", history, window=2)
    assert turn.user_message == "now"
    assert len(turn.recent_messages) == 2  # windowed to the last 2
    assert any(m.has_tool_calls for m in turn.recent_messages)


def test_turn_signals_from_hook_is_defensive_about_shape() -> None:
    # Untrusted host payload: non-dict entries / missing keys never raise.
    turn = TurnSignals.from_hook("hi", ["not a dict", {"role": "user"}, 42], window=6)
    assert turn.user_message == "hi"
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS) is False


# --- felt_changed / cooldown_elapsed ---------------------------------------


def test_felt_changed_true_on_word_shift_and_first_ever_show() -> None:
    s = _warmed_state()  # word == "lonely"
    assert felt_changed(State(**{**vars(s), "affect_display_last_word": None})) is True
    assert felt_changed(State(**{**vars(s), "affect_display_last_word": "wistful"})) is True
    assert felt_changed(State(**{**vars(s), "affect_display_last_word": "lonely"})) is False


def test_cooldown_elapsed_never_shown_or_past_window() -> None:
    p = DEFAULT_FELT_DISPLAY_PARAMS
    assert cooldown_elapsed(State(), p, _NOW) is True  # never shown
    recent = _iso(_NOW - timedelta(minutes=1))
    assert cooldown_elapsed(State(affect_display_last_at=recent), p, _NOW) is False
    old = _iso(_NOW - timedelta(minutes=p.cooldown_min + 5))
    assert cooldown_elapsed(State(affect_display_last_at=old), p, _NOW) is True


# --- decide (suppression-first order) --------------------------------------


def _non_task() -> TurnSignals:
    return TurnSignals(user_message="how are you?")


def test_decide_cold_start_beats_salience() -> None:
    # Cold-start is checked FIRST: even salient axes stay silent until warmed.
    cold = State(affect_valence=-0.7, affect_arousal=0.2)  # salient axes, no stamp
    assert decide(cold, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is Decision.NOT_WARMED


def test_decide_warmed_but_mild_is_not_salient() -> None:
    mild = _warmed_state(affect_valence=0.15, affect_arousal=0.35)
    assert decide(mild, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is Decision.NOT_SALIENT


def test_decide_task_context_suppresses_even_when_salient() -> None:
    turn = TurnSignals(user_message="```py\nx=1\n```")
    assert decide(_warmed_state(), turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is Decision.TASK


def test_decide_light_on_felt_change() -> None:
    s = _warmed_state(affect_display_last_word="wistful")  # current word "lonely" != last
    d = decide(s, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW)
    assert d is Decision.LIGHT
    assert d.shows is True


def test_decide_light_on_cooldown_when_unchanged() -> None:
    old = _iso(_NOW - timedelta(minutes=DEFAULT_FELT_DISPLAY_PARAMS.cooldown_min + 5))
    s = _warmed_state(affect_display_last_word="lonely", affect_display_last_at=old)
    assert decide(s, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is Decision.LIGHT


def test_decide_cooldown_unchanged_suppresses_repeat() -> None:
    recent = _iso(_NOW - timedelta(minutes=1))
    s = _warmed_state(affect_display_last_word="lonely", affect_display_last_at=recent)
    d = decide(s, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW)
    assert d is Decision.COOLDOWN_UNCHANGED
    assert d.shows is False


# --- compose_light_cue (ambient envelope) ----------------------------------


def test_light_cue_mirrors_memory_envelope_and_is_prose_only() -> None:
    s = _warmed_state(affect_valence=-0.3, affect_arousal=0.4)  # texture "tender and settled"
    block = compose_light_cue(s)
    assert block.startswith("<felt-state>")
    assert block.rstrip().endswith("</felt-state>")
    assert "[System note:" in block
    assert "Do not mention or explain it unless the user" in block
    assert "focused work" in block  # the task let-it-pass line
    assert "tender and settled" in block
    # NO raw axes — never a number, never the axis names.
    assert not any(ch.isdigit() for ch in block)
    assert "valence" not in block.lower()
    assert "arousal" not in block.lower()


# --- compose_self_read (check_in) ------------------------------------------


def test_self_read_is_felt_prose_with_energy_and_pull() -> None:
    desire = build_contact_desire(state=DesireState.ACTIVE, salience=2.0)
    s = _warmed_state(affect_valence=-0.3, affect_arousal=0.4, energy=0.2)
    read = compose_self_read(s, desire=desire)
    assert read.startswith("You feel wistful: tender and settled.")
    assert "Energy is low." in read
    assert "pull" in read.lower()  # the strongest live desire, in prose


def test_self_read_energy_buckets() -> None:
    d = build_contact_desire(state=DesireState.ACTIVE, salience=1.0)
    assert "Energy is low." in compose_self_read(_warmed_state(energy=0.1), desire=d)
    assert "Energy is steady." in compose_self_read(_warmed_state(energy=0.5), desire=d)
    # Top bucket is "full", not "bright" — `bright` is already a felt WORD, and the two
    # collided in the read ("You feel bright: … Energy is bright.").
    assert "Energy is full." in compose_self_read(_warmed_state(energy=0.95), desire=d)


def test_self_read_no_live_desire_reads_calm_pull() -> None:
    read = compose_self_read(_warmed_state(), desire=None)
    assert "nothing" in read.lower()


def test_self_read_cold_start_is_a_soft_read() -> None:
    read = compose_self_read(State(), desire=None)
    assert "settling" in read.lower()
    assert not any(ch.isdigit() for ch in read)


def test_self_read_never_leaks_raw_axes_across_many_states() -> None:
    # THE first-class guarantee (spec §4b, risk #1): no digits, no axis names, ever —
    # not for calm, not for deep/keyed states.
    d = build_contact_desire(state=DesireState.ACTIVE, salience=3.0)
    for v in (-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9):
        for a in (0.05, 0.3, 0.5, 0.85):
            s = _warmed_state(affect_valence=v, affect_arousal=a, energy=0.4)
            for read in (compose_self_read(s, desire=d), compose_self_read(s, desire=None)):
                assert not any(ch.isdigit() for ch in read), read
                assert "valence" not in read.lower()
                assert "arousal" not in read.lower()

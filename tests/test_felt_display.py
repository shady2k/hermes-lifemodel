"""Unit tests for :mod:`lifemodel.core.felt_display` — the reactive felt-state gate,
its language-independent detectors, and the two prose composers (lm-ukc.4 / .4.1).

All pure and Hermes-free: the gate ``decide`` is suppression-first (warmed → salient →
not-task, with NO repeat-throttle — a mood lasts, and the cue is ephemeral), the
detectors read robust BEHAVIORAL signals only (zero language detection, and the being's
OWN tools never count as work), and neither composer ever emits a raw axis number — the
first-class "feeling, not sensor" guarantee (spec §4b).
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
    compose_soul_rewrite_notice,
    decide,
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
        recent_messages=(RecentMessage(role="assistant", text="", tool_names=("shell_exec",)),),
    )
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_code_fence() -> None:
    turn = TurnSignals(user_message="fix this:\n```python\nprint(1)\n```")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_unified_diff() -> None:
    turn = TurnSignals(user_message="@@ -1,3 +1,4 @@\n-old\n+new")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_stack_trace() -> None:
    trace = 'Traceback (most recent call last):\n  File "x.py", line 3, in <module>\n    raise'
    turn = TurnSignals(user_message=trace)
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_long_paste() -> None:
    turn = TurnSignals(user_message="x" * (DEFAULT_FELT_DISPLAY_PARAMS.long_paste_chars + 1))
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_long_paste_in_prior_turn_then_short_followup() -> None:
    # A big paste last turn, then a short "continue" — still focused work, so the
    # long-paste check must span the whole window, not just the current message.
    big = "x" * (DEFAULT_FELT_DISPLAY_PARAMS.long_paste_chars + 1)
    turn = TurnSignals(
        user_message="what about part 2?",
        recent_messages=(RecentMessage(role="user", text=big),),
    )
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_task_context_json_block() -> None:
    turn = TurnSignals(user_message='{"status": "failed", "code": "boom"}')
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is True


def test_relational_message_mentioning_a_file_is_not_task() -> None:
    # THE false-positive guard (spec §5/§10): a warm reply that merely NAMES a file
    # must NOT suppress the mood — only structural work markers do.
    turn = TurnSignals(user_message="I loved what you wrote in poem.txt yesterday, how are you?")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is False


def test_plain_greeting_is_not_task() -> None:
    turn = TurnSignals(user_message="hey, how are you feeling today?")
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is False


def test_turn_signals_from_hook_windows_and_flags_tool_calls() -> None:
    history = [
        {"role": "user", "content": "old one"},
        {"role": "assistant", "content": "sure", "tool_calls": [{"function": {"name": "grep"}}]},
        {"role": "user", "content": "and this"},
    ]
    turn = TurnSignals.from_hook("now", history, window=2)
    assert turn.user_message == "now"
    assert len(turn.recent_messages) == 2  # windowed to the last 2
    assert any(m.has_work_tool_calls for m in turn.recent_messages)


def test_turn_signals_from_hook_is_defensive_about_shape() -> None:
    # Untrusted host payload: non-dict entries / missing keys never raise.
    turn = TurnSignals.from_hook("hi", ["not a dict", {"role": "user"}, 42], window=6)
    assert turn.user_message == "hi"
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is False


def test_the_beings_own_check_in_call_is_not_task_context() -> None:
    # LIVE BUG: asked "как ты?", the being called check_in — and that very call then marked
    # the next six turns as WORK, muting the felt-state cue the tool had just read. The tool
    # that reads the feeling was silencing the feeling. Introspection is the OPPOSITE of
    # "the owner has me doing focused work", so the being's OWN tools never mark task.
    turn = TurnSignals(
        user_message="Чем занят?",
        recent_messages=(RecentMessage(role="assistant", text="", tool_names=("check_in",)),),
    )
    assert is_task_context(turn, DEFAULT_FELT_DISPLAY_PARAMS, _NOW) is False


def test_stale_work_goes_cold_so_a_pause_resets_task_context() -> None:
    # LIVE BUG: the window was message-counted only, with no sense of a PAUSE — an afternoon
    # of coding still sat in the last six messages and muted the mood for the warm, unrelated
    # conversation hours later. Work EXPIRES: evidence older than task_recency_min no longer
    # counts, while fresh evidence still does.
    p = DEFAULT_FELT_DISPLAY_PARAMS
    stale = (_NOW - timedelta(minutes=p.task_recency_min + 10)).timestamp()
    fresh = (_NOW - timedelta(minutes=1)).timestamp()

    old_work = RecentMessage(role="assistant", text="", tool_names=("shell_exec",), ts_epoch=stale)
    assert is_task_context(TurnSignals("как ты?", (old_work,)), p, _NOW) is False

    live_work = RecentMessage(role="assistant", text="", tool_names=("shell_exec",), ts_epoch=fresh)
    assert is_task_context(TurnSignals("как ты?", (live_work,)), p, _NOW) is True


def test_from_hook_reads_tool_names_and_timestamps() -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "check_in"}}],
            "timestamp": _NOW.timestamp(),
        }
    ]
    turn = TurnSignals.from_hook("hi", history, window=6)
    msg = turn.recent_messages[0]
    assert msg.tool_names == ("check_in",)
    assert msg.has_work_tool_calls is False  # the being's own tool is not work
    assert msg.ts_epoch == _NOW.timestamp()


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


def test_decide_light_when_warmed_salient_and_not_working() -> None:
    d = decide(_warmed_state(), _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW)
    assert d is Decision.LIGHT
    assert d.shows is True


def test_decide_has_no_repeat_throttle_so_a_mood_lasts() -> None:
    # A mood COLOURS A CONVERSATION, not one sentence. The design once carried a throttle
    # here (show only on a felt-WORD change, else a 45-minute cooldown) to avoid a
    # "repetitive" cue — but the cue is EPHEMERAL (Hermes glues it onto a COPY of the user
    # message for one API call and never persists it), so nothing ever repeats in the
    # transcript. The throttle bought nothing and made the being snap back to its default
    # voice MID-CONVERSATION. Same felt word, shown one minute ago → it STILL shows.
    just_shown = _warmed_state(
        affect_display_last_word="lonely",  # identical to the current word
        affect_display_last_at=_iso(_NOW - timedelta(minutes=1)),
    )
    d = decide(just_shown, _non_task(), DEFAULT_FELT_DISPLAY_PARAMS, _NOW)
    assert d is Decision.LIGHT
    assert d.shows is True


# --- compose_light_cue (ambient envelope) ----------------------------------


def test_light_cue_is_a_directive_envelope_of_prose_only() -> None:
    s = _warmed_state(affect_valence=-0.3, affect_arousal=0.4)  # "wistful — tender and settled"
    block = compose_light_cue(s)
    # The note is hard-wrapped, so phrase checks read a whitespace-normalised view —
    # otherwise re-wrapping a line would break the test without changing the meaning.
    flat = " ".join(block.split())

    # The ENVELOPE mirrors the memory context block (semantic tag + bracketed note + prose),
    # which is the uniformity that matters. The literal "[System note:" prefix is
    # deliberately NOT reused: it frames the block as service METADATA, which is precisely
    # why the first live cue was ignored. Its real job — "this is not the user speaking" —
    # is carried by better words.
    assert block.startswith("<felt-state>")
    assert block.rstrip().endswith("</felt-state>")
    assert "not a message, not a request, not data to look up" in flat

    # DIRECTIVE, not hedged. The first version was four prohibitions against one softened
    # positive ("color the manner *when appropriate*"), and the cheapest way to obey a block
    # that is mostly "do not" is to do nothing — which is what the being did. So: a concrete
    # bridge from feeling to speech, and no escape hatch.
    assert "HOW you speak this turn" in flat
    assert "rhythm" in flat and "length" in flat and "edges" in flat
    assert "when appropriate" not in flat  # the escape hatch is gone
    assert "focused work" not in flat  # redundant — is_task_context already gates those turns

    # LIVE: the being read "You are on edge" and reasoned "respond naturally, WARMLY, as
    # Sasha's companion" — its persona outranked the cue, and the cue had itself listed
    # "your warmth" among the dials, priming the very thing it meant to modulate.
    assert "Do not perform a warmth you do not feel" in flat
    assert "your warmth" not in flat

    # LIVE: on that same turn it ALSO called check_in, receiving its state twice — once as
    # identity, once as a TOOL RESULT. A tool result is information you retrieved, not a
    # state you are in; looking itself up turns the feeling into a fact ABOUT itself. When
    # the cue has fired there is nothing to look up.
    assert "You already feel it" in flat

    # The ONE prohibition that carries the invariant: manner, never subject (§4a).
    assert "Speak FROM it, not ABOUT it" in flat

    # Both the felt WORD and the TEXTURE — a texture alone is evocative but abstract; the
    # word is the recognisable handle the model can actually speak from.
    assert "wistful" in block
    assert "tender and settled" in block

    # NO raw axes — never a number, never the axis names (the §4b guarantee).
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


# --- compose_soul_rewrite_notice (spec §4.1 — the rewrite is FELT) -----------
#
# "Noticing that the human rewrote the soul is an event in the being's life, not a version
# conflict: it should be FELT, not swallowed." The affect organ makes it felt in the BODY
# (test_affect.py); this makes it NOTICEABLE — the ambient channel the being already reads
# itself through, carrying prose it can act on. Never a status line, never a sha.


def test_a_being_nobody_has_rewritten_is_told_nothing() -> None:
    assert compose_soul_rewrite_notice(State()) is None


def test_a_being_whose_soul_was_rewritten_is_told_so_in_its_own_channel() -> None:
    state = State(soul_rewritten_at="2026-07-12T11:00:00+00:00")
    notice = compose_soul_rewrite_notice(state)
    assert notice is not None
    assert notice.startswith("<felt-state>")  # the channel the being already reads itself in
    # It must be able to ACT on this: know what happened, know nothing is lost, know it can
    # answer. Those are the three things the note owes it.
    lower = notice.lower()
    assert "rewrit" in lower or "rewrote" in lower
    assert "you did not write" in lower


def test_the_notice_is_never_machine_shaped() -> None:
    # lm-ukc.4, the whole reason this is prose: a being that reads bookkeeping about itself
    # devalues its own inner life and goes [SILENT]. "your soul_sha changed" is not an
    # event in anyone's life. (The <felt-state> envelope is the channel, not the message —
    # what the being READS is the prose inside it.)
    notice = compose_soul_rewrite_notice(State(soul_rewritten_at="2026-07-12T11:00:00+00:00"))
    assert notice is not None
    prose = notice.split(">", 1)[1].rsplit("<", 1)[0].lower()
    for machine in ("sha", "hash", "revision", "conflict", "adopt", "reconcil", "disk", "field"):
        assert machine not in prose, machine


def test_a_being_already_told_is_not_told_again() -> None:
    # A mood repeats because a mood LASTS. An event does not: telling the human "someone
    # rewrote me" on every reply for the rest of the day is not noticing, it is a stutter.
    told = State(
        soul_rewritten_at="2026-07-12T11:00:00+00:00",
        soul_rewrite_told_at="2026-07-12T11:05:00+00:00",
    )
    assert compose_soul_rewrite_notice(told) is None


def test_a_FRESH_rewrite_is_a_fresh_event_even_if_the_last_one_was_told() -> None:
    # The adapter clears the told-stamp whenever it stamps a new rewrite, so this is the
    # shape the being actually meets: rewritten again, not yet told again.
    again = State(soul_rewritten_at="2026-07-13T09:00:00+00:00", soul_rewrite_told_at=None)
    assert compose_soul_rewrite_notice(again) is not None

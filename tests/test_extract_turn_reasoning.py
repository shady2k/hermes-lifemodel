"""``_extract_turn_reasoning`` (lm-hg7) — the being's chain of thought for THIS
turn, pulled from ``conversation_history`` the way ``agent/turn_finalizer.py`` does.

The reasoning is not a top-level ``post_llm_call`` kwarg; it rides on the assistant
message (``msg["reasoning"]``). We take the MOST RECENT non-empty reasoning, walking
backwards but STOPPING at the ``user`` message that opened the turn, so a prior turn's
reasoning never leaks in. Defensive on the untrusted host shape → ``""``.
"""

from __future__ import annotations

from lifemodel.hooks import _extract_turn_reasoning


def test_takes_the_most_recent_assistant_reasoning_in_the_turn() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "reasoning": "they greeted me; be warm", "content": "Привет."},
    ]
    assert _extract_turn_reasoning(history) == "they greeted me; be warm"


def test_takes_the_tool_call_step_reasoning_when_the_final_answer_step_has_none() -> None:
    # Many providers emit reasoning on the tool-call step and leave the final
    # answer with reasoning=None — walking backwards must not stop at the first
    # (reasoning-less) assistant and silently drop legitimate same-turn reasoning.
    history = [
        {"role": "user", "content": "what's the weather"},
        {"role": "assistant", "reasoning": "need the weather tool", "tool_calls": [{}]},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "reasoning": None, "content": "It's sunny."},
    ]
    assert _extract_turn_reasoning(history) == "need the weather tool"


def test_stops_at_the_turn_boundary_and_never_crosses_into_a_prior_turn() -> None:
    history = [
        {"role": "assistant", "reasoning": "PRIOR turn reasoning", "content": "earlier"},
        {"role": "user", "content": "this turn's message"},  # boundary
        {"role": "assistant", "reasoning": None, "content": "ok"},
    ]
    # This turn's only assistant has no reasoning; the prior turn's is walled off.
    assert _extract_turn_reasoning(history) == ""


def test_empty_when_no_assistant_reasoning() -> None:
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Привет."},  # no reasoning key
    ]
    assert _extract_turn_reasoning(history) == ""


def test_defensive_on_untrusted_shapes() -> None:
    assert _extract_turn_reasoning(None) == ""
    assert _extract_turn_reasoning("not a list") == ""
    assert _extract_turn_reasoning([None, 42, "x"]) == ""  # non-dict entries skipped, no crash


def test_stringifies_a_non_str_reasoning_payload() -> None:
    history = [{"role": "assistant", "reasoning": ["a", "b"]}]
    assert _extract_turn_reasoning(history) == "['a', 'b']"

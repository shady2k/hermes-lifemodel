from __future__ import annotations

import re
from datetime import UTC, datetime

from lifemodel.core.thought_view import build_thought
from lifemodel.core.wake_packet import (
    GUIDANCE,
    RECENT_THOUGHTS_HEADER,
    THOUGHTS_RENDER_LIMIT,
    ProactivePrompt,
    build_wake_packet,
    render_situational_brief,
)

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)


def test_packet_carries_desire_frame_and_guidance() -> None:
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="corr-1")
    assert isinstance(p, ProactivePrompt)
    assert GUIDANCE in p.prompt
    # the desire-frame phrasing for this band appears in the prompt
    assert "мыслях о нём" in p.prompt or "услышать, как он" in p.prompt
    assert p.correlation_id == "corr-1"
    assert p.projection_id.startswith("contact.")


def test_packet_has_no_raw_numbers() -> None:
    p = build_wake_packet(value=3.4, theta=1.0, correlation_id="c")
    assert not re.search(r"\d", p.prompt)  # never leaks the value/hours


def test_guidance_permits_silence_and_owns_the_wish() -> None:
    # the guidance must invite [SILENT] and frame the motive as desire, not a timer
    assert "[SILENT]" in GUIDANCE
    assert "хочешь" in GUIDANCE.lower()


# --- lm-27n.6: Recent Thoughts render (behavior-neutral when empty) ----------


def test_no_thoughts_is_byte_identical_to_before() -> None:
    # The load-bearing behavior-neutrality proof: with no thoughts the prompt is
    # byte-identical to the no-thoughts default, and carries NO block.
    base = build_wake_packet(value=2.0, theta=1.0, correlation_id="c")
    with_empty = build_wake_packet(value=2.0, theta=1.0, correlation_id="c", thoughts=())
    assert with_empty.prompt == base.prompt
    assert RECENT_THOUGHTS_HEADER not in base.prompt


def test_thoughts_render_a_recent_thoughts_block_content_only_no_id() -> None:
    thoughts = [
        build_thought(id="t-a", content="did the owner hear back about the flat", salience=0.8),
        build_thought(id="t-b", content="I keep circling the same worry", salience=0.4),
    ]
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="c", thoughts=thoughts)
    assert RECENT_THOUGHTS_HEADER in p.prompt
    assert "— did the owner hear back about the flat" in p.prompt
    assert "— I keep circling the same worry" in p.prompt
    # the internal id is NEVER shown to the model (anti-echo — codex, lm-27n.6)
    assert "t-a" not in p.prompt
    assert "t-b" not in p.prompt
    # still carries the desire-frame + guidance (the block is appended, not replacing)
    assert GUIDANCE in p.prompt


def test_thoughts_block_is_bounded() -> None:
    thoughts = [
        build_thought(id=f"t{i}", content=f"thought number {i}", salience=1.0 - i * 0.01)
        for i in range(THOUGHTS_RENDER_LIMIT + 5)
    ]
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="c", thoughts=thoughts)
    rendered = sum(1 for line in p.prompt.splitlines() if line.startswith("— "))
    assert rendered == THOUGHTS_RENDER_LIMIT  # top-N only, order preserved from the caller


# --- lm-8o3: situational brief woven into the wake packet -------------------


def test_brief_frames_elapsed_in_words() -> None:
    brief = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=1.0
    )
    assert "несколько часов назад" in brief  # 180 min
    assert "вспомни, на чём вы остановились" in brief


def test_brief_fresh_history_does_not_fabricate() -> None:
    brief = render_situational_brief(last_exchange_at=None, now=NOW, decline_count=0, energy=1.0)
    assert "вы ещё толком не общались" in brief
    assert "не выдумывай повод" in brief
    assert "вспомни, на чём вы остановились" not in brief  # nothing to mine


def test_brief_rebuff_tone_only_when_declined() -> None:
    hot = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=2, energy=1.0
    )
    cold = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=1.0
    )
    assert "не дави" in hot
    assert "не дави" not in cold


def test_brief_energy_restraint_only_when_low() -> None:
    low = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00", now=NOW, decline_count=0, energy=0.1
    )
    assert "Сил сейчас немного" in low


def test_wake_packet_weaves_brief_and_keeps_no_digits() -> None:
    p = build_wake_packet(
        value=2.0,
        theta=1.0,
        correlation_id="c",
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=1,
        energy=0.1,
    )
    assert "несколько часов назад" in p.prompt
    assert "не дави" in p.prompt
    assert "Сил сейчас немного" in p.prompt
    assert re.search(r"\d", p.prompt) is None  # global invariant


def test_wake_packet_without_now_is_brief_free() -> None:
    # back-compat: no `now` -> no situational brief in the prompt
    p = build_wake_packet(value=2.0, theta=1.0, correlation_id="c")
    assert "Вы общались" not in p.prompt
    assert "вспомни, на чём вы остановились" not in p.prompt


# --- lm-8o3.1 Task 9: unanswered-bid line in the situational brief ----------


def test_brief_unanswered_bid_line_present_when_count_at_least_one() -> None:
    brief = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
        unanswered_outbound_count=1,
    )
    assert "пока без ответа" in brief
    assert "не повторяйся ради самого жеста" in brief


def test_brief_unanswered_bid_line_absent_when_count_zero() -> None:
    brief = render_situational_brief(
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
        unanswered_outbound_count=0,
    )
    assert "пока без ответа" not in brief
    assert "не повторяйся ради самого жеста" not in brief


def test_wake_packet_weaves_unanswered_bid_and_keeps_no_digits() -> None:
    p = build_wake_packet(
        value=2.0,
        theta=1.0,
        correlation_id="c",
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
        unanswered_outbound_count=2,
    )
    assert "пока без ответа" in p.prompt
    assert re.search(r"\d", p.prompt) is None  # global invariant


def test_build_wake_packet_unanswered_bid_defaults_to_absent() -> None:
    # back-compat: the default (0) leaves the prompt unchanged from before this task
    with_default = build_wake_packet(
        value=2.0,
        theta=1.0,
        correlation_id="c",
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
    )
    explicit_zero = build_wake_packet(
        value=2.0,
        theta=1.0,
        correlation_id="c",
        last_exchange_at="2026-07-08T09:00:00+00:00",
        now=NOW,
        decline_count=0,
        energy=1.0,
        unanswered_outbound_count=0,
    )
    assert with_default.prompt == explicit_zero.prompt
    assert "пока без ответа" not in with_default.prompt

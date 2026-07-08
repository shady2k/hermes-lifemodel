from __future__ import annotations

import re

from lifemodel.core.thought_view import build_thought
from lifemodel.core.wake_packet import (
    GUIDANCE,
    RECENT_THOUGHTS_HEADER,
    THOUGHTS_RENDER_LIMIT,
    ProactivePrompt,
    build_wake_packet,
)


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

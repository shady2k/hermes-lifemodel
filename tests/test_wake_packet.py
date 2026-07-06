from __future__ import annotations

import re

from lifemodel.core.wake_packet import GUIDANCE, ProactivePrompt, build_wake_packet


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

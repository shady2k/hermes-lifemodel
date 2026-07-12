"""The ``q_event`` exchange-quality classifier — desire-model spec §6.

Maps a lane event (actor + label) to a scalar quality ``q`` that drives
satiation of the contact urge. Satiation later uses ``β · max(q, 0)`` so only
positive-quality exchanges reduce the urge.
"""

from __future__ import annotations

from typing import Literal

Actor = Literal["user", "assistant", "proactive_internal"]
Label = Literal["two_way", "ack", "monologue", "rejection"]

# Quality by label (spec §6 table). Only positive values satiate.
_QUALITY_BY_LABEL: dict[Label, float] = {
    "two_way": 1.0,
    "ack": 0.5,
    "monologue": 0.0,
    "rejection": -0.5,
}


def quality_of(*, actor: Actor, label: Label) -> float:
    """Return the exchange quality ``q`` for a lane event.

    Load-bearing rule: an internal proactive impulse is *never* user contact
    (``q = 0``), whatever its label — this stops the being from satiating its
    own urge with its own nudges.
    """
    if actor == "proactive_internal":
        return 0.0
    return _QUALITY_BY_LABEL[label]

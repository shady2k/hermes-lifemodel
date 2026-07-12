"""The ``q_event`` exchange-quality classifier (desire-model spec §6).

Pure, Hermes-free. Maps a lane event (actor + label) to a scalar quality ``q``
that drives satiation of the contact urge. Load-bearing rule: an internal
proactive impulse is *never* user contact (``q = 0``), whatever its label — this
is exactly what stops the being from satiating its own urge with its own nudges.
"""

from __future__ import annotations

from lifemodel.core.quality import quality_of


def test_genuine_two_way_exchange_has_quality_one() -> None:
    assert quality_of(actor="user", label="two_way") == 1.0


def test_low_effort_ack_has_half_quality() -> None:
    assert quality_of(actor="user", label="ack") == 0.5


def test_assistant_monologue_has_zero_quality() -> None:
    assert quality_of(actor="assistant", label="monologue") == 0.0


def test_rejection_is_negative() -> None:
    assert quality_of(actor="user", label="rejection") == -0.5


def test_internal_proactive_impulse_is_never_user_contact() -> None:
    # actor=proactive_internal forces q=0 regardless of label, so the being
    # cannot satiate its own urge with its own internal nudges.
    assert quality_of(actor="proactive_internal", label="two_way") == 0.0
    assert quality_of(actor="proactive_internal", label="ack") == 0.0

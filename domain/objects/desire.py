"""``Desire`` — a felt pull toward some object of wanting (BDI, HLA §4.1).

A desire is what the being *wants*: an appraised pull with an intensity, a
valence, and a risk profile, sprung from a drive, a thought, or both. It is the
raw material the act-gate and intention-former weigh. State machine: an
``active`` desire may be deferred, satisfied, or dropped; a ``deferred`` one may
reactivate, be satisfied, dropped, or expire. ``satisfied``/``dropped``/
``expired`` are terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import (
    BaseFields,
    BaseObject,
    opt_float,
    req_enum,
    req_float,
    req_str,
    req_str_tuple,
    state_set,
)


class DesireState(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    SATISFIED = "satisfied"
    DROPPED = "dropped"
    EXPIRED = "expired"


class DesireSpring(StrEnum):
    """Where the desire sprang from (named ``spring`` to avoid the ``source`` column).

    ``GENESIS`` is the one spring that is NOT an appraisal of the being's situation but
    of the being itself (Phase 4, spec §6.2): it wants to reach out because it is nobody
    yet. It exists so that wake can never be mistaken for a DRIVE-sprung longing — a
    newborn has no relationship to have a deficit in, so its ``u`` is 0 and must stay 0.
    Marking the desire honestly is what keeps the contact model clean downstream: the
    pure-longing anti-repeat counter (``unanswered_outbound_count``) does not count it,
    the async staleness rule does not judge it by pressure it never had, and the
    wake-packet carries the ``<genesis>`` ritual instead of "I miss them".
    """

    DRIVE = "drive"
    THOUGHT = "thought"
    MIXED = "mixed"
    GENESIS = "genesis"


#: The explicit transition table (held by the registry). Terminal states are
#: present as keys with empty out-sets.
DESIRE_TRANSITIONS: dict[str, frozenset[str]] = {
    DesireState.ACTIVE: state_set(DesireState.DEFERRED, DesireState.SATISFIED, DesireState.DROPPED),
    DesireState.DEFERRED: state_set(
        DesireState.ACTIVE, DesireState.SATISFIED, DesireState.DROPPED, DesireState.EXPIRED
    ),
    DesireState.SATISFIED: state_set(),
    DesireState.DROPPED: state_set(),
    DesireState.EXPIRED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class Desire(BaseObject):
    object: str
    spring: DesireSpring
    source_drive: float | None
    source_thought_ids: tuple[str, ...]
    intensity: float
    valence: str
    urgency: float
    satiation_condition: str
    risk_if_acted: float
    risk_if_ignored: float

    KIND: ClassVar[str] = "desire"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "object": self.object,
            "spring": str(self.spring),
            "source_drive": self.source_drive,
            "source_thought_ids": list(self.source_thought_ids),
            "intensity": self.intensity,
            "valence": self.valence,
            "urgency": self.urgency,
            "satiation_condition": self.satiation_condition,
            "risk_if_acted": self.risk_if_acted,
            "risk_if_ignored": self.risk_if_ignored,
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            object=req_str(payload, "object"),
            spring=req_enum(payload, "spring", DesireSpring),
            source_drive=opt_float(payload, "source_drive"),
            source_thought_ids=req_str_tuple(payload, "source_thought_ids"),
            intensity=req_float(payload, "intensity"),
            valence=req_str(payload, "valence"),
            urgency=req_float(payload, "urgency"),
            satiation_condition=req_str(payload, "satiation_condition"),
            risk_if_acted=req_float(payload, "risk_if_acted"),
            risk_if_ignored=req_float(payload, "risk_if_ignored"),
        )

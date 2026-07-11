"""``UserModel`` — our derived model of the Other (the user) (spec §8, HLA §4.1).

A UserModel holds our *derived* cache of the correspondent — the learned
interaction norms and receptivity we infer about the owner: the cadence that
works, the good/bad hours, response-valence patterns, privacy boundaries, topic
sensitivities, intimacy depth, acceptable styles, explicit preferences. It is
the model of "the Other", NOT the being's own ``AgentState`` (the self) and NOT
a live counter store: egress counters (unanswered-outbound count, backoff,
action-pending) stay in ``runtime_state``; a UserModel *reads* them elsewhere and
never duplicates them (the split-brain guard, HLA §4.1). State machine:
``active`` may only be ``archived`` (terminal).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import (
    BaseFields,
    BaseObject,
    req_float,
    req_int_tuple,
    req_str,
    req_str_tuple,
    state_set,
)


class UserModelState(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


USER_MODEL_TRANSITIONS: dict[str, frozenset[str]] = {
    UserModelState.ACTIVE: state_set(UserModelState.ARCHIVED),
    UserModelState.ARCHIVED: state_set(),
}


@dataclass(frozen=True, kw_only=True)
class UserModel(BaseObject):
    cadence: str
    good_hours: tuple[int, ...]
    bad_hours: tuple[int, ...]
    response_valence_pattern: str
    privacy_boundaries: tuple[str, ...]
    topic_sensitivity: tuple[str, ...]
    intimacy_depth: float
    reply_latency_norm: str
    known_load: str
    acceptable_styles: tuple[str, ...]
    explicit_preferences: tuple[str, ...]

    KIND: ClassVar[str] = "user_model"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "cadence": self.cadence,
            "good_hours": list(self.good_hours),
            "bad_hours": list(self.bad_hours),
            "response_valence_pattern": self.response_valence_pattern,
            "privacy_boundaries": list(self.privacy_boundaries),
            "topic_sensitivity": list(self.topic_sensitivity),
            "intimacy_depth": self.intimacy_depth,
            "reply_latency_norm": self.reply_latency_norm,
            "known_load": self.known_load,
            "acceptable_styles": list(self.acceptable_styles),
            "explicit_preferences": list(self.explicit_preferences),
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            cadence=req_str(payload, "cadence"),
            good_hours=req_int_tuple(payload, "good_hours"),
            bad_hours=req_int_tuple(payload, "bad_hours"),
            response_valence_pattern=req_str(payload, "response_valence_pattern"),
            privacy_boundaries=req_str_tuple(payload, "privacy_boundaries"),
            topic_sensitivity=req_str_tuple(payload, "topic_sensitivity"),
            intimacy_depth=req_float(payload, "intimacy_depth"),
            reply_latency_norm=req_str(payload, "reply_latency_norm"),
            known_load=req_str(payload, "known_load"),
            acceptable_styles=req_str_tuple(payload, "acceptable_styles"),
            explicit_preferences=req_str_tuple(payload, "explicit_preferences"),
        )

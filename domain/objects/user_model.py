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

**Per-field inference metadata (spec §8):** every semantic field is an
:class:`~lifemodel.domain.objects.inference.InferredField` — ``{value,
inferred_at, ttl}`` — not a bare scalar, because the UserModel is a DERIVED
cache with a shelf life. A field whose ``inferred_at + ttl`` has passed is stale
and reads as ``UNKNOWN`` (never the old value); an owner-SET / authoritative
field carries no ttl and never expires.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Self

from ..memory import JsonObject
from .base import BaseFields, BaseObject, state_set
from .inference import (
    InferredField,
    pack_inferred,
    req_inferred_float,
    req_inferred_int_tuple,
    req_inferred_str,
    req_inferred_str_tuple,
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
    cadence: InferredField[str]
    good_hours: InferredField[tuple[int, ...]]
    bad_hours: InferredField[tuple[int, ...]]
    response_valence_pattern: InferredField[str]
    privacy_boundaries: InferredField[tuple[str, ...]]
    topic_sensitivity: InferredField[tuple[str, ...]]
    intimacy_depth: InferredField[float]
    reply_latency_norm: InferredField[str]
    known_load: InferredField[str]
    acceptable_styles: InferredField[tuple[str, ...]]
    explicit_preferences: InferredField[tuple[str, ...]]

    KIND: ClassVar[str] = "user_model"
    SCHEMA_VERSION: ClassVar[int] = 1

    def _semantic_payload(self) -> JsonObject:
        return {
            "cadence": pack_inferred(self.cadence, self.cadence.value),
            "good_hours": pack_inferred(self.good_hours, list(self.good_hours.value)),
            "bad_hours": pack_inferred(self.bad_hours, list(self.bad_hours.value)),
            "response_valence_pattern": pack_inferred(
                self.response_valence_pattern, self.response_valence_pattern.value
            ),
            "privacy_boundaries": pack_inferred(
                self.privacy_boundaries, list(self.privacy_boundaries.value)
            ),
            "topic_sensitivity": pack_inferred(
                self.topic_sensitivity, list(self.topic_sensitivity.value)
            ),
            "intimacy_depth": pack_inferred(self.intimacy_depth, self.intimacy_depth.value),
            "reply_latency_norm": pack_inferred(
                self.reply_latency_norm, self.reply_latency_norm.value
            ),
            "known_load": pack_inferred(self.known_load, self.known_load.value),
            "acceptable_styles": pack_inferred(
                self.acceptable_styles, list(self.acceptable_styles.value)
            ),
            "explicit_preferences": pack_inferred(
                self.explicit_preferences, list(self.explicit_preferences.value)
            ),
        }

    @classmethod
    def _rebuild(cls, base: BaseFields, payload: JsonObject) -> Self:
        return cls(
            **base,
            cadence=req_inferred_str(payload, "cadence"),
            good_hours=req_inferred_int_tuple(payload, "good_hours"),
            bad_hours=req_inferred_int_tuple(payload, "bad_hours"),
            response_valence_pattern=req_inferred_str(payload, "response_valence_pattern"),
            privacy_boundaries=req_inferred_str_tuple(payload, "privacy_boundaries"),
            topic_sensitivity=req_inferred_str_tuple(payload, "topic_sensitivity"),
            intimacy_depth=req_inferred_float(payload, "intimacy_depth"),
            reply_latency_norm=req_inferred_str(payload, "reply_latency_norm"),
            known_load=req_inferred_str(payload, "known_load"),
            acceptable_styles=req_inferred_str_tuple(payload, "acceptable_styles"),
            explicit_preferences=req_inferred_str_tuple(payload, "explicit_preferences"),
        )

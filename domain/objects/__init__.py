"""The BDI object-core substrate — typed kinds over the memory envelope (§4.1).

A typed Belief-Desire-Intention layer sitting *on top of* the generic
``memory_records`` envelope (:mod:`lifemodel.domain.memory`). Four kinds
(:class:`Desire`, :class:`Intention`, :class:`Relationship`, :class:`Thought`)
subclass a shared :class:`BaseObject`; the :class:`KindRegistry` is the single
door for every encode/decode/transition. :func:`default_registry` is the blessed
factory for the closed four-kind catalog — feature code takes one and cannot add
kinds. (The ``KindRegistry(specs)`` constructor stays as the extension/test seam,
imported from :mod:`lifemodel.domain.objects.registry` directly, but is not part
of this package's advertised surface.) Provenance carries W3C-traceparent-
compatible creation context (definitions + validation only — task .1; minting
and propagation are task .2). Imports nothing from Hermes, ``ports/``, or
``core/`` — its only intra-repo dependency is :mod:`lifemodel.domain.memory`.
"""

from __future__ import annotations

from .base import (
    CONTACT_DESIRE_ID,
    CONTACT_INTENTION_ID,
    BaseObject,
    derive_id,
    qualified_id,
)
from .desire import Desire, DesireSpring, DesireState
from .errors import (
    InvalidPayload,
    InvalidTransition,
    ObjectCoreError,
    UnknownKind,
)
from .intention import Intention, IntentionState
from .provenance import (
    Provenance,
    Sensitivity,
    format_traceparent,
    parse_traceparent,
)
from .registry import KindRegistry, default_registry
from .relationship import Relationship, RelationshipState
from .thought import Thought, ThoughtState

__all__ = [
    "CONTACT_DESIRE_ID",
    "CONTACT_INTENTION_ID",
    "BaseObject",
    "Desire",
    "DesireSpring",
    "DesireState",
    "InvalidPayload",
    "InvalidTransition",
    "Intention",
    "IntentionState",
    "KindRegistry",
    "ObjectCoreError",
    "Provenance",
    "Relationship",
    "RelationshipState",
    "Sensitivity",
    "Thought",
    "ThoughtState",
    "UnknownKind",
    "default_registry",
    "derive_id",
    "format_traceparent",
    "parse_traceparent",
    "qualified_id",
]

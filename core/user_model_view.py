"""The owner user-model view — the single door onto the singleton
``kind='user_model'`` row ``owner`` (our derived model of the owner, HLA §4.1).

Spec §8 makes the being's derived model of its owner a first-class typed row:
good/bad hours, cadence, privacy boundaries, acceptable styles, explicit prefs,
plus a ``confidence`` marking whether the row was EXPLICITLY set by the owner
(its boundaries hard-veto) or is a low-confidence seed/inference (its weak norms
only down-weight). Like the desire (:mod:`lifemodel.core.desire_view`) and
intention (:mod:`lifemodel.core.intention_view`) singletons, its lifecycle lives
in a ``kind='user_model'`` record ``owner`` (HLA §4.1), created/updated ONLY
through the intent bus (a :class:`~lifemodel.core.intents.PutRecord` upsert; a
:class:`~lifemodel.core.intents.TransitionRecord` only for archive) — never a
hand-built draft, never direct SQL/config.

This module is the ONE place that reads that row back into a typed
:class:`~lifemodel.domain.objects.UserModel`, and the sole constructor of the
singleton, so every appraisal site asks the SAME question. Two readers, one
predicate, mirroring the desire/intention views:

* :func:`live_owner_user_model` reads the start-of-tick records snapshot
  (:attr:`~lifemodel.core.component.TickContext.objects`) — what aggregation and
  cognition consume in-tick;
* :func:`read_owner_user_model` reads a
  :class:`~lifemodel.ports.memory.MemoryPort` point-in-time — what the debug view
  uses.

**Behavior-neutrality (lm-27n.5):** when NO owner user-model row exists (or it is
archived), appraisal uses the permissive :data:`DEFAULT_USER_MODEL` — empty
boundaries, neutral valence, low confidence — so the appraisal returns
``allowed=True, pressure_multiplier=1.0`` and the being sends EXACTLY as it did
before this task (.4). A seeded/default row NEVER hard-vetoes.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import MemoryDraft, MemoryRecord
from ..domain.objects import (
    OWNER_USER_MODEL_ID,
    UserModel,
    UserModelState,
    default_registry,
)
from ..ports.memory import MemoryPort

#: The kind of the owner user-model record (``kind`` column, HLA §4.1).
USER_MODEL_KIND = "user_model"

#: The non-terminal states — a user-model in one of these is *live*; anything
#: else (``archived``, or absent) reads as absence → the permissive
#: :data:`DEFAULT_USER_MODEL` is used by appraisal.
LIVE_USER_MODEL_STATES: frozenset[str] = frozenset({UserModelState.ACTIVE.value})

#: Confidence at/above which a user-model's boundaries are treated as EXPLICIT
#: owner boundaries — they may HARD-veto proactive contact (quiet hours, cadence,
#: no-contact privacy). Below it (seeded default / a future inference), the same
#: fields only SOFT down-weight, so a seeded row can never hard-veto (sovereignty:
#: "the being doesn't disappear").
EXPLICIT_CONFIDENCE = 0.9

#: The seeded/default confidence — low, so the default row never hard-vetoes.
DEFAULT_CONFIDENCE = 0.1

#: Built once; :func:`default_registry` validates its four-kind catalog on every
#: call, so the per-tick readers reuse one instance rather than rebuild it.
_REGISTRY = default_registry()


def build_owner_user_model(
    *,
    state: UserModelState = UserModelState.ACTIVE,
    cadence: str = "",
    good_hours: tuple[int, ...] = (),
    bad_hours: tuple[int, ...] = (),
    response_valence_pattern: str = "neutral",
    privacy_boundaries: tuple[str, ...] = (),
    topic_sensitivity: tuple[str, ...] = (),
    intimacy_depth: float = 0.0,
    reply_latency_norm: str = "",
    known_load: str = "",
    acceptable_styles: tuple[str, ...] = (),
    explicit_preferences: tuple[str, ...] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    source: str = "owner-user-model",
) -> UserModel:
    """Construct the singleton owner :class:`UserModel`.

    The one constructor for the ``owner`` user-model row. Every field defaults to
    its permissive/empty value, so ``build_owner_user_model()`` with no arguments
    IS :data:`DEFAULT_USER_MODEL`; the owner's explicit-set path passes the
    populated fields plus ``confidence=EXPLICIT_CONFIDENCE`` so its boundaries
    hard-veto.
    """
    return UserModel(
        id=OWNER_USER_MODEL_ID,
        state=str(state),
        source=source,
        confidence=confidence,
        cadence=cadence,
        good_hours=good_hours,
        bad_hours=bad_hours,
        response_valence_pattern=response_valence_pattern,
        privacy_boundaries=privacy_boundaries,
        topic_sensitivity=topic_sensitivity,
        intimacy_depth=intimacy_depth,
        reply_latency_norm=reply_latency_norm,
        known_load=known_load,
        acceptable_styles=acceptable_styles,
        explicit_preferences=explicit_preferences,
    )


#: The permissive default appraisal uses when NO owner user-model row exists (or
#: it is archived). Empty boundaries + neutral valence + low confidence ⇒
#: :func:`~lifemodel.core.receptivity.appraise_receptivity` returns
#: ``allowed=True, pressure_multiplier=1.0`` ⇒ the being behaves exactly as .4.
DEFAULT_USER_MODEL: UserModel = build_owner_user_model()


def _decode_live(record: MemoryRecord | None) -> UserModel | None:
    """Decode *record* into the live owner :class:`UserModel`, or ``None``.

    ``None`` when the record is absent, is not the owner user-model singleton, or
    is terminal (``archived``). Decoding goes through the registry (the single
    read door), so a malformed row surfaces as its
    :class:`~lifemodel.domain.objects.InvalidPayload`, never a silent miss.
    """
    if record is None or record.kind != USER_MODEL_KIND or record.id != OWNER_USER_MODEL_ID:
        return None
    if record.state not in LIVE_USER_MODEL_STATES:
        return None
    user_model = _REGISTRY.decode(record)
    return user_model if isinstance(user_model, UserModel) else None


def live_owner_user_model(objects: Sequence[MemoryRecord]) -> UserModel | None:
    """The live (``active``) owner user-model in a records snapshot, or ``None``.

    Scans the start-of-tick :attr:`~lifemodel.core.component.TickContext.objects`
    snapshot for the ``owner`` user-model and returns it typed, or ``None`` when
    there is no live one (callers then fall back to :data:`DEFAULT_USER_MODEL`).
    """
    for record in objects:
        user_model = _decode_live(record)
        if user_model is not None:
            return user_model
    return None


def read_owner_user_model(memory: MemoryPort) -> UserModel | None:
    """The live owner user-model read point-in-time from a :class:`MemoryPort`.

    For out-of-band readers (the debug view) that hold a store rather than a tick
    snapshot. ``get`` by the singleton id, then the same live/terminal predicate
    as :func:`live_owner_user_model`.
    """
    return _decode_live(memory.get(USER_MODEL_KIND, OWNER_USER_MODEL_ID))


def encode_owner_user_model(user_model: UserModel) -> MemoryDraft:
    """Encode *user_model* through the registry (the single write door)."""
    return _REGISTRY.encode(user_model)

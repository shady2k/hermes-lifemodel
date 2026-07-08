"""``KindRegistry`` — the single door for every typed-kind operation (lm-27n.1).

No feature code constructs a raw ``MemoryDraft``/``MemoryRecord`` for a typed
kind, and no feature code decides a lifecycle edge by hand: every
create/decode/transition goes through this registry. The catalog is **closed at
construction** — there is no public ``register()`` to call — so the set of
kinds is exactly ``{desire, intention, relationship, thought}``.

Registration validates each kind at construction: semantic field names must not
use the reserved ``_`` prefix, the transition table must be non-empty, and every
target state must itself be a declared state (no dangling edges). These are
design-time guards and raise :class:`ValueError` (a mis-built registry), as
distinct from the runtime :class:`~lifemodel.domain.objects.errors` family.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields

from ..memory import MemoryDraft, MemoryRecord
from .base import RESERVED_KEYS, BaseFields, BaseObject, pack_envelope, unpack_envelope
from .desire import DESIRE_TRANSITIONS, Desire
from .errors import InvalidPayload, InvalidTransition, UnknownKind
from .intention import INTENTION_TRANSITIONS, Intention
from .relationship import RELATIONSHIP_TRANSITIONS, Relationship
from .thought import THOUGHT_TRANSITIONS, Thought

#: The envelope field names, computed once — anything on a kind that is *not*
#: one of these is a semantic field (the reserved-prefix guard checks those).
_ENVELOPE_FIELD_NAMES = frozenset(f.name for f in fields(BaseObject))


@dataclass(frozen=True)
class KindSpec:
    """A kind's class + its explicit transition table, as fed to the registry."""

    cls: type[BaseObject]
    transitions: Mapping[str, frozenset[str]]

    @property
    def kind(self) -> str:
        return self.cls.KIND


class KindRegistry:
    """The closed catalog of typed kinds and the only codec/transition door."""

    def __init__(self, specs: Iterable[KindSpec]) -> None:
        self._classes: dict[str, type[BaseObject]] = {}
        self._transitions: dict[str, dict[str, frozenset[str]]] = {}
        self._semantic_fields: dict[str, frozenset[str]] = {}
        for spec in specs:
            self._register(spec)

    def _register(self, spec: KindSpec) -> None:
        kind = spec.kind
        if kind in self._classes:
            raise ValueError(f"duplicate kind {kind!r} in registry")
        self._check_semantic_fields(spec.cls)
        self._check_transitions(kind, spec.transitions)
        self._classes[kind] = spec.cls
        self._transitions[kind] = dict(spec.transitions)
        self._semantic_fields[kind] = self._semantic_field_names(spec.cls)

    @staticmethod
    def _semantic_field_names(cls: type[BaseObject]) -> frozenset[str]:
        """The kind's own (non-envelope) instance field names."""
        return frozenset(f.name for f in fields(cls)) - _ENVELOPE_FIELD_NAMES

    @staticmethod
    def _check_semantic_fields(cls: type[BaseObject]) -> None:
        for field in fields(cls):
            if field.name in _ENVELOPE_FIELD_NAMES:
                continue
            if field.name.startswith("_") or field.name in RESERVED_KEYS:
                raise ValueError(
                    f"kind {cls.KIND!r} declares a reserved-prefixed semantic field "
                    f"{field.name!r}; semantic fields must not start with '_'"
                )

    @staticmethod
    def _check_transitions(kind: str, transitions: Mapping[str, frozenset[str]]) -> None:
        if not transitions:
            raise ValueError(f"kind {kind!r} has an empty transition table")
        states = set(transitions)
        for from_state, to_states in transitions.items():
            dangling = to_states - states
            if dangling:
                raise ValueError(
                    f"kind {kind!r} state {from_state!r} points at undeclared "
                    f"state(s) {sorted(dangling)}"
                )

    def kinds(self) -> frozenset[str]:
        """The closed set of known kinds."""
        return frozenset(self._classes)

    def is_known(self, kind: str) -> bool:
        return kind in self._classes

    def _require(self, kind: str) -> type[BaseObject]:
        cls = self._classes.get(kind)
        if cls is None:
            raise UnknownKind(f"unknown kind {kind!r}; known kinds: {sorted(self._classes)}")
        return cls

    def states_of(self, kind: str) -> frozenset[str]:
        """Every valid state of *kind* (raises :class:`UnknownKind` if unknown)."""
        self._require(kind)
        return frozenset(self._transitions[kind])

    def terminal_states_of(self, kind: str) -> frozenset[str]:
        """The *terminal* states of *kind* — those with an empty transition
        out-set (no legal edge leaves them). Its complement in
        :meth:`states_of` is the kind's non-terminal (live) states."""
        self._require(kind)
        return frozenset(s for s, outs in self._transitions[kind].items() if not outs)

    def live_states(self) -> frozenset[str]:
        """The union of every kind's NON-terminal states — the *live* state-set.

        A state is terminal for a kind iff its transition out-set is empty
        (:meth:`terminal_states_of`); non-terminal (live) otherwise. This unions
        the live states across the whole catalog, so a single ``find(state=s)``
        sweep over it surfaces every non-terminal row of every kind — for the
        four-kind catalog that is ``{active, deferred, pending, parked}``.

        Safe because our catalog is *terminal-consistent*: no state string is
        terminal for one kind and non-terminal for another (asserted by the
        object-kinds tests), so a fetch by this union never returns a row that is
        terminal for its own kind."""
        live: set[str] = set()
        for transitions in self._transitions.values():
            live.update(s for s, outs in transitions.items() if outs)
        return frozenset(live)

    def validate_transition(self, kind: str, from_state: str, to_state: str) -> None:
        """Raise unless *kind* allows the edge ``from_state -> to_state``.

        :class:`UnknownKind` if the kind is not registered; :class:`InvalidTransition`
        if either endpoint is not a state of the kind, or the edge is not allowed.
        """
        self._require(kind)
        transitions = self._transitions[kind]
        if from_state not in transitions:
            raise InvalidTransition(f"{kind!r} has no state {from_state!r}")
        if to_state not in transitions:
            raise InvalidTransition(f"{kind!r} has no state {to_state!r}")
        if to_state not in transitions[from_state]:
            raise InvalidTransition(f"{kind!r} does not allow {from_state!r} -> {to_state!r}")

    def encode(self, obj: BaseObject) -> MemoryDraft:
        """Encode a typed object into a :class:`MemoryDraft` (the only writer).

        The write door validates too, not just decode: an object of the wrong
        class for its ``KIND`` or holding a state its machine does not know is
        rejected (:class:`InvalidPayload`) rather than persisted — otherwise the
        "only door" would let invalid state out on write and only catch it later
        on read.
        """
        cls = self._require(obj.KIND)
        if not isinstance(obj, cls):
            raise InvalidPayload(
                f"object for kind {obj.KIND!r} is not a {cls.__name__} ({type(obj).__name__})"
            )
        if str(obj.state) not in self._transitions[obj.KIND]:
            raise InvalidPayload(f"kind {obj.KIND!r} has no state {obj.state!r}")
        draft = pack_envelope(obj, obj._semantic_payload())
        # Write-door closure: the draft must be one that :meth:`decode` would
        # accept. Static typing does not guarantee field types at runtime (an
        # object built from LLM output, or past a ``# type: ignore``, could hold
        # a non-float ``intensity``), so run the draft back through the same
        # strict decoder — a malformed typed object is rejected on WRITE, not
        # only on a later read. Reuses one validation path; the result is discarded.
        self._validate_encodable(draft, cls)
        return draft

    def _validate_encodable(self, draft: MemoryDraft, cls: type[BaseObject]) -> None:
        """Assert *draft* decodes cleanly (raising :class:`InvalidPayload` if not)."""
        probe = MemoryRecord(
            kind=draft.kind,
            id=draft.id,
            state=draft.state,
            payload=draft.payload,
            source=draft.source,
            recipient_id=draft.recipient_id,
            salience=draft.salience,
            confidence=draft.confidence,
            expires_at=draft.expires_at,
            created_at="",
            updated_at="",
            revision=0,
            schema_version=cls.SCHEMA_VERSION,
        )
        self.decode(probe)

    def decode(self, record: MemoryRecord) -> BaseObject:
        """Decode a :class:`MemoryRecord` into its typed kind (the only reader).

        Rejects an unknown kind (:class:`UnknownKind`); a ``schema_version`` that
        differs from the kind's (no back-compat), an unknown ``state``, or a
        malformed payload (all :class:`InvalidPayload`).
        """
        cls = self._require(record.kind)
        if record.schema_version != cls.SCHEMA_VERSION:
            raise InvalidPayload(
                f"kind {record.kind!r} schema_version={record.schema_version} is not "
                f"supported by this build (expects {cls.SCHEMA_VERSION})"
            )
        if record.state not in self._transitions[record.kind]:
            raise InvalidPayload(f"kind {record.kind!r} has no state {record.state!r}")
        base: BaseFields
        base, semantic = unpack_envelope(record)
        # After the reserved envelope keys are popped, whatever is left must be
        # exactly the kind's declared semantic fields — anything else (a stray
        # ``_schema_version`` / other ``_``-reserved key that slipped the
        # envelope, or an unknown/typo'd semantic key) is a malformed payload,
        # not a silently-ignored extra.
        extra = set(semantic) - self._semantic_fields[record.kind]
        if extra:
            raise InvalidPayload(
                f"kind {record.kind!r} record has unexpected payload key(s) {sorted(extra)}"
            )
        return cls._rebuild(base, semantic)


#: The four-kind catalog, wired to each kind's module-level transition table.
_CATALOG: tuple[KindSpec, ...] = (
    KindSpec(cls=Desire, transitions=DESIRE_TRANSITIONS),
    KindSpec(cls=Intention, transitions=INTENTION_TRANSITIONS),
    KindSpec(cls=Relationship, transitions=RELATIONSHIP_TRANSITIONS),
    KindSpec(cls=Thought, transitions=THOUGHT_TRANSITIONS),
)


def default_registry() -> KindRegistry:
    """Build the closed BDI catalog. Feature code receives one; it cannot add kinds."""
    return KindRegistry(_CATALOG)

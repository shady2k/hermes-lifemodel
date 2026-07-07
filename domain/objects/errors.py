"""Typed errors for the BDI object core (lm-27n.1, HLA §4.1).

A small taxonomy so callers can distinguish *why* a typed-kind operation
failed. Mirrors :class:`~lifemodel.state.errors.StateError` and
:class:`~lifemodel.domain.memory.MemoryPortError` as the root of a per-package
family. This module imports nothing — it sits at the bottom of the objects
package so every other module can depend on it without an import cycle.
"""

from __future__ import annotations


class ObjectCoreError(Exception):
    """Base class for every typed-object failure."""


class UnknownKind(ObjectCoreError):
    """Raised when a kind method is given a kind the registry does not know.

    The catalog is closed at construction (exactly the four BDI kinds), so an
    unregistered kind is always a programming error surfaced here rather than a
    silent no-op.
    """


class InvalidPayload(ObjectCoreError):
    """Raised when a record cannot be decoded into its typed kind.

    Covers a missing/mis-typed/non-JSON semantic field, an unknown ``state`` or
    ``sensitivity`` value, a ``schema_version`` mismatch, or malformed W3C
    trace-context fields. The decode boundary is strict: bad data fails loud
    rather than round-tripping a half-built object.
    """


class InvalidTransition(ObjectCoreError):
    """Raised when a state transition is not permitted by the kind's machine.

    Either endpoint is not a state of the kind, or the edge from-state ->
    to-state is not in the kind's explicit transition table (terminal states
    have empty out-sets, so any move off them lands here).
    """

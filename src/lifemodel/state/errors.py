"""Typed errors for the state store.

A small taxonomy so callers can distinguish *why* a load failed and react
appropriately (fail loud vs. recover):

* :class:`StateSchemaError` — the file is well-formed JSON but its
  ``schema_version`` is not one this build knows how to read. Full migrations
  are Phase 7 (HLA §9 / FR16); until then any mismatch fails loud.
* :class:`StateCorruptError` — the file cannot be parsed or interpreted as a
  :class:`~lifemodel.state.model.State` (bad JSON, invalid UTF-8, wrong shape,
  wrong types, non-finite floats). Snapshot recovery is Phase 7 (HLA §9 / FR12);
  until then this fails loud.
* :class:`StateSerializationError` — a commit was refused because the in-memory
  ``State`` cannot be serialized to *valid* JSON (e.g. a non-finite float).
  Raised before anything is written, so the previous good state.json stands.

This module imports nothing — it sits at the bottom of the state package so
both the model and the adapter can depend on it without import cycles.
"""

from __future__ import annotations


class StateError(Exception):
    """Base class for all state-store failures."""


class StateSchemaError(StateError):
    """Raised when a persisted state carries an unsupported ``schema_version``."""


class StateCorruptError(StateError):
    """Raised when a persisted state cannot be parsed or interpreted."""


class StateSerializationError(StateError):
    """Raised when a ``State`` cannot be serialized to valid JSON for commit.

    The failing case in Phase 1 is a non-finite float (``NaN``/``Infinity``),
    which ``json`` would otherwise emit as a non-standard token that is not
    valid JSON. Refusing to write it keeps poison values out of the store.
    """

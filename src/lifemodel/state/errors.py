"""Typed errors for the state store.

A small taxonomy so callers can distinguish *why* a load failed and react
appropriately (fail loud vs. recover):

* :class:`StateSchemaError` — the file is well-formed JSON but its
  ``schema_version`` is not one this build knows how to read. Full migrations
  are Phase 7 (HLA §9 / FR16); until then any mismatch fails loud.
* :class:`StateCorruptError` — the file cannot be parsed or interpreted as a
  :class:`~lifemodel.state.model.State` (bad JSON, wrong shape, wrong types).
  Snapshot recovery is Phase 7 (HLA §9 / FR12); until then this fails loud.

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

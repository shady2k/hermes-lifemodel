"""State store — the single source of truth for the being's state (HLA §4/§9).

Public surface: the :class:`StatePort` boundary, the :class:`State` model and
its ``SCHEMA_VERSION``, the :class:`JsonStateStore` adapter, and the typed
errors. Everything here is Hermes-free and stdlib-only.
"""

from __future__ import annotations

from .errors import (
    StateCorruptError,
    StateError,
    StateSchemaError,
    StateSerializationError,
)
from .json_store import JsonStateStore
from .model import SCHEMA_VERSION, State
from .port import StatePort

__all__ = [
    "SCHEMA_VERSION",
    "JsonStateStore",
    "State",
    "StateCorruptError",
    "StateError",
    "StatePort",
    "StateSchemaError",
    "StateSerializationError",
]

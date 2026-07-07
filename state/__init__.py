"""State store — the single source of truth for the being's state (HLA §4/§9).

Public surface: the :class:`StatePort` boundary, the :class:`State` model and
its ``SCHEMA_VERSION``, and the typed errors. The concrete live adapter,
:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore` (lm-fib.6.2), is
imported directly from its module rather than re-exported here — it also
implements ``MemoryPort``/``PressureSensorPort`` (HLA §4.1/D7), so it does not
belong solely to this package's catalogue. Everything here is Hermes-free and
stdlib-only.
"""

from __future__ import annotations

from .errors import (
    StateCorruptError,
    StateError,
    StateSchemaError,
    StateSerializationError,
)
from .model import SCHEMA_VERSION, State
from .port import StatePort

__all__ = [
    "SCHEMA_VERSION",
    "State",
    "StateCorruptError",
    "StateError",
    "StatePort",
    "StateSchemaError",
    "StateSerializationError",
]

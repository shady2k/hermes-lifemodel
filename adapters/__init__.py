"""Adapters — concrete, Hermes-free implementations of the ports (HLA §13).

These sit on the outside of the hexagon and implement a port each: the system
clock, a no-op delivery stub, and the durable file signal bus. The SQLite
runtime store (:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`,
lm-fib.6.2) is the state-store adapter (also ``MemoryPort``/``PressureSensorPort``,
HLA §4.1/D7), kept in its cohesive :mod:`lifemodel.state` package and
re-exported here so the adapter layer is one catalogue. It replaces the
retired ``JsonStateStore``/``state.json``.

Adapters that speak to *Hermes* (the real gateway delivery) are constructed at
the plugin boundary (:func:`lifemodel.register`) and injected into the
composition root — they are deliberately not imported here, so this layer, like
the core, imports no Hermes.
"""

from __future__ import annotations

from ..state.sqlite_store import SQLiteRuntimeStore
from .clock import SystemClock
from .delivery import NoopDelivery
from .signal_bus import FileSignalBus

__all__ = [
    "FileSignalBus",
    "NoopDelivery",
    "SQLiteRuntimeStore",
    "SystemClock",
]

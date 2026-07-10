"""Adapters — concrete, Hermes-free implementations of the ports (HLA §13).

These sit on the outside of the hexagon and implement a port each: the system
clock and a no-op delivery stub. The SQLite runtime store
(:class:`~lifemodel.state.sqlite_store.SQLiteRuntimeStore`, lm-fib.6.2) is the
state-store adapter (also ``MemoryPort``/``PressureSensorPort``, HLA §4.1/D7),
kept in its cohesive :mod:`lifemodel.state` package and re-exported here so the
adapter layer is one catalogue. It replaces the retired ``JsonStateStore``/
``state.json``. There is no durable signal bus: the nervous flow is ephemeral
(spec §2/§3) — signals live inside one :class:`~lifemodel.core.frame.SignalFrame`.

Adapters that speak to *Hermes* (the real gateway delivery) are constructed at
the plugin boundary (:func:`lifemodel.register`) and injected into the
composition root — they are deliberately not imported here, so this layer, like
the core, imports no Hermes.
"""

from __future__ import annotations

from ..state.sqlite_store import SQLiteRuntimeStore
from .clock import SystemClock
from .delivery import NoopDelivery

__all__ = [
    "NoopDelivery",
    "SQLiteRuntimeStore",
    "SystemClock",
]

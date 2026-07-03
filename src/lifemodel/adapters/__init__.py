"""Adapters — concrete, Hermes-free implementations of the ports (HLA §13).

These sit on the outside of the hexagon and implement a port each: the system
clock, a no-op delivery stub, and the durable file signal bus. The JSON state
store (:class:`~lifemodel.state.json_store.JsonStateStore`) is the state-store
adapter, kept in its cohesive :mod:`lifemodel.state` package (task 0.2) and
re-exported here so the adapter layer is one catalogue.

Adapters that speak to *Hermes* (the real gateway delivery) are constructed at
the plugin boundary (:func:`lifemodel.register`) and injected into the
composition root — they are deliberately not imported here, so this layer, like
the core, imports no Hermes.
"""

from __future__ import annotations

from ..state.json_store import JsonStateStore
from .clock import SystemClock
from .delivery import NoopDelivery
from .signal_bus import FileSignalBus

__all__ = [
    "FileSignalBus",
    "JsonStateStore",
    "NoopDelivery",
    "SystemClock",
]

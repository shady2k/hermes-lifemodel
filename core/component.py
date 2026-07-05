"""The per-tick component seam (spec §3, §7.2).

A component is anything the CoreLoop schedules on a tick — a neuron, an
aggregation stage, the personality, cognition. It reads an immutable
:class:`TickContext` (state snapshot + clock + bus) and returns intents; it
never mutates state. Kept deliberately minimal so every layer can implement it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..state.model import State
from .intents import Intent
from .signal_bus import SignalBus


@dataclass(frozen=True)
class TickContext:
    """Read-only inputs handed to every component on a tick."""

    state: State
    now: datetime
    bus: SignalBus


@runtime_checkable
class Component(Protocol):
    """A schedulable unit. ``id`` is stable and unique within a registry."""

    id: str

    def step(self, ctx: TickContext) -> Sequence[Intent]: ...

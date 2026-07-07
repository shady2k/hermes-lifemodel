"""Test doubles shipped with the package so every task reuses the same fakes.

Importable as ``from lifemodel.testing import FakeClock, FakeDelivery,
FakeMemoryStore, FakePressureSensor, FakeSignalBus, FakeStateStore``. See
:mod:`lifemodel.testing.fakes`.
"""

from __future__ import annotations

from .desires import contact_desire_objects, contact_desire_record
from .fakes import (
    FakeClock,
    FakeDelivery,
    FakeMemoryStore,
    FakePressureSensor,
    FakeSignalBus,
    FakeStateStore,
)

__all__ = [
    "FakeClock",
    "FakeDelivery",
    "FakeMemoryStore",
    "FakePressureSensor",
    "FakeSignalBus",
    "FakeStateStore",
    "contact_desire_objects",
    "contact_desire_record",
]

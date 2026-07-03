"""Test doubles shipped with the package so every task reuses the same fakes.

Importable as ``from lifemodel.testing import FakeClock, FakeDelivery,
FakeSignalBus, FakeStateStore``. See :mod:`lifemodel.testing.fakes`.
"""

from __future__ import annotations

from .fakes import FakeClock, FakeDelivery, FakeSignalBus, FakeStateStore

__all__ = [
    "FakeClock",
    "FakeDelivery",
    "FakeSignalBus",
    "FakeStateStore",
]

"""Test doubles shipped with the package so every task reuses the same fakes.

Importable as ``from lifemodel.testing import FakeClock, FakeDelivery,
FakeMemoryStore, FakePressureSensor, FakeSignalBus, FakeStateStore``. See
:mod:`lifemodel.testing.fakes`.
"""

from __future__ import annotations

from .desires import contact_desire_objects, contact_desire_record
from .fakes import (
    FakeActiveSpan,
    FakeClock,
    FakeDelivery,
    FakeMemoryStore,
    FakePressureSensor,
    FakeSignalBus,
    FakeSpanLogger,
    FakeStateStore,
    FakeTracer,
)
from .harness import (
    IntegrationHarness,
    RecordingEgress,
    Step,
    TickRecord,
)
from .intentions import contact_intention_objects, contact_intention_record
from .relationships import owner_relationship_objects, owner_relationship_record
from .thoughts import thought_objects, thought_record

__all__ = [
    "FakeActiveSpan",
    "FakeClock",
    "FakeDelivery",
    "FakeMemoryStore",
    "FakePressureSensor",
    "FakeSignalBus",
    "FakeSpanLogger",
    "FakeStateStore",
    "FakeTracer",
    "IntegrationHarness",
    "RecordingEgress",
    "Step",
    "TickRecord",
    "contact_desire_objects",
    "contact_desire_record",
    "contact_intention_objects",
    "contact_intention_record",
    "owner_relationship_objects",
    "owner_relationship_record",
    "thought_objects",
    "thought_record",
]

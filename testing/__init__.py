"""Test doubles shipped with the package so every task reuses the same fakes.

Importable as ``from lifemodel.testing import FakeClock, FakeDelivery,
FakeMemoryStore, FakePressureSensor, FakeStateStore``. See
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
from .thoughts import thought_objects, thought_record
from .user_model import owner_user_model_objects, owner_user_model_record

__all__ = [
    "FakeActiveSpan",
    "FakeClock",
    "FakeDelivery",
    "FakeMemoryStore",
    "FakePressureSensor",
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
    "owner_user_model_objects",
    "owner_user_model_record",
    "thought_objects",
    "thought_record",
]

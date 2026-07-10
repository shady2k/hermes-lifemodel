"""Tests that the Phase-1 ports are satisfied by their adapters and fakes.

The ports are ``runtime_checkable`` Protocols, so ``isinstance`` verifies the
structural contract each adapter/fake must meet. This is the "imitations before
code" guarantee: a fake and its real adapter answer to the same interface.
Imports no Hermes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from lifemodel.adapters.clock import SystemClock
from lifemodel.adapters.delivery import NoopDelivery
from lifemodel.ports import (
    ClockPort,
    DeliveryPort,
    MemoryPort,
    PressureSensorPort,
    StatePort,
    TickCommitPort,
)
from lifemodel.ports.clock import ClockPort as ClockPortDirect
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing import (
    FakeClock,
    FakeDelivery,
    FakeMemoryStore,
    FakePressureSensor,
    FakeStateStore,
)


def test_ports_package_re_exports_state_port() -> None:
    # StatePort lives in the state package (0.2) but is catalogued in ports/.
    assert StatePort is not None
    assert ClockPort is ClockPortDirect


def test_clock_port_is_satisfied_by_adapter_and_fake() -> None:
    assert isinstance(SystemClock(), ClockPort)
    assert isinstance(FakeClock(datetime.now(UTC)), ClockPort)


def test_delivery_port_is_satisfied_by_adapter_and_fake() -> None:
    assert isinstance(NoopDelivery(), DeliveryPort)
    assert isinstance(FakeDelivery(), DeliveryPort)


def test_state_port_is_satisfied_by_adapter_and_fake(tmp_path: object) -> None:
    clock = FakeClock(datetime.now(UTC))
    assert isinstance(SQLiteRuntimeStore(tmp_path, clock=clock), StatePort)  # type: ignore[arg-type]
    assert isinstance(FakeStateStore(), StatePort)


def test_memory_port_is_satisfied_by_sqlite_store_and_fake(tmp_path: object) -> None:
    clock = FakeClock(datetime.now(UTC))
    assert isinstance(SQLiteRuntimeStore(tmp_path, clock=clock), MemoryPort)  # type: ignore[arg-type]
    assert isinstance(FakeMemoryStore(clock=clock), MemoryPort)


def test_pressure_sensor_port_is_satisfied_by_sqlite_store_and_fakes(tmp_path: object) -> None:
    clock = FakeClock(datetime.now(UTC))
    store = SQLiteRuntimeStore(tmp_path, clock=clock)  # type: ignore[arg-type]
    fake_store = FakeMemoryStore(clock=clock)
    assert isinstance(store, PressureSensorPort)
    assert isinstance(fake_store, PressureSensorPort)
    assert isinstance(FakePressureSensor(fake_store), PressureSensorPort)


def test_tick_commit_port_is_satisfied_by_sqlite_store_and_fake(tmp_path: object) -> None:
    # One real adapter implements every port, so a tick's commit spans vitals +
    # entities in one transaction; the state fake honours the same contract.
    clock = FakeClock(datetime.now(UTC))
    assert isinstance(SQLiteRuntimeStore(tmp_path, clock=clock), TickCommitPort)  # type: ignore[arg-type]
    assert isinstance(FakeStateStore(), TickCommitPort)


def test_clock_adapter_returns_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == datetime.now(UTC).utcoffset()


def test_noop_delivery_drops_the_send_but_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="lifemodel.delivery"):
        NoopDelivery().send("author", "hello there")

    events = [r.getMessage() for r in caplog.records if r.getMessage().startswith("delivery_noop")]
    assert len(events) == 1
    assert "channel=author" in events[0]
    assert f"text_len={len('hello there')}" in events[0]

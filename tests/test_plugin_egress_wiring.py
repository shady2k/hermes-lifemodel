from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import lifemodel


class FakeCtx:
    profile_name = "test-being"

    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}

    def register_command(
        self, name: str, handler: Callable[..., Any], description: str = "", args_hint: str = ""
    ) -> None:
        self.commands[name] = handler


def test_register_starts_service_when_home_origin_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A home origin exists -> start the in-process service. We do NOT gate on
    # reachin_available() at register time (adapters aren't wired yet); the loop
    # decides at runtime. So the service is registered whenever origin is present.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "115679831")

    started: list[str] = []
    monkeypatch.setattr(lifemodel, "default_runner_accessor", lambda: object())
    monkeypatch.setattr(
        lifemodel,
        "register_gateway_service",
        lambda runner, key, factory, **kw: started.append(key) or True,
    )
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: None)

    ctx = FakeCtx()
    lifemodel.register(ctx)  # must not raise
    assert "lifemodel" in ctx.commands
    assert started == ["lifemodel-egress"]


def test_register_skips_service_but_registers_cron_without_home_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No home origin (no TELEGRAM_HOME_CHANNEL) -> no reach-in target, so the
    # in-process service is NOT started; the cron heartbeat is still registered
    # (it is the always-on fallback brain).
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    started: list[str] = []
    heartbeat: list[bool] = []
    monkeypatch.setattr(
        lifemodel, "register_gateway_service", lambda *a, **k: started.append("x") or True
    )
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: heartbeat.append(True))

    ctx = FakeCtx()
    lifemodel.register(ctx)
    assert started == []  # service NOT started (no origin)
    assert heartbeat == [True]  # cron fallback registered

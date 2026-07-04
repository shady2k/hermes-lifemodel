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


def test_register_starts_service_when_reachin_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "115679831")

    started: list[str] = []
    monkeypatch.setattr(lifemodel, "reachin_available", lambda runner: True)
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


def test_register_falls_back_to_cron_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(lifemodel, "reachin_available", lambda runner: False)
    monkeypatch.setattr(lifemodel, "default_runner_accessor", lambda: None)

    started: list[str] = []
    heartbeat: list[bool] = []
    monkeypatch.setattr(
        lifemodel, "register_gateway_service", lambda *a, **k: started.append("x") or True
    )
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: heartbeat.append(True))

    ctx = FakeCtx()
    lifemodel.register(ctx)
    assert started == []  # service NOT started
    assert heartbeat == [True]  # cron fallback registered

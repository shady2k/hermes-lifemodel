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
        self.hooks: list[tuple[str, Callable[..., Any]]] = []

    def register_command(
        self, name: str, handler: Callable[..., Any], description: str = "", args_hint: str = ""
    ) -> None:
        self.commands[name] = handler

    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        self.hooks.append((hook_name, callback))


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


def test_register_defers_service_to_session_start_when_no_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # At register() time there is no running loop (sync gateway startup), so the
    # first register_gateway_service attempt fails; the plugin defers arming to the
    # first on_session_start, where the loop is running.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "115679831")
    monkeypatch.setattr(lifemodel, "default_runner_accessor", lambda: object())
    monkeypatch.setattr(lifemodel, "register_heartbeat", lambda *a, **k: None)

    attempts: list[int] = []

    def _reg(runner: Any, key: str, factory: Any, **kw: Any) -> bool:
        attempts.append(1)
        return len(attempts) >= 2  # register-time fails; the deferred arm succeeds

    monkeypatch.setattr(lifemodel, "register_gateway_service", _reg)

    ctx = FakeCtx()
    lifemodel.register(ctx)

    assert any(h == "on_session_start" for h, _ in ctx.hooks)  # deferred
    cb = next(cb for h, cb in ctx.hooks if h == "on_session_start")
    cb()  # simulate first session start -> arms the service
    assert len(attempts) == 2
    cb()  # idempotent: does not re-register
    assert len(attempts) == 2

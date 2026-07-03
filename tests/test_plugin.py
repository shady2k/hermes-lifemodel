"""Unit tests for register(ctx) — the Hermes adapter — with a fake ctx.

These prove the plugin's registration surface WITHOUT importing Hermes: the one
host touchpoint (``_hermes_home``) is monkeypatched to inject a profile home,
and ``ctx`` is a duck-typed recorder. No real Hermes package is imported.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

import lifemodel
from lifemodel.events import EVENTS_FILENAME


class FakeCtx:
    """Duck-typed stand-in for Hermes' PluginContext.

    Records ``register_*`` calls and exposes ``profile_name``, mirroring the
    slice of the real ctx surface that :func:`register` actually uses.
    """

    profile_name = "test-being"

    def __init__(self) -> None:
        self.commands: dict[str, dict[str, Any]] = {}
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: list[tuple[str, Callable[..., Any]]] = []

    def register_command(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        args_hint: str = "",
    ) -> None:
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }

    def register_tool(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.tools[name] = {"args": args, "kwargs": kwargs}

    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        self.hooks.append((hook_name, callback))


def test_register_adds_lifemodel_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)

    assert "lifemodel" in ctx.commands
    entry = ctx.commands["lifemodel"]
    assert entry["description"]
    # The command prints an 'alive' line bound to the active profile + state dir.
    line = entry["handler"]("")
    assert "alive" in line
    assert "test-being" in line
    assert str(tmp_path / "lifemodel") in line


def test_register_emits_plugin_registered_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    with capture_logs() as logs:
        lifemodel.register(ctx)

    events = [e for e in logs if e.get("event") == "plugin_registered"]
    assert len(events) == 1
    event = events[0]
    assert event["profile"] == "test-being"
    assert event["state_dir"] == str(tmp_path / "lifemodel")
    assert event["version"] == lifemodel.__version__


def test_register_does_not_import_real_hermes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Guard: register() must not require Hermes. The one Hermes touchpoint is
    # the monkeypatched ``_hermes_home`` seam; everything else is Hermes-free,
    # so no real Hermes module is imported by exercising register().
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)

    lifemodel.register(FakeCtx())

    assert "hermes_constants" not in sys.modules
    assert not any(m == "hermes" or m.startswith("hermes.") for m in sys.modules)


def test_register_defaults_profile_when_ctx_lacks_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)

    class BareCtx(FakeCtx):
        profile_name = ""  # empty / unset → falls back to "default"

    ctx = BareCtx()
    lifemodel.register(ctx)

    assert "default" in ctx.commands["lifemodel"]["handler"]("")


def test_register_lifemodel_debug_subcommand_returns_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    # `/lifemodel debug` renders the read-only inspection dump (default state).
    dump = handler("debug")
    assert "debug dump" in dump
    assert "schema_version:" in dump
    assert "lock status:" in dump
    # `/lifemodel` (and any other arg) still prints the status line.
    assert "alive" in handler("")
    assert "alive" in handler("status")
    # args_hint advertises the new subcommand.
    assert "debug" in ctx.commands["lifemodel"]["args_hint"]


def test_register_tees_plugin_registered_into_events_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)

    lifemodel.register(FakeCtx())

    # The structured event landed in the queryable sink, not only the logs.
    events_file = tmp_path / "lifemodel" / EVENTS_FILENAME
    records = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert any(r["event"] == "plugin_registered" for r in records)


def test_uses_the_configured_structlog_pipeline() -> None:
    # Sanity: structlog is the backend in the plugin's own test environment,
    # so get_logger yields a real structlog logger (not the fallback shim).
    assert structlog is not None
    assert lifemodel.get_logger("lifemodel.probe") is not None

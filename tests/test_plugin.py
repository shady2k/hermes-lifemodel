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
    assert str(tmp_path / "workspace" / "lifemodel") in line


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
    assert event["state_dir"] == str(tmp_path / "workspace" / "lifemodel")
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
    assert "read-only" in dump
    assert "PHYSIOLOGY" in dump
    # `/lifemodel status` (and any other unrecognized arg) prints the status line.
    assert "alive" in handler("status")
    # args_hint advertises the new subcommand.
    assert "debug" in ctx.commands["lifemodel"]["args_hint"]


def test_register_lifemodel_help_subcommand_lists_subcommands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    text = handler("help")
    # Every registered subcommand shows up with its one-line description —
    # the registry is the single source of truth for this text.
    for name, info in lifemodel._SUBCOMMANDS.items():
        assert name in text
        assert info.description in text


def test_register_lifemodel_bare_command_includes_full_command_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    bare = handler("")
    # Bare `/lifemodel` keeps the status line...
    assert "alive" in bare
    # ...and surfaces the full subcommand list (same as `/lifemodel help`),
    # not a truncated footer — discoverability without a second round trip.
    for name, info in lifemodel._SUBCOMMANDS.items():
        assert name in bare
        assert info.description in bare


def test_register_lifemodel_args_hint_derived_from_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)

    args_hint = ctx.commands["lifemodel"]["args_hint"]
    # args_hint must be built from the registry, not a separately hardcoded
    # string, so the two can never drift apart.
    for name in lifemodel._SUBCOMMANDS:
        assert name in args_hint


def test_register_lifemodel_help_flags_mutating_subcommands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    lines = {line.strip(): line for line in handler("help").splitlines()}
    for name, info in lifemodel._SUBCOMMANDS.items():
        line = next(text for text in lines.values() if text.strip().startswith(name))
        if info.mutating:
            assert "[mutating]" in line, line
        else:
            assert "[mutating]" not in line, line


def test_register_lifemodel_nudge_subcommand_mutates_state_via_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("nudge 2.5")

    assert "(mutating)" in out
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    assert json.loads(state_file.read_text())["u"] == 2.5


def test_register_lifemodel_reset_subcommand_writes_a_fresh_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    handler("nudge 5")  # dirty the state first
    handler("reset")

    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    persisted = json.loads(state_file.read_text())
    assert persisted["u"] == 0.0
    assert persisted["tick_count"] == 0
    assert persisted["proactive_send_log"] == []


def test_register_lifemodel_set_subcommand_rejects_unwhitelisted_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("set tick_count 99")

    assert "not writable" in out
    state_file = tmp_path / "workspace" / "lifemodel" / "state.json"
    assert not state_file.exists()  # rejected before ever touching the store


def test_register_tees_plugin_registered_into_events_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)

    lifemodel.register(FakeCtx())

    # The structured event landed in the queryable sink, not only the logs.
    events_file = tmp_path / "workspace" / "lifemodel" / EVENTS_FILENAME
    records = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert any(r["event"] == "plugin_registered" for r in records)


def test_uses_the_configured_structlog_pipeline() -> None:
    # Sanity: structlog is the backend in the plugin's own test environment,
    # so get_logger yields a real structlog logger (not the fallback shim).
    assert structlog is not None
    assert lifemodel.get_logger("lifemodel.probe") is not None

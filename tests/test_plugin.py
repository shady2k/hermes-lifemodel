"""Unit tests for register(ctx) — the Hermes adapter — with a fake ctx.

These prove the plugin's registration surface WITHOUT importing Hermes: the one
host touchpoint (``_hermes_home``) is monkeypatched to inject a profile home,
and ``ctx`` is a duck-typed recorder. No real Hermes package is imported.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

import lifemodel
import lifemodel.log as lm_logging
from lifemodel.adapters.clock import SystemClock
from lifemodel.config import write_log_level
from lifemodel.events import EVENTS_FILENAME
from lifemodel.state.errors import StateSchemaError
from lifemodel.state.model import State
from lifemodel.state.sqlite_store import SQLiteRuntimeStore


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
    # Every registered subcommand shows up bold (new /status-style house
    # format: **name** — description) with its one-line description — the
    # registry is the single source of truth for this text.
    for name, info in lifemodel._SUBCOMMANDS.items():
        assert f"**{name}**" in text
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
        assert f"**{name}**" in bare
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
        line = next(text for text in lines.values() if text.strip().startswith(f"**{name}**"))
        if info.mutating:
            assert "[mutating]" in line, line
        else:
            assert "[mutating]" not in line, line


def test_register_lifemodel_help_command_list_has_no_column_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The old `_command_list()` space-padded a column to align descriptions
    (``f"  {name:<{width}}  {description}"``), which goes ragged in
    Telegram's proportional font. The new /status-style rendering drops that
    padding entirely: no line should contain a run of 2+ spaces, and every
    command name renders bold (``**name**``), matching debug.py's
    ``**label:** value`` convention."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    text = handler("help")
    for name in lifemodel._SUBCOMMANDS:
        assert f"**{name}**" in text
    for line in text.splitlines():
        assert "  " not in line, line


def _committed_state(tmp_path: Path) -> State:
    """Read back the being's persisted state through a fresh ``StatePort``
    handle over the same profile-scoped dir ``register(ctx)`` wired — the
    SQLite equivalent of the old "read state.json back" assertion."""
    sdir = tmp_path / "workspace" / "lifemodel"
    return SQLiteRuntimeStore(sdir, clock=SystemClock()).load()


def test_register_lifemodel_nudge_subcommand_mutates_state_via_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("nudge 2.5")

    assert "(mutating)" in out
    assert _committed_state(tmp_path).u == 2.5


def test_register_lifemodel_think_subcommand_seeds_a_thought(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lifemodel.core.thought_view import read_live_thoughts

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("think did the owner ever hear back")

    assert "(mutating)" in out
    store = SQLiteRuntimeStore(tmp_path / "workspace" / "lifemodel", clock=SystemClock())
    thoughts = read_live_thoughts(store)
    assert len(thoughts) == 1
    assert thoughts[0].content == "did the owner ever hear back"


def test_register_lifemodel_reset_subcommand_writes_a_fresh_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    handler("nudge 5")  # dirty the state first
    handler("reset")

    persisted = _committed_state(tmp_path)
    assert persisted.u == 0.0
    assert persisted.tick_count == 0
    assert persisted.proactive_send_log == []


def test_register_lifemodel_set_subcommand_rejects_unwhitelisted_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("set tick_count 99")

    assert "not writable" in out
    # Rejected before ever committing: unlike JsonStateStore (which never
    # touched the filesystem until commit), constructing SQLiteRuntimeStore
    # always creates lifemodel.sqlite (recovery/migration need the file) —
    # so "untouched" is checked at the State level, not file existence.
    assert _committed_state(tmp_path) == State()


def test_register_lifemodel_command_catches_handler_exception_and_returns_error_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """lm-zhh: when a subcommand handler RAISES (e.g. StateSchemaError, the
    confirmed incident mechanism -- '/lifemodel set ...' loaded state, hit a
    newer/unsupported schema, and raised), that exception must NOT propagate
    out of ``lifemodel_command``. Left uncaught, Hermes' gateway degrades it
    into a misleading generic "Unknown command /lifemodel" notice instead of
    the real reason. The command boundary must catch it and return a
    readable, owner-facing error string carrying the actual reason, and must
    log the failure too."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    def _boom(*args: Any, **kwargs: Any) -> str:
        raise StateSchemaError("schema_version=99 is newer than this build supports")

    monkeypatch.setattr(lifemodel, "set_field_for_dir", _boom)

    with capture_logs() as logs:
        out = handler("set u 1")

    # Owner-facing text: readable, prefixed as a lifemodel error, carries the
    # real reason -- never the generic "unknown command" degradation.
    assert "команда не выполнена" in out
    assert "schema_version=99 is newer than this build supports" in out
    # The failure is also recorded, not only shown to the owner.
    failures = [e for e in logs if e.get("event") == "lifemodel_command_failed"]
    assert len(failures) == 1
    assert "schema_version=99" in failures[0]["error"]


def test_register_lifemodel_command_success_path_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression guard: normal (non-raising) subcommands are unaffected by
    the exception-catching boundary -- bare/help/status still return their
    usual output, not an error string."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    assert "alive" in handler("status")
    assert "alive" in handler("")
    assert "**help**" in handler("help")


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


# --- loglevel (lm-j2w B2): persisted log level + `/lifemodel loglevel` -------


def test_register_boots_at_the_persisted_log_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    sdir = tmp_path / "workspace" / "lifemodel"
    write_log_level(sdir, "warning")
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)

    lifemodel.register(FakeCtx())

    assert lm_logging._effective_level == logging.WARNING


def test_register_boots_at_info_when_no_config_is_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)

    lifemodel.register(FakeCtx())

    assert lm_logging._effective_level == logging.INFO


def test_loglevel_in_subcommands_registry() -> None:
    assert "loglevel" in lifemodel._SUBCOMMANDS
    assert lifemodel._SUBCOMMANDS["loglevel"].mutating is True


def test_register_lifemodel_loglevel_no_arg_returns_current_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("loglevel")

    assert "info" in out


def test_register_lifemodel_loglevel_sets_and_persists_and_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    # Guard against leaking the real (non-monkeypatched) log.configure() call
    # this test triggers into later tests: monkeypatch reverts _effective_level
    # to its pre-test value on teardown regardless of what happens in between.
    monkeypatch.setattr(lm_logging, "_effective_level", logging.INFO)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("loglevel debug")

    assert "info" in out and "debug" in out  # old -> new echoed
    sdir = tmp_path / "workspace" / "lifemodel"
    from lifemodel.config import read_log_level

    assert read_log_level(sdir) == "debug"
    assert lm_logging._effective_level == logging.DEBUG


def test_register_does_not_raise_on_invalid_persisted_log_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A hand-edited config.json with a plausible-but-invalid level name (e.g.
    # a "warn" typo for "warning") must never take the plugin down at load —
    # register() degrades to the default level instead of letting
    # parse_log_level() raise ValueError out of registration.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    sdir = tmp_path / "workspace" / "lifemodel"
    from lifemodel.config import write_config

    write_config(sdir, {"log_level": "warn"})
    monkeypatch.setattr(lm_logging, "_effective_level", logging.DEBUG)

    lifemodel.register(FakeCtx())  # must not raise

    assert lm_logging._effective_level == logging.INFO


def test_register_lifemodel_loglevel_invalid_arg_returns_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("loglevel loud")

    for name in lm_logging.LOG_LEVEL_NAMES:
        assert name in out

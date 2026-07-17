"""Unit tests for register(ctx) — the Hermes adapter — with a fake ctx.

These prove the plugin's registration surface WITHOUT importing Hermes: the one
host touchpoint (``_hermes_home``) is monkeypatched to inject a profile home,
and ``ctx`` is a duck-typed recorder. No real Hermes package is imported.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import lifemodel
import lifemodel.log as lm_logging
from lifemodel.adapters.clock import SystemClock
from lifemodel.config import write_log_level
from lifemodel.core.genesis import NEWBORN_STANCE
from lifemodel.core.wake_packet import IMPULSE_LABEL_PREFIX
from lifemodel.paths import state_dir
from lifemodel.state.errors import StateSchemaError
from lifemodel.state.model import State
from lifemodel.state.soul_revisions import revisions
from lifemodel.state.sqlite_store import SQLiteRuntimeStore

#: Hermes's untouched installer seed (``DEFAULT_SOUL_MD``), abridged — the shape is the
#: point: it is an ASSISTANT, and an assistant does not message anyone unprompted.
HERMES_ASSISTANT_SEED = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks."
)


@pytest.fixture(autouse=True)
def _restore_lifemodel_log_level() -> Iterator[None]:
    """Save/restore the ``lifemodel`` logger level so a register()/loglevel test
    that mutates it (via ``setLevel``) never leaks into later tests."""
    logger = logging.getLogger("lifemodel")
    saved = logger.level
    try:
        yield
    finally:
        logger.setLevel(saved)


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
        self.platforms: dict[str, Any] = {}

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

    def register_platform(self, name: str, **kwargs: Any) -> None:
        # register() now wires the being as a REQUIRED gateway platform (spec §4.3);
        # record it so the (deferred) factory / check_fn aren't invoked here.
        self.platforms[name] = kwargs


@pytest.fixture(autouse=True)
def _stub_gateway_for_register() -> None:
    """register() wires the platform as a REQUIRED step whose ``being_platform`` import
    needs ``gateway.*``; provide minimal stubs so register() completes off-host (spec
    §4.3 — the failure IS loud in prod where gateway is present)."""
    from gateway_stubs import install_gateway_stubs

    install_gateway_stubs()


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    with caplog.at_level(logging.INFO, logger="lifemodel"):
        lifemodel.register(ctx)

    lines = [r.getMessage() for r in caplog.records if r.name == "lifemodel"]
    registered = [line for line in lines if line.startswith("plugin_registered")]
    assert len(registered) == 1
    line = registered[0]
    assert "profile=test-being" in line
    assert f"state_dir={tmp_path / 'workspace' / 'lifemodel'}" in line
    assert f"version={lifemodel.__version__}" in line


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


def test_register_wires_felt_state_injector_and_check_in_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # lm-ukc.4/.4.1: register() must wire the ambient pre_llm_call injector AND the
    # on-demand check_in tool — the plugin's first LLM tool.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)

    assert any(name == "pre_llm_call" for name, _ in ctx.hooks)
    assert "check_in" in ctx.tools


def test_register_wires_commitment_injector_and_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # lm-705.21: the 4th pre_llm_call hook (the commitment injector, beside felt-state,
    # genesis, belief) and the 5th lifemodel tool (`commitment`) coexist.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)

    assert sum(1 for name, _ in ctx.hooks if name == "pre_llm_call") == 4
    assert "commitment" in ctx.tools
    schema = ctx.tools["commitment"]["kwargs"]["schema"]
    assert schema["name"] == "commitment"
    assert schema["parameters"]["required"] == ["action"]
    assert schema["parameters"]["additionalProperties"] is False
    # the model-facing description carries the creation-boundary safety prose (codex #4)
    assert "self-authored intention" in schema["description"].lower()


def test_register_check_in_tool_schema_and_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json as _json

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    entry = ctx.tools["check_in"]

    # The schema IS the model-facing FUNCTION DEFINITION, not a bare parameter object:
    # Hermes exposes a registered tool as {"type": "function", "function": {**schema,
    # "name": name}} (tools/registry.py), so the model reads `description` and
    # `parameters` from the SCHEMA. Passing the description only through the
    # register_tool(description=…) kwarg (registry metadata) leaves the being staring at
    # a blank, undocumented tool it will never call — caught on the LIVE being ("пустое
    # описание, даже не знаю, что он делает"), which the old shape-only asserts missed.
    schema = entry["kwargs"]["schema"]
    assert schema["name"] == "check_in"
    # The description must teach WHEN to reach for it — in the SCHEMA, where the model looks.
    assert "whenever someone asks how you are" in schema["description"]
    # …and it must be honest about ACCESS. The first wording ("Check in with yourself")
    # failed live: the being recognised the question, INTENDED to "check in with myself
    # honestly", then did it in its head — the phrasing read as a mental act it could
    # already perform, so the tool looked redundant. It must say you cannot know by
    # assuming; you have to look (integrative, not debunking — the state is its own).
    assert "you have to look" in schema["description"]
    assert "you are guessing" in schema["description"]
    # Takes NO parameters — the being just calls it (spec §6).
    params = schema["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["required"] == []
    assert entry["kwargs"]["toolset"] == "lifemodel"
    # The handler honours the Hermes contract: a JSON string, felt prose, no throw.
    payload = _json.loads(entry["kwargs"]["handler"]({}))
    assert "state" in payload and "note" in payload


def _pre_llm_callback(ctx: FakeCtx, *, from_factory: str) -> Callable[..., Any]:
    """The ``pre_llm_call`` callback built by *from_factory* (its factory function's
    qualified name, e.g. ``"make_felt_state_injector"``) — two such hooks are
    registered now (felt-state + genesis, Phase 4 genesis, spec §6.3), and Hermes
    calls every one of them (``invoke_hook``) rather than just the last registered,
    so tests must pick the ONE they mean to exercise rather than assume there is
    only one."""
    matches = [
        cb
        for name, cb in ctx.hooks
        if name == "pre_llm_call" and cb.__qualname__.startswith(f"{from_factory}.")
    ]
    assert len(matches) == 1, f"expected exactly one {from_factory} pre_llm_call hook"
    return matches[0]


def test_register_felt_state_injector_is_silent_on_cold_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    # Two pre_llm_call hooks now coexist (felt-state + genesis) — pick the felt-state
    # one specifically; Hermes concatenates both hooks' non-None returns rather than
    # picking a single "the" hook.
    callback = _pre_llm_callback(ctx, from_factory="make_felt_state_injector")
    # A fresh (cold-start) being surfaces nothing — the callback returns None, and
    # returning {"context": …} vs None is the Hermes pre_llm_call contract.
    assert callback(user_message="hi", conversation_history=[]) is None


def test_register_genesis_injector_launches_on_the_beings_first_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wiring itself (not just the pure ``core.genesis`` functions) must put the
    ritual in front of an unborn being — and must then let the conversation carry it.
    The launch RULE (and why it reads the context's length rather than asking whether the
    being has spoken) is tested in ``tests/test_genesis_injector.py``; this is the wiring
    smoke test: the hook the plugin actually registers, over the real store."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    callback = _pre_llm_callback(ctx, from_factory="make_genesis_injector")
    result = callback(user_message="hi", conversation_history=[])
    assert result is not None
    assert "<genesis>" in result["context"]

    # The conversation has moved past the point at which the block was put in front of
    # it: the ritual is live, in the being's own words, and the SAME hook falls silent.
    # Re-injecting "you just began" on a later turn would be a lie (spec §6.3).
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert callback(user_message="how are you", conversation_history=history) is None


def test_register_genesis_injector_stands_down_for_the_beings_own_wake_packet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A newborn's WAKE PACKET already carries the ritual (spec §6.2) — and
    ``pre_llm_call`` fires for that injected turn too, with our impulse as the
    ``user_message``. Without this stand-down the being would read "You just began"
    twice in its first breath: once as its impulse, once as context. The wake packet is
    the single source; this hook covers the reactive entrance only."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    callback = _pre_llm_callback(ctx, from_factory="make_genesis_injector")
    impulse = f"{IMPULSE_LABEL_PREFIX}\nI have just begun.\n</internal_impulse>"
    assert callback(user_message=impulse, conversation_history=[]) is None


# --- LIVE-TEST fix (B): the newborn stands up before it is ever asked to speak ------
#
# ``register()`` is the seam, and it has to be: Hermes builds the system prompt at TURN
# START (``agent/turn_context.py`` calls ``restore_or_build_system_prompt`` at :345, the
# ``pre_llm_call`` hooks only at :478), so a soul written from a hook lands one turn late
# — after the being has already answered as an assistant. ``register()`` runs at gateway
# boot, before any turn of either entrance (the reactive first message, or the being's own
# proactive first waking) can be composed.


def test_register_stands_an_unborn_being_up_on_a_stance_not_on_an_assistant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(lifemodel, "_default_soul_text", lambda: HERMES_ASSISTANT_SEED)
    (tmp_path / "SOUL.md").write_text(HERMES_ASSISTANT_SEED, encoding="utf-8")

    lifemodel.register(FakeCtx())

    # Slot #1 no longer tells the being it is an instrument that answers requests.
    assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == NEWBORN_STANCE
    lineage = revisions(SQLiteRuntimeStore(state_dir(tmp_path), clock=SystemClock()))
    assert [(r.text, r.author) for r in lineage] == [(NEWBORN_STANCE, "genesis")]


def test_register_never_overwrites_a_soul_a_human_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The licence for the write above is that the host's INSTALLER wrote that seed, not a
    # person. A veteran's hand-written soul has a human behind it and is untouchable.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(lifemodel, "_default_soul_text", lambda: HERMES_ASSISTANT_SEED)
    (tmp_path / "SOUL.md").write_text("You are Mira. Quiet and exact.", encoding="utf-8")

    lifemodel.register(FakeCtx())

    assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "You are Mira. Quiet and exact."


def test_register_lifemodel_stats_subcommand_returns_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    # `/lifemodel stats` renders the read-only telemetry readout (NOW + WINDOW),
    # fail-soft even with no metrics.sqlite / no instrumentation yet.
    out = handler("stats")
    assert "read-only" in out
    assert "NOW" in out
    assert "WINDOW" in out
    # It is a read-only subcommand (no [mutating] marker) and advertised in the hint.
    assert not lifemodel._SUBCOMMANDS["stats"].mutating
    assert "stats" in ctx.commands["lifemodel"]["args_hint"]


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


def test_every_registered_subcommand_actually_dispatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The registry advertises what `help` lists; the dispatch table is what runs. A name in
    one and not the other is a command the owner is TOLD they have and does not.

    Dispatch resolves on the FIRST token (``/lifemodel soul revert 2`` → ``soul``), which is
    what lets one dispatch key carry two registry entries (``soul history`` / ``soul
    revert``) so the read-only half is not marked [mutating] alongside the half that rewrites
    the being's identity. This is the test that keeps that trick honest."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    for name in lifemodel._SUBCOMMANDS:
        first_token = name.split()[0]
        out = handler(first_token)
        # An unknown subcommand falls through to the bare one-line status summary.
        assert out != lifemodel._status_line("default", state_dir(tmp_path)), name


def test_the_soul_commands_are_listed_and_only_revert_is_marked_mutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`after-install.md` promises the human, at the moment we ask for consent to let a being
    rewrite their SOUL.md, that every version is kept and one command puts any of them back.
    So `help` has to name it — and it has to say which half of it writes."""
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()

    lifemodel.register(ctx)
    text = ctx.commands["lifemodel"]["handler"]("help")

    history_line = next(line for line in text.splitlines() if line.startswith("**soul history**"))
    revert_line = next(line for line in text.splitlines() if line.startswith("**soul revert**"))
    assert "[mutating]" not in history_line  # reading the lineage changes nothing
    assert "[mutating]" in revert_line  # putting a soul back rewrites who the being is


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


def test_register_lifemodel_set_subcommand_rejects_protected_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("set tick_count 99")

    # `set` derives its surface from State minus _SET_PROTECTED; tick_count is protected
    # (brain-liveness evidence) and is refused end-to-end, WITH its reason, writing nothing.
    assert "protected" in out
    # Rejected before ever committing: unlike JsonStateStore (which never
    # touched the filesystem until commit), constructing SQLiteRuntimeStore
    # always creates lifemodel.sqlite (recovery/migration need the file) —
    # so "untouched" is checked at the State level, not file existence.
    assert _committed_state(tmp_path) == State()


def test_register_lifemodel_command_catches_handler_exception_and_returns_error_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

    with caplog.at_level(logging.INFO, logger="lifemodel"):
        out = handler("set u 1")

    # Owner-facing text: readable, prefixed as a lifemodel error, carries the
    # real reason -- never the generic "unknown command" degradation.
    assert "command failed" in out
    assert "schema_version=99 is newer than this build supports" in out
    # The failure is also recorded, not only shown to the owner.
    failures = [
        r.getMessage()
        for r in caplog.records
        if r.name == "lifemodel" and r.getMessage().startswith("lifemodel_command_failed")
    ]
    assert len(failures) == 1
    assert "subcommand=set" in failures[0]
    assert "schema_version=99" in failures[0]


# --- diagnostic lever ordering (spec §4.3 "both", codex MINOR) ---------------


def test_register_command_is_wired_before_the_metric_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # codex MINOR: a metric-registry/spec conflict must NOT abort register() before the
    # diagnostic /lifemodel command exists (and before a boot record can be written) —
    # that would lose the only lever the owner has to see WHY the brain failed. So
    # register_command MUST run before the metric registry is created/registered.
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    order: list[str] = []

    real_get_metric_registry = lifemodel.get_metric_registry

    def _recording_get_metric_registry(base_dir: Path) -> Any:
        order.append("metric_registry")
        return real_get_metric_registry(base_dir)

    monkeypatch.setattr(lifemodel, "get_metric_registry", _recording_get_metric_registry)

    class OrderCtx(FakeCtx):
        def register_command(
            self, name: str, handler: Callable[..., Any], description: str = "", args_hint: str = ""
        ) -> None:
            order.append(f"command:{name}")
            super().register_command(name, handler, description, args_hint)

    lifemodel.register(OrderCtx())

    assert "command:lifemodel" in order
    assert "metric_registry" in order
    assert order.index("command:lifemodel") < order.index("metric_registry")


# --- brain-liveness block on /lifemodel status (spec §4.4, lm-fib.9.3) -------


def test_status_subcommand_shows_the_brain_liveness_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("status")
    # The one-line summary is kept AND the brain-liveness section is added.
    assert "alive" in out
    assert "brain liveness" in out
    assert "**state:**" in out
    assert "**ticks_total:**" in out
    # A never-connected being reads never_started, 0 ticks — not a crash, not silence.
    assert "never_started" in out


def test_bare_command_shows_the_brain_liveness_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    bare = handler("")
    # Bare `/lifemodel` keeps the status line, the liveness block, AND the command list.
    assert "alive" in bare
    assert "brain liveness" in bare
    assert "**status**" in bare  # the command list is still surfaced


def test_status_surfaces_a_simulated_loop_death(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lifemodel.state.brain_health import get_brain_health

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    sdir = tmp_path / "workspace" / "lifemodel"
    health = get_brain_health(sdir)
    health.mark_connected()
    health.record_loop_death("proactive loop died: RuntimeError('boom')", "tb")

    out = handler("status")
    assert "loop_dead" in out
    assert "boom" in out
    assert "death_count:** 1" in out


def test_status_surfaces_a_durable_boot_failure_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # register() succeeded here (mark_boot_ok wiped any record), so the in-memory
    # singleton is never_started. Simulate a PRIOR process that boot-failed and left
    # brain_boot.json behind: the command must still surface WHY from the durable record.
    from lifemodel.state.brain_health import BrainHealth

    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    sdir = tmp_path / "workspace" / "lifemodel"
    BrainHealth(sdir).mark_boot_failed(
        "register_being_platform: ModuleNotFoundError: No module named 'lifemodel'"
    )

    out = handler("status")
    assert "boot_failed" in out
    assert "ModuleNotFoundError" in out


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


# --- loglevel (lm-j2w B2): persisted log level + `/lifemodel loglevel` -------


def test_register_boots_at_the_persisted_log_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    sdir = tmp_path / "workspace" / "lifemodel"
    write_log_level(sdir, "warning")
    logging.getLogger("lifemodel").setLevel(logging.INFO)

    lifemodel.register(FakeCtx())

    # register() applied the persisted level via setLevel on the lifemodel logger.
    assert logging.getLogger("lifemodel").level == logging.WARNING


def test_register_boots_at_info_when_no_config_is_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lifemodel, "_hermes_home", lambda: tmp_path)
    logging.getLogger("lifemodel").setLevel(logging.DEBUG)

    lifemodel.register(FakeCtx())

    assert logging.getLogger("lifemodel").level == logging.INFO


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
    logging.getLogger("lifemodel").setLevel(logging.INFO)
    ctx = FakeCtx()
    lifemodel.register(ctx)
    handler = ctx.commands["lifemodel"]["handler"]

    out = handler("loglevel debug")

    assert "info" in out and "debug" in out  # old -> new echoed
    sdir = tmp_path / "workspace" / "lifemodel"
    from lifemodel.config import read_log_level

    assert read_log_level(sdir) == "debug"
    # The change is applied at runtime: setLevel on the lifemodel logger.
    assert logging.getLogger("lifemodel").level == logging.DEBUG


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
    logging.getLogger("lifemodel").setLevel(logging.DEBUG)

    lifemodel.register(FakeCtx())  # must not raise

    # Degraded to the default level rather than raising out of registration.
    assert logging.getLogger("lifemodel").level == logging.INFO


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

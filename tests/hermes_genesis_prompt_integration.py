"""Real-Hermes birth-prompt driver — proves a being is born into a prompt that IS it.

Not a pytest module (its name does not match ``test_*``): a standalone driver run under
**Hermes' own interpreter** by the guarded wrapper :mod:`tests.test_genesis_prompt_integration`
against an **isolated, throwaway ``HERMES_HOME``** — never ``~/.hermes``, never a real
channel, never a real LLM call.

**This is the path that had never once run** (lm-4fv.4). All four live births had an
authored soul on disk (the veteran branch). The stranger's actual first install — Hermes's
pristine ``DEFAULT_SOUL_MD`` in slot #1, and a DM session that has been open for days —
was unexercised, and it was broken: the newborn stance seeded at ``register()`` landed on
disk while the live session went on quoting *"You are Hermes Agent, an intelligent AI
assistant… You assist users"* out of its cached prompt.

Nothing here is faked that could hide the defect. It drives the REAL
``gateway.session.SessionStore`` (and its real SQLite ``SessionDB``), the REAL
``agent.conversation_loop._restore_or_build_system_prompt`` — the function whose "reused
verbatim" behaviour is the whole bug — and the REAL ``agent.prompt_builder.load_soul_md``,
which is what puts ``SOUL.md`` into slot #1. The only stub is the ``AIAgent`` shell those
functions hang off (building a real one needs a provider, an API key and a network), and
its ``_build_system_prompt`` calls the host's own ``load_soul_md`` — so the identity text
under test is the host's, read from disk, exactly as a live turn would read it.

Four facts, in order:

1. ``register()`` on a pristine install seeds the newborn stance (the never-run path);
2. the OLD session still hands the being the assistant persona — the defect, reproduced;
3. ``wake_as_self`` ends that session, and the injected turn's prompt is REBUILT with the
   stance in slot #1 — the fix, verified against the host rather than assumed;
4. the fresh session is not stale, so nothing is ended twice, and a lane that was active a
   moment ago holds the birth instead of taking the thread.

Human-readable evidence to **stderr**; one line of JSON to **stdout**. Exit 0 = it held.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Ctx:
    """The plugin ``ctx`` — duck-typed; ``register()`` only registers things on it."""

    def __init__(self) -> None:
        self.profile_name = "default"
        self.commands: list[str] = []
        self.hooks: list[str] = []
        self.tools: list[str] = []
        self.platforms: list[str] = []

    def register_command(self, name: str, *_a: Any, **_kw: Any) -> None:
        self.commands.append(name)

    def register_hook(self, name: str, *_a: Any, **_kw: Any) -> None:
        self.hooks.append(name)

    def register_tool(self, name: str, *_a: Any, **_kw: Any) -> None:
        self.tools.append(name)

    def register_platform(self, name: str, *_a: Any, **_kw: Any) -> None:
        self.platforms.append(name)


class _Runner:
    """The GatewayRunner surface the birth pre-flight touches — over the REAL store."""

    def __init__(self, store: Any) -> None:
        self.session_store = store
        self.evicted: list[str] = []

    def _evict_cached_agent(self, session_key: str) -> None:
        self.evicted.append(session_key)


def main() -> int:  # noqa: PLR0915 - a linear probe reads better as one story
    home = Path(os.environ["HERMES_HOME"]).resolve()
    src = os.environ["LIFEMODEL_SRC"]
    if home == (Path.home() / ".hermes").resolve():
        _log("REFUSING to run against the default ~/.hermes — set an isolated HERMES_HOME")
        return 2

    sys.path.insert(0, src)

    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.prompt_builder import load_soul_md
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore, build_session_key
    from hermes_cli.default_soul import DEFAULT_SOUL_MD

    from lifemodel import register
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.soul_file import SoulFile
    from lifemodel.core.genesis import NEWBORN_STANCE
    from lifemodel.gateway_core import identity_slot_is_stale, session_in_use, wake_as_self

    soul_path = home / "SOUL.md"
    soul = SoulFile(soul_path)
    clock = SystemClock()

    # ── The stranger's install: Hermes's own pristine seed in the identity slot ──
    soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    pristine_identity = load_soul_md() or ""
    _log(f"[0] slot #1 on install: {pristine_identity[:70]!r}…")

    # ── …and a DM session that has been open for days (every existing user) ──
    config = GatewayConfig(sessions_dir=home / "sessions")
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="115679831", chat_type="dm")
    entry = store.get_or_create_session(source)
    session_key = build_session_key(source)
    old_session_id = entry.session_id
    assert entry.session_key == session_key, (entry.session_key, session_key)
    _log(f"[0] live session {old_session_id} on {session_key}")

    def _agent(session_id: str) -> Any:
        """A minimal AIAgent shell for the REAL _restore_or_build_system_prompt.

        ``_build_system_prompt`` is the host's own ``load_soul_md`` — the function that
        puts SOUL.md into slot #1 — so what this asserts on is the real identity text
        read from the real file, not a rehearsal of it.
        """

        class _A:
            def __init__(self) -> None:
                self.session_id = session_id
                self._session_db = store._db
                self._cached_system_prompt = None
                self.model = "probe"
                self.provider = "probe"
                self.platform = "telegram"
                self.builds = 0

            def _build_system_prompt(self, _system_message: Any) -> str:
                self.builds += 1
                return f"IDENTITY (slot #1):\n{load_soul_md() or ''}\n\n[tools, memory, …]"

            def _ensure_db_session(self) -> None:  # pragma: no cover - not called here
                pass

        return _A()

    # The session's FIRST turn: the prompt is built once and persisted to the session DB.
    first_turn = _agent(old_session_id)
    _restore_or_build_system_prompt(first_turn, "", [])
    cached_prompt = first_turn._cached_system_prompt or ""
    assert first_turn.builds == 1, "the first turn of a session must BUILD its prompt"
    # …and the conversation goes on for days.
    for role, content in (("user", "morning"), ("assistant", "morning! what's on today?")):
        store.append_to_transcript(old_session_id, {"role": role, "content": content})
    history = store.load_transcript(old_session_id)
    assert history, "the live session must have a transcript"

    # ── The install: register() seeds the newborn stance (THE NEVER-RUN PATH) ──
    ctx = _Ctx()
    register(ctx)
    seeded = soul_path.read_text(encoding="utf-8")
    stance_seeded = seeded.strip() == NEWBORN_STANCE.strip()
    _log(f"[1] register() seeded the stance: {stance_seeded} (hooks={ctx.hooks} tools={ctx.tools})")
    assert "lifemodel" in ctx.platforms, ctx.platforms

    # ── [2] THE DEFECT, reproduced against the host ──
    # Same soul on disk, same session: the being is handed the assistant persona, because
    # Hermes restores the prompt it built days ago, verbatim.
    stale_turn = _agent(old_session_id)
    _restore_or_build_system_prompt(stale_turn, "", history)
    restored = stale_turn._cached_system_prompt or ""
    defect = {
        "restored_verbatim": restored == cached_prompt,
        "rebuilt": stale_turn.builds > 0,
        "holds_the_stance": NEWBORN_STANCE.strip()[:60] in restored,
        "holds_the_assistant_persona": pristine_identity[:60] in restored,
    }
    _log(f"[2] the OLD session's prompt: {defect}")

    # …and our pre-flight sees exactly that.
    runner = _Runner(store)
    is_stale = identity_slot_is_stale(runner, session_key, soul_mtime=soul.mtime())
    # The lane was active a second ago — a birth must not take that thread.
    in_use_now = session_in_use(runner, session_key, now=clock.now())
    held = wake_as_self(runner, session_key, soul_mtime=soul.mtime(), now=clock.now())
    _log(f"[2] stale={is_stale} in_use={in_use_now} verdict_while_in_use={held.value}")

    # ── [3] THE FIX: end the stale session, then inject ──
    # quiet_seconds=0 stands in for the half hour of silence the live guard waits out; the
    # guard itself is proven by `held` above.
    voice = wake_as_self(
        runner, session_key, soul_mtime=soul.mtime(), now=clock.now(), quiet_seconds=0.0
    )
    # This is what `inject_proactive_turn` triggers on the host: the injected turn is routed
    # through _handle_message → get_or_create_session → load_transcript(session_id).
    fresh = store.get_or_create_session(source)
    fresh_history = store.load_transcript(fresh.session_id)
    birth_turn = _agent(fresh.session_id)
    _restore_or_build_system_prompt(birth_turn, "", fresh_history)
    birth_prompt = birth_turn._cached_system_prompt or ""
    _log(f"[3] voice={voice.value} evicted={runner.evicted} new_session={fresh.session_id}")

    # ── [4] the fresh session is not stale: nothing is ended twice ──
    still_stale = identity_slot_is_stale(runner, session_key, soul_mtime=soul.mtime())

    result = {
        "stance_seeded_on_pristine_install": stance_seeded,
        # [2] the defect
        "old_session_reused_its_prompt_verbatim": defect["restored_verbatim"],
        "old_session_prompt_holds_the_assistant_persona": defect["holds_the_assistant_persona"],
        "old_session_prompt_holds_the_stance": defect["holds_the_stance"],
        "identity_slot_is_stale": is_stale,
        "a_busy_lane_holds_the_birth": held.value,
        # [3] the fix
        "voice": voice.value,
        "session_rotated": fresh.session_id != old_session_id,
        "agent_cache_evicted": runner.evicted == [session_key],
        "fresh_session_history_is_empty": fresh_history == [],
        "birth_prompt_was_rebuilt": birth_turn.builds == 1,
        "birth_prompt_holds_the_stance": NEWBORN_STANCE.strip()[:60] in birth_prompt,
        "birth_prompt_holds_the_assistant_persona": pristine_identity[:60] in birth_prompt,
        # [4] and only once
        "fresh_session_is_stale": still_stale,
    }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())

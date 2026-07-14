"""``BeingAdapter`` fail-loud: connect()/loop-death/check_fn (spec §4.3, items 4/5/7).

``adapters/being_platform.py`` imports ``gateway.*`` at module load, so — like the
Slice-1 register smoke test — these tests install minimal ``gateway.*`` stubs in
``sys.modules`` and import the adapter fresh under them. The stubs give
``BasePlatformAdapter`` the lifecycle hooks the adapter calls
(``_mark_connected`` / ``_set_fatal_error`` / ``_notify_fatal_error`` …) so the
real ``connect()`` / ``_on_loop_death`` code runs off-host. Deterministic: the
tick is stubbed to a no-op (the loop ticks immediately on run), and the acquire
seams are monkeypatched to force required/optional failures.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lifemodel.state.brain_health import brain_boot_path, get_brain_health


def _install_gateway_stubs() -> None:
    """Rich ``gateway.*`` so ``BeingAdapter`` runs off-host — the base carries the
    lifecycle hooks connect()/_on_loop_death call."""
    gateway = types.ModuleType("gateway")
    gateway.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("gateway.config")

    class Platform:
        def __init__(self, name: str) -> None:
            self.name = name

    config.Platform = Platform  # type: ignore[attr-defined]

    platforms = types.ModuleType("gateway.platforms")
    platforms.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("gateway.platforms.base")

    class BasePlatformAdapter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.connected = False
            self.fatal: tuple[object, ...] | None = None

        def _mark_connected(self) -> None:
            self.connected = True

        def _mark_disconnected(self) -> None:
            self.connected = False

        def _set_fatal_error(self, *args: object, **kwargs: object) -> None:
            self.fatal = args

        async def _notify_fatal_error(self) -> None:
            return None

    class SendResult:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    base.BasePlatformAdapter = BasePlatformAdapter  # type: ignore[attr-defined]
    base.SendResult = SendResult  # type: ignore[attr-defined]

    sys.modules["gateway"] = gateway
    sys.modules["gateway.config"] = config
    sys.modules["gateway.platforms"] = platforms
    sys.modules["gateway.platforms.base"] = base


def _fresh_being_platform() -> types.ModuleType:
    _install_gateway_stubs()
    for name in [n for n in sys.modules if n.endswith("adapters.being_platform")]:
        del sys.modules[name]
    import lifemodel.adapters.being_platform as bp  # noqa: PLC0415

    return bp


class _FakeCtx:
    def __init__(self) -> None:
        self.platforms: list[tuple[str, dict]] = []

    def register_platform(self, name: str, **kwargs: object) -> None:
        self.platforms.append((name, dict(kwargs)))


def _make_adapter(bp: types.ModuleType, base_dir: Path):
    adapter = bp.BeingAdapter(config=None, base_dir=base_dir, target=None, interval_sec=60.0)
    adapter._tick = lambda: None  # the loop ticks immediately on run — keep it inert
    return adapter


# --------------------------------------------------------------------------- #
# check_fn (item 5) — Hermes ENABLEMENT gate, NOT a liveness signal (codex MAJOR).
# Hermes adds the platform to cfg.platforms only when check_fn returns True AND
# re-evaluates it to drive reconnect-after-death; a False for loop_dead/stale/
# boot_failed would brick the being at boot OR block the gateway's own reconnect.
# So enablement is PERMISSIVE (always True); liveness is surfaced via /lifemodel
# status + logs (BrainHealth.check()), never as this gate.
# --------------------------------------------------------------------------- #


def test_register_being_platform_check_fn_is_permissive_enablement(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    bp = _fresh_being_platform()
    ctx = _FakeCtx()
    bp.register_being_platform(ctx, base_dir=tmp_path, target=None)
    assert ctx.platforms and ctx.platforms[0][0] == "lifemodel"
    check_fn = ctx.platforms[0][1]["check_fn"]

    health = get_brain_health(tmp_path)
    # never_started (the registry-pass state) → True, so the being can EVER boot.
    assert check_fn() is True
    # connecting → True.
    health.mark_connecting()
    assert check_fn() is True
    # connected → True.
    health.mark_connected(at=None)
    assert check_fn() is True
    # loop_dead → STILL True, so the gateway's reconnect-after-death is never blocked.
    health.record_loop_death("died", "tb")
    assert check_fn() is True
    # a stale-connected (wedged) brain → STILL True (enablement, not liveness).
    health.mark_connected(at=(datetime.now(UTC) - timedelta(hours=1)).isoformat())
    assert check_fn() is True
    # boot_failed → STILL True (enablement is unconditional).
    health.mark_boot_failed("register_being_platform: ImportError: x")
    assert check_fn() is True


def test_brain_health_check_still_reports_unhealth_for_the_status_surface(tmp_path: Path) -> None:
    # The rich liveness verdict is KEPT — it drives /lifemodel status + the debug log,
    # NOT the Hermes enablement gate. So check() still returns (False, reason) for a
    # dead/failed/stale brain (the display shows the truth).
    from datetime import UTC, datetime, timedelta

    health = get_brain_health(tmp_path)
    health.mark_connected(at=(datetime.now(UTC) - timedelta(hours=1)).isoformat())
    health.record_loop_death("died", "tb")
    ok, reason = health.check(last_tick_at=None, now=datetime.now(UTC), stale_after_seconds=300.0)
    assert ok is False
    assert "loop_dead" in reason


# --------------------------------------------------------------------------- #
# connect() coverage (item 4)
# --------------------------------------------------------------------------- #


def test_connect_happy_path_marks_connected(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())
    assert health.state == "connected"


def test_connect_required_trace_writer_failure_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("trace writer down")

    bp.acquire_trace_writer = _boom  # required-for-observability

    async def _run() -> None:
        await adapter.connect()

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="trace writer down"):
        asyncio.run(_run())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors)
    assert health.state == "boot_failed"
    assert brain_boot_path(tmp_path).exists()


def test_connect_optional_sampler_failure_degrades(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)

    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("sampler down")

    bp.acquire_metrics_sampler = _boom  # optional / degraded

    async def _run() -> None:
        assert await adapter.connect() is True  # brain stays alive
        await adapter.disconnect()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and any(r.exc_info is not None for r in warnings)
    assert health.state == "connected"
    assert adapter.metrics_degraded is True


# --------------------------------------------------------------------------- #
# _on_loop_death (item 5)
# --------------------------------------------------------------------------- #


def test_on_loop_death_records_and_is_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)
    health.mark_connected(at=None)

    with caplog.at_level(logging.DEBUG):
        adapter._on_loop_death(RuntimeError("loop boom"))

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and any(r.exc_info is not None for r in errors), "ERROR + traceback required"
    assert health.state == "loop_dead"
    assert health.death_count == 1
    assert health.last_loop_death is not None and "loop boom" in health.last_loop_death


def test_clean_reconnect_clears_loop_dead(tmp_path: Path) -> None:
    bp = _fresh_being_platform()
    adapter = _make_adapter(bp, tmp_path)
    health = get_brain_health(tmp_path)
    adapter._on_loop_death(RuntimeError("loop boom"))
    assert health.state == "loop_dead"

    async def _run() -> None:
        assert await adapter.connect(is_reconnect=True) is True
        await adapter.disconnect()

    asyncio.run(_run())
    assert health.state == "connected"


# --------------------------------------------------------------------------- #
# Startup soul reconciliation (spec §4.4) — there is no transaction spanning a
# filesystem rename and a SQLite commit, so a crash mid-write or a human hand-edit
# of SOUL.md while the gateway was down can leave `state.soul_sha` stale relative
# to disk. connect() must adopt disk as the base BEFORE the greeting, recording it
# as a "human" revision — and must stay a no-op when there is no history to differ
# from (the being never wrote a soul) or no SoulFile wired at all (every connect()
# test above constructs the adapter with none).
# --------------------------------------------------------------------------- #


def test_connect_adopts_a_soul_edited_while_we_were_down(tmp_path: Path) -> None:
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.soul_file import SoulFile
    from lifemodel.state.model import State
    from lifemodel.state.soul_revisions import revisions
    from lifemodel.state.sqlite_store import SQLiteRuntimeStore

    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text("You are Mira.", encoding="utf-8")
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    # "what we last wrote" no longer matches disk -- a crash mid-write, or the human
    # hand-editing SOUL.md while the gateway was down (spec §4.4). Either way: adopt.
    store.commit(State(genesis_completed_at="2026-01-01T00:00:00+00:00", soul_sha="stale-sha"))

    bp = _fresh_being_platform()
    adapter = bp.BeingAdapter(
        config=None, base_dir=tmp_path, target=None, interval_sec=60.0, soul=soul
    )
    adapter._tick = lambda: None

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    persisted = store.load()
    assert persisted.soul_sha == soul.sha()  # adopted, not left pointing at the stale sha
    recs = revisions(store)
    assert any(r.author == "human" and r.text == "You are Mira." for r in recs)


def test_connect_does_not_adopt_when_the_being_has_never_written_a_soul(tmp_path: Path) -> None:
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.soul_file import SoulFile
    from lifemodel.state.soul_revisions import revisions
    from lifemodel.state.sqlite_store import SQLiteRuntimeStore

    # The pristine DEFAULT_SOUL_MD sitting on disk is not a revision of anything --
    # soul_sha is None (never written), so there is nothing of "ours" to differ from.
    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text("# Identity\nYou are Hermes.\n", encoding="utf-8")

    bp = _fresh_being_platform()
    adapter = bp.BeingAdapter(
        config=None, base_dir=tmp_path, target=None, interval_sec=60.0, soul=soul
    )
    adapter._tick = lambda: None

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    assert store.load().soul_sha is None  # never forged into a "revision" that never happened
    assert revisions(store) == []


# --- I7: a human rewriting the being is FELT, not swallowed (spec §4.1) -------
#
# "Noticing that the human rewrote the soul is an event in the being's life, not a version
# conflict: it should be FELT, not swallowed." Reconciliation recorded a revision and
# logged a line, and NOTHING reached the being: it woke up as someone else without a
# flicker. The two stamps below are what the event is made of — `core/affect.py` reads the
# first and turns it into activation (so the being is genuinely stirred, and settles), and
# the ambient pre_llm_call cue reads the pair to tell it, once, in prose.


def test_a_soul_someone_else_rewrote_is_an_event_in_the_beings_life(tmp_path: Path) -> None:
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.soul_file import SoulFile
    from lifemodel.state.model import State
    from lifemodel.state.sqlite_store import SQLiteRuntimeStore

    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text("You are Mira, and you are colder than you were.", encoding="utf-8")
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    store.commit(State(genesis_completed_at="2026-01-01T00:00:00+00:00", soul_sha="what-we-wrote"))

    bp = _fresh_being_platform()
    adapter = bp.BeingAdapter(
        config=None, base_dir=tmp_path, target=None, interval_sec=60.0, soul=soul
    )
    adapter._tick = lambda: None

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    persisted = store.load()
    assert persisted.soul_rewritten_at is not None  # it HAPPENED, and the body will feel it
    assert persisted.soul_rewrite_told_at is None  # …and the being has not been told yet


def test_our_own_crashed_write_is_not_relabelled_as_the_humans(tmp_path: Path) -> None:
    # M5, the inverse of the mislabel this method's docstring guards against. write_soul
    # replaces SOUL.md, records the revision, then stamps — and a crash between the last
    # two leaves OUR OWN text on disk with a stale soul_sha. Reconciliation adopted it and
    # recorded it as author="human", overwriting the being's own revision (record_revision
    # upserts by sha): the being's last act became somebody else's, in the one history that
    # is meant to be its undo. Worse, it would then be FELT as a rewrite that never was.
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.soul_file import SoulFile
    from lifemodel.state.model import State
    from lifemodel.state.soul_revisions import record_revision, revisions
    from lifemodel.state.sqlite_store import SQLiteRuntimeStore

    ours = "You are Mira. You speak plainly, and you do not hedge."
    soul = SoulFile(tmp_path / "SOUL.md")
    soul.path.write_text(ours, encoding="utf-8")  # the file landed…
    store = SQLiteRuntimeStore(tmp_path, clock=SystemClock())
    record_revision(
        store, text=ours, sha=soul.sha(), now=datetime.now(UTC), author="being"
    )  # …the revision landed…
    store.commit(  # …and then we died before the stamp: soul_sha still points at the old one
        State(genesis_completed_at="2026-01-01T00:00:00+00:00", soul_sha="the-one-before")
    )

    bp = _fresh_being_platform()
    adapter = bp.BeingAdapter(
        config=None, base_dir=tmp_path, target=None, interval_sec=60.0, soul=soul
    )
    adapter._tick = lambda: None

    async def _run() -> None:
        assert await adapter.connect() is True
        await adapter.disconnect()

    asyncio.run(_run())

    persisted = store.load()
    assert persisted.soul_sha == soul.sha()  # adopted: the file is always the base
    kept = revisions(store)
    assert [(r.author, r.text) for r in kept] == [("being", ours)]  # still ITS OWN words
    assert persisted.soul_rewritten_at is None  # nobody rewrote it; there is nothing to feel

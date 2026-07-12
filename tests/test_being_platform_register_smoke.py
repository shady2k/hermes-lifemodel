"""Gateway-stubbed ``register()`` smoke test (spec §4.1, best-effort).

The faithful-load test (``test_real_loader_import``) covers the whole Hermes-free
runtime surface but EXCLUDES ``adapters/being_platform.py`` — the one runtime
module that imports ``gateway.*`` at load. This test closes that last gap: it
installs minimal ``gateway.*`` (+ ``hermes_constants``) stubs, sets up the same
isolated ``hermes_plugins.lifemodel`` namespace with ``lifemodel`` scrubbed, then
runs the plugin's real ``register(fake_ctx)`` and asserts the platform wiring
path was exercised — ``register_being_platform`` ran and called
``ctx.register_platform("lifemodel", …)`` rather than being swallowed by the
``being_platform_registration_skipped`` guard.

Because ``being_platform`` imports ``from ..state.metrics_store import …``, this
also re-catches the incident from the *adapter* side: a pre-fix absolute
self-import would make the import fail → the platform registration would be
swallowed → the ``register_platform`` assertion would fail.

The gateway stubs are deliberately tiny (``BasePlatformAdapter`` must be a real
subclassable class; ``Platform``/``SendResult`` are only referenced inside method
bodies never run here), so this is maintainable rather than fragile — the
in-Hermes path is exercised without needing the gateway package. Stdlib-only.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_CHECKOUT_PARENT = _PKG_ROOT.parent


class _FakeCtx:
    """Duck-typed Hermes registration context — records what the plugin wires."""

    def __init__(self) -> None:
        self.profile_name = "smoke-profile"
        self.commands: list[tuple[str, object]] = []
        self.hooks: list[tuple[str, object]] = []
        self.tools: list[tuple[str, dict]] = []
        self.platforms: list[tuple[str, dict]] = []

    def register_command(self, name: str, fn: object, **kwargs: object) -> None:
        self.commands.append((name, fn))

    def register_hook(self, name: str, cb: object) -> None:
        self.hooks.append((name, cb))

    def register_tool(self, name: str, **kwargs: object) -> None:
        self.tools.append((name, kwargs))

    def register_platform(self, name: str, **kwargs: object) -> None:
        self.platforms.append((name, kwargs))


def _install_gateway_stubs() -> None:
    """Minimal ``gateway.*`` so ``being_platform`` imports off-host. Only the
    module-level names it binds are needed; the classes are never instantiated
    during ``register()`` (the adapter factory is a deferred lambda)."""
    gateway = types.ModuleType("gateway")
    gateway.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("gateway.config")

    class Platform:  # noqa: D401 - stub
        def __init__(self, name: str) -> None:
            self.name = name

    config.Platform = Platform  # type: ignore[attr-defined]

    platforms = types.ModuleType("gateway.platforms")
    platforms.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("gateway.platforms.base")

    class BasePlatformAdapter:  # real class so `class BeingAdapter(...)` works
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class SendResult:  # only referenced in method bodies, never run here
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    base.BasePlatformAdapter = BasePlatformAdapter  # type: ignore[attr-defined]
    base.SendResult = SendResult  # type: ignore[attr-defined]

    sys.modules["gateway"] = gateway
    sys.modules["gateway.config"] = config
    sys.modules["gateway.platforms"] = platforms
    sys.modules["gateway.platforms.base"] = base


def _install_hermes_constants_stub(home: Path) -> None:
    hc = types.ModuleType("hermes_constants")
    hc.get_hermes_home = lambda: home  # type: ignore[attr-defined]
    sys.modules["hermes_constants"] = hc


def _setup_isolated_package() -> types.ModuleType:
    """Scrub ``lifemodel`` and build ``hermes_plugins.lifemodel`` from the
    checkout __init__.py (mirrors the faithful-load harness). Returns the pkg."""
    sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _CHECKOUT_PARENT)]
    for name in [n for n in sys.modules if n == "lifemodel" or n.startswith("lifemodel.")]:
        del sys.modules[name]
    assert importlib.util.find_spec("lifemodel") is None, "isolation incomplete"

    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []  # type: ignore[attr-defined]
        ns.__package__ = "hermes_plugins"
        sys.modules["hermes_plugins"] = ns

    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.lifemodel",
        _PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(_PKG_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_plugins.lifemodel"
    module.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    sys.modules["hermes_plugins.lifemodel"] = module
    spec.loader.exec_module(module)
    return module


def test_register_exercises_platform_path_with_gateway_stubbed(tmp_path: Path) -> None:
    home = tmp_path / "hermes-home"
    # register() resolves state_dir(home) = <home>/workspace/lifemodel and, on
    # the post_llm wiring, opens observability.sqlite there — make it writable.
    sdir = home / "workspace" / "lifemodel"
    sdir.mkdir(parents=True, exist_ok=True)

    orig_path = sys.path[:]
    orig_modules = dict(sys.modules)
    try:
        pkg = _setup_isolated_package()
        _install_gateway_stubs()
        _install_hermes_constants_stub(home)

        ctx = _FakeCtx()
        pkg.register(ctx)  # must NOT raise; must reach the platform wiring

        assert ("hermes_plugins.lifemodel.adapters.being_platform") in sys.modules, (
            "being_platform was never imported — the platform path was not exercised"
        )
        # register_being_platform ran (not swallowed by the skipped-guard).
        assert any(name == "lifemodel" for name, _ in ctx.platforms), (
            "register_platform('lifemodel', …) was not called — platform wiring "
            "was skipped (import/registration failure swallowed)"
        )
        # Sanity: the /lifemodel diagnostic command was wired too.
        assert any(name == "lifemodel" for name, _ in ctx.commands)

        # Tidy up the trace-writer daemon register() acquired for post_llm.
        ts = sys.modules.get("hermes_plugins.lifemodel.state.trace_store")
        if ts is not None:
            with contextlib.suppress(Exception):
                ts.release_trace_writer(ts.observability_db_path(sdir))
    finally:
        sys.path[:] = orig_path
        for k in list(sys.modules):
            if k not in orig_modules:
                del sys.modules[k]
        for k, v in orig_modules.items():
            sys.modules[k] = v

"""Shared minimal ``gateway.*`` stubs for dev tests that drive ``register()``.

``adapters/being_platform.py`` imports ``gateway.*`` at module load, and
``register()`` now wires the platform as a REQUIRED step (spec §4.3) — so a dev
run under ``uv`` (no ``gateway`` package) makes that step re-raise, which is the
correct fail-loud behavior in prod but breaks off-host command-surface tests.
These tiny stubs give ``being_platform`` an importable ``gateway`` so ``register()``
completes off-host, exercising the real (now-strict) wiring path. Not a test
module (no ``test_`` prefix); imported by test fixtures.
"""

from __future__ import annotations

import sys
import types


def install_gateway_stubs() -> None:
    """Install minimal ``gateway.*`` in ``sys.modules`` (idempotent).

    Only the names ``being_platform`` binds at module level are provided; the
    classes are never instantiated during ``register()`` (the adapter factory /
    check_fn are deferred callables).
    """
    existing = sys.modules.get("gateway.platforms.base")
    if existing is not None and hasattr(existing, "BasePlatformAdapter"):
        return  # already installed (by this helper or a richer per-test stub)

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
            pass

    class SendResult:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    base.BasePlatformAdapter = BasePlatformAdapter  # type: ignore[attr-defined]
    base.SendResult = SendResult  # type: ignore[attr-defined]

    sys.modules["gateway"] = gateway
    sys.modules["gateway.config"] = config
    sys.modules["gateway.platforms"] = platforms
    sys.modules["gateway.platforms.base"] = base

"""Faithful-load smoke test: import the runtime surface the way Hermes does.

The incident (spec §1/§4.1): under the real Hermes loader a directory plugin is
imported as ``hermes_plugins.<slug>`` (``hermes_cli/plugins.py``), so the checkout
name ``lifemodel`` is NOT a top-level module. An absolute self-import
(``from lifemodel.core.metrics import …`` in ``state/metrics_store.py``) therefore
raised ``ModuleNotFoundError`` in prod — but ``make check`` stayed green because
``conftest.py`` puts the package parent on ``sys.path``, so ``lifemodel`` resolves
*in tests*.

This test reproduces the loader FAITHFULLY so that harness shim cannot mask the
bug: for the duration it

* removes the checkout parent from ``sys.path`` and deletes ``lifemodel`` +
  every ``lifemodel.*`` from ``sys.modules`` — so a stray ``from lifemodel…``
  fails exactly as in prod (asserted: ``lifemodel`` is un-findable inside the
  block);
* builds ``hermes_plugins.lifemodel`` from ``__init__.py`` via
  ``spec_from_file_location`` (``submodule_search_locations=[pkg_root]``), then
  imports every Hermes-free runtime module under that namespace — including
  ``state.metrics_store`` (the module that carried the incident);
* restores ``sys.path`` and purges the injected namespace in a ``finally``.

``adapters/being_platform.py`` is the one runtime module that imports ``gateway.*``
at module load (absent under uv), so it is excluded here and covered by the
gateway-stubbed smoke test.

Stdlib-only (``importlib``/``ast``-free ``pathlib`` walk) — runs inside Hermes'
own venv too.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_CHECKOUT_PARENT = _PKG_ROOT.parent

#: Runtime source dirs (Hermes-free engine + the adapter boundary), scanned
#: recursively. Curated on purpose — a blind ``rglob`` of the package root would
#: descend into ``.venv``/``docs``/``.git``. Mirrors the linter's scan set.
_RUNTIME_DIRS = ("core", "domain", "state", "adapters", "ports", "sim")

# The one runtime module that imports gateway.* at load — excluded (see docstring).
_SKIP_REL = {Path("adapters/being_platform.py")}


def _runtime_dotted_names() -> list[str]:
    """Every Hermes-free runtime module, as dotted names relative to the package
    (e.g. ``state.metrics_store``): the runtime dirs (recursive) plus root-level
    ``*.py``. Excludes the gateway-bound adapter, ``conftest.py``, and the
    package ``__init__`` (which is set up as the package, not walked)."""
    paths: list[Path] = []
    for d in _RUNTIME_DIRS:
        paths.extend((_PKG_ROOT / d).rglob("*.py"))
    paths.extend(_PKG_ROOT.glob("*.py"))  # root-level modules only (non-recursive)

    names: list[str] = []
    for path in sorted(paths):
        rel = path.relative_to(_PKG_ROOT)
        if "__pycache__" in rel.parts or rel.name == "conftest.py" or rel in _SKIP_REL:
            continue
        if rel.name == "__init__.py":
            dotted = ".".join(rel.parts[:-1])  # the (sub)package itself
        else:
            dotted = ".".join((*rel.parts[:-1], rel.stem))
        if dotted:  # root __init__.py -> "" -> set up as the package, not walked
            names.append(dotted)
    return names


def _import_runtime_under_hermes_namespace() -> list[str]:
    """Set up ``hermes_plugins.lifemodel`` from the checkout and import the whole
    Hermes-free runtime surface under it, with ``lifemodel`` scrubbed so an
    absolute self-import cannot resolve. Returns the loaded dotted-module names.

    Caller owns save/restore of ``sys.path`` / ``sys.modules`` (see the test's
    ``finally``); this function only does the in-block mutation + imports.
    """
    # 1) Isolate: drop the checkout parent (conftest's shim) and purge lifemodel*.
    sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _CHECKOUT_PARENT)]
    for name in [n for n in sys.modules if n == "lifemodel" or n.startswith("lifemodel.")]:
        del sys.modules[name]

    # The isolation must be REAL: if `lifemodel` is still findable, the scrub is
    # incomplete and this test would false-pass. Fail loudly instead.
    assert importlib.util.find_spec("lifemodel") is None, (
        "isolation incomplete: top-level `lifemodel` is still importable — the "
        "faithful-load test cannot prove the fix. Scrub sys.path/sys.modules."
    )

    # 2) `hermes_plugins` namespace package.
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []  # PEP 420-ish: a package with no on-disk search dir
        ns.__package__ = "hermes_plugins"
        sys.modules["hermes_plugins"] = ns

    # 3) `hermes_plugins.lifemodel` package, built from the checkout __init__.py.
    init_path = _PKG_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.lifemodel",
        init_path,
        submodule_search_locations=[str(_PKG_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_plugins.lifemodel"
    module.__path__ = [str(_PKG_ROOT)]  # so submodules resolve via PathFinder
    sys.modules["hermes_plugins.lifemodel"] = module  # BEFORE exec_module
    spec.loader.exec_module(module)

    # 4) Import the Hermes-free runtime surface under the namespace. A pre-fix
    #    `from lifemodel.core.metrics import …` raises ModuleNotFoundError here.
    loaded: list[str] = []
    for dotted in _runtime_dotted_names():
        full = f"hermes_plugins.lifemodel.{dotted}"
        importlib.import_module(full)
        loaded.append(dotted)
    return loaded


def test_runtime_surface_imports_under_real_hermes_namespace() -> None:
    # metrics_store is THE module that carried the incident — guard that the
    # walk still includes it, so a refactor can't silently drop the coverage.
    assert "state.metrics_store" in _runtime_dotted_names()

    orig_path = sys.path[:]
    orig_modules = dict(sys.modules)
    try:
        loaded = _import_runtime_under_hermes_namespace()

        # The whole point: this module imported cleanly under the namespace,
        # which means its `from ..core.metrics import …` resolved relatively.
        assert "state.metrics_store" in loaded
        ms = sys.modules["hermes_plugins.lifemodel.state.metrics_store"]
        core_metrics = sys.modules["hermes_plugins.lifemodel.core.metrics"]
        # It bound to the NAMESPACED core.metrics (relative), not a stray
        # absolute `lifemodel.core.metrics`.
        assert ms.Counter is core_metrics.Counter

        # No absolute `lifemodel` module leaked in while loading.
        assert "lifemodel" not in sys.modules
        assert not any(n.startswith("lifemodel.") for n in sys.modules)
    finally:
        sys.path[:] = orig_path
        for k in list(sys.modules):
            if k not in orig_modules:
                del sys.modules[k]
        for k, v in orig_modules.items():
            sys.modules[k] = v

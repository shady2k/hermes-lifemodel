"""Faithful proof that Hermes' PluginManager can load the flat-layout package.

The repo directory IS the ``lifemodel`` package (no ``src/`` layer): a plain
``git clone`` of this repo (what ``hermes plugins install/update`` does under
the hood) drops the package straight into the install dir, and Hermes' loader
imports that install dir's own ``__init__.py`` under a *namespaced* module name
(``hermes_plugins.lifemodel``) — never a top-level ``lifemodel``.

This is exactly what makes step 2's absolute→relative import fix load-bearing:
a stray ``from lifemodel.x import ...`` resolves fine in this dev checkout
(``lifemodel`` really is importable, top-level, via the root ``conftest.py``)
but would crash in prod, because there is no top-level ``lifemodel`` there —
only ``hermes_plugins.lifemodel``. So the test below drives the load in a
**clean subprocess** (no inherited ``sys.path``/``PYTHONPATH`` from this dev
checkout) using the *exact* loading sequence Hermes' PluginManager uses, and
only then imports two internal submodules — if any internal import were still
absolute, that import would ``ModuleNotFoundError`` right here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Entries at the repo root that are dev/VCS/tooling, not part of the
#: installable plugin package — a real `hermes plugins install` (git clone)
#: never puts these in front of Hermes' loader either.
_NON_PACKAGE_NAMES = {
    "tests",
    "docs",
    "node_modules",
    ".venv",
    ".beads",
    ".superpowers",
    "pyproject.toml",
    "Makefile",
    "conftest.py",
    "uv.lock",
    ".gitignore",
}


def _simulate_install(dest: Path) -> Path:
    """Copy the flat package into ``dest/lifemodel/`` — a stand-in for what
    ``hermes plugins install`` (a full git clone) drops into the install dir.

    Excludes VCS/dev-tooling/cache entries (``.git``, dotfiles generally,
    ``*.md`` docs, ``__pycache__`` at any depth, ...) that a real clone of the
    tracked tree would never contain, so only actual package files land in the
    simulated install dir.
    """
    install_dir = dest / "lifemodel"
    install_dir.mkdir(parents=True)
    for entry in sorted(_REPO_ROOT.iterdir()):
        name = entry.name
        if name in _NON_PACKAGE_NAMES or name.startswith(".") or name.endswith(".md"):
            continue
        if entry.is_dir():
            shutil.copytree(entry, install_dir / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(entry, install_dir / name)
    return install_dir


#: The exact loading sequence Hermes' PluginManager runs for a directory
#: plugin: `spec_from_file_location` under a namespaced module name (never a
#: top-level `lifemodel`), `module_from_spec`, wire `__package__`/`__path__`,
#: register in `sys.modules`, then `exec_module`. `hermes_plugins` itself is
#: bootstrapped as an empty namespace parent first, exactly as the real host's
#: plugin loader package would be, so `importlib.import_module` can resolve
#: dotted children under it afterwards.
_LOADER_SCRIPT = textwrap.dedent(
    """\
    import importlib
    import importlib.util
    import sys
    import types

    install_dir = sys.argv[1]
    name = "hermes_plugins.lifemodel"

    parent = types.ModuleType("hermes_plugins")
    parent.__path__ = []
    sys.modules["hermes_plugins"] = parent

    spec = importlib.util.spec_from_file_location(
        name, install_dir + "/__init__.py", submodule_search_locations=[install_dir]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name
    module.__path__ = [install_dir]
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert callable(module.register), "register(ctx) must be callable"

    tick_mod = importlib.import_module("hermes_plugins.lifemodel.tick")
    decision_mod = importlib.import_module("hermes_plugins.lifemodel.core.decision")
    assert tick_mod is not None
    assert decision_mod is not None

    print("OK")
    """
)


def test_hermes_loads_the_flat_plugin(tmp_path: Path) -> None:
    install_dir = _simulate_install(tmp_path / "install")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", _LOADER_SCRIPT, str(install_dir)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )

    assert result.returncode == 0, (
        f"loader failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_plugin_yaml_at_repo_root() -> None:
    assert (_REPO_ROOT / "plugin.yaml").is_file()
    assert (_REPO_ROOT / "__init__.py").is_file()


def test_register_importable_in_dev() -> None:
    from lifemodel import register

    assert callable(register)

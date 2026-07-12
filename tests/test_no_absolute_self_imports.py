"""Static guard: runtime code must never import itself *absolutely* (spec §4.1).

The incident this guards against: ``state/metrics_store.py`` did
``from lifemodel.core.metrics import …`` (a level-0 absolute self-import). Under
the real Hermes loader the package is imported as ``hermes_plugins.lifemodel``
(``hermes_cli/plugins.py:_load_directory_module`` maps a directory plugin to
``hermes_plugins.<slug>``), so a top-level ``lifemodel`` module does not exist →
``ModuleNotFoundError`` → the being's brain silently never started.

``make check`` stayed GREEN because ``conftest.py`` inserts the package parent on
``sys.path``, so ``lifemodel`` resolves *in tests* — the exact thing fatal in
prod. This AST linter closes that gap: runtime code (``core domain state adapters
ports`` + package-root ``*.py``) must use **relative** imports
(``from ..core.metrics``), never ``lifemodel.…`` nor the loader namespace
``hermes_plugins.lifemodel.…``. Tests / ``conftest.py`` / ``testing/`` are exempt
(test-only, run under the harness shim).

Stdlib-only (``ast``, ``pathlib``) so it runs inside Hermes' own venv too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The package root is the directory that contains the runtime dirs (this test
# lives in ``<pkg>/tests/``).
_PKG_ROOT = Path(__file__).resolve().parent.parent

#: Runtime source dirs — Hermes-free engine + the adapter boundary. Scanned
#: recursively (``domain/objects/`` etc. count).
_RUNTIME_DIRS = ("core", "domain", "state", "adapters", "ports")

#: Files under the package root that are runtime code, scanned non-recursively.
#: ``conftest.py`` is a pytest harness file, not runtime, so it is exempt.
_ROOT_EXCLUDE = {"conftest.py"}


def _self_ref(name: str | None) -> bool:
    """True iff ``name`` is (or is a submodule of) the plugin's own package,
    referenced ABSOLUTELY — either the checkout name ``lifemodel`` or the loader
    namespace ``hermes_plugins.lifemodel``. Both are wrong in runtime code:
    only relative imports resolve under whatever name the host binds us to."""
    if not name:
        return False
    for pkg in ("lifemodel", "hermes_plugins.lifemodel"):
        if name == pkg or name.startswith(pkg + "."):
            return True
    return False


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for d in _RUNTIME_DIRS:
        files.extend(sorted((_PKG_ROOT / d).rglob("*.py")))
    for f in sorted(_PKG_ROOT.glob("*.py")):
        if f.name not in _ROOT_EXCLUDE:
            files.append(f)
    return files


def _offenders_in(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, offending-source-snippet)`` for every absolute
    self-import anywhere in the file's AST — module level, inside functions,
    or under ``TYPE_CHECKING``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # `from lifemodel… import …` / `from hermes_plugins.lifemodel… import …`
        # (level 0 == absolute; a relative `from ..core` has level > 0 and
        # module="core.…", so it never matches).
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and _self_ref(node.module):
                found.append((node.lineno, f"from {node.module} import ..."))
        # `import lifemodel[.x]` / `import hermes_plugins.lifemodel[.x]`
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _self_ref(alias.name):
                    found.append((node.lineno, f"import {alias.name}"))
        # Literal-string dynamic self-imports:
        #   importlib.import_module("lifemodel…") / import_module("lifemodel…")
        #   __import__("lifemodel…")
        elif isinstance(node, ast.Call):
            fn = node.func
            fn_name = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else fn.id
                if isinstance(fn, ast.Name)
                else None
            )
            if fn_name in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and _self_ref(first.value):
                    found.append((node.lineno, f"{fn_name}({first.value!r})"))
    return found


def test_no_absolute_self_imports_in_runtime_code() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(_PKG_ROOT)
        for lineno, snippet in _offenders_in(path):
            offenders.append(f"{rel}:{lineno}: {snippet}")

    assert not offenders, (
        "Runtime code must use RELATIVE imports, never absolute self-imports "
        "(`lifemodel.…` / `hermes_plugins.lifemodel.…`) — they resolve in tests "
        "(conftest shim) but ModuleNotFoundError under the real Hermes loader. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )


# --- The linter's own detection branches ---------------------------------
# The real tree exercises only the `from lifemodel…` case, so pin every other
# banned form (and the relative-import negatives) against synthetic source, or
# a regression could quietly blind the linter to a whole class.

_BANNED = [
    "from lifemodel.core.metrics import Counter",
    "from lifemodel import x",
    "import lifemodel",
    "import lifemodel.core.metrics",
    "from hermes_plugins.lifemodel.core import metrics",
    "import hermes_plugins.lifemodel",
    "import hermes_plugins.lifemodel.core.metrics",
    'importlib.import_module("lifemodel.core.metrics")',
    'import_module("hermes_plugins.lifemodel.core.metrics")',
    '__import__("lifemodel.core.metrics")',
    # Anywhere in the AST — inside a function and under TYPE_CHECKING.
    "def f():\n    from lifemodel.core import metrics\n    return metrics",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import lifemodel.core",
]

_ALLOWED = [
    "from ..core.metrics import Counter",  # the correct relative form
    "from .metrics import Counter",
    "from . import metrics",
    "import logging",
    "from collections.abc import Callable",
    "from hermes_constants import get_hermes_home",  # a real host module, not self
    "import lifemodella",  # not a self-ref: distinct top-level name
    "from lifemodella import x",
    'importlib.import_module("some.other.module")',
    "importlib.import_module(dynamic_name)",  # non-literal first arg: not our concern
]


@pytest.mark.parametrize("src", _BANNED)
def test_linter_flags_every_banned_form(tmp_path: Path, src: str) -> None:
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    assert _offenders_in(f), f"linter missed a banned self-import: {src!r}"


@pytest.mark.parametrize("src", _ALLOWED)
def test_linter_ignores_legitimate_imports(tmp_path: Path, src: str) -> None:
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    assert not _offenders_in(f), f"linter false-positived on legitimate code: {src!r}"

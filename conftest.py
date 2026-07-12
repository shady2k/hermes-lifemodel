"""Dev-checkout test setup for the flat root-layout, **independent of the checkout
directory's name** (dev worktrees may be named anything — the leaf dir need not be
literally ``lifemodel``).

The repo directory IS the ``lifemodel`` package (the root holds ``__init__.py`` — it is
the plugin entrypoint, with module-level *relative* imports that must stay relative so
deploy can load it as ``hermes_plugins.lifemodel``). Two things must hold in a dev
checkout, and both would otherwise depend on the leaf dir being named ``lifemodel``:

1. **Tests import the package absolutely** (``from lifemodel.core... import ...``), so
   ``lifemodel`` must be importable by name. When the leaf dir is ``lifemodel`` a plain
   parent-on-path is enough (and matches deploy); otherwise we bind the directory under
   its canonical name via ``importlib``.

2. **pytest's ``Package.setup()`` imports the root ``__init__.py``.** Because the root
   has ``__init__.py``, pytest builds a ``Package`` node for it and, at setup, does
   ``importlib.import_module(<name>)`` where ``<name>`` is derived from the leaf dir:
   an identifier leaf (``lm_probe``) → that name; a non-identifier leaf (``lm-probe``)
   → bare ``__init__``. In a non-canonical checkout that re-execs the file outside any
   package and crashes on the relative imports. We **pre-seed ``sys.modules[<name>]``**
   to the already-bound ``lifemodel`` module, so importlib returns it from cache instead
   of re-exec'ing the file. (mypy gets its own canonical view via the ``make mypy``
   symlink — see Makefile.)
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# os.path.abspath (not resolve) matches the path spelling pytest uses.
_HERE = Path(os.path.abspath(__file__)).parent


def _bind_lifemodel() -> ModuleType:
    """Bind this flat-layout directory as the ``lifemodel`` package (idempotent)."""
    existing = sys.modules.get("lifemodel")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        "lifemodel",
        _HERE / "__init__.py",
        submodule_search_locations=[str(_HERE)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot bind {_HERE} as the 'lifemodel' package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lifemodel"] = module  # register before exec so submodules resolve
    spec.loader.exec_module(module)
    return module


if _HERE.name == "lifemodel":
    # Canonical checkout / deploy layout: discoverable by name once its parent is on
    # the path, and pytest imports the root as `lifemodel`. Leave this fast path alone.
    sys.path.insert(0, str(_HERE.parent))
else:
    _module = _bind_lifemodel()
    # Pre-seed the name pytest's Package.setup() will import for root/__init__.py so it
    # resolves to the already-bound `lifemodel` module (no bare re-exec, no crash).
    _pytest_root_name = _HERE.name if _HERE.name.isidentifier() else "__init__"
    sys.modules[_pytest_root_name] = _module

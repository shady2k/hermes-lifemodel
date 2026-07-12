"""Guarded integration test: the proactive reach-in primitive against real Hermes.

Drives :mod:`tests.hermes_reachin_integration` out of process under Hermes' own
interpreter (the venv that has ``gateway.*`` on its path) against a throwaway
``HERMES_HOME`` — never ``~/.hermes``, never a real channel. It proves
:func:`lifemodel.gateway_core.inject_proactive_turn` with its REAL default seams
builds a genuine ``MessageEvent(internal=True, message_id=None)`` from a real
:class:`SessionSource` and selects the right adapter — the unit fakes can't prove
that wire shape. Where Hermes is not installed (CI, a fresh clone) it **skips**
cleanly so ``make check`` stays green everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from _hermes_probe import find_hermes_python

_DRIVER = Path(__file__).resolve().parent / "hermes_reachin_integration.py"
# Flat root-layout: the repo dir IS the `lifemodel` package. The out-of-process
# driver does `import lifemodel`, so it needs a directory that CONTAINS a `lifemodel`
# package on its sys.path. The checkout dir may be named anything (dev worktrees), so
# the test hands the driver a per-test symlink dir (`<tmp>/src/lifemodel -> repo root`)
# instead of the repo's parent — see the LIFEMODEL_SRC setup below.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_reachin_builds_real_message_event(tmp_path: Path) -> None:
    hermes_py = find_hermes_python(tmp_path / "probe")
    if hermes_py is None:
        pytest.skip("no Hermes interpreter with gateway.* found; reach-in integration needs Hermes")

    home = tmp_path / "iso-home"
    home.mkdir()
    # Sanity: never the user's live being.
    assert home.resolve() != (Path.home() / ".hermes").resolve()

    # Present the flat-layout package under its canonical name for the driver's
    # `import lifemodel`, independent of the checkout dir's name (dev worktrees).
    src = tmp_path / "src"
    src.mkdir()
    (src / "lifemodel").symlink_to(_REPO_ROOT, target_is_directory=True)

    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "LIFEMODEL_SRC": str(src),
    }
    result = subprocess.run(
        [hermes_py, str(_DRIVER)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"driver failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    # The real default make_event must produce an internal, message_id-less TEXT
    # event, and the primitive must resolve a source + adapter and report DELIVERED.
    assert payload["event_internal"] is True, payload
    assert payload["message_id"] is None, payload
    assert payload["message_type"] == "text", payload
    assert payload["outcome"] == "delivered", payload

"""Guarded integration test: a being is born into a prompt that IS it (lm-4fv.4).

Drives :mod:`tests.hermes_genesis_prompt_integration` out of process under Hermes' own
interpreter (the venv that has ``gateway.*`` / ``agent.*`` on its path) against a throwaway
``HERMES_HOME`` — never ``~/.hermes``, never a real channel, never a real LLM call.

**It exercises the path that had never once run.** All four live births had an authored
soul on disk (the veteran branch); the stranger's first install — Hermes's pristine
``DEFAULT_SOUL_MD`` in slot #1, and a DM session already open — was unexercised, and
broken. The unit tests here drive every seam with fakes, and fakes cannot prove the one
claim the whole fix rests on: that a session ended BEFORE the turn is injected actually
yields a REBUILT system prompt carrying the new soul. This test proves it against the
host's own ``SessionStore`` and its own ``_restore_or_build_system_prompt`` — the function
whose "reused verbatim" behaviour is the defect. Where Hermes is not installed (CI, a
fresh clone) it **skips** cleanly so ``make check`` stays green everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from _hermes_probe import find_hermes_python

_DRIVER = Path(__file__).resolve().parent / "hermes_genesis_prompt_integration.py"
# Flat root-layout: the repo dir IS the `lifemodel` package, and the checkout dir may be
# named anything (dev worktrees) — so the driver is handed a symlink dir that presents it
# under its canonical name (see tests/test_reachin_integration.py, same shape).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_a_newborn_is_not_born_into_a_stale_prompt(tmp_path: Path) -> None:
    hermes_py = find_hermes_python(tmp_path / "probe")
    if hermes_py is None:
        pytest.skip("no Hermes interpreter with gateway.*/agent.* found; this needs Hermes")

    home = tmp_path / "iso-home"
    home.mkdir()
    assert home.resolve() != (Path.home() / ".hermes").resolve()  # never the live being

    src = tmp_path / "src"
    src.mkdir()
    (src / "lifemodel").symlink_to(_REPO_ROOT, target_is_directory=True)

    result = subprocess.run(
        [hermes_py, str(_DRIVER)],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(home), "LIFEMODEL_SRC": str(src)},
        timeout=180,
    )
    assert result.returncode == 0, (
        f"driver failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    p = json.loads(result.stdout.strip().splitlines()[-1])

    # The never-run path: a pristine install stands the unborn being on the stance.
    assert p["stance_seeded_on_pristine_install"] is True, p

    # THE DEFECT, against real host code: the live session hands the being the host's
    # assistant persona — the prompt is restored verbatim and never rebuilt, so the stance
    # on disk is nowhere in it.
    assert p["old_session_reused_its_prompt_verbatim"] is True, p
    assert p["old_session_prompt_holds_the_assistant_persona"] is True, p
    assert p["old_session_prompt_holds_the_stance"] is False, p
    assert p["identity_slot_is_stale"] is True, p

    # …and a lane somebody is using is not taken from them for it.
    assert p["a_busy_lane_holds_the_birth"] == "in_use", p

    # THE FIX — verified, not assumed (this is the assumption that was wrong last time):
    # the session ends, the agent cache is evicted, the next turn opens on an EMPTY history,
    # and *that* is what makes the host build the prompt again — with the stance in slot #1.
    assert p["voice"] == "ended", p
    assert p["session_rotated"] is True, p
    assert p["agent_cache_evicted"] is True, p
    assert p["fresh_session_history_is_empty"] is True, p
    assert p["birth_prompt_was_rebuilt"] is True, p
    assert p["birth_prompt_holds_the_stance"] is True, p
    assert p["birth_prompt_holds_the_assistant_persona"] is False, p

    # And exactly once: the session it was born into is not stale, so nothing ends twice.
    assert p["fresh_session_is_stale"] is False, p

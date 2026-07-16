"""Guarded integration test: the internal-cognition seam against real Hermes (lm-705.6).

Drives :mod:`tests.hermes_internal_cognition_integration` out of process under
Hermes' own interpreter (the venv that has ``gateway.*``/``agent.*`` on its path)
against a throwaway ``HERMES_HOME`` — never ``~/.hermes``, never a real channel,
never a real LLM call. See that module's docstring for exactly what "real" means
here and where the line against a live model call is drawn.

**Manual setup** (only needed to run this LOCALLY against a real Hermes install —
CI/a fresh clone has none of this and the test skips cleanly):

1. A working Hermes Agent install with its own venv, e.g. ``~/.hermes/hermes-agent/venv``
   (importable ``gateway``, ``agent``, ``hermes_cli``, ``cron`` packages).
2. Nothing else — the driver never reads real credentials or touches the
   network; it scripts the host's ``PluginLlm`` transport directly
   (``agent.plugin_llm.make_plugin_llm_for_test``).
3. Run: ``uv run pytest tests/test_internal_cognition_integration.py -v``
   (or the whole suite via ``make check`` — this test SKIPS, never fails, when
   no Hermes interpreter is found, so it never blocks the gate off-host).

Where Hermes is not installed (CI, a fresh clone) this **skips** cleanly so
``make check`` stays green everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from _hermes_probe import find_hermes_python

_DRIVER = Path(__file__).resolve().parent / "hermes_internal_cognition_integration.py"
# Flat root-layout: the repo dir IS the `lifemodel` package. The out-of-process
# driver does `import lifemodel`, so it needs a directory that CONTAINS a `lifemodel`
# package on its sys.path. The checkout dir may be named anything (dev worktrees), so
# the test hands the driver a per-test symlink dir (`<tmp>/src/lifemodel -> repo root`)
# instead of the repo's parent — mirrors ``test_reachin_integration.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_internal_cognition_seam_against_real_hermes(tmp_path: Path) -> None:
    hermes_py = find_hermes_python(tmp_path / "probe")
    if hermes_py is None:
        pytest.skip(
            "no Hermes interpreter with gateway.*/agent.* found; "
            "internal-cognition integration needs Hermes"
        )

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

    # Part A — aux-task registration + ctx.llm against the REAL host validation.
    assert payload["aux_task_registered"] is True, payload
    assert payload["aux_task_plugin"] == "lifemodel", payload
    assert payload["ctx_llm_resolved"] is True, payload

    # Part B1 — non-delivery, gateway-loop execution, typed re-entry, FR20 budget.
    assert payload["b1_launch_within_budget_accepted"] is True, payload
    assert payload["b1_pending_set_synchronously"] is True, payload
    assert payload["b1_host_call_used_real_message_building"] is True, payload
    assert payload["b1_apply_saw_typed_result"] is True, payload
    assert payload["b1_pending_cleared_after_completion"] is True, payload
    assert payload["b1_note_persisted"] is True, payload
    assert payload["b1_egress_never_called"] is True, payload  # non-delivery is structural
    assert payload["b1_second_call_denied_over_budget"] is True, payload
    assert payload["b1_budget_consumed_exactly_once"] is True, payload

    # Part B2 — a failed call still clears pending (no strand).
    assert payload["b2_failed_call_still_clears_pending"] is True, payload
    assert payload["b2_apply_saw_empty_result_on_failure"] is True, payload

    # Part B3 — stale-pending recovery at "connect".
    assert payload["b3_stale_pending_cleared_at_connect"] is True, payload

    # Part B4 — clean shutdown cancellation.
    assert payload["b4_task_actually_running_before_cancel"] is True, payload
    assert payload["b4_no_leaked_task_after_cancel_all"] is True, payload

    # Part B5 — codex #2 regression: an unrelated proactive launch still dispatches.
    assert payload["b5_unrelated_proactive_launch_was_dispatched"] is True, payload
    assert payload["b5_internal_pending_still_cleared"] is True, payload

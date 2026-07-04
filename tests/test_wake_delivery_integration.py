"""Guarded integration test: the Phase-1 capstone wake + delivery loop.

Drives the **real** ``cron.scheduler`` out of process under Hermes' interpreter
(the venv that has ``cron.*`` / ``run_agent`` on its path) against a **throwaway
``HERMES_HOME``** under ``tmp_path`` — never the user's ``~/.hermes``, never a real
channel. It runs :mod:`tests.hermes_wake_delivery_integration`, which registers the
heartbeat, ticks the real scheduler until the threshold crossing wakes the agent,
and asserts all seven epic acceptance criteria (see that module's docstring).

Where Hermes is not installed (CI, a fresh clone) it **skips** cleanly so
``make check`` stays green everywhere.

**Real vs stub (stated for the reviewer):** the cron machinery is REAL — the
``--script`` subprocess (our tick), ``_parse_wake_gate``, ``_build_job_prompt``
(wake-packet injection), the toolset resolution, and the delivery decision. The
**LLM turn is stubbed** (a canned one-line reply) and the provider resolver + the
final send are patched to a no-op, so no real model call and no outbound message
can happen. The Phase-1 point is the LOOP, not LLM quality (roadmap 1.4).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _hermes_probe import find_hermes_python

_DRIVER = Path(__file__).resolve().parent / "hermes_wake_delivery_integration.py"
_SRC = Path(__file__).resolve().parent.parent / "src"

# The seven epic acceptance criteria (roadmap 1.4), mapped to the driver's checks.
_REQUIRED_CHECKS = (
    "tick_fired_at_least_twice",
    "pressure_persists_between_ticks",
    "threshold_crossing_woke_agent",
    "woken_turn_received_wake_packet",
    "exactly_one_delivery_to_author",
    "woken_turn_is_text_only_no_tools",
    "pressure_drained_on_wake",
    "cooldown_opened_on_wake",
    "cooldown_vetoes_second_fire",
    "below_threshold_zero_llm",
    "exactly_one_llm_call_total",
)


def test_wake_delivery_loop_on_isolated_hermes(tmp_path: Path) -> None:
    hermes_py = find_hermes_python(tmp_path / "probe")
    if hermes_py is None:
        pytest.skip("no Hermes interpreter with cron.* found; integration needs a real Hermes")

    home = tmp_path / "iso-home"
    home.mkdir()
    # Sanity: never the user's live being.
    assert home.resolve() != (Path.home() / ".hermes").resolve()

    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "LIFEMODEL_SRC": str(_SRC),
    }
    result = subprocess.run(
        [hermes_py, str(_DRIVER)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"driver failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok", payload

    # Every epic criterion must hold.
    checks = payload["checks"]
    for name in _REQUIRED_CHECKS:
        assert checks.get(name) is True, f"criterion {name!r} failed: {checks}"

    # Spot-check the headline numbers directly (defence in depth over the flags).
    assert payload["delivery_count"] == 1, payload
    assert payload["resolved_toolsets"] == [], "woken turn must have zero tools"
    assert payload["llm_constructions"] == 1, "exactly one wake => exactly one LLM turn"
    assert payload["wake_tick_index"] is not None

    # Print the driver evidence so `pytest -s` / CI logs carry the real-scheduler proof.
    print("\n[wake+delivery integration] driver stderr:\n" + result.stderr, file=sys.stderr)

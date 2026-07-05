"""Guarded integration test: fire the heartbeat on a real, isolated Hermes.

The plugin's own test interpreter (the uv project venv) does **not** have the
Hermes ``cron`` package, so this test drives the real scheduler out-of-process:
it discovers Hermes' interpreter (the venv with ``cron.jobs`` on its path) and
runs :mod:`tests.hermes_heartbeat_integration` as a subprocess against a
**throwaway ``HERMES_HOME``** under ``tmp_path`` — never the user's ``~/.hermes``.

Where Hermes is not installed (CI, a fresh clone) it **skips** cleanly, so
``make check`` stays green everywhere while still exercising the full
cron + ``--script`` + wake-gate path wherever a real Hermes exists.

Chosen mechanism (per the roadmap-1.1 acceptance note): rather than wait real
wall-clock minutes for the in-process ticker, it invokes the scheduler's own
``run_job`` path programmatically — this still runs the real script subprocess
and the real ``_parse_wake_gate``, only without the 60s timer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_DRIVER = Path(__file__).resolve().parent / "hermes_heartbeat_integration.py"
# Flat root-layout: the repo dir IS the `lifemodel` package, so its *parent*
# (not a `src/` dir — there isn't one anymore) must go on the driver's
# sys.path for `import lifemodel` to resolve, mirroring the root conftest.py.
_SRC = Path(__file__).resolve().parent.parent.parent


def _candidate_pythons() -> list[Path]:
    """Ordered candidates for a Hermes-capable interpreter."""
    candidates: list[Path] = []
    override = os.environ.get("LIFEMODEL_HERMES_PYTHON")
    if override:
        candidates.append(Path(override))
    launcher = shutil.which("hermes")
    if launcher:
        # The launcher is a shell wrapper: `exec "<venv>/bin/hermes" "$@"`.
        try:
            text = Path(launcher).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        for token in text.split('"'):
            if token.endswith("/bin/hermes"):
                candidates.append(Path(token).with_name("python"))
    candidates.append(Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python")
    # De-dup, keep order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _find_hermes_python(probe_home: Path) -> str | None:
    """Return the first candidate interpreter that can import the Hermes cron API."""
    for py in _candidate_pythons():
        if not py.exists():
            continue
        probe = subprocess.run(
            [str(py), "-c", "import cron.jobs, cron.scheduler"],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HOME": str(probe_home)},
            timeout=60,
        )
        if probe.returncode == 0:
            return str(py)
    return None


def test_heartbeat_fires_twice_on_isolated_hermes(tmp_path: Path) -> None:
    hermes_py = _find_hermes_python(tmp_path / "probe")
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
        timeout=180,
    )
    # Surface the driver's stderr evidence on failure for a debuggable report.
    assert result.returncode == 0, (
        f"driver failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok", payload
    assert payload["job_count"] == 1, "register() must be idempotent (one heartbeat job)"
    assert all(payload["checks"].values()), payload["checks"]

    tick_counts = [f["tick_count"] for f in payload["fires"]]
    assert tick_counts == [1, 2], "state must persist and advance across ticks"
    assert all(f["silent"] for f in payload["fires"]), "every tick silent => zero LLM"

    # Print evidence so `pytest -s` / CI logs carry the real-scheduler proof.
    print("\n[heartbeat integration] driver stderr:\n" + result.stderr, file=sys.stderr)

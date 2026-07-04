"""Shared discovery of a Hermes-capable interpreter for the guarded integration
tests (roadmap 1.1 / 1.4).

The plugin's own test interpreter (the uv project venv) does **not** have the
Hermes ``cron`` package, so the real-scheduler integrations drive it out of
process under Hermes' own interpreter. This module locates that interpreter and
is shared by the heartbeat (1.1) and wake+delivery (1.4) wrappers so the probe
logic lives in exactly one place. It is not a test module (the leading underscore
keeps pytest from collecting it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def candidate_pythons() -> list[Path]:
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


def find_hermes_python(probe_home: Path) -> str | None:
    """Return the first candidate interpreter that can import the Hermes cron API."""
    for py in candidate_pythons():
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

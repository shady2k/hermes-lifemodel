"""Real-Hermes integration driver for the heartbeat (roadmap 1.1 acceptance).

This is **not** a pytest module (its name does not match ``test_*``); it is a
standalone driver meant to run under **Hermes' own interpreter** (its venv has
``cron.jobs`` / ``cron.scheduler`` on the path via an editable install). The
guarded wrapper :mod:`tests.test_heartbeat_integration` invokes it as a
subprocess against an **isolated, throwaway ``HERMES_HOME``**; you can also run
it by hand for evidence:

    HERMES_HOME=/tmp/iso LIFEMODEL_SRC=$(dirname "$PWD") \\
        ~/.hermes/hermes-agent/venv/bin/python tests/hermes_heartbeat_integration.py

What it proves against the real scheduler (never the default ``~/.hermes``):

* ``register(ctx)`` is **idempotent** — called twice, exactly one
  ``lifemodel-heartbeat`` cron job exists.
* Firing the job through the real ``cron.scheduler.run_job`` **twice** advances
  ``state.json`` (``tick_count`` 1 → 2, ``last_tick_at`` moves) — state is read
  and written and **persists between ticks**.
* Both fires take the wake-gate ``wakeAgent=false`` path → ``run_job`` returns
  the ``[SILENT]`` marker, so **zero LLM calls** happened.

Human-readable evidence goes to **stderr**; a single-line JSON result goes to
**stdout** for the wrapper to parse. Exit code 0 = all assertions held.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class _FakeCtx:
    """Duck-typed stand-in for Hermes' PluginContext (records nothing we need)."""

    profile_name = "integration-being"

    def register_command(self, *args: Any, **kwargs: Any) -> None: ...
    def register_tool(self, *args: Any, **kwargs: Any) -> None: ...
    def register_hook(self, *args: Any, **kwargs: Any) -> None: ...


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    home = Path(os.environ["HERMES_HOME"]).resolve()
    src = os.environ["LIFEMODEL_SRC"]
    # Hard guard: refuse to run against the user's real being.
    if home == (Path.home() / ".hermes").resolve():
        _log("REFUSING to run against the default ~/.hermes — set an isolated HERMES_HOME")
        return 2

    sys.path.insert(0, src)
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "scripts").mkdir(parents=True, exist_ok=True)

    import lifemodel
    from lifemodel.heartbeat import HEARTBEAT_JOB_NAME
    from lifemodel.paths import state_dir

    _log(f"[1] register(ctx) x2 against isolated HERMES_HOME={home}")
    lifemodel.register(_FakeCtx())
    lifemodel.register(_FakeCtx())  # idempotency: must NOT create a second job

    from cron.jobs import list_jobs

    heartbeats = [
        j for j in list_jobs(include_disabled=True) if j.get("name") == HEARTBEAT_JOB_NAME
    ]
    _log(f"    heartbeat jobs found: {len(heartbeats)}")
    if len(heartbeats) != 1:
        _log(f"FAIL: expected exactly one heartbeat job, got {len(heartbeats)}")
        return 1
    job = heartbeats[0]
    _log(
        f"    job id={job['id']} schedule={job.get('schedule')} script={job.get('script')} "
        f"no_agent={job.get('no_agent')}"
    )

    from cron.scheduler import SILENT_MARKER, run_job

    state_file = state_dir(home) / "state.json"
    fires: list[dict[str, Any]] = []
    for i in range(2):
        ok, _doc, final, err = run_job(job)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        fires.append(
            {
                "ok": ok,
                "final": final,
                "silent": final == SILENT_MARKER,
                "err": err,
                "tick_count": state["tick_count"],
                "last_tick_at": state["last_tick_at"],
            }
        )
        _log(
            f"[2] fire #{i + 1}: ok={ok} final={final!r} "
            f"tick_count={state['tick_count']} last_tick_at={state['last_tick_at']}"
        )

    r0, r1 = fires
    checks = {
        "both_ran_ok": bool(r0["ok"] and r1["ok"]),
        "both_silent_zero_llm": bool(r0["silent"] and r1["silent"]),
        "tick_count_advanced": r0["tick_count"] == 1 and r1["tick_count"] == 2,
        "last_tick_at_advanced": r1["last_tick_at"] != r0["last_tick_at"]
        and r0["last_tick_at"] is not None,
        "state_written_under_isolated_home": state_file.is_file()
        and str(state_file).startswith(str(home)),
    }
    _log(f"[3] checks: {checks}")

    passed = all(checks.values())
    result = {
        "status": "ok" if passed else "fail",
        "job_count": len(heartbeats),
        "job_id": job["id"],
        "checks": checks,
        "fires": fires,
    }
    print(json.dumps(result), flush=True)  # single-line JSON for the wrapper
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

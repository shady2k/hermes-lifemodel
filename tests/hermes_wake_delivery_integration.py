"""Real-scheduler wake + delivery driver — the Phase-1 capstone (roadmap 1.4).

This is **not** a pytest module (its name does not match ``test_*``); it is a
standalone driver run under **Hermes' own interpreter** by the guarded wrapper
:mod:`tests.test_wake_delivery_integration` against an **isolated, throwaway
``HERMES_HOME``** (never ``~/.hermes``, never a real channel). It proves the whole
walking-skeleton loop end-to-end through the **real** ``cron.scheduler`` — the real
``create_job``, the real ``--script`` subprocess (our tick), the real
``_parse_wake_gate`` / ``_build_job_prompt``, the real toolset resolution and the
real delivery decision — closing all seven epic acceptance criteria:

1. the tick fires ≥ 2 times;
2. pressure persists between ticks;
3. threshold crossing flips the gate to ``wakeAgent: true``;
4. the woken turn RECEIVES the wake-packet (the injected prompt contains it);
5. exactly ONE message is delivered to the author (local) channel, text-only;
6. pressure drains to 0 after the wake and a cooldown prevents an immediate
   second fire (even when pressure is pushed back above threshold);
7. below threshold there are ZERO LLM calls.

**Safety.** Nothing may leave the process. The isolated home is guarded against
``~/.hermes``; the job's ``deliver`` is forced to ``"local"`` (no outbound); and,
belt-and-suspenders, we monkeypatch the delivery, the LLM agent, and the provider
resolver so there is no path to a real send or a real model call:

* ``run_agent.AIAgent`` → a stub that records the injected prompt + the resolved
  ``enabled_toolsets`` and returns a canned one-line message (**stubbed cognition
  — no real LLM**; the Phase-1 point is the LOOP, not LLM quality);
* ``hermes_cli.runtime_provider.resolve_runtime_provider`` → a benign dummy so the
  real run_job reaches the (stubbed) agent instead of failing on missing auth;
* ``cron.scheduler._deliver_result`` → records ``(deliver, text)`` and returns
  ``None`` (no send), so we can *count* deliveries exactly.

Human-readable evidence goes to **stderr**; a single-line JSON result goes to
**stdout** for the wrapper to parse. Exit code 0 = every criterion held.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


#: The canned one-line message the stubbed cognition "sends". Text only.
_CANNED_MESSAGE = "Hi — my inner pressure built up enough that I wanted to reach out and say hello."


class _StubAgent:
    """Stub for ``run_agent.AIAgent`` — records inputs, returns a canned reply.

    Construction count is our zero-LLM proof: below threshold the scheduler
    short-circuits at the wake gate and never constructs the agent, so this stays
    at 0 until the crossing tick.
    """

    constructions: list[dict[str, Any]] = []
    prompts: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        _StubAgent.constructions.append(kwargs)

    def run_conversation(self, prompt: str) -> dict[str, Any]:
        _StubAgent.prompts.append(prompt)
        return {
            "final_response": _CANNED_MESSAGE,
            "completed": True,
            "failed": False,
            "turn_exit_reason": "",
        }

    def close(self) -> None: ...

    def get_activity_summary(self) -> dict[str, Any]:
        return {}

    def interrupt(self, _reason: str) -> None: ...

    @staticmethod
    def _format_turn_completion_explanation(_reason: str) -> str:
        return ""


def _install_stubs() -> list[tuple[str, str]]:
    """Patch the LLM agent, provider resolver, and delivery. Returns the (shared)
    list that records each delivery as ``(deliver_value, text)``."""
    import run_agent

    run_agent.AIAgent = _StubAgent  # type: ignore[misc, assignment]

    import hermes_cli.runtime_provider as _rp

    def _stub_resolve(**_kwargs: Any) -> dict[str, Any]:
        # provider="" keeps the credential-pool + provider-drift guards inert.
        return {
            "provider": "",
            "api_key": "stub",
            "base_url": None,
            "api_mode": None,
            "command": None,
            "args": None,
        }

    _rp.resolve_runtime_provider = _stub_resolve  # type: ignore[assignment]

    import cron.scheduler as _sched

    deliveries: list[tuple[str, str]] = []

    def _recording_deliver(job: dict, content: str, adapters: Any = None, loop: Any = None) -> None:
        # Record intent; never actually send. deliver is already "local" (no
        # target), but this guarantees no outbound even if that ever changed.
        deliveries.append((str(job.get("deliver")), content))
        return None

    _sched._deliver_result = _recording_deliver  # type: ignore[assignment]
    return deliveries


def _read_state(home: Path) -> dict[str, Any]:
    from lifemodel.paths import state_dir

    state_file = state_dir(home) / "state.json"
    return json.loads(state_file.read_text(encoding="utf-8"))


def main() -> int:
    home = Path(os.environ["HERMES_HOME"]).resolve()
    src = os.environ["LIFEMODEL_SRC"]
    if home == (Path.home() / ".hermes").resolve():
        _log("REFUSING to run against the default ~/.hermes — set an isolated HERMES_HOME")
        return 2

    sys.path.insert(0, src)
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "scripts").mkdir(parents=True, exist_ok=True)
    # A model must resolve for the (stubbed) LLM path; pin it via env so no real
    # provider/config is consulted.
    os.environ["HERMES_MODEL"] = "stub-model"

    from cron.jobs import create_job, list_jobs
    from cron.scheduler import _resolve_cron_enabled_toolsets, run_one_job

    import lifemodel  # noqa: F401  (import side effects: none, but proves it loads)
    from lifemodel.core.aggregator import DEFAULT_WAKE_THRESHOLD, WAKE_REASON
    from lifemodel.heartbeat import ensure_heartbeat_job
    from lifemodel.paths import state_dir
    from lifemodel.state.json_store import JsonStateStore

    deliveries = _install_stubs()

    # --- Register the heartbeat with delivery FORCED to local (captured). -----
    _log(f"[setup] register heartbeat on isolated HERMES_HOME={home} (deliver=local)")
    job = ensure_heartbeat_job(
        home=home,
        src_dir=Path(src),
        create_job=create_job,
        list_jobs=list_jobs,
        deliver="local",  # capture — nothing leaves the process
    )
    # Pin model/provider on the in-memory job so the stubbed run_job path is
    # deterministic and the provider/model drift guards never engage.
    job["model"] = "stub-model"
    job["provider"] = "stub-provider"
    job["model_snapshot"] = None
    job["provider_snapshot"] = None

    threshold = float(DEFAULT_WAKE_THRESHOLD)
    _log(f"[setup] job id={job['id']} deliver={job['deliver']} threshold={threshold}")

    # No-tools rail (FINDING 3), proven at TWO layers with the REAL Hermes code:
    #  (1) the scheduler resolver reduces our ["no_mcp"] job to an empty toolset;
    #  (2) the tool-definition layer then yields a genuinely EMPTY tool schema for
    #      that empty allowlist — so the woken agent is handed zero tools.
    # Residual (documented): model_tools force-appends `kanban` to any non-None
    # allowlist when HERMES_KANBAN_TASK is set; a normal cron process does not set
    # it, so we clear it here to prove the real floor deterministically.
    os.environ.pop("HERMES_KANBAN_TASK", None)
    resolved_toolsets = _resolve_cron_enabled_toolsets(job, {})
    from cron.scheduler import _resolve_cron_disabled_toolsets
    from model_tools import get_tool_definitions

    tool_schema = get_tool_definitions(
        enabled_toolsets=resolved_toolsets,
        disabled_toolsets=_resolve_cron_disabled_toolsets({}),
        quiet_mode=True,
    )
    tool_names = [t.get("function", {}).get("name") for t in tool_schema]
    _log(
        f"[setup] enabled_toolsets stored={job.get('enabled_toolsets')} "
        f"resolved={resolved_toolsets} tool_schema_size={len(tool_names)} names={tool_names[:8]}"
    )

    # --- Drive real ticks until the crossing tick wakes. ----------------------
    max_ticks = int(threshold) + 4
    ticks: list[dict[str, Any]] = []
    wake_tick_index: int | None = None
    for i in range(1, max_ticks + 1):
        deliveries_before = len(deliveries)
        constructions_before = len(_StubAgent.constructions)
        run_one_job(job)
        state = _read_state(home)
        delivered = len(deliveries) - deliveries_before
        constructed = len(_StubAgent.constructions) - constructions_before
        rec = {
            "i": i,
            "tick_count": state["tick_count"],
            "pressure": state["pressure"],
            "cooldown_until": state.get("cooldown_until"),
            "last_contact_at": state.get("last_contact_at"),
            "delivered": delivered,
            "constructed": constructed,
        }
        ticks.append(rec)
        _log(
            f"[tick {i}] tick_count={rec['tick_count']} pressure={rec['pressure']} "
            f"delivered={delivered} llm_constructed={constructed} "
            f"cooldown_until={rec['cooldown_until']}"
        )
        if delivered:
            wake_tick_index = i
            break

    # --- Criterion 3 + 4: the injected prompt carries the wake-packet. --------
    # The prompt the (stubbed) agent received IS the scheduler's real
    # _build_job_prompt output, which injects our tick's stdout (the wake-gate
    # line). Asserting on it proves both the gate flipped (``wakeAgent: true``)
    # and the woken turn received the full packet — with no extra script run that
    # would double-advance state.
    wake_prompt = _StubAgent.prompts[-1] if _StubAgent.prompts else ""
    packet_in_prompt = (
        '"wakeAgent": true' in wake_prompt
        and WAKE_REASON in wake_prompt
        and '"pressure":' in wake_prompt
        and '"threshold":' in wake_prompt
    )
    _log(f"[wake] prompt carries packet={packet_in_prompt}")
    _log(f"[wake] captured prompt head:\n{wake_prompt[:400]}")

    # --- Criterion 6 (direct cooldown proof): push pressure back above the ----
    # threshold WHILE the cooldown is active and fire once more — it must stay
    # silent (no delivery, no LLM), proving the cooldown vetoes a would-be wake.
    store = JsonStateStore(state_dir(home))
    s = store.load()
    s.pressure = threshold + 5.0
    store.commit(s)
    deliveries_before = len(deliveries)
    constructions_before = len(_StubAgent.constructions)
    run_one_job(job)
    cooldown_probe = {
        "pressure_after": _read_state(home)["pressure"],
        "delivered": len(deliveries) - deliveries_before,
        "constructed": len(_StubAgent.constructions) - constructions_before,
        "cooldown_until": _read_state(home).get("cooldown_until"),
    }
    _log(f"[cooldown probe] forced pressure>={threshold}, result={cooldown_probe}")

    # --- FINDING 1+2 end-to-end: a crashing tick FAILS CLOSED through the real ---
    # --script subprocess. Corrupt state.json with a tz-naive cooldown_until (also
    # pressure far above threshold): the tick's state load raises StateCorruptError,
    # yet the shim must still exit 0 with {"wakeAgent": false} so the REAL scheduler
    # stays silent — no wake, no crash-message delivery. Proven via the real
    # _run_job_script + _parse_wake_gate. This is the last probe (it leaves state
    # corrupt on purpose).
    from cron.scheduler import _parse_wake_gate, _run_job_script

    (state_dir(home) / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pressure": threshold + 99.0,
                "cooldown_until": "2026-07-04T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    fc_ok, fc_out = _run_job_script(job["script"])
    fc_last = next((ln for ln in reversed(fc_out.splitlines()) if ln.strip()), "")
    fail_closed = (
        fc_ok  # script exited 0 despite the internal crash
        and _parse_wake_gate(fc_out) is False  # scheduler would skip the agent
        and json.loads(fc_last) == {"wakeAgent": False}
    )
    _log(f"[fail-closed probe] script_ok={fc_ok} last_line={fc_last!r} fail_closed={fail_closed}")

    # --- Assemble checks. -----------------------------------------------------
    pre_wake = ticks[:-1] if wake_tick_index is not None else ticks
    wake_rec = ticks[-1] if wake_tick_index is not None else None
    pressures = [t["pressure"] for t in ticks]

    checks = {
        # 1. tick fired >= 2 times
        "tick_fired_at_least_twice": len(ticks) >= 2 and ticks[-1]["tick_count"] >= 2,
        # 2. pressure persists / accumulates between ticks (strictly grew each
        #    pre-wake tick from the committed state, not recomputed from zero)
        "pressure_persists_between_ticks": all(
            pressures[k] > pressures[k - 1] for k in range(1, len(pre_wake))
        )
        and len(pre_wake) >= 1,
        # 3. threshold crossing flipped the gate to wakeAgent:true (seen in the
        #    injected prompt) and it happened at/after pressure reached threshold
        "threshold_crossing_woke_agent": wake_tick_index is not None
        and '"wakeAgent": true' in wake_prompt,
        # 4. the woken turn received the full wake-packet
        "woken_turn_received_wake_packet": packet_in_prompt,
        # 5. exactly ONE delivery, to the author (local) channel, of the text msg
        "exactly_one_delivery_to_author": len(deliveries) == 1
        and deliveries[0][0] == "local"
        and deliveries[0][1] == _CANNED_MESSAGE,
        # 5b. text-only: the woken agent was handed ZERO tools — proven at the
        #     real resolver AND the real tool-definition layer (FINDING 3).
        "woken_turn_is_text_only_no_tools": resolved_toolsets == [],
        "woken_turn_tool_schema_is_empty": tool_names == [],
        # 6. drained to 0 on the wake, cooldown opened, and the cooldown vetoes a
        #    forced above-threshold second fire (no 2nd delivery / no LLM)
        "pressure_drained_on_wake": wake_rec is not None and wake_rec["pressure"] == 0.0,
        "cooldown_opened_on_wake": wake_rec is not None and wake_rec["cooldown_until"] is not None,
        "cooldown_vetoes_second_fire": cooldown_probe["delivered"] == 0
        and cooldown_probe["constructed"] == 0,
        # 7. below threshold => zero LLM (agent never constructed) and no delivery
        "below_threshold_zero_llm": all(t["constructed"] == 0 for t in pre_wake)
        and all(t["delivered"] == 0 for t in pre_wake),
        # exactly one LLM construction across the whole run (the single wake)
        "exactly_one_llm_call_total": len(_StubAgent.constructions) == 1,
        # a crashing tick fails CLOSED end-to-end (silent gate, exit 0) — the real
        # scheduler would NOT wake/deliver on a corrupt-state crash.
        "crash_fails_closed_silent": fail_closed,
    }

    passed = all(checks.values())
    _log(f"[checks] {json.dumps(checks, indent=2)}")

    result = {
        "status": "ok" if passed else "fail",
        "job_id": job["id"],
        "threshold": threshold,
        "wake_tick_index": wake_tick_index,
        "resolved_toolsets": resolved_toolsets,
        "tool_schema_names": tool_names,
        "delivery_count": len(deliveries),
        "delivered": deliveries,
        "llm_constructions": len(_StubAgent.constructions),
        "cooldown_probe": cooldown_probe,
        "fail_closed_probe": {"script_ok": fc_ok, "last_line": fc_last},
        "checks": checks,
        "ticks": ticks,
    }
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

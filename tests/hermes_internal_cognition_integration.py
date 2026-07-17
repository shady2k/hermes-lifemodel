"""Real-Hermes driver — proves the non-delivered internal-cognition seam (lm-705.6).

Not a pytest module (its name does not match ``test_*``): a standalone driver run
under **Hermes' own interpreter** by the guarded wrapper
:mod:`tests.test_internal_cognition_integration` against an **isolated, throwaway
``HERMES_HOME``** — never ``~/.hermes``, never a real channel, never a real LLM
call (mirrors ``hermes_genesis_prompt_integration.py``'s house rule).

**What "real" means here, and where the line is drawn:**

* **Part A — aux-task registration against the REAL host validation.** Builds a
  genuine ``hermes_cli.plugins.PluginContext`` (real ``PluginManifest`` +
  ``PluginManager`` — both cheap, no I/O) and runs the plugin's REAL
  ``lifemodel.register(ctx)`` against it. Proves ``ctx.register_auxiliary_task
  ("lifemodel_internal", ...)`` succeeds against the host's real key-format /
  builtin-shadow validation (``hermes_cli/plugins.py:1047``) and that
  ``register()`` resolves a real ``ctx.llm`` without raising. This is as far as
  "custom-slot routing" can be proven honestly: as documented in
  ``adapters/plugin_llm_adapter.py``, the CURRENT ``ctx.llm.acomplete_structured``
  hard-codes ``task=None`` — a registered aux-task key is not yet reachable
  through it. Nothing here calls a model.
* **Part B — the seam's real mechanics, over a REAL SQLite store, driven by the
  REAL host's structured-completion code path with a SCRIPTED transport.** Uses
  ``agent.plugin_llm.make_plugin_llm_for_test(..., async_caller=...)` — a
  host-provided test seam — to build a genuine ``PluginLlm`` whose message
  building / JSON-schema handling / response parsing (``_build_structured_messages``,
  ``_parse_structured_text``, ``_extract_text``/``_extract_usage``) all run for
  real, but the actual network hop is a scripted async function. Proves, against
  the real store and the real completion-frame code:
  non-delivery (the egress is never called for the internal path itself);
  gateway-loop task execution + retention;
  typed re-entry under the lock (the scripted JSON round-trips through
  ``PluginLlmPort`` into the injected ``apply`` component);
  timeout/failure → pending still clears (no strand);
  stale-pending recovery at "connect";
  clean shutdown cancellation (``cancel_all``);
  the FR20 hard-budget ceiling denying the N+1 call;
  and the codex-#2 regression — a completion frame whose (unrelated) proactive
  launch reaches the egress, dispatched, while the internal call itself never did.

Human-readable evidence to **stderr**; one line of JSON to **stdout**. Exit 0 =
every assertion held.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


_TARGET: dict[str, str | None] = {"platform": "test", "chat_id": "1", "thread_id": None}
_BORN_AT = "2026-01-01T10:00:00+00:00"
_ORIGIN_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def _fake_openai_response(text: str, *, model: str = "fake-model") -> Any:
    class _Msg:
        content = text

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 5
        completion_tokens = 5
        total_tokens = 10

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    _Resp.model = model  # type: ignore[attr-defined]
    return _Resp()


async def main_async() -> dict[str, Any]:
    import asyncio
    from datetime import UTC, datetime

    from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    from lifemodel import register
    from lifemodel.adapters.clock import SystemClock
    from lifemodel.adapters.internal_runner import InternalCognitionRunner
    from lifemodel.adapters.plugin_llm_adapter import PluginLlmPort
    from lifemodel.composition import build_lifemodel
    from lifemodel.core.commitment_view import read_live_commitments
    from lifemodel.core.component import ComponentLayer, TickContext
    from lifemodel.core.intents import Intent, LaunchProactive, PutRecord
    from lifemodel.core.internal_cognition import NullInternalApply
    from lifemodel.core.llm_port import InternalCognitionRequest
    from lifemodel.core.noticing import NOTICING_JSON_SCHEMA, NoticingApply
    from lifemodel.core.noticing_buffer import NoticingBuffer
    from lifemodel.core.registry import ComponentManifest
    from lifemodel.core.taxonomy import KIND_INTERNAL_RESULT, read_internal_result
    from lifemodel.core.thought_processing import ThoughtProcessingApply
    from lifemodel.core.thought_view import (
        build_thought,
        encode_thought,
        read_thought,
        seed_thought_id,
    )
    from lifemodel.domain.egress import ReachOutcome
    from lifemodel.domain.memory import MemoryDraft, PutOp
    from lifemodel.domain.objects import ThoughtState
    from lifemodel.hooks import make_felt_state_injector, make_post_llm_observer
    from lifemodel.state.model import State

    result: dict[str, Any] = {}

    # ---- Part A: aux-task registration against the REAL host validation ----
    home = Path(os.environ["HERMES_HOME"]).resolve()
    sdir = home / "workspace" / "lifemodel"
    sdir.mkdir(parents=True, exist_ok=True)

    manager = PluginManager()
    manifest = PluginManifest(name="lifemodel")
    ctx = PluginContext(manifest, manager)
    register(ctx)  # the REAL register() — real aux-task registration, real ctx.llm

    result["aux_task_registered"] = "lifemodel_internal" in manager._aux_tasks
    aux_entry = manager._aux_tasks.get("lifemodel_internal") or {}
    result["aux_task_display_name"] = aux_entry.get("display_name")
    result["aux_task_plugin"] = aux_entry.get("plugin")
    result["ctx_llm_resolved"] = ctx.llm is not None
    _log(f"[A] aux_task_registered={result['aux_task_registered']} entry={aux_entry}")

    class FakeEgress:
        def __init__(self, outcome: ReachOutcome = ReachOutcome.DELIVERED) -> None:
            self.outcome = outcome
            self.calls: list[tuple[Any, str]] = []

        def reach_out(self, target: Any, impulse: str) -> ReachOutcome:
            self.calls.append((target, impulse))
            return self.outcome

    class RecordingApply:
        id = "recording-apply"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def step(self, tick_ctx: TickContext) -> list[Intent]:
            intents: list[Intent] = []
            for sig in tick_ctx.signals:
                if sig.kind == KIND_INTERNAL_RESULT:
                    read = read_internal_result(sig)
                    self.seen.append(read.raw)
                    intents.append(
                        PutRecord(
                            op=PutOp(
                                draft=MemoryDraft(
                                    kind="note",
                                    id=f"n-{read.correlation_id}",
                                    state="active",
                                    payload={"raw": read.raw},
                                    source="integration-driver",
                                )
                            )
                        )
                    )
            return intents

    def _build_lm_factory(base_dir: Path):
        def _build():
            return build_lifemodel(base_dir=base_dir, clock=SystemClock())

        return _build

    def _commit_born(base_dir: Path, **fields: object) -> None:
        base: dict[str, object] = dict(genesis_completed_at=_BORN_AT)
        base.update(fields)
        _build_lm_factory(base_dir)().state.commit(State(**base))  # type: ignore[arg-type]

    # ---- Part B1: non-delivery + gateway-loop execution + typed re-entry ----
    b1_dir = home / "seam-b1"
    b1_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(b1_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_b1 = _build_lm_factory(b1_dir)

    calls_b1: list[dict[str, Any]] = []

    async def _async_caller_b1(**kwargs: Any) -> Any:
        calls_b1.append(kwargs)
        # PluginLlm._invoke_async's injected async_caller contract (agent/plugin_llm.py):
        # returns (provider, model, response) — a bare response raises inside the host's
        # own `real_provider, real_model, response = await self._invoke_async(...)` unpack.
        return (
            "fake-provider",
            "fake-model",
            _fake_openai_response('{"gist": "noticed something real"}'),
        )

    plugin_llm_b1 = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_b1,
    )
    egress_b1 = FakeEgress()
    apply_b1 = RecordingApply()
    runner_b1 = InternalCognitionRunner(
        build_lm_b1,
        PluginLlmPort(plugin_llm_b1),
        egress_b1,
        _TARGET,
        daily_ceiling=1,
        gateway_loop=asyncio.get_running_loop(),
        apply=apply_b1,
        timeout=10.0,
    )
    ok_first = runner_b1.launch(
        InternalCognitionRequest(instructions="notice", input_text="the segment"), "c-1"
    )
    result["b1_launch_within_budget_accepted"] = ok_first
    result["b1_pending_set_synchronously"] = build_lm_b1().state.load().pending_internal_id == "c-1"
    for task in list(runner_b1._tasks):
        await task
    result["b1_host_call_used_real_message_building"] = bool(calls_b1 and calls_b1[0]["messages"])
    result["b1_apply_saw_typed_result"] = apply_b1.seen == ['{"gist": "noticed something real"}']
    result["b1_pending_cleared_after_completion"] = (
        build_lm_b1().state.load().pending_internal_id is None
    )
    result["b1_note_persisted"] = build_lm_b1().state.get("note", "n-c-1") is not None
    result["b1_egress_never_called"] = egress_b1.calls == []  # non-delivery is structural

    # ---- Part B1b: FR20 hard-budget ceiling denies the N+1 call ----
    ok_second = runner_b1.launch(
        InternalCognitionRequest(instructions="notice", input_text="another segment"), "c-2"
    )
    result["b1_second_call_denied_over_budget"] = ok_second is False
    result["b1_budget_consumed_exactly_once"] = build_lm_b1().state.load().internal_calls_today == 1
    _log(f"[B1] {result}")

    # ---- Part B2: a failed/timed-out call still clears pending (no strand) ----
    b2_dir = home / "seam-b2"
    b2_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(b2_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_b2 = _build_lm_factory(b2_dir)

    async def _async_caller_b2(**kwargs: Any) -> Any:
        raise RuntimeError("simulated provider failure")

    plugin_llm_b2 = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_b2,
    )
    apply_b2 = RecordingApply()
    runner_b2 = InternalCognitionRunner(
        build_lm_b2,
        PluginLlmPort(plugin_llm_b2),
        FakeEgress(),
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=apply_b2,
        timeout=10.0,
    )
    runner_b2.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-fail")
    for task in list(runner_b2._tasks):
        await task
    result["b2_failed_call_still_clears_pending"] = (
        build_lm_b2().state.load().pending_internal_id is None
    )
    result["b2_apply_saw_empty_result_on_failure"] = apply_b2.seen == [""]
    _log(f"[B2] {result['b2_failed_call_still_clears_pending']=} {apply_b2.seen=}")

    # ---- Part B3: stale-pending recovery at "connect" ----
    b3_dir = home / "seam-b3"
    b3_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(b3_dir, pending_internal_id="stale-from-a-dead-process")
    build_lm_b3 = _build_lm_factory(b3_dir)
    runner_b3 = InternalCognitionRunner(
        build_lm_b3,
        PluginLlmPort(plugin_llm_b2),  # unused here — recover_stale never calls it
        FakeEgress(),
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=NullInternalApply(),
    )
    runner_b3.recover_stale(build_lm_b3())
    result["b3_stale_pending_cleared_at_connect"] = (
        build_lm_b3().state.load().pending_internal_id is None
    )
    _log(f"[B3] {result['b3_stale_pending_cleared_at_connect']=}")

    # ---- Part B4: clean shutdown cancellation ----
    b4_dir = home / "seam-b4"
    b4_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(b4_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_b4 = _build_lm_factory(b4_dir)

    async def _hanging_caller(**kwargs: Any) -> Any:
        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled first")

    plugin_llm_b4 = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_hanging_caller,
    )
    runner_b4 = InternalCognitionRunner(
        build_lm_b4,
        PluginLlmPort(plugin_llm_b4),
        FakeEgress(),
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=NullInternalApply(),
        timeout=3600.0,
    )
    runner_b4.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-hang")
    await asyncio.sleep(0)
    result["b4_task_actually_running_before_cancel"] = len(runner_b4._tasks) == 1
    await asyncio.wait_for(runner_b4.cancel_all(), timeout=10.0)
    result["b4_no_leaked_task_after_cancel_all"] = runner_b4._tasks == set()
    _log(
        f"[B4] {result['b4_task_actually_running_before_cancel']=} "
        f"{result['b4_no_leaked_task_after_cancel_all']=}"
    )

    # ---- Part B5: codex #2 — an unrelated proactive launch is still dispatched ----
    b5_dir = home / "seam-b5"
    b5_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(b5_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_b5 = _build_lm_factory(b5_dir)

    class FakeProactiveLauncher:
        id = "fake-proactive-launcher"

        def step(self, tick_ctx: TickContext) -> list[Intent]:
            return [
                LaunchProactive(
                    prompt="an unrelated proactive impulse",
                    correlation_id="proactive-from-b5",
                    origin_traceparent=_ORIGIN_TP,
                )
            ]

    lm_b5 = build_lm_b5()
    lm_b5.registry.register(
        FakeProactiveLauncher(),
        ComponentManifest(
            id=FakeProactiveLauncher.id,
            type="cognition",
            layer=ComponentLayer.COGNITION,
            metric_surface=(),
            accepts_signals=False,
        ),
    )
    egress_b5 = FakeEgress()

    async def _async_caller_b5(**kwargs: Any) -> Any:
        return "fake-provider", "fake-model", _fake_openai_response("")

    plugin_llm_b5 = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_b5,
    )
    runner_b5 = InternalCognitionRunner(
        lambda: lm_b5,  # SAME registry every call, so the fake launcher stays registered
        PluginLlmPort(plugin_llm_b5),
        egress_b5,
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=NullInternalApply(),
        timeout=10.0,
    )
    runner_b5.launch(InternalCognitionRequest(instructions="i", input_text="t"), "c-b5")
    for task in list(runner_b5._tasks):
        await task
    result["b5_unrelated_proactive_launch_was_dispatched"] = len(egress_b5.calls) == 1 and (
        egress_b5.calls[0][1] == "an unrelated proactive impulse"
    )
    result["b5_internal_pending_still_cleared"] = lm_b5.state.load().pending_internal_id is None
    _log(f"[B5] {result['b5_unrelated_proactive_launch_was_dispatched']=}")

    # ---- Part B-processing: a real thought seeded, processed, and resolved ----
    bp_dir = home / "seam-bp"
    bp_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(bp_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_bp = _build_lm_factory(bp_dir)
    build_lm_bp().state.put(
        encode_thought(
            build_thought(
                id="thought:seed:p",
                content="the owner mentioned a trip on Friday",
                state=ThoughtState.ACTIVE,
                salience=0.8,
            )
        )
    )

    async def _async_caller_bp(**kwargs: Any) -> Any:
        return (
            "fake-provider",
            "fake-model",
            _fake_openai_response('{"outcome": "resolve", "reflection": "thought it through"}'),
        )

    plugin_llm_bp = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_bp,
    )
    egress_bp = FakeEgress()
    runner_bp = InternalCognitionRunner(
        build_lm_bp,
        PluginLlmPort(plugin_llm_bp),
        egress_bp,
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=ThoughtProcessingApply(),
        timeout=10.0,
    )
    runner_bp.launch(
        InternalCognitionRequest(
            instructions="ruminate", input_text="the thought", json_schema={"type": "object"}
        ),
        "c-proc",
        subject_id="thought:seed:p",
    )
    # Single-flight (checked BEFORE awaiting the first task, so a denial here can
    # only be the in-flight gate — daily_ceiling=5 rules out the budget instead).
    ok_bp_second = runner_bp.launch(
        InternalCognitionRequest(instructions="x", input_text="y"),
        "c-proc-2",
        subject_id="thought:seed:p",
    )
    result["bp_single_flight_denied_concurrent"] = ok_bp_second is False
    for task in list(runner_bp._tasks):
        await task
    result["bp_pending_cleared"] = build_lm_bp().state.load().pending_internal_id is None
    result["bp_subject_cleared"] = build_lm_bp().state.load().pending_internal_subject_id is None
    result["bp_thought_resolved"] = read_thought(build_lm_bp().state, "thought:seed:p") is None
    result["bp_egress_never_called"] = egress_bp.calls == []  # non-delivery is structural
    _log(f"[BP] {result}")

    # ---- Part B-crystallize: a real thought crystallized into a real Commitment ----
    bc_dir = home / "seam-bc"
    bc_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(bc_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_bc = _build_lm_factory(bc_dir)
    build_lm_bc().state.put(
        encode_thought(
            build_thought(
                id="thought:seed:bc",
                content="the owner mentioned a job interview on Friday",
                state=ThoughtState.ACTIVE,
                salience=0.8,
            )
        )
    )

    async def _async_caller_bc(**kwargs: Any) -> Any:
        return (
            "fake-provider",
            "fake-model",
            _fake_openai_response(
                '{"outcome": "crystallize_commitment", '
                '"reflection": "I want to check in on that", '
                '"commitment": {"content": "ask how their interview went", '
                '"basis": "follow_up", "trigger_kind": "event", '
                '"trigger_value": "next time we talk"}}'
            ),
        )

    plugin_llm_bc = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_bc,
    )
    egress_bc = FakeEgress()
    runner_bc = InternalCognitionRunner(
        build_lm_bc,
        PluginLlmPort(plugin_llm_bc),
        egress_bc,
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=ThoughtProcessingApply(),
        timeout=10.0,
    )
    runner_bc.launch(
        InternalCognitionRequest(
            instructions="ruminate", input_text="the thought", json_schema={"type": "object"}
        ),
        "c-cryst",
        subject_id="thought:seed:bc",
    )
    for task in list(runner_bc._tasks):
        await task
    commitments_bc = read_live_commitments(build_lm_bc().state)
    result["bc_commitment_created"] = any(
        c.content == "ask how their interview went" for c in commitments_bc
    )
    commitment_bc = next(
        (c for c in commitments_bc if c.content == "ask how their interview went"), None
    )
    result["bc_commitment_links_source"] = commitment_bc is not None and (
        "thought:seed:bc" in commitment_bc.source_thought_ids
    )
    result["bc_thought_resolved"] = read_thought(build_lm_bc().state, "thought:seed:bc") is None
    result["bc_egress_never_called"] = egress_bc.calls == []  # non-delivery is structural
    result["bc_pending_cleared"] = build_lm_bc().state.load().pending_internal_id is None
    _log(f"[BC] {result}")

    # ---- Part B-noticing: a real noticing pass seeds thoughts, non-delivered ----
    bn_dir = home / "seam-bn"
    bn_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(bn_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_bn = _build_lm_factory(bn_dir)

    # The noticing buffer is process-owned (never per-graph, see
    # core/noticing_buffer.py) -- seed a closed segment through its OWN public API,
    # exactly as the pre_llm/post_llm hooks would (Task 3/E3), rather than a fresh
    # NoticingTrigger heartbeat: this proves the COMPLETION half of the seam
    # (NoticingApply reading the segment back via ``segment_through`` + validating +
    # creating real thoughts) against the REAL host's structured-completion path, the
    # same way Part B-processing/B-crystallize drive ``ThoughtProcessingApply``
    # directly rather than waiting on ``ThoughtProcessingSelector``.
    bn_buffer = NoticingBuffer()
    bn_now = datetime(2026, 1, 1, 10, 5, tzinfo=UTC)
    bn_buffer.open_pending("bn-session", user_text="I have a big interview Friday", now=bn_now)
    bn_buffer.complete("bn-session", "bn-turn-1", assistant_text="Good luck!", now=bn_now)
    bn_correlation_id = f"notice-bn-session@bn-turn-1@{bn_now.isoformat()}"

    async def _async_caller_bn(**kwargs: Any) -> Any:
        return (
            "fake-provider",
            "fake-model",
            _fake_openai_response(
                '{"seeds": [{"gist": "they have a big interview Friday", '
                '"source_message_ids": ["bn-turn-1"], "turn_id": "bn-turn-1", '
                '"salience": 0.7}]}'
            ),
        )

    plugin_llm_bn = make_plugin_llm_for_test(
        plugin_id="lifemodel",
        policy=_TrustPolicy(plugin_id="lifemodel"),
        async_caller=_async_caller_bn,
    )
    egress_bn = FakeEgress()
    runner_bn = InternalCognitionRunner(
        build_lm_bn,
        PluginLlmPort(plugin_llm_bn),
        egress_bn,
        _TARGET,
        daily_ceiling=5,
        gateway_loop=asyncio.get_running_loop(),
        apply=NoticingApply(bn_buffer),
        timeout=10.0,
    )
    runner_bn.launch(
        InternalCognitionRequest(
            instructions="notice", input_text="the segment", json_schema=NOTICING_JSON_SCHEMA
        ),
        bn_correlation_id,
        # subject_id defaults to None -- subjectless, the noticing pass's own
        # disambiguator against ThoughtProcessingApply (core/noticing.py).
    )
    for task in list(runner_bn._tasks):
        await task
    bn_thought = read_thought(
        build_lm_bn().state, seed_thought_id("they have a big interview Friday")
    )
    result["bn_thought_created"] = bn_thought is not None
    result["bn_thought_has_source"] = bn_thought is not None and (
        bn_thought.provenance is not None
        and bn_thought.provenance.source_object_ids == ("bn-turn-1",)
        and bn_thought.provenance.turn_id == "bn-turn-1"
    )
    result["bn_egress_never_called"] = egress_bn.calls == []  # non-delivery is structural
    result["bn_pending_cleared"] = build_lm_bn().state.load().pending_internal_id is None
    _log(f"[BN] {result}")

    # ---- Part BNS: opus-I2 -- the REAL pre_llm_call/post_llm_call host contract ----
    # Part B-noticing above seeds the buffer directly through its own API and only
    # drives the COMPLETION half of the seam against the real host; the live
    # pre_llm/post_llm HOOK seam -- the code that rides every real turn -- was only
    # ever unit-tested with hand-passed literal kwargs (tests/test_noticing_seam.py).
    # This part drives that other half through the REAL PluginContext.register_hook +
    # PluginManager.invoke_hook dispatch (never calling our hook closures directly),
    # with the SAME kwarg names the real host actually passes: agent/turn_context.py's
    # pre_llm_call call site passes session_id/task_id/turn_id/user_message/
    # conversation_history/is_first_turn/model/platform/sender_id, and
    # agent/turn_finalizer.py's post_llm_call call site passes session_id/task_id/
    # turn_id/user_message/assistant_response/conversation_history/model/platform.
    # `_maybe_complete_buffer_entry` degrades to a SILENT no-op on an empty
    # turn_id/session_id (hooks.py) -- so a host-contract drift (a renamed or
    # missing kwarg) would make noticing silently never capture a single live turn
    # while every hand-literal-kwarg unit test kept passing. This proves the real
    # dispatch mechanism hands both hooks a non-empty, CONSISTENT session_id/turn_id
    # and that the buffer actually closes a segment from it.
    bns_dir = home / "seam-bn-seam"
    bns_dir.mkdir(parents=True, exist_ok=True)
    _commit_born(bns_dir, internal_calls_today=0, internal_calls_day="")
    build_lm_bns = _build_lm_factory(bns_dir)

    bns_buffer = NoticingBuffer()
    bns_manager = PluginManager()
    bns_ctx = PluginContext(PluginManifest(name="lifemodel"), bns_manager)
    bns_ctx.register_hook("pre_llm_call", make_felt_state_injector(build_lm_bns, buffer=bns_buffer))
    bns_ctx.register_hook("post_llm_call", make_post_llm_observer(build_lm_bns, buffer=bns_buffer))

    bns_session_id = "bns-session"
    bns_turn_id = "bns-turn-1"
    # Mirrors agent/turn_context.py's real pre_llm_call invoke_hook call site.
    bns_manager.invoke_hook(
        "pre_llm_call",
        session_id=bns_session_id,
        task_id="bns-task",
        turn_id=bns_turn_id,
        user_message="driving the real host contract",
        conversation_history=[],
        is_first_turn=True,
        model="fake-model",
        platform="test",
        sender_id="owner",
    )
    # Mirrors agent/turn_finalizer.py's real post_llm_call invoke_hook call site.
    bns_manager.invoke_hook(
        "post_llm_call",
        session_id=bns_session_id,
        task_id="bns-task",
        turn_id=bns_turn_id,
        user_message="driving the real host contract",
        assistant_response="a genuine reply, not a decline",
        conversation_history=[],
        model="fake-model",
        platform="test",
    )
    bns_segment = bns_buffer.closed_segment(bns_session_id, now=datetime.now(UTC))
    result["bns_closed_segment_has_one_entry"] = len(bns_segment) == 1
    result["bns_captured_session_id_nonempty"] = (
        bool(bns_segment) and bns_segment[0].session_id == bns_session_id
    )
    result["bns_captured_turn_id_nonempty"] = (
        bool(bns_segment) and bns_segment[0].turn_id == bns_turn_id
    )
    _log(f"[BNS] {result['bns_closed_segment_has_one_entry']=} {bns_segment=}")

    return result


#: Every result key that MUST be ``True`` for the driver to report success —
#: kept explicit (not "every bool key") so a new descriptive (non-assertion)
#: field can be added to ``result`` later without silently becoming a gate.
_REQUIRED_TRUE_KEYS = (
    "aux_task_registered",
    "ctx_llm_resolved",
    "b1_launch_within_budget_accepted",
    "b1_pending_set_synchronously",
    "b1_host_call_used_real_message_building",
    "b1_apply_saw_typed_result",
    "b1_pending_cleared_after_completion",
    "b1_note_persisted",
    "b1_egress_never_called",
    "b1_second_call_denied_over_budget",
    "b1_budget_consumed_exactly_once",
    "b2_failed_call_still_clears_pending",
    "b2_apply_saw_empty_result_on_failure",
    "b3_stale_pending_cleared_at_connect",
    "b4_task_actually_running_before_cancel",
    "b4_no_leaked_task_after_cancel_all",
    "b5_unrelated_proactive_launch_was_dispatched",
    "b5_internal_pending_still_cleared",
    "bp_single_flight_denied_concurrent",
    "bp_pending_cleared",
    "bp_subject_cleared",
    "bp_thought_resolved",
    "bp_egress_never_called",
    "bc_commitment_created",
    "bc_commitment_links_source",
    "bc_thought_resolved",
    "bc_egress_never_called",
    "bc_pending_cleared",
    "bn_thought_created",
    "bn_thought_has_source",
    "bn_egress_never_called",
    "bn_pending_cleared",
    "bns_closed_segment_has_one_entry",
    "bns_captured_session_id_nonempty",
    "bns_captured_turn_id_nonempty",
)


def main() -> int:
    import asyncio

    home = Path(os.environ["HERMES_HOME"]).resolve()
    src = os.environ["LIFEMODEL_SRC"]
    if home == (Path.home() / ".hermes").resolve():
        _log("REFUSING to run against the default ~/.hermes — set an isolated HERMES_HOME")
        return 2

    sys.path.insert(0, src)

    result = asyncio.run(main_async())
    print(json.dumps(result), flush=True)

    failed = [k for k in _REQUIRED_TRUE_KEYS if result.get(k) is not True]
    if failed:
        _log(f"FAILED assertions: {failed}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())

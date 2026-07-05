from __future__ import annotations

from typing import Any

from lifemodel.adapters.reachin import ReachInEgress
from lifemodel.domain.egress import ReachOutcome
from lifemodel.ports import ProactiveEgressPort

_TARGET = {"platform": "telegram", "chat_id": "115679831", "thread_id": None}


class _Runner:
    def __init__(self, *, busy: bool = False, complete: bool = True) -> None:
        self._running_agents = {"x"} if busy else set()
        if complete:
            self._gateway_loop = object()
            self._build_process_event_source = lambda evt: object()
            self.adapters = {"telegram": object()}
            self._running = True
            self._draining = False


def test_is_proactive_egress_port() -> None:
    egress = ReachInEgress(runner_accessor=lambda: None)
    assert isinstance(egress, ProactiveEgressPort)


def test_unavailable_when_no_runner() -> None:
    egress = ReachInEgress(runner_accessor=lambda: None)
    assert egress.reach_out(_TARGET, "hi") is ReachOutcome.UNAVAILABLE


def test_reach_out_does_not_self_skip_on_running_agents() -> None:
    # Busy ownership is the caller's gate now (the ``busy`` arg threaded into
    # ``decide_reachout`` by the service loop) — the adapter must not
    # second-guess it via the stale ``runner._running_agents`` heuristic, which
    # stays truthy while a session is merely OPEN (not mid-turn).
    calls: list[Any] = []
    egress = ReachInEgress(
        runner_accessor=lambda: _Runner(busy=True),
        inject=lambda *a, **k: calls.append((a, k)) or ReachOutcome.DELIVERED,
    )
    out = egress.reach_out(_TARGET, "hi")
    assert out is not ReachOutcome.SKIPPED_BUSY
    assert out is ReachOutcome.DELIVERED
    assert calls  # inject IS called even though _running_agents is truthy


def test_delegates_to_inject_when_idle() -> None:
    captured: dict[str, Any] = {}

    def fake_inject(runner: Any, target: Any, prompt: str, **kwargs: Any) -> ReachOutcome:
        captured.update(runner=runner, target=target, prompt=prompt, kwargs=kwargs)
        return ReachOutcome.DELIVERED

    runner = _Runner()
    egress = ReachInEgress(runner_accessor=lambda: runner, inject=fake_inject)
    assert egress.reach_out(_TARGET, "impulse-text") is ReachOutcome.DELIVERED
    assert captured["runner"] is runner
    assert captured["target"] == _TARGET
    assert captured["prompt"] == "impulse-text"
    assert captured["kwargs"].get("message_id") is None

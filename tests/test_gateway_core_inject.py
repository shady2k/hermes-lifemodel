from __future__ import annotations

from typing import Any

from lifemodel.domain.egress import ReachOutcome
from lifemodel.gateway_core import inject_proactive_turn, reachin_available

_TARGET = {"platform": "telegram", "chat_id": "115679831", "thread_id": None}


class _Source:
    def __init__(self, platform: str = "telegram", chat_id: str = "115679831", profile: str = "") -> None:
        self.platform = platform
        self.chat_id = chat_id
        self.profile = profile


class _FakeAdapter:
    """Mirrors a real platform adapter: ``handle_message`` is async, so calling it
    yields a coroutine the scheduler runs on the gateway loop (spec §3.1:
    ``handle_message`` is async, base.py:4513). A bare ``object()`` cannot stand in
    here — calling ``object().handle_message(...)`` raises ``AttributeError``
    eagerly, so the recorded-coroutine assertion below would never hold."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def handle_message(self, event: Any) -> None:
        self.events.append(event)


class _FakeRunner:
    """Duck-types only what inject_proactive_turn touches."""

    def __init__(self, *, source: Any = None, running: bool = True) -> None:
        self._gateway_loop = object()
        self._running = running
        self._draining = False
        self._running_agents: set[Any] = set()
        self._source = source if source is not None else _Source()
        self.adapters = {"telegram": _FakeAdapter()}
        self._profile_adapters: dict[str, dict[str, Any]] = {}
        self.built_with: Any = None

    def _build_process_event_source(self, evt: dict[str, Any]) -> Any:
        self.built_with = evt
        return self._source


def _make_event(text: str, source: Any, message_id: int | None) -> dict[str, Any]:
    return {"text": text, "source": source, "message_id": message_id, "internal": True}


def test_reachin_available_true_for_complete_runner() -> None:
    assert reachin_available(_FakeRunner()) is True
    assert reachin_available(None) is False

    class Partial:  # missing _gateway_loop etc.
        adapters: dict[str, Any] = {}

    assert reachin_available(Partial()) is False


def test_inject_builds_internal_event_and_schedules_on_loop() -> None:
    runner = _FakeRunner()
    scheduled: list[tuple[Any, Any]] = []

    def schedule(coro: Any, loop: Any) -> None:
        scheduled.append((coro, loop))

    outcome = inject_proactive_turn(
        runner, _TARGET, "hello", make_event=_make_event, schedule=schedule
    )

    assert outcome is ReachOutcome.DELIVERED
    assert runner.built_with == {"platform": "telegram", "chat_id": "115679831", "thread_id": None}
    assert len(scheduled) == 1
    coro, loop = scheduled[0]
    assert loop is runner._gateway_loop
    coro.close()  # the fake adapter's async handle_message yielded this coroutine


def test_inject_unavailable_when_no_source() -> None:
    runner = _FakeRunner(source=None)
    runner._source = None  # _build_process_event_source returns None -> lane unknown
    outcome = inject_proactive_turn(runner, _TARGET, "hi", make_event=_make_event, schedule=lambda c, l: None)
    assert outcome is ReachOutcome.UNAVAILABLE


def test_inject_unavailable_when_draining() -> None:
    runner = _FakeRunner()
    runner._draining = True
    outcome = inject_proactive_turn(runner, _TARGET, "hi", make_event=_make_event, schedule=lambda c, l: None)
    assert outcome is ReachOutcome.UNAVAILABLE

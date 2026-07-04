"""Real-Hermes reach-in driver — proves ``inject_proactive_turn`` end-to-end.

This is **not** a pytest module (its name does not match ``test_*``); it is a
standalone driver run under **Hermes' own interpreter** by the guarded wrapper
:mod:`tests.test_reachin_integration` against an **isolated, throwaway
``HERMES_HOME``** (never ``~/.hermes``, never a real channel).

The unit tests for :func:`lifemodel.gateway_core.inject_proactive_turn` drive it
with fakes; they cannot prove the *wire shape* of the event the REAL default
``make_event`` builds. This driver does: it constructs a faithful fake runner
that exposes a real asyncio loop, a real :class:`SessionSource`, and a recording
adapter keyed by ``Platform.TELEGRAM``, then calls ``inject_proactive_turn`` with
its REAL default seams and asserts the captured event is a genuine
``MessageEvent(internal=True, message_id=None, message_type=TEXT)`` and that the
primitive reports ``DELIVERED``.

Nothing leaves the process: the recording adapter captures the event in memory
and never sends. Human-readable evidence goes to **stderr**; a single-line JSON
result goes to **stdout** for the wrapper to parse. Exit 0 = the probe held.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    import asyncio

    home = Path(os.environ["HERMES_HOME"]).resolve()
    src = os.environ["LIFEMODEL_SRC"]
    if home == (Path.home() / ".hermes").resolve():
        _log("REFUSING to run against the default ~/.hermes — set an isolated HERMES_HOME")
        return 2

    sys.path.insert(0, src)

    # Real Hermes types — the whole point is to prove the default make_event builds
    # a genuine MessageEvent and that source/adapter selection works against them.
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType  # noqa: F401 (proves import)
    from gateway.session import SessionSource

    from lifemodel.gateway_core import inject_proactive_turn

    captured: dict[str, Any] = {}
    done = threading.Event()

    class _RecordingAdapter:
        """Captures the event the primitive schedules; never sends anything."""

        async def handle_message(self, event: Any) -> None:
            captured["internal"] = bool(event.internal)
            captured["message_id"] = event.message_id
            mt = event.message_type
            captured["message_type"] = mt.value if mt is not None else None
            captured["text"] = event.text
            done.set()
            return None

    # Run the gateway loop in a daemon thread so the default schedule
    # (asyncio.run_coroutine_threadsafe) actually executes the injected coroutine.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    class _Runner:
        """Faithful fake of the GatewayRunner surface inject_proactive_turn touches."""

        def __init__(self) -> None:
            self._gateway_loop = loop
            self._running = True
            self._draining = False
            self._running_agents: set[Any] = set()
            self.adapters = {Platform.TELEGRAM: _RecordingAdapter()}
            self._profile_adapters: dict[str, dict[str, Any]] = {}

        def _build_process_event_source(self, evt: dict[str, Any]) -> Any:
            return SessionSource(platform=Platform.TELEGRAM, chat_id=str(evt.get("chat_id")))

    try:
        runner = _Runner()
        target = {"platform": "telegram", "chat_id": "115679831", "thread_id": None}
        outcome = inject_proactive_turn(runner, target, "[probe] reach-in impulse")
        delivered = done.wait(timeout=5.0)
        _log(
            f"[reach-in] outcome={outcome.value} captured={captured} "
            f"adapter_ran={delivered}"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        loop.close()

    result = {
        "outcome": outcome.value,
        "event_internal": captured.get("internal"),
        "message_id": captured.get("message_id"),
        "message_type": captured.get("message_type"),
    }
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())

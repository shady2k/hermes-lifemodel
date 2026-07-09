"""Primary proactive-egress adapter: native reach-in turn (spec §3.1/§6).

:class:`ReachInEgress` is the being's delivery side, driven by the supervised
platform adapter's tick: it resolves the live :class:`GatewayRunner` and delegates
to :func:`~lifemodel.gateway_core.inject_proactive_turn`. Fail-closed: every path
returns a :class:`~lifemodel.domain.egress.ReachOutcome`, never raises.

This adapter does not second-guess "busy": it used to gate on
``runner._running_agents``, but that attribute stays truthy while a session is
merely OPEN (not actively mid-turn), so it silently dropped every reach-out.
Removed outright — delivery is decided once, upstream, by the wake gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..domain.egress import ReachOutcome
from ..gateway_core import inject_proactive_turn, reachin_available

RunnerAccessor = Callable[[], Any | None]
InjectFn = Callable[..., ReachOutcome]


def default_runner_accessor() -> Any | None:
    """Lazily read the live GatewayRunner (weakref singleton, run.py:2588)."""
    try:
        import gateway.run as grun

        ref = getattr(grun, "_gateway_runner_ref", None)
        return ref() if callable(ref) else None
    except Exception:  # noqa: BLE001 - not in a gateway process / import failure
        return None


class ReachInEgress:
    """Deliver a proactive turn by injecting an internal user turn in the live session."""

    def __init__(
        self,
        *,
        runner_accessor: RunnerAccessor,
        inject: InjectFn = inject_proactive_turn,
    ) -> None:
        self._runner_accessor = runner_accessor
        self._inject = inject

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        runner = self._runner_accessor()
        if not reachin_available(runner):
            return ReachOutcome.UNAVAILABLE
        return self._inject(runner, target, impulse, message_id=None)

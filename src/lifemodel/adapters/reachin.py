"""Primary proactive-egress adapter: native reach-in turn (spec §3.1/§6).

:class:`ReachInEgress` is the in-process brain's delivery side: it resolves the
live :class:`GatewayRunner` and delegates to
:func:`~lifemodel.gateway_core.inject_proactive_turn`. Fail-closed: every path
returns a :class:`~lifemodel.domain.egress.ReachOutcome`, never raises.

**Busy ownership is centralized in the wake gate, not here** (task 7, HLA/spec
RC2): the caller (``proactive_service_loop``) computes ``busy`` once from the
accurate runner state and threads it into ``decide_reachout(..., busy=...)`` —
the no-wake-in-flight gate. This adapter used to *also* second-guess busy via
``runner._running_agents``, but that attribute stays truthy while a session is
merely OPEN (not actively mid-turn), so it silently dropped every reach-out.
Removed outright — one accurate ``busy`` source, decided once, upstream of
delivery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..domain.egress import ReachOutcome
from ..gateway_core import inject_proactive_turn, reachin_available
from ..logging import EventLogger, get_logger

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
        logger: EventLogger | None = None,
    ) -> None:
        self._runner_accessor = runner_accessor
        self._inject = inject
        self._log = logger or get_logger("lifemodel.reachin")

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        runner = self._runner_accessor()
        if not reachin_available(runner):
            return ReachOutcome.UNAVAILABLE
        return self._inject(runner, target, impulse, message_id=None, logger=self._log)

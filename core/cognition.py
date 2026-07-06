"""Cognition — decides WHEN to wake the being's native turn and HOW to frame it
(spec §13, model A).

Cognition does not call an LLM: it emits a ``LaunchProactive`` intent carrying a
desire-framed wake-packet, and the being's own Hermes turn is the act-gate
(message = FULFILL, ``[SILENT]`` = REJECT — fed back by the ``post_llm`` hook in
Phase E). It launches only for a live, un-acted desire, and only if the proactive
turn's energy is affordable — otherwise it holds (emergent shutoff, spec §8).
"""

from __future__ import annotations

from collections.abc import Sequence

from .component import TickContext
from .energy import cost_real, reserve
from .intents import Intent, LaunchProactive, UpdateState
from .wake_packet import build_wake_packet


class Cognition:
    """The cognition layer: launch a proactive turn for a live desire, gated by
    energy. Idempotent via ``pending_proactive_id``."""

    def __init__(
        self, *, fast_cost: float, send_cost: float, alpha: float, id: str = "cognition"
    ) -> None:
        self.id = id
        self._fast_cost = fast_cost
        self._send_cost = send_cost
        self._alpha = alpha

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        if state.desire_status != "active" or state.pending_proactive_id is not None:
            return []

        estimate = cost_real(self._fast_cost + self._send_cost, state.fatigue, alpha=self._alpha)
        reserved = reserve(state.energy, estimate)
        if reserved is None:
            return []  # can't afford a proactive turn -> hold (emergent shutoff)
        energy_after, _reservation = reserved

        correlation_id = f"proactive-{ctx.now.isoformat()}"
        packet = build_wake_packet(value=state.u, theta=1.0, correlation_id=correlation_id)
        return [
            LaunchProactive(
                prompt=packet.prompt, correlation_id=correlation_id, reserved_energy=estimate
            ),
            UpdateState(
                {
                    "energy": energy_after,
                    "pending_proactive_id": correlation_id,
                    "pending_proactive_since": ctx.now.isoformat(),
                }
            ),
        ]

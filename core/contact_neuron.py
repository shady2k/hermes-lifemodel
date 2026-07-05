"""ContactNeuron — the AUTONOMIC contact sensor (spec §2.1, §11).

A dumb sensor: it measures contact deprivation by accumulating the certified
drive ``u`` over elapsed silence (``sim.drive.Drive.rise``) and resets it on a
real exchange (``Drive.satiate`` with ``sim.quality.quality_of``). It emits the
raw ``{value, delta}`` as a transient ``contact`` signal and writes only ``u``.
It computes no salience, thresholds, or gates — those are AGGREGATION's job
(the lower layer is never smarter than the layer above it).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.drive import Drive
from ..sim.quality import quality_of
from .component import TickContext
from .intents import EmitSignal, Intent, UpdateState
from .taxonomy import KIND_EXCHANGE, contact_signal, is_kind, read_exchange


class ContactNeuron:
    """The v1 first neuron: contact-deprivation sensor. Sole writer of ``u``."""

    def __init__(self, *, alpha: float, beta: float, u_max: float, id: str = "contact") -> None:
        self.id = id
        self._alpha = alpha
        self._beta = beta
        self._u_max = u_max

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        from .timeutil import minutes_between

        dt = minutes_between(ctx.state.last_tick_at, ctx.now)
        drive = Drive(alpha=self._alpha, beta=self._beta, u_max=self._u_max, u=ctx.state.u)
        if dt > 0:
            drive.rise(dt=dt)
        for signal in ctx.signals:
            if is_kind(signal, KIND_EXCHANGE):
                actor, label = read_exchange(signal)
                drive.satiate(q=quality_of(actor=actor, label=label))

        delta = drive.u - ctx.state.u
        emit = contact_signal(
            origin_id=f"contact-{ctx.now.isoformat()}",
            value=drive.u,
            delta=delta,
            timestamp=ctx.now.isoformat(),
        )
        return [UpdateState({"u": drive.u}), EmitSignal(emit)]

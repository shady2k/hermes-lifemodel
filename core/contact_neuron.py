"""PresenceNeuron — the INSTANTANEOUS contact-channel sensor (spec §3/§8, T2 split).

A dumb, stateless receptor: it measures the contact channel RIGHT NOW — elapsed
silence (``dt``) plus the quality of each exchange this tick
(:func:`lifemodel.sim.quality.quality_of`) — and emits that raw, unintegrated
reading as a transient ``contact_presence`` signal. It does NOT accumulate, does
NOT own or write ``u``, holds no durable state. Integrating the drive deficit into
``u`` is the NEXT component's job (:class:`lifemodel.core.solitude_drive.SolitudeDrive`)
— the receptor senses, the center integrates (osmoreceptor vs thirst, spec §3).

This is the T2 split of the old :class:`ContactNeuron`, which mixed four jobs
(clock reading, drive integration, exchange sensing + quality, ``u`` ownership).
Sensing is now here; integration + ``u`` ownership moved to the AUTONOMIC drive.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.quality import quality_of
from .component import TickContext
from .intents import EmitSignal, Intent
from .taxonomy import KIND_EXCHANGE, contact_presence_signal, is_kind, read_exchange


class PresenceNeuron:
    """The instantaneous contact-channel sensor. Stateless; emits a raw reading."""

    def __init__(self, *, id: str = "contact") -> None:
        self.id = id

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        from .timeutil import minutes_between

        # A pure measurement of the channel NOW: how long silent + each exchange's
        # contact quality. Nothing is accumulated, no state written, no ``u`` touched.
        dt = minutes_between(ctx.state.last_tick_at, ctx.now)
        qualities: list[float] = []
        for signal in ctx.signals:
            if is_kind(signal, KIND_EXCHANGE):
                actor, label = read_exchange(signal)
                # The being's own proactive impulse carries q=0 (quality_of) and so
                # reads as "no real contact" — satiation only happens on a genuine
                # exchange, decided downstream by the drive.
                qualities.append(quality_of(actor=actor, label=label))
        emit = contact_presence_signal(
            origin_id=f"presence-{ctx.now.isoformat()}",
            dt=dt,
            qualities=qualities,
            timestamp=ctx.now.isoformat(),
        )
        return [EmitSignal(emit)]

"""ContactSensor — the INSTANTANEOUS contact-channel sensor (spec §4, afferent).

A dumb, stateless afferent transducer: it measures the contact channel RIGHT NOW —
elapsed silence (``dt``) plus the quality of each contact observed this frame
(:func:`lifemodel.sim.quality.quality_of`) — and emits that raw, unintegrated
reading as a transient ``contact_presence`` signal. It does NOT accumulate, does
NOT own or write ``u``, holds no durable state. Integrating the drive deficit into
``u`` is the NEXT component's job (:class:`lifemodel.core.solitude_drive.SolitudeDrive`)
— the sensor senses, the drive integrates (osmoreceptor vs thirst, spec §4).

Band-pass at the sensor (spec §4): control commands never reach here as
``contact_observed`` — they are filtered UPSTREAM at the hook boundary
(:mod:`lifemodel.hooks`) before a frame is ever started, "as the ear does not hear
ultrasound". The sensor only ever sees genuine contact readings.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.quality import quality_of
from .component import TickContext
from .intents import EmitSignal, Intent
from .taxonomy import (
    KIND_CONTACT_OBSERVED,
    contact_presence_signal,
    is_kind,
    read_contact_observed,
)

#: The historical ``contact`` slot id. Named so the coreloop can identify the
#: load-bearing consumer of ``contact_observed`` when gating the idempotency-ring
#: record on its success (spec §8): an inbound is only durably remembered once THIS
#: component has processed it, so a fault does not lose the retry.
CONTACT_SENSOR_ID = "contact"


class ContactSensor:
    """The instantaneous contact-channel sensor. Stateless; emits a raw reading."""

    def __init__(self, *, id: str = CONTACT_SENSOR_ID) -> None:
        self.id = id

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        from .timeutil import minutes_between

        # A pure measurement of the channel NOW: how long silent + each observed
        # contact's quality. Nothing is accumulated, no state written, no ``u`` touched.
        dt = minutes_between(ctx.state.last_tick_at, ctx.now)
        qualities: list[float] = []
        for signal in ctx.signals:
            if is_kind(signal, KIND_CONTACT_OBSERVED):
                actor, label = read_contact_observed(signal)
                # The being's own proactive impulse carries q=0 (quality_of) and so
                # reads as "no real contact" — satiation only happens on a genuine
                # contact, decided downstream by the drive.
                qualities.append(quality_of(actor=actor, label=label))
        emit = contact_presence_signal(
            origin_id=f"presence-{ctx.now.isoformat()}",
            dt=dt,
            qualities=qualities,
            timestamp=ctx.now.isoformat(),
        )
        return [EmitSignal(emit)]

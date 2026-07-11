"""SolitudeDrive — the AUTONOMIC contact-deficit integrator (spec §3/§8, T2 split).

Owns and writes the vital ``u`` (``runtime_state``): it consumes the fresh
``contact_presence`` reading from :class:`ContactSensor` (elapsed silence ``dt`` +
this tick's exchange qualities) and integrates the CERTIFIED drive
(:class:`lifemodel.sim.drive.Drive`) — ``rise`` over genuine silence, ``satiate``
ONLY on a real exchange — then emits the fresh ``u`` as the transient
``contact_pressure`` signal :class:`~lifemodel.core.aggregation.ContactAggregation`
reads (T3: the drive-output kind, replacing the legacy ``contact`` kind).

The snapshot-per-tick seam (spec §4): ``UpdateState({"u":…})`` is only visible AFTER
the tick's atomic commit, so aggregation — which sees start-of-tick ``ctx.state`` —
must read the fresh ``u`` from this transient signal, not from ``ctx.state.u``.

Satiation is ONLY on a real (positive-quality) exchange — the being's own proactive
impulse carries ``q = 0`` (computed upstream by ContactSensor via ``quality_of``)
and never self-satiates. The drive math is the certified ``sim`` model, unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..sim.drive import Drive
from .component import TickContext
from .intents import EmitSignal, Intent, UpdateState
from .metrics import MetricSpec
from .taxonomy import contact_pressure_signal, read_contact_presence
from .timeutil import to_iso

#: The being's current contact-solitude drive level ``u`` — the first live domain
#: metric emitted through ``ctx.observe`` (telemetry-core §4.3). This is genuine
#: component knowledge (the integrated deficit only the drive owns), NOT something
#: the harness can snap from outside. Declared in the drive's ``metric_surface``
#: (composition root) so the registry knows it and the surface check admits it.
CONTACT_DRIVE_U = "lifemodel_contact_drive_u"
CONTACT_DRIVE_U_SPEC = MetricSpec(
    name=CONTACT_DRIVE_U,
    kind="gauge",
    unit="",
    help="The being's current contact-solitude drive level u (SolitudeDrive).",
)


class SolitudeDrive:
    """The AUTONOMIC integrator that owns and writes the vital ``u``."""

    def __init__(
        self, *, alpha: float, beta: float, u_max: float, id: str = "solitude-drive"
    ) -> None:
        self.id = id
        self._alpha = alpha
        self._beta = beta
        self._u_max = u_max

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        # Read the instantaneous channel measurement the sensor emitted THIS tick
        # (ContactSensor runs before the drive). Absent (first wiring / corrupt) →
        # no rise, no satiate: the drive holds its value rather than guessing.
        presence = read_contact_presence(ctx.signals)
        dt = presence.dt if presence is not None else 0.0
        qualities = presence.qualities if presence is not None else ()

        # Integrate the certified drive from the start-of-tick u: rise over silence,
        # then satiate per exchange quality IN ORDER (per-step clamp at 0 — the
        # certified math, preserved exactly from the pre-split ContactNeuron).
        drive = Drive(alpha=self._alpha, beta=self._beta, u_max=self._u_max, u=ctx.state.u)
        if dt > 0:
            drive.rise(dt=dt)
        for q in qualities:
            drive.satiate(q=q)

        delta = drive.u - ctx.state.u
        # Publish the freshly integrated drive level as a domain metric (§4.3). Guard
        # on the channel: a bare unit-test context (no graph) carries observe=None.
        if ctx.observe is not None:
            ctx.observe.set(CONTACT_DRIVE_U, drive.u)
        emit = contact_pressure_signal(
            origin_id=f"contact-pressure-{to_iso(ctx.now)}",
            value=drive.u,
            delta=delta,
            timestamp=to_iso(ctx.now),
        )
        # Write u (visible NEXT tick via state) AND emit the fresh u this tick (so
        # aggregation sees the same-tick value, not the stale start-of-tick one).
        return [UpdateState({"u": drive.u}), EmitSignal(emit)]

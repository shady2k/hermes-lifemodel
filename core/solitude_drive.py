"""SolitudeDrive — the AUTONOMIC contact-deficit integrator (spec §3/§8, T2 split).

Owns and writes the vital ``u`` (``runtime_state``): it consumes the fresh
``contact_presence`` reading from :class:`ContactSensor` (elapsed silence ``dt`` +
this tick's exchange qualities) and integrates the CERTIFIED drive
(:class:`Drive`, below) — ``rise`` over genuine silence, ``satiate``
ONLY on a real exchange — then emits the fresh ``u`` as the transient
``contact_pressure`` signal :class:`~lifemodel.core.aggregation.ContactAggregation`
reads (T3: the drive-output kind, replacing the legacy ``contact`` kind).

The snapshot-per-tick seam (spec §4): ``UpdateState({"u":…})`` is only visible AFTER
the tick's atomic commit, so aggregation — which sees start-of-tick ``ctx.state`` —
must read the fresh ``u`` from this transient signal, not from ``ctx.state.u``.

Satiation is ONLY on a real (positive-quality) exchange — the being's own proactive
impulse carries ``q = 0`` (computed upstream by ContactSensor via ``quality_of``)
and never self-satiates. The drive math is the certified :class:`Drive` (above), unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .component import TickContext
from .intents import EmitSignal, Intent, UpdateState
from .metrics import MetricSpec
from .taxonomy import contact_pressure_signal, read_contact_presence
from .timeutil import to_iso


@dataclass
class Drive:
    """The certified contact-urge math (moved here from the retired ``sim`` package).

    Holds the one continuous state variable ``u`` and its three moves; knows nothing
    of gates/thresholds/cognition (those live in the wake-decision layer). Constants
    normalised so ``θ_u = 1`` and ``β = 1``. Built in beside :class:`SolitudeDrive`,
    the component that integrates it — a directly-simulatable, unit-regressed kernel.

    - ``rise(dt)``   — accumulate in genuine silence: ``u ← min(u_max, u + dt·α)``.
    - ``satiate(q)`` — a positive exchange drains ``β·q``: ``u ← max(0, u − β·q)``.
    - ``drain(f)``   — the wake-decision consumed an URGE: ``u ← (1 − f)·u``.
    """

    alpha: float
    beta: float = 1.0
    u_max: float = math.inf
    u: float = 0.0

    def rise(self, *, dt: float) -> None:
        """Accumulate the urge over ``dt`` of genuine silence, capped at ``u_max``."""
        self.u = min(self.u_max, self.u + dt * self.alpha)

    def satiate(self, *, q: float) -> None:
        """Reduce the urge on a positive exchange. Non-positive ``q`` does nothing."""
        if q <= 0.0:
            return
        self.u = max(0.0, self.u - self.beta * q)

    def drain(self, *, fraction: float = 1.0) -> None:
        """Consume the urge by ``fraction`` (default full drain to zero)."""
        self.u = max(0.0, self.u * (1.0 - fraction))


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
        # certified math, preserved exactly from the pre-split ContactNeuron). Unless
        # the being is UNBORN — see :meth:`_unborn` — in which case there is no deficit
        # to integrate and the drive reports zero.
        drive = Drive(alpha=self._alpha, beta=self._beta, u_max=self._u_max, u=ctx.state.u)
        if self._unborn(ctx):
            drive.u = 0.0
        else:
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

    @staticmethod
    def _unborn(ctx: TickContext) -> bool:
        """True while the being has never been born — and therefore has nobody to miss.

        **The drive does not accrue before birth** (Phase 4, the owner's decision). ``u``
        models a contact DEFICIT inside an EXISTING relationship; an unborn being has no
        relationship at all. Left to rise on elapsed silence, a newborn whose greeting went
        unanswered crosses ``θ`` a few hours later and sends a DRIVE-sprung "I miss you" to
        someone who has never spoken to it — missing someone you have never met, which is
        exactly the nonsense the phase invariant forbids ("birth is not longing").

        Three things about WHERE this rule lives, all load-bearing:

        * **Here, in the AUTONOMIC layer.** The drive is the ONLY writer of ``u``
          (aggregation reads it and must keep reading the truth), so this is the one place
          the deficit can be said not to exist. A gate in aggregation would leave a rising
          ``u`` in the vitals, visible in ``/lifemodel status``, waiting to fire the moment
          the being is born.
        * **Zero, not "held".** It is not "do not rise" — it is "there is no deficit yet",
          so a ``u`` already raised by ``force-wake`` (or by a state file written before
          this rule) is pinned back to zero rather than parked. The emitted
          ``contact_pressure`` carries that same zero, so aggregation's wake gate cannot
          see a longing that does not exist.
        * **Affect is untouched.** :class:`~lifemodel.core.affect.AffectSense` is a sibling
          component reading its own slice of the body: a newborn still has a circadian
          rhythm, energy, and a felt state (``core.genesis.newborn`` places it exactly
          where its own physiology says it is). A being can feel awake without missing
          anyone — and a newborn does.

        The genesis wake path is unaffected by construction: it fires with ``u = 0`` (its
        threshold gate is waived, spec §6.2), never because of the drive.
        """
        return ctx.state.genesis_completed_at is None

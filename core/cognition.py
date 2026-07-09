"""CognitionLauncher — the 0-LLM launcher that wakes the being's native turn (spec §3/§13).

Honest about what it is: a LAUNCHER, not a thinker. It never calls an LLM. For a
live, un-acted desire it reserves the proactive turn's energy, builds a
desire-framed wake-packet, and emits a ``LaunchProactive`` intent plus the typed
:class:`~lifemodel.domain.objects.Intention` (the Bratman/Rubicon decision record).
The being's own ASYNC Hermes turn is the real act-gate: a real message = FULFILL,
``[SILENT]`` = REJECT — fed back as a verdict signal by the ``post_llm`` hook and
resolved by aggregation next tick (spec §4.5). There is no synchronous LLM in core,
and none appears here.

T4 re-cut (spec §8): the hidden suppressors are gone. The launch jitter (a
deterministic ~20% HOLD seeded off the correlation id) is removed — it was an
invisible gate with no observability. The receptivity re-check is removed
(receptivity was cut in T3; appropriateness is the async act-gate's job, not this
launcher's). The launch gate is now exactly: a live ACTIVE desire + no turn in
flight + affordable energy. A HOLD is no longer the absence of a record: it emits a
suppression span (spec §5) — ``pending_proactive`` when a turn is already in flight,
``energy_unaffordable`` on emergent shutoff.

lm-27n.4: the intention is born directly ``active`` (snapshot-visible next tick), an
upsert on the singleton ``contact:owner``. Creation provenance is IMMUTABLE per
episode: on a delivery-fail retry the launcher re-emits ``PutRecord(intention
active)`` while the intention is still live → it PRESERVES the birth provenance
rather than re-stamping it with the retry tick's trace.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.memory import PutOp
from ..domain.objects import (
    CONTACT_DESIRE_ID,
    DesireState,
    IntentionState,
    qualified_id,
)
from .component import TickContext
from .desire_view import live_contact_desire
from .energy import cost_real, reserve
from .intention_view import (
    build_contact_intention,
    encode_contact_intention,
    live_contact_intention,
)
from .intents import Intent, LaunchProactive, PutRecord, UpdateState
from .suppression import SuppressionReason, emit_suppression_span
from .trace import creation_provenance
from .wake_packet import build_wake_packet


class CognitionLauncher:
    """The 0-LLM launcher: wake a proactive turn for a live desire, gated by energy.

    Idempotent via ``pending_proactive_id``. Emits a suppression span on each HOLD
    (a turn in flight / energy unaffordable) so a held launch is a logged decision,
    not silence.
    """

    def __init__(
        self,
        *,
        fast_cost: float,
        send_cost: float,
        alpha: float,
        id: str = "cognition-launcher",
    ) -> None:
        self.id = id
        self._fast_cost = fast_cost
        self._send_cost = send_cost
        self._alpha = alpha

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        state = ctx.state
        desire = live_contact_desire(ctx.objects)
        # Nothing to launch — no live ACTIVE desire. (Not this launcher's suppression:
        # aggregation already recorded why no desire was born / why it is not active.)
        if desire is None or desire.state != DesireState.ACTIVE:
            return []
        # A turn is already in flight — idempotent hold (never double-launch).
        if state.pending_proactive_id is not None:
            self._emit_suppression(ctx, SuppressionReason.PENDING_PROACTIVE)
            return []
        # Emergent shutoff: can't afford the proactive turn → hold the desire.
        estimate = cost_real(self._fast_cost + self._send_cost, state.fatigue, alpha=self._alpha)
        reserved = reserve(state.energy, estimate)
        if reserved is None:
            self._emit_suppression(ctx, SuppressionReason.ENERGY_UNAFFORDABLE)
            return []
        energy_after, _reservation = reserved

        correlation_id = f"proactive-{ctx.now.isoformat()}"
        # T6: thoughts are no longer born in the tick (the thought machinery moved
        # to Phase 6 in T7), so the launcher passes no thoughts — build_wake_packet
        # renders no "Recent Thoughts" block when none are supplied (empty-safe).
        # The wake packet is the fixed owner-approved felt impulse: it carries no
        # situational/procedural brief (the [SILENT]-regression cure), so no
        # last-exchange/decline/energy context is threaded into the prompt.
        packet = build_wake_packet(
            value=state.u,
            theta=1.0,
            correlation_id=correlation_id,
        )
        # Creation provenance is IMMUTABLE per episode (lm-27n.11). This PutRecord is
        # an upsert on the singleton intention: on a delivery-fail RETRY it re-emits
        # ``PutRecord(intention active)`` while the intention is STILL LIVE in
        # ctx.objects → PRESERVE its birth provenance. Only a FIRST crystallize (no
        # live intention) stamps a fresh trace. ``source_object_ids`` is the ONE new
        # causal stamp — the Intention→Desire edge the domain has no typed field for.
        existing_intention = live_contact_intention(ctx.objects)
        provenance = (
            existing_intention.provenance
            if existing_intention is not None
            else creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="cognition",
                reason="crystallized contact intention",
                source_object_ids=(qualified_id("desire", CONTACT_DESIRE_ID),),
            )
        )
        # 0-LLM crystallization: record the committed decision (Bratman act-gate),
        # born directly ``active`` so it gates + is snapshot-visible next tick.
        intention = build_contact_intention(
            state=IntentionState.ACTIVE,
            commitment_strength=desire.salience,
            salience=desire.salience,
            source_drive=desire.source_drive,
            provenance=provenance,
        )
        return [
            PutRecord(op=PutOp(draft=encode_contact_intention(intention))),
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

    def _emit_suppression(self, ctx: TickContext, reason: SuppressionReason) -> None:
        """Log a HOLD as a suppression span (spec §5) — only when a logger + span are
        wired (the live tick); a bare unit-test ``TickContext`` skips it."""
        if ctx.logger is None or ctx.trace is None:
            return
        emit_suppression_span(
            logger=ctx.logger,
            reason=reason,
            component=self.id,
            span=ctx.trace,
            tick=ctx.state.tick_count + 1,
        )

"""Cognition — decides WHEN to wake the being's native turn and HOW to frame it
(spec §13, model A).

Cognition does not call an LLM: it emits a ``LaunchProactive`` intent carrying a
desire-framed wake-packet, and the being's own Hermes turn is the act-gate
(message = FULFILL, ``[SILENT]`` = REJECT — fed back by the ``post_llm`` hook in
Phase E). It launches only for a live, un-acted desire, and only if the proactive
turn's energy is affordable — otherwise it holds (emergent shutoff, spec §8).

lm-27n.4 inserts a typed :class:`~lifemodel.domain.objects.Intention` (the
Bratman/Rubicon decision record) as the gate-owner between the active desire and
the launch. Crystallization is **0-LLM** and **behavior-neutral on send timing**:
the launch gate is unchanged (live active desire + no turn in flight + affordable),
and *whenever* it fires cognition also emits ``PutRecord(intention active)`` — an
upsert on the singleton ``contact:owner`` intention, born directly ``active`` so
it is visible in the next tick's snapshot. The Rubicon fields are computed
deterministically and recorded for auditability; they do NOT change *when* the
being sends (that stays this gate + aggregation's upstream gates).
"""

from __future__ import annotations

import hashlib
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
from .receptivity import appraise_receptivity
from .relationship_view import DEFAULT_RELATIONSHIP, live_owner_relationship
from .thought_view import live_thoughts
from .trace import creation_provenance
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
        desire = live_contact_desire(ctx.objects)
        if (
            desire is None
            or desire.state != DesireState.ACTIVE
            or state.pending_proactive_id is not None
        ):
            return []

        # Receptivity re-check (lm-27n.5): the relationship/state may have changed
        # since the desire was born (e.g. quiet hours just started). Re-appraise
        # with the SAME pure fn aggregation used — if an explicit boundary now
        # hard-vetoes, HOLD (return []): do not launch this tick; the live desire
        # persists for a later, admissible tick. With the permissive default this
        # is always ``allowed`` → behaviour-identical to .4.
        relationship = live_owner_relationship(ctx.objects) or DEFAULT_RELATIONSHIP
        appraisal = appraise_receptivity(relationship, state, ctx.now)
        if not appraisal.allowed:
            return []

        estimate = cost_real(self._fast_cost + self._send_cost, state.fatigue, alpha=self._alpha)
        reserved = reserve(state.energy, estimate)
        if reserved is None:
            return []  # can't afford a proactive turn -> hold (emergent shutoff)
        energy_after, _reservation = reserved

        correlation_id = f"proactive-{ctx.now.isoformat()}"
        # Launch jitter (lm-8o3, design point 6): a small, deterministic fraction of
        # ticks HOLD here — one more beat of human unpredictability, not a timer. It
        # is seeded off ``correlation_id`` (== ctx.now.isoformat()) via sha256, so it
        # is fully reproducible and testable, NEVER a random-module/wall-clock read.
        # Placed strictly AFTER the receptivity re-check and the energy reservation
        # above: jitter only ever delays an otherwise-permitted launch, it never
        # overrides a respect or energy gate. On hold, nothing is emitted — the
        # desire is NOT resolved, so the next admissible tick launches normally.
        jitter_seed = hashlib.sha256(correlation_id.encode()).digest()[0]
        if jitter_seed % 5 == 0:  # ~20% of correlation-ids deferred by one tick
            return []
        # Render the live thoughts (active/parked, most-salient first) as the
        # first-person "Recent Thoughts" CONTEXT block. No thoughts → no block →
        # the prompt is byte-identical to before (behavior-neutral, lm-27n.6).
        packet = build_wake_packet(
            value=state.u,
            theta=1.0,
            correlation_id=correlation_id,
            thoughts=live_thoughts(ctx.objects),
            last_exchange_at=state.last_exchange_at,
            now=ctx.now,
            decline_count=state.decline_count,
            energy=state.energy,
        )
        # Creation provenance is IMMUTABLE per episode (lm-27n.11). This PutRecord is
        # an upsert on the singleton intention: on a delivery-fail RETRY it re-emits
        # ``PutRecord(intention active)`` while the intention is STILL LIVE in
        # ctx.objects → PRESERVE its birth provenance (do NOT rewrite the birth trace
        # with this retry tick's). Only a FIRST crystallize (no live intention) stamps
        # a fresh trace. Decide from the snapshot, never a hidden read.
        existing_intention = live_contact_intention(ctx.objects)
        provenance = (
            existing_intention.provenance
            if existing_intention is not None
            else creation_provenance(
                ctx.trace,
                created_by=self.id,
                component="cognition",
                reason="crystallized contact intention",
                # The ONE new causal stamp (lm-27n.10): the Intention→Desire edge — the
                # only lineage the domain has no typed field for ("same id, different
                # kind" is too implicit for an audit reader). Stamped on the FRESH birth
                # provenance ONLY; the preserve-on-retry branch keeps its birth
                # provenance (which already carries this edge), so a retry never rewrites
                # it. The typed edges (source_thought_ids / parent_id) stay the truth —
                # they are NOT mirrored here, so no edge is authoritative twice.
                source_object_ids=(qualified_id("desire", CONTACT_DESIRE_ID),),
            )
        )
        # 0-LLM crystallization: record the committed decision (Bratman act-gate).
        # ``commitment_strength`` is the effective pressure the desire crystallized
        # on (its salience). Born directly ``active`` so it gates + is snapshot-visible
        # next tick. An upsert on the singleton — on a delivery-fail retry this simply
        # re-stamps the still-active intention, so the send timing is unchanged.
        intention = build_contact_intention(
            state=IntentionState.ACTIVE,
            commitment_strength=desire.salience,
            salience=desire.salience,
            source_drive=desire.source_drive,
            extra_constraints=appraisal.constraints,
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

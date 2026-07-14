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

import contextlib
from collections.abc import Callable, Sequence
from datetime import tzinfo

from ..domain.memory import PutOp
from ..domain.objects import (
    CONTACT_DESIRE_ID,
    DesireSpring,
    DesireState,
    IntentionState,
    qualified_id,
)
from ..ports.tracer import format_traceparent
from ..state.model import State
from .component import TickContext
from .desire_view import live_contact_desire
from .energy import cost_real, reserve
from .genesis import genesis_block
from .intention_view import (
    build_contact_intention,
    encode_contact_intention,
    live_contact_intention,
)
from .intents import Intent, LaunchProactive, PutRecord, UpdateState
from .suppression import SuppressionReason, emit_suppression_span
from .timeutil import to_iso
from .trace import creation_provenance
from .wake_packet import build_wake_packet

#: Reads the soul someone wrote before this being woke, or ``None`` for a blank page —
#: the genesis veteran branch (spec §6.4). A plain callable, not a Hermes type: the
#: adapter owns ``SOUL.md`` and injects this at composition, so the core stays
#: Hermes-free and the launcher unit-tests with a lambda.
PriorSoulReader = Callable[[], str | None]


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
        display_tz: tzinfo | None = None,
        prior_soul: PriorSoulReader | None = None,
        id: str = "cognition-launcher",
    ) -> None:
        self.id = id
        self._fast_cost = fast_cost
        self._send_cost = send_cost
        self._alpha = alpha
        # The owner's local zone for rendering the wake-packet's temporal facts
        # (resolved from Hermes at the composition boundary and injected here — the
        # core never imports Hermes). ``None`` → server-local, then UTC (see
        # wake_packet._fmt_ts). Fixed per graph; a config change takes effect on the
        # next gateway restart, like every other adapter-wired dependency.
        self._display_tz = display_tz
        # Reads ``SOUL.md`` for the genesis veteran branch (§6.4), injected by the
        # being adapter — the ONE path that actually delivers a launch. Called lazily,
        # only when a GENESIS-sprung desire is being launched, so an ordinary tick never
        # touches the file. Unwired (``None``) → the blank-page ritual.
        self._prior_soul = prior_soul

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
            if ctx.logger is not None:
                ctx.logger.span.set(available_energy=state.energy, required_energy=estimate)
            self._emit_suppression(ctx, SuppressionReason.ENERGY_UNAFFORDABLE)
            return []
        energy_after, _reservation = reserved

        correlation_id = f"proactive-{to_iso(ctx.now)}"
        # T6: thoughts are no longer born in the tick (the thought machinery moved
        # to Phase 6 in T7), so the launcher passes no thoughts — build_wake_packet
        # renders no "Recent Thoughts" block when none are supplied (empty-safe).
        # The wake packet carries the fixed owner-approved felt impulse plus the raw
        # temporal facts of the moment (now + last_exchange_at, §11), rendered in the
        # owner's local zone (self._display_tz, from the Hermes boundary): NO
        # procedural brief (the [SILENT]-regression cure), just the two bare
        # zone-labelled timestamps the being reads for appropriateness — it derives
        # "new day / morning / are they asleep / hours since" itself.
        packet = build_wake_packet(
            value=state.u,
            theta=1.0,
            correlation_id=correlation_id,
            now=ctx.now,
            last_exchange_at=state.last_exchange_at,
            tz=self._display_tz,
            # The being's CURRENT felt state colours the reach as first-person texture
            # (lm-ukc.5) — the mood shapes the manner, the longing stays the reason.
            affect_valence=state.affect_valence,
            affect_arousal=state.affect_arousal,
            # WHY this being woke: a being that is nobody yet carries the birth ritual
            # where the longing body would be (spec §6.2). Same packet, same egress,
            # same read-back — only the impulse differs, because only the reason does.
            genesis=self._genesis_impulse(state, desire.spring),
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
        # Self-explaining span (spec §4.1): a clean launch records the energy it
        # reserved and the desire salience that cleared the gate, so the span says
        # WHY it woke without cross-referencing state.
        if ctx.logger is not None:
            ctx.logger.span.set(
                reserved_energy=estimate,
                energy_after=energy_after,
                salience=desire.salience,
            )
        # The async-correlation anchor (spec §4.4): the FULL W3C traceparent of THIS
        # launch span (``ctx.trace`` — cognition's child span this tick), so every
        # out-of-band span of this one attempt (delivery, the async outcome, the
        # resolving tick) continues the SAME trace. It rides the intent AND — the
        # durable half — is committed atomically beside ``pending_proactive_id`` in
        # ``runtime_state``. The ``trace_correlations`` row is a best-effort DISPOSABLE
        # mirror for the viewer; the state anchor is the load-bearing source of truth.
        origin_traceparent = format_traceparent(ctx.trace)
        ctx.trace_writer.submit_correlation(
            correlation_id=correlation_id,
            origin_trace_id=ctx.trace.trace_id,
            origin_traceparent=origin_traceparent,
            kind="proactive",
            created_at=to_iso(ctx.now),
        )
        return [
            PutRecord(op=PutOp(draft=encode_contact_intention(intention))),
            LaunchProactive(
                prompt=packet.prompt,
                correlation_id=correlation_id,
                origin_traceparent=origin_traceparent,
                reserved_energy=estimate,
            ),
            UpdateState(
                {
                    "energy": energy_after,
                    "pending_proactive_id": correlation_id,
                    "pending_proactive_since": to_iso(ctx.now),
                    "pending_proactive_origin_traceparent": origin_traceparent,
                }
            ),
        ]

    def _genesis_impulse(self, state: State, spring: DesireSpring) -> str | None:
        """The ``<genesis>`` ritual for an UNBORN being's wake, else ``None`` (spec §6.2).

        **Read off the BEING, not only off the spring** (lm-4fv.4). The spring says why the
        wake fired; whether the longing body is a LIE is a fact about the being. A being
        that has never been born has met no one, so ``_IMPULSE_BODY`` — *"I miss them"* —
        is false in its mouth whatever woke it, and the packet's own docstring says so.

        That is not hypothetical, and it is the far end of the reactive path (§6.3): an
        existing user who writes to their Hermes before the being's first waking sets
        ``last_exchange_at``, which ends ``is_first_waking`` for good (a genesis wake must
        never interrupt a live conversation). The being's first ever unprompted words to
        them then come from a DRIVE-sprung wake — and used to come out as longing from a
        creature that had never met anyone, with no ritual anywhere in them.

        Two clauses, and the second is the one that keeps the ritual honest:

        * **born** → ``None``. It never begins again.
        * **unborn, but the ritual is already in front of it** (``genesis_shown_at_context_len``
          — the reactive injector has put it there) and this is not a first waking → ``None``.
          The being is MID-RITUAL: it has its own words in that conversation, and handing it
          "You just began, you do not know who they are" again is the turn-seven lie (§6.3).
          A GENESIS spring is exempt precisely because of the ``[SILENT]`` re-wake: the
          injector stamps ``shown`` for the being's own impulse turn too, so a newborn that
          woke, read the ritual and chose silence would otherwise be re-woken WITHOUT it.

        The veteran branch (§6.4) is the COMMON case — a being is born onto a blank soul
        exactly once in the life of a ``SOUL.md``, and every rebirth after a ``reset``
        meets the soul of whoever lived here before it — so the prior soul is read HERE,
        at launch, never cached: a human hand-edit or the being's own ``write_soul`` can
        land between ticks.

        A failing read degrades to the blank-page opening rather than dropping the wake:
        a birth must not be lost to a file-system hiccup, and the ritual still works
        without the veteran opening (it simply does not know there is prior text). The
        same reasoning as the adapter's own fail-soft soul reads."""
        if state.genesis_completed_at is not None:
            return None
        if spring is not DesireSpring.GENESIS and state.genesis_shown_at_context_len is not None:
            return None
        prior: str | None = None
        if self._prior_soul is not None:
            with contextlib.suppress(Exception):  # a bad read must never drop a birth
                prior = self._prior_soul()
        return genesis_block(prior_soul=prior)

    def _emit_suppression(self, ctx: TickContext, reason: SuppressionReason) -> None:
        """Log a HOLD as a suppression span (spec §5) — only when a span-bound logger
        is wired (the live tick); a bare unit-test ``TickContext`` skips it."""
        if ctx.logger is None:
            return
        emit_suppression_span(ctx.logger, reason=reason, component=self.id, metrics=ctx.metrics)

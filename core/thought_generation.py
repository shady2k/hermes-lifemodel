"""ThoughtGeneration — the 0-LLM generative stream (lm-27n.8).

The being's thoughts come *alive* here: each tick this component may mint at most
**one** new thought from three triggers — an external **event** (an exchange, the
being's own send-verdict, or a genuine contact-drive crossing), the **chaining**
development of the thought :mod:`~lifemodel.core.thought_attention` just attended,
or low-salience **idle** mind-wandering. Content is **templated and deterministic**
(:mod:`lifemodel.core.thought_templates`) — the plugin cannot call an LLM, so a
thought is a pure function of the object/event that spawned it (rich prose is a
later надстройка). Every thought is born through a single ``PutRecord`` and is
visible only NEXT tick (snapshot-per-tick).

**The anti-runaway spine (codex).** Generation is *object metabolism*, not
cognition (it never launches a turn and never mints a contact desire — a thought
becoming a desire is lm-27n.9's Rubicon gate). It is bounded so it can neither
flood nor drive contact:

* ``max_new_per_tick = 1`` — triggers are evaluated in priority order
  **event > chaining > idle** and the FIRST warranted + affordable one is minted,
  then we STOP. So no emitted thought's ``parent_id`` can be another emitted
  thought's id, and same-tick recursion is structurally impossible (it reads only
  the start-of-tick ``ctx.objects`` snapshot, never its own output).
* deterministic **idempotent ids** (:func:`~lifemodel.domain.objects.derive_id`)
  — a retried event, the same idle window, or the same parent upserts ONE row.
  We skip a trigger whose id is already live in the snapshot.
* ``MAX_LIVE_THOUGHTS`` — at the cap, generate nothing (let the .7 brake drain the
  table first).
* a small non-LLM ``THOUGHT_GEN_COST`` reserved from ``state.energy`` (fatigue
  inflates it) — unaffordable ⇒ skip with **no debit** (a tired being wanders
  less: emergent shutoff). The debit rides the same tick's ``UpdateState``.
* chaining is gated HARD (parent active, not near the .7 brake, depth-bounded, no
  live child already) and idle fires only when genuinely quiet + sparse and is
  born LOW-salience (weak anti-frivolity — a fresh idle thought alone must not
  become a strong contact reason).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ..domain.memory import MemoryRecord, PutOp
from ..domain.objects import Thought, ThoughtState, derive_id
from .component import TickContext
from .energy import cost_real, reserve
from .intents import Intent, PutRecord, UpdateState
from .taxonomy import (
    KIND_CONTACT,
    KIND_EXCHANGE,
    KIND_VERDICT,
    read_exchange,
    read_verdict,
)
from .thought_score import PARK_AFTER
from .thought_templates import (
    chain_content,
    event_drive_content,
    event_exchange_content,
    event_verdict_content,
    idle_about_desire_content,
    idle_about_thought_content,
    idle_self_check_content,
)
from .thought_view import (
    THOUGHT_KIND,
    build_thought,
    encode_thought,
    live_thoughts,
    selected_thoughts,
)
from .timeutil import minutes_between

# --- Hard caps (the anti-runaway spine, codex). ------------------------------
#: At/above this many live thoughts in the snapshot, generate NOTHING — let the
#: .7 brake (decay/park/expire) drain the table before adding more.
MAX_LIVE_THOUGHTS = 48
#: A softer live-thought ceiling above which *idle* mind-wandering stops (a busy
#: mind does not wander); events/chaining still fire up to ``MAX_LIVE_THOUGHTS``.
IDLE_SOFT_CAP = 12
#: Small, non-LLM energy price of minting one thought — reserved (fatigue-inflated)
#: from ``state.energy``; unaffordable ⇒ no generation, no debit.
THOUGHT_GEN_COST = 0.005
#: Deepest chain lineage: chaining is refused once the parent already has this
#: many ancestors, so a chain is at most ``root → child → grandchild``.
MAX_DEPTH = 2
#: A parent one step from the .7 park brake (``no_progress_count`` this high) is
#: NOT developed — generation must not fuel a loop the brake is trying to stop.
CHAIN_NEAR_BRAKE = PARK_AFTER - 1
#: A child's salience is this fraction of its parent's (a developed thought pulls
#: less than the thought it came from — chains fade, they do not amplify).
CHAIN_SALIENCE_FACTOR = 0.45

# --- Salience / appraisal bands per trigger. ---------------------------------
#: Event appraisal salience (moderate, 0.3–0.5): an external event is relevant but
#: does not by itself out-shout a live desire.
_EVENT_EXCHANGE_SALIENCE = 0.40
_EVENT_VERDICT_SALIENCE = 0.35
_EVENT_DRIVE_SALIENCE = 0.45
_EVENT_ACTIONABILITY = 0.30
_EVENT_OTHER_REGARDING = 0.40
#: Idle mind-wandering salience (LOW, 0.10–0.20) + low actionability — the weak
#: anti-frivolity teaser: a fresh idle thought must not be a strong contact reason.
_IDLE_SALIENCE = 0.15
_IDLE_ACTIONABILITY = 0.05
_IDLE_OTHER_REGARDING = 0.10

# --- Idle-quiet gate. --------------------------------------------------------
#: Minutes of silence since the last real exchange before the mind is "quiet"
#: enough to wander (a prior exchange must exist — a being with no history idles
#: not, keeping the trigger conservative).
IDLE_QUIET_MIN = 30.0
#: Coarse cooldown window (minutes): the idle id buckets ``now`` into this window,
#: so at most one idle thought is minted per window (upsert on the bucket id).
IDLE_COOLDOWN_MIN = 60.0

#: The fixed origin token for the transient contact drive (its per-tick signal
#: origin churns, so the event id uses a stable token → one live drive thought).
_DRIVE_ORIGIN = "contact"

_GEN_SOURCE = "thought-generation"


class ThoughtGeneration:
    """Mints ≤1 templated, energy-costed thought per tick (0-LLM, bounded)."""

    def __init__(
        self,
        *,
        alpha: float,
        theta: float = 1.0,
        gen_cost: float = THOUGHT_GEN_COST,
        max_live: int = MAX_LIVE_THOUGHTS,
        id: str = "thought-generation",
    ) -> None:
        self.id = id
        self._alpha = alpha
        self._theta = theta
        self._gen_cost = gen_cost
        self._max_live = max_live

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        # Cap 1: at the live-thought ceiling, generate nothing (drain first).
        live = live_thoughts(ctx.objects)
        if len(live) >= self._max_live:
            return []

        # Cap 2: affordability gate — reserve the (fatigue-inflated) cost. A tired
        # being cannot afford to wander (emergent shutoff). Unaffordable ⇒ no debit.
        estimate = cost_real(self._gen_cost, ctx.state.fatigue, alpha=self._alpha)
        reserved = reserve(ctx.state.energy, estimate)
        if reserved is None:
            return []
        energy_after, _reservation = reserved

        # The set of thought ids already live in the snapshot — the idempotency
        # guard: a trigger whose deterministic id is already present is skipped, so
        # a retry / same window / same parent never mints a second row.
        existing = {r.id for r in ctx.objects if r.kind == THOUGHT_KIND}

        # Priority order: event > chaining > idle. Mint the FIRST warranted one.
        thought = (
            self._event_thought(ctx, existing)
            or self._chain_thought(ctx, existing)
            or self._idle_thought(ctx, existing, len(live))
        )
        if thought is None:
            return []  # nothing warranted → no thought, no energy debit

        return [
            PutRecord(op=PutOp(draft=encode_thought(thought))),
            UpdateState({"energy": energy_after}),
        ]

    # --- Trigger 1: external event appraisal ---------------------------------
    def _event_thought(self, ctx: TickContext, existing: set[str]) -> Thought | None:
        """The first not-yet-minted appraisal of a real event this tick.

        Priority within events: a real exchange, then the being's own send-verdict,
        then a genuine upward crossing of the contact drive. One thought per event
        origin (deterministic id) — a re-delivered/re-appraised event upserts it."""
        for candidate in self._event_candidates(ctx):
            if candidate is not None and candidate.id not in existing:
                return candidate
        return None

    def _event_candidates(self, ctx: TickContext) -> list[Thought | None]:
        return [
            self._exchange_candidate(ctx),
            self._verdict_candidate(ctx),
            self._drive_candidate(ctx),
        ]

    def _exchange_candidate(self, ctx: TickContext) -> Thought | None:
        for sig in ctx.signals:
            if sig.kind != KIND_EXCHANGE:
                continue
            actor, label = read_exchange(sig)
            if actor == "proactive_internal":
                continue  # the being's own internal turn is not an external event
            return self._event(
                origin=("exchange", sig.origin_id),
                content=event_exchange_content(actor, label),
                salience=_EVENT_EXCHANGE_SALIENCE,
            )
        return None

    def _verdict_candidate(self, ctx: TickContext) -> Thought | None:
        for sig in ctx.signals:
            if sig.kind != KIND_VERDICT:
                continue
            return self._event(
                origin=("verdict", sig.origin_id),
                content=event_verdict_content(read_verdict(sig)),
                salience=_EVENT_VERDICT_SALIENCE,
            )
        return None

    def _drive_candidate(self, ctx: TickContext) -> Thought | None:
        """A drive event only on a GENUINE upward θ crossing this tick — not every
        tick the drive sits above θ (that would mint one thought per tick)."""
        for sig in ctx.signals:
            if sig.kind != KIND_CONTACT:
                continue
            value = _as_float(sig.payload.get("value"))
            delta = _as_float(sig.payload.get("delta"))
            if value is None or delta is None:
                continue
            crossed_up = value >= self._theta > (value - delta)
            if not crossed_up:
                continue
            return self._event(
                origin=("drive", _DRIVE_ORIGIN),
                content=event_drive_content(),
                salience=_EVENT_DRIVE_SALIENCE,
            )
        return None

    def _event(self, *, origin: tuple[str, str], content: str, salience: float) -> Thought:
        kind, origin_id = origin
        return build_thought(
            id=derive_id(THOUGHT_KIND, "event", kind, origin_id),
            content=content,
            trigger="event",
            parent_id=None,
            salience=salience,
            actionability=_EVENT_ACTIONABILITY,
            other_regarding_value=_EVENT_OTHER_REGARDING,
            source=_GEN_SOURCE,
        )

    # --- Trigger 2: chaining (develop the .7-attended thought) ---------------
    def _chain_thought(self, ctx: TickContext, existing: set[str]) -> Thought | None:
        """A CHILD developing the single thought .7 attended, gated HARD."""
        selected = selected_thoughts(ctx.objects, ctx.now, limit=1)
        if not selected:
            return None
        parent = selected[0]  # active-only (selected_thoughts filters parked out)
        child_id = derive_id(THOUGHT_KIND, "chain", parent.id)
        if (
            child_id in existing
            or parent.no_progress_count >= CHAIN_NEAR_BRAKE
            or self._depth(parent.id, ctx.objects) >= MAX_DEPTH
            or self._has_live_child(parent.id, ctx.objects)
        ):
            return None
        return build_thought(
            id=child_id,
            content=chain_content(parent.content),
            trigger=f"thought:{parent.id}",
            parent_id=parent.id,
            salience=parent.salience * CHAIN_SALIENCE_FACTOR,
            actionability=parent.actionability * CHAIN_SALIENCE_FACTOR,
            other_regarding_value=parent.other_regarding_value,
            source=_GEN_SOURCE,
        )

    def _parents(self, objects: Sequence[MemoryRecord]) -> dict[str, str | None]:
        """Map every live thought's id → its ``parent_id`` (for lineage walks)."""
        out: dict[str, str | None] = {}
        for record in objects:
            if record.kind != THOUGHT_KIND:
                continue
            parent = record.payload.get("parent_id")
            out[record.id] = parent if isinstance(parent, str) else None
        return out

    def _depth(self, thought_id: str, objects: Sequence[MemoryRecord]) -> int:
        """Ancestor count of *thought_id* via the snapshot's parent links.

        Cycle- and orphan-safe: a broken/looping ``parent_id`` chain terminates at
        ``MAX_DEPTH`` (which blocks chaining anyway) rather than spinning."""
        parents = self._parents(objects)
        depth = 0
        seen: set[str] = {thought_id}
        current = parents.get(thought_id)
        while current is not None and depth < MAX_DEPTH:
            if current in seen:
                break  # defensive: a malformed cycle cannot loop forever
            seen.add(current)
            depth += 1
            current = parents.get(current)
        return depth

    def _has_live_child(self, parent_id: str, objects: Sequence[MemoryRecord]) -> bool:
        """Does any live thought in the snapshot already develop *parent_id*?"""
        for record in objects:
            if record.kind == THOUGHT_KIND and record.payload.get("parent_id") == parent_id:
                return True
        return False

    # --- Trigger 3: idle mind-wandering --------------------------------------
    def _idle_thought(
        self, ctx: TickContext, existing: set[str], live_count: int
    ) -> Thought | None:
        """A low-salience wandering thought, only when quiet + sparse + off-cooldown."""
        state = ctx.state
        if state.pending_proactive_id is not None:
            return None  # a turn is in flight — not idle
        if live_count >= IDLE_SOFT_CAP:
            return None  # the mind is already busy — do not add wandering
        if self._exchange_this_tick(ctx):
            return None  # a real exchange just happened — not idle
        if not self._quiet_enough(state.last_exchange_at, ctx.now):
            return None  # too soon after the last exchange (or none ever)

        # Cooldown via a coarse time bucket: at most one idle thought per window.
        bucket = int(ctx.now.timestamp() // (IDLE_COOLDOWN_MIN * 60.0))
        idle_id = derive_id(THOUGHT_KIND, "idle", str(bucket))
        if idle_id in existing:
            return None  # already wandered this window (idempotent)

        return build_thought(
            id=idle_id,
            content=self._idle_content(ctx),
            trigger="idle",
            parent_id=None,
            salience=_IDLE_SALIENCE,
            actionability=_IDLE_ACTIONABILITY,
            other_regarding_value=_IDLE_OTHER_REGARDING,
            source=_GEN_SOURCE,
        )

    def _idle_content(self, ctx: TickContext) -> str:
        """Wander about the most salient live thing, else a generic self-check."""
        thoughts = live_thoughts(ctx.objects)
        active = [t for t in thoughts if t.state == ThoughtState.ACTIVE.value]
        if active:
            return idle_about_thought_content(active[0].content)
        if self._has_live_desire(ctx.objects):
            return idle_about_desire_content()
        return idle_self_check_content()

    def _has_live_desire(self, objects: Sequence[MemoryRecord]) -> bool:
        return any(record.kind == "desire" for record in objects)

    @staticmethod
    def _exchange_this_tick(ctx: TickContext) -> bool:
        for sig in ctx.signals:
            if sig.kind != KIND_EXCHANGE:
                continue
            actor, _label = read_exchange(sig)
            if actor != "proactive_internal":
                return True
        return False

    @staticmethod
    def _quiet_enough(last_exchange_at: str | None, now: datetime) -> bool:
        if last_exchange_at is None:
            return False  # no exchange history → conservatively not "quiet"
        return minutes_between(last_exchange_at, now) >= IDLE_QUIET_MIN


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)

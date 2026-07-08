"""ThoughtCrystallization — the TOP-DOWN desire spring, the Rubicon gate (lm-27n.9).

The capstone of the thought engine: a deliberated :class:`~lifemodel.domain.objects.Thought`
crossing the **Rubicon** (Heckhausen/Gollwitzer — the move from *deliberating* a
motive to *committing* to act on it) becomes the being's singleton contact
:class:`~lifemodel.domain.objects.Desire`. Contact then springs from a GENUINE
reason — a surviving, other-serving, actionable thought — not a timer.

**A PURE PROPOSER — writes NOTHING (the one-writer crux, codex).** The contact
desire has ONE writer (:class:`~lifemodel.core.aggregation.ContactAggregation`)
and a thought has ONE writer (:class:`~lifemodel.core.thought_attention.ThoughtAttention`);
a snapshot-per-tick engine cannot have two components mutate the same object in a
tick. So this component reads the start-of-tick snapshot, applies the deterministic
:func:`should_crystallize` gate, and — when it fires — emits a single transient
:func:`~lifemodel.core.taxonomy.thought_contact_proposal_signal` (an
:class:`~lifemodel.core.intents.EmitSignal`). It emits **no** ``PutRecord``/
``TransitionRecord``. Aggregation (which runs NEXT, same tick — the pipeline is
reordered so crystallization precedes it) folds the proposal into the desire;
attention resolves the source thought on the same signal. This component is the
proposer; the two writers commit.

**Anti-frivolity (0-LLM, deterministic).** A fresh idle thought can NEVER mint
contact: it fails the salience bar AND the persistence bar (``sustained_attention_count``,
bumped only by attention on a *viable* candidate), and idle NEVER bypasses
persistence. Crystallization takes either K ticks of sustained viable attention OR
a strong external event. See :func:`should_crystallize`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.objects import Thought
from .component import TickContext
from .desire_view import live_contact_desire
from .intents import EmitSignal, Intent
from .receptivity import appraise_receptivity
from .relationship_view import DEFAULT_RELATIONSHIP, live_owner_relationship
from .taxonomy import thought_contact_proposal_signal
from .thought_score import attention_score
from .thought_view import selected_thoughts

# --- The Rubicon gate calibration (0-LLM; codex). ----------------------------
#: Personal-relevance floor: a thought under this salience is mind-wandering, not a
#: reason to reach out (the first anti-frivolity bar — an idle thought is ~0.15).
CRYSTALLIZE_SALIENCE = 0.55
#: Other-regarding floor: at/above this the thought serves the OWNER, a legitimate
#: reason to make contact (the primary, non-self-centred spring).
OTHER_REGARDING_MIN = 0.55
#: Actionability floor: at/above this the thought is concrete enough to act on now.
ACTIONABILITY_MIN = 0.65
#: Persistence floor: ticks of *sustained viable attention* before an ordinary
#: thought may cross — proves it survived deliberation, not a one-tick flash.
PERSIST_MIN = 2

#: Strong-event bypass: a genuinely urgent EXTERNAL event may cross without waiting
#: for persistence (deliberation already happened out there). Idle never qualifies.
STRONG_EVENT_SALIENCE = 0.75
STRONG_EVENT_OTHER_REGARDING = 0.70
STRONG_EVENT_ACTIONABILITY = 0.75

#: Own-longing (legitimate but tightly gated): the being's OWN accumulated pull is a
#: real reason, but — to keep it from being fake altruism's backdoor — it needs a
#: high-salience drive/event thought, STRONGER persistence than the other-serving
#: path, and low recent send pressure (no ActionPending inhibition).
STRONG_LONGING_SALIENCE = 0.75
STRONG_LONGING_PERSIST = PERSIST_MIN + 1


def strong_event_trigger(thought: Thought) -> bool:
    """A strong EXTERNAL event that may cross the Rubicon without waiting for
    persistence — an ``event``-triggered, high-salience, other-serving OR
    high-actionability thought. Idle mind-wandering can never qualify (its trigger
    is ``idle``, not ``event``), so it never bypasses the persistence bar."""
    return (
        thought.trigger.startswith("event")
        and thought.salience >= STRONG_EVENT_SALIENCE
        and (
            thought.other_regarding_value >= STRONG_EVENT_OTHER_REGARDING
            or thought.actionability >= STRONG_EVENT_ACTIONABILITY
        )
    )


def strong_own_longing(thought: Thought, *, action_pending: bool) -> bool:
    """The gated own-longing reason: the being's OWN drive/event pull is legitimate,
    but crosses only with high salience, STRONGER persistence than the other-serving
    path, and no recent send pressure (``action_pending`` False). This keeps a
    self-centred urge from masquerading as an other-serving reason."""
    return (
        thought.trigger.startswith(("drive", "event"))
        and thought.salience >= STRONG_LONGING_SALIENCE
        and thought.sustained_attention_count >= STRONG_LONGING_PERSIST
        and not action_pending
    )


def should_crystallize(
    thought: Thought,
    *,
    receptivity_allowed: bool,
    pending: bool,
    action_pending: bool,
    persist_min: int = PERSIST_MIN,
) -> bool:
    """The deterministic Rubicon gate — ``True`` iff *thought* may spring contact.

    All of:

    * ``receptivity_allowed`` — no explicit-boundary hard veto (pre-scored to avoid
      noisy proposals; aggregation is the AUTHORITATIVE veto, not this pre-check);
    * ``not pending`` — no proactive turn already in flight;
    * ``thought.state == active`` — a live, competing thought;
    * ``salience >= CRYSTALLIZE_SALIENCE`` — personal-relevance bar (anti-frivolity);
    * a REASON — it serves the owner (``other_regarding_value``), is concrete
      (``actionability``), OR is a tightly-gated own-longing;
    * PERSISTENCE — it survived ``persist_min`` ticks of viable attention, OR is a
      strong external event (idle NEVER bypasses persistence)."""
    if not receptivity_allowed or pending:
        return False
    if thought.state != "active":
        return False
    if thought.salience < CRYSTALLIZE_SALIENCE:
        return False
    has_reason = (
        thought.other_regarding_value >= OTHER_REGARDING_MIN
        or thought.actionability >= ACTIONABILITY_MIN
        or strong_own_longing(thought, action_pending=action_pending)
    )
    if not has_reason:
        return False
    persisted = thought.sustained_attention_count >= persist_min or strong_event_trigger(thought)
    return persisted


def _reason(thought: Thought) -> str:
    """A short, human ``reason`` for the proposal (from the winning gate branch)."""
    if thought.other_regarding_value >= OTHER_REGARDING_MIN:
        return "other-serving"
    if thought.actionability >= ACTIONABILITY_MIN:
        return "actionable"
    if strong_event_trigger(thought):
        return "strong-event"
    return "own-longing"


class ThoughtCrystallization:
    """Proposes the top-down contact desire from a deliberated thought (0-LLM).

    Reads the snapshot, decides via :func:`should_crystallize`, and emits at most
    ONE transient proposal signal. Never writes a record (the writers are
    aggregation + attention). Behavior-neutral when there is no live active thought,
    a live contact desire already exists (the singleton is taken — proposing would
    resolve a thought whose reason never became a desire), or the gate fails."""

    def __init__(
        self, *, persist_min: int = PERSIST_MIN, id: str = "thought-crystallization"
    ) -> None:
        self.id = id
        self._persist_min = persist_min

    def step(self, ctx: TickContext) -> Sequence[Intent]:
        # Singleton guard: if a live contact desire already exists, do NOT propose —
        # aggregation would dedup the create, yet attention would still resolve the
        # thought, orphaning its reason. Leave the thought alone until the desire
        # clears.
        if live_contact_desire(ctx.objects) is not None:
            return []

        selected = selected_thoughts(ctx.objects, ctx.now, limit=1)
        if not selected:
            return []
        thought = selected[0]  # active-only (selected_thoughts filters parked out)

        # Pre-score receptivity to avoid noisy proposals (aggregation re-appraises and
        # is the authoritative veto — this is only to keep the proposal stream clean).
        relationship = live_owner_relationship(ctx.objects) or DEFAULT_RELATIONSHIP
        appraisal = appraise_receptivity(relationship, ctx.state, ctx.now)

        if not should_crystallize(
            thought,
            receptivity_allowed=appraisal.allowed,
            pending=ctx.state.pending_proactive_id is not None,
            action_pending=ctx.state.action_pending_since is not None,
            persist_min=self._persist_min,
        ):
            return []

        # The desire's salience is the deliberated pull — the max of the thought's
        # salience and its attention score (the pull it competed at). A proposal,
        # not a command: aggregation folds it, this component writes nothing.
        score = max(thought.salience, attention_score(thought, ctx.now))
        signal = thought_contact_proposal_signal(
            origin_id=self.id,
            thought_id=thought.id,
            score=score,
            reason=_reason(thought),
            other_regarding=thought.other_regarding_value,
            actionability=thought.actionability,
            salience=thought.salience,
            timestamp=ctx.now.isoformat(),
        )
        return [EmitSignal(signal=signal)]

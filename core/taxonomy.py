"""Signal taxonomy — the typed vocabulary of the pipeline (spec §4).

Phase B1 defines two kinds:
- ``contact`` — the neuron's *transient* intra-tick output: the unipolar drive
  value ``[0..u_max]`` plus its per-tick ``delta``. Never persisted.
- ``exchange`` — a *durable* external input: a real lane event (actor + label,
  per :mod:`lifemodel.sim.quality`) that the neuron reads to satiate the drive.

Builders keep payloads JSON-native and uniform; readers validate on the way out.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from ..domain.signal import Signal
from ..sim.aggregation import Verdict
from ..sim.quality import Actor, Label

KIND_CONTACT = "contact"
KIND_EXCHANGE = "exchange"
KIND_VERDICT = "verdict"
KIND_IN_FLIGHT = "in_flight"
#: The transient top-down desire spring (lm-27n.9): ThoughtCrystallization emits it
#: mid-tick when a deliberated thought clears the Rubicon gate. It is a *proposal*
#: (not a command) — ContactAggregation, the SOLE desire writer, folds it into the
#: singleton contact desire, and ThoughtAttention resolves the source thought on
#: seeing it. Never persisted: EmitSignal threads it in-tick to later components
#: (crystallization runs BEFORE aggregation + attention), never to the durable bus.
KIND_THOUGHT_CONTACT_PROPOSAL = "thought_contact_proposal"

Lane = Literal["control", "sensor"]

#: Load-bearing lifecycle events — never salience-shed (spec §5.1). Includes a
#: forward-looking ``delivery_result`` kind (used by phases D/E) so the lane is
#: stable before that signal exists.
CONTROL_KINDS: frozenset[str] = frozenset(
    {KIND_EXCHANGE, KIND_VERDICT, KIND_IN_FLIGHT, "delivery_result"}
)


def lane_of(kind: str) -> Lane:
    """Backpressure lane for a signal kind. Unknown kinds are sensors (never
    control) so an unknown flood cannot claim the lossless lane."""
    return "control" if kind in CONTROL_KINDS else "sensor"


_VERDICTS: dict[str, Verdict] = {v.value: v for v in Verdict}

_ACTORS: frozenset[str] = frozenset({"user", "assistant", "proactive_internal"})
_LABELS: frozenset[str] = frozenset({"two_way", "ack", "monologue", "rejection"})


def is_kind(signal: Signal, kind: str) -> bool:
    return signal.kind == kind


def contact_signal(*, origin_id: str, value: float, delta: float, timestamp: str | None) -> Signal:
    """Build a transient contact signal carrying the drive value and its delta."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_CONTACT,
        payload={"value": float(value), "delta": float(delta)},
        timestamp=timestamp,
    )


def exchange_signal(*, origin_id: str, actor: Actor, label: Label, timestamp: str | None) -> Signal:
    """Build a durable exchange-input signal from a lane event."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_EXCHANGE,
        payload={"actor": actor, "label": label},
        timestamp=timestamp,
    )


def read_exchange(signal: Signal) -> tuple[Actor, Label]:
    """Validate and extract ``(actor, label)`` from an exchange signal."""
    if signal.kind != KIND_EXCHANGE:
        raise ValueError(f"not an exchange signal: kind={signal.kind!r}")
    actor = signal.payload.get("actor")
    label = signal.payload.get("label")
    if actor not in _ACTORS or label not in _LABELS:
        raise ValueError(f"invalid exchange payload: {signal.payload!r}")
    return cast(Actor, actor), cast(Label, label)


def verdict_signal(
    *, origin_id: str, verdict: Verdict, timestamp: str | None, correlation_id: str = ""
) -> Signal:
    """Build a durable verdict-input signal (cognition's decision on a desire)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_VERDICT,
        payload={"verdict": verdict.value, "correlation_id": correlation_id},
        timestamp=timestamp,
    )


def read_verdict(signal: Signal) -> Verdict:
    """Validate and extract the ``Verdict`` from a verdict signal."""
    if signal.kind != KIND_VERDICT:
        raise ValueError(f"not a verdict signal: kind={signal.kind!r}")
    raw = signal.payload.get("verdict")
    if raw not in _VERDICTS:
        raise ValueError(f"invalid verdict payload: {signal.payload!r}")
    return _VERDICTS[raw]


def read_verdict_correlation(signal: Signal) -> str:
    """The correlation id a verdict resolves (``""`` if absent)."""
    if signal.kind != KIND_VERDICT:
        raise ValueError(f"not a verdict signal: kind={signal.kind!r}")
    raw = signal.payload.get("correlation_id", "")
    return raw if isinstance(raw, str) else ""


def in_flight_signal(*, origin_id: str, value: bool, timestamp: str | None) -> Signal:
    """Build a durable in-flight input (a turn is running/queued)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_IN_FLIGHT,
        payload={"value": bool(value)},
        timestamp=timestamp,
    )


def is_in_flight(signals: Iterable[Signal]) -> bool:
    """True if any in-flight signal in the batch reports a running turn."""
    return any(s.kind == KIND_IN_FLIGHT and bool(s.payload.get("value")) for s in signals)


def contact_value(signals: Iterable[Signal], *, default: float) -> float:
    """The most recent transient contact value in the batch, or ``default``."""
    value = default
    for s in signals:
        if s.kind == KIND_CONTACT:
            raw = s.payload.get("value", default)
            value = float(raw) if isinstance(raw, int | float) else default
    return value


@dataclass(frozen=True)
class ThoughtContactProposal:
    """A crystallized thought's *proposal* to spring the contact desire (lm-27n.9).

    Read from the transient ``thought_contact_proposal`` signal by aggregation (the
    desire writer) and attention (the thought writer). Carries the source thought's
    id, the proposal ``score`` (the desire's salience), a human ``reason`` (why it
    crossed the Rubicon), and the source appraisal scores — enough for aggregation
    to fold it into the singleton and for lm-8o3 to frame the wake later, no more."""

    thought_id: str
    score: float
    reason: str
    other_regarding: float
    actionability: float
    salience: float


def thought_contact_proposal_signal(
    *,
    origin_id: str,
    thought_id: str,
    score: float,
    reason: str,
    other_regarding: float,
    actionability: float,
    salience: float,
    timestamp: str | None,
) -> Signal:
    """Build the transient top-down desire-spring proposal (a proposal, not a command)."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_THOUGHT_CONTACT_PROPOSAL,
        payload={
            "thought_id": thought_id,
            "score": float(score),
            "reason": reason,
            "other_regarding": float(other_regarding),
            "actionability": float(actionability),
            "salience": float(salience),
        },
        timestamp=timestamp,
    )


def read_thought_contact_proposal(signals: Iterable[Signal]) -> ThoughtContactProposal | None:
    """The last well-formed contact proposal in the batch, or ``None``.

    Mirrors :func:`contact_value` — a later component reads the freshest proposal
    an earlier one emitted this tick. A malformed payload is skipped (never a
    partial proposal): a missing/ill-typed ``thought_id`` or numeric field is
    ignored, so a corrupt in-tick signal degrades to "no proposal", not a crash."""
    latest: ThoughtContactProposal | None = None
    for s in signals:
        if s.kind != KIND_THOUGHT_CONTACT_PROPOSAL:
            continue
        thought_id = s.payload.get("thought_id")
        reason = s.payload.get("reason", "")
        if not isinstance(thought_id, str) or not isinstance(reason, str):
            continue
        try:
            latest = ThoughtContactProposal(
                thought_id=thought_id,
                score=_as_float(s.payload.get("score")),
                reason=reason,
                other_regarding=_as_float(s.payload.get("other_regarding")),
                actionability=_as_float(s.payload.get("actionability")),
                salience=_as_float(s.payload.get("salience")),
            )
        except (TypeError, ValueError):
            continue
    return latest


#: Emitted by aggregation ONLY when it actually creates a top-down/mixed contact
#: desire FROM a proposal (lm-27n.9) — so ThoughtAttention resolves the source
#: thought on genuine CREATION, never on a mere proposal aggregation then suppressed
#: (via silence window / backoff / in-flight). A suppressed reason stays live and is
#: handled by normal decay/parking ("not nagged" without silently dropping it).
KIND_THOUGHT_CONTACT_CREATED = "thought_contact_created"


def thought_contact_created_signal(
    *, origin_id: str, thought_id: str, timestamp: str | None
) -> Signal:
    """Build the transient "a contact desire was created from this thought" signal."""
    return Signal(
        origin_id=origin_id,
        kind=KIND_THOUGHT_CONTACT_CREATED,
        payload={"thought_id": thought_id},
        timestamp=timestamp,
    )


def read_thought_contact_created(signals: Iterable[Signal]) -> str | None:
    """The source ``thought_id`` of a top-down desire created this tick, or ``None``."""
    created: str | None = None
    for s in signals:
        if s.kind != KIND_THOUGHT_CONTACT_CREATED:
            continue
        thought_id = s.payload.get("thought_id")
        if isinstance(thought_id, str):
            created = thought_id
    return created


def _as_float(raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"expected a number, got {type(raw).__name__}")
    return float(raw)

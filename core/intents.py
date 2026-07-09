"""Intents — the sole channel for state mutation (spec §6).

Layers, neurons and Hermes hooks never write state directly; they return (or
enqueue) `Intent`s, and the single :class:`~lifemodel.core.state_actor.StateActor`
applies them atomically at end of tick. Intents are immutable value objects.

Phase A defines the subset the skeleton actually routes: `UpdateState` (a
validated patch on :class:`~lifemodel.state.model.State`), `EmitSignal` (append
to the durable bus), and `CheckpointState` (the marker the actor emits for
observability when it commits). The energy / cognition / user-model intents
from spec §6 arrive in their own phases against this same base.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.memory import PutOp, TransitionOp
from ..domain.signal import Signal


class Intent:
    """Marker base for every intent. Carries no fields of its own."""

    __slots__ = ()


@dataclass(frozen=True)
class UpdateState(Intent):
    """Patch the model state. ``changes`` maps ``State`` field names to new
    values; the state-actor validates the field names and applies the merge."""

    changes: Mapping[str, Any]


@dataclass(frozen=True)
class EmitSignal(Intent):
    """Append a signal to the durable bus (handled by the CoreLoop, not the
    state-actor — bus writes are immediate, state mutation is end-of-tick,
    spec §7.4)."""

    signal: Signal


@dataclass(frozen=True)
class CheckpointState(Intent):
    """Observability marker for a committed checkpoint. The state-actor emits
    it implicitly on mutation (spec §6); callers need not construct it."""


@dataclass(frozen=True)
class LaunchProactive(Intent):
    """Launch a proactive turn (the being's native Hermes turn) with this
    desire-framed prompt. Consumed by the egress in Phase E.

    ``origin_traceparent`` is the MANDATORY async-correlation anchor (spec §4.4):
    the full W3C ``traceparent`` (trace_id + span_id + flags) of the launch span,
    so every downstream span of this one attempt — delivery, the async
    ``post_llm`` outcome, the resolving tick — can ``child_of`` it and land under
    ONE ``trace_id``. The type forbids launching without an origin trace (§3 law 1):
    an untraceable async launch is structurally impossible.
    """

    prompt: str
    correlation_id: str
    origin_traceparent: str
    reserved_energy: float = 0.0


@dataclass(frozen=True)
class PutRecord(Intent):
    """Request a memory ``put``. The :class:`~lifemodel.core.state_actor.StateActor`
    collects these (in emission order) and hands them to the tick's atomic
    committer alongside the merged state patch — never a direct store write. No
    live component emits one yet (lm-27n.2 installs the machinery; .3 wires
    emitters)."""

    op: PutOp


@dataclass(frozen=True)
class TransitionRecord(Intent):
    """Request a guarded memory ``transition``. Collected by the
    :class:`~lifemodel.core.state_actor.StateActor` into the tick's one atomic
    commit; a stale ``from_state`` rolls back the whole tick (state included).
    No live component emits one yet (lm-27n.2)."""

    op: TransitionOp

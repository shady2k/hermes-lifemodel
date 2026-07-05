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

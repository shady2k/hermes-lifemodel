"""ExecutionFrame + SignalFrame — the unit of processing (spec §2/§3).

The being's nervous flow is **ephemeral**: an afferent impulse lives no longer
than the frame that carries it, and nothing is written to a durable log. Durable
state is only where biology keeps a trace — the body (``AgentState``), memory
(``Memory``/BDI) and the observability record (``Trace``). This module holds the
two ideas that replace the old durable "signal bus":

* :class:`SignalFrame` — the in-memory bus for ONE frame. Seeded with the frame's
  trigger signals, grown as components emit into it, then discarded when the frame
  commits. A signal lives ``<=`` one frame (not ``<=`` one tick).
* :func:`run_frame` — run one ExecutionFrame, serialized through a single
  process-wide state-actor lock so no two frames can interleave their
  snapshot→commit. A frame is triggered by a heartbeat, an incoming Hermes event,
  the completion of an async cognition turn, or an admin mutation
  (:class:`FrameTrigger`); an async completion commits its outcome **immediately**
  (its own frame), not at the next heartbeat.

Stdlib only; imports no Hermes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import TYPE_CHECKING

from ..domain.signal import Signal

if TYPE_CHECKING:
    from .coreloop import CoreLoop, TickReport


class FrameTrigger(StrEnum):
    """Why an :class:`ExecutionFrame` ran — the closed set of occasions (spec §3).

    * ``HEARTBEAT`` — the periodic tick (empty world: the drive rises, aggregation
      watches the threshold).
    * ``EVENT`` — an incoming Hermes event (the user wrote).
    * ``ASYNC_COMPLETION`` — an async cognition turn finished (``sent``/``silent``…);
      its outcome commits in its OWN frame, immediately, not at the next heartbeat.
    * ``ADMIN`` — an admin/debug command that mutates state.
    """

    HEARTBEAT = "heartbeat"
    EVENT = "event"
    ASYNC_COMPLETION = "async_completion"
    ADMIN = "admin"


class SignalFrame:
    """The in-memory ephemeral signal bus for ONE ExecutionFrame (spec §2/§3).

    Seeded with the frame's initial (trigger) signals; components emit into it in
    order during the frame; it is discarded when the frame commits — there is no
    durable log, and nothing survives a restart ("lost consciousness → don't replay
    stale impulses", spec §2). A :meth:`snapshot` handed to a component is a frozen
    copy, so a later emission never mutates an earlier component's view.
    """

    __slots__ = ("_signals",)

    def __init__(self, initial: Sequence[Signal] = ()) -> None:
        self._signals: list[Signal] = list(initial)

    def emit(self, signal: Signal) -> None:
        """Append *signal* to the frame — visible to every LATER component this frame."""
        self._signals.append(signal)

    def snapshot(self) -> tuple[Signal, ...]:
        """A frozen copy of the signals emitted so far this frame."""
        return tuple(self._signals)

    def __len__(self) -> int:
        return len(self._signals)


#: The one process-wide state-actor lock (spec §3). EVERY frame — heartbeat, event,
#: async-completion, admin — acquires it, so two frames can never interleave their
#: snapshot→commit even when a heartbeat and an inbound event coincide. Re-entrant so
#: a frame that itself runs a nested reconciliation (e.g. an egress rollback commit)
#: does not deadlock against its own outer frame.
_STATE_ACTOR_LOCK = threading.RLock()


@contextmanager
def state_actor_lock() -> Iterator[None]:
    """Serialize a NON-frame writer against every frame — the SAME one lock (spec §3).

    Almost everything that touches ``State`` is a frame, and :func:`run_frame` takes this
    lock for it. The being's SOUL is the exception: ``write_soul`` runs inside an agent
    turn (an executor thread) and startup reconciliation runs at ``connect()`` — neither
    is a frame, and both must still write ``genesis_completed_at``/``soul_sha``.

    Left unserialized, the interleave costs a birth: the ~60s tick loads its snapshot,
    the being is born, the tick commits the snapshot it loaded BEFORE the birth (its
    ``commit`` is an unconditional whole-``State`` UPSERT), and the being now has a soul
    on disk and no birth — it re-runs the ritual and reads its OWN soul as "someone wrote
    this before you woke". The reverse ordering silently discards the tick's u/energy/
    affect instead.

    This is deliberately NOT a second lock — a soul writer takes the one state-actor lock,
    so a frame and a soul write can never straddle each other's snapshot→commit. It is
    re-entrant, so a soul write made from INSIDE a frame would not deadlock against it.
    Note this is only half of the answer: holding the lock stops a frame from clobbering
    the stamps, and the store's ``stamp_soul`` (a field-level merge, never a whole-State
    commit) stops the soul path from clobbering the frame's work. Both directions matter.
    """
    with _STATE_ACTOR_LOCK:
        yield


def run_frame(
    coreloop: CoreLoop,
    initial_signals: Sequence[Signal] = (),
    *,
    trigger: FrameTrigger = FrameTrigger.HEARTBEAT,
) -> TickReport:
    """Run ONE ExecutionFrame, serialized through the one state-actor lock (spec §3).

    Seeds the frame's :class:`SignalFrame` with *initial_signals*, runs the pipeline,
    and commits its intents atomically at end of frame — all under
    :data:`_STATE_ACTOR_LOCK` so frames are strictly serialized (no split-brain). An
    async-completion frame therefore commits its outcome the moment cognition finishes,
    not at the next heartbeat.
    """
    with _STATE_ACTOR_LOCK:
        return coreloop.tick(initial_signals, trigger=trigger)

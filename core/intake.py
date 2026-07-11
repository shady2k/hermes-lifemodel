"""Priority-class backpressure — the AGGREGATION gate's defensive layer (spec §7).

The ephemeral bus has no "defer overflow to the next tick" — an impulse lives ``<=``
one frame (spec §2/§3), so backpressure cannot live in the bus. It lives in the
CONSUMER: the AGGREGATION gate (the thalamus) folds the frame's signals down here
before it reduces them, off each signal's taxonomy ``kind`` (spec §7):

* ``must_process`` — the load-bearing signals aggregation must never lose:
  ``contact_observed`` / ``proactive_outcome`` / the ``in_flight`` safety interlock
  (``taxonomy.CONTROL_KINDS``) **plus the drive's own output** (``contact_pressure`` /
  the legacy ``contact``) — the PANIC/GRIEF ``u`` the gate reads to decide the wake.
  These are NEVER shed, no matter how full the frame is. (This set is deliberately a
  SUPERSET of ``CONTROL_KINDS``: ``lane_of`` classes the drive signal as ``sensor``
  for the coarse public taxonomy, but the gate must not shed the very drive it
  defends cognition for — spec §7: "safety/PANIC/GRIEF drive signal — never shed".)
* ``best_effort`` — sensor noise / low-salience observations: under load COALESCED to
  a bounded count (:data:`MAX_BEST_EFFORT`), shedding the LOWEST-salience first, so a
  flood does not each individually drive a separate expensive downstream step.

The frame itself (``core/frame.py``) stays a dumb in-memory blackboard: everything
hits the bus; the gate is the only place that decides what reaches cognition.

Stdlib only; imports no Hermes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..domain.signal import Signal
from .taxonomy import CONTROL_KINDS, KIND_CONTACT, KIND_CONTACT_PRESSURE

#: The two priority classes (spec §7). ``must_process`` is lossless; ``best_effort``
#: is coalesced/shed under load.
PriorityClass = Literal["must_process", "best_effort"]
MUST_PROCESS: PriorityClass = "must_process"
BEST_EFFORT: PriorityClass = "best_effort"

#: The drive's OWN output kinds — the transient ``u`` the gate reads to decide the
#: wake. Never shed: shedding these blinds the gate to its own drive (spec §7).
_DRIVE_KINDS: frozenset[str] = frozenset({KIND_CONTACT_PRESSURE, KIND_CONTACT})

#: The kinds the gate must never shed: the public lifecycle set (``CONTROL_KINDS``)
#: UNIONED with the drive-output kinds. Broader than ``CONTROL_KINDS`` on purpose —
#: see the module docstring.
MUST_PROCESS_KINDS: frozenset[str] = CONTROL_KINDS | _DRIVE_KINDS

#: The bounded cap on best_effort signals reaching the reducer. The spec names no
#: specific number for v1 ("with one sensor, MAX_INTAKE is almost unnecessary —
#: classification is enough", §7); this is a sensible bounded constant so a
#: pathological sensor flood collapses to a handful of the most-salient readings.
MAX_BEST_EFFORT: int = 8


def priority_class(kind: str) -> PriorityClass:
    """The priority class of a signal *kind* (spec §7).

    Unknown kinds are ``best_effort`` (never ``must_process``) so an unknown flood
    cannot claim the lossless class — mirrors :func:`~lifemodel.core.taxonomy.lane_of`.
    """
    return MUST_PROCESS if kind in MUST_PROCESS_KINDS else BEST_EFFORT


@dataclass(frozen=True)
class IntakeResult:
    """The gate's verdict for one frame (spec §7).

    * ``signals`` — the gated signals the reducer processes: EVERY ``must_process``
      signal plus the surviving ``best_effort`` ones, in original FRAME ORDER (so the
      gate's "latest wins" reads are unperturbed).
    * ``must_process`` — count of lossless signals (all kept).
    * ``best_effort_kept`` — best_effort signals that survived the coalescing cap.
    * ``best_effort_shed`` — best_effort signals dropped under load.

    Invariant: ``must_process + best_effort_kept == len(signals)`` and
    ``best_effort_kept + best_effort_shed`` == the frame's total best_effort count.
    """

    signals: tuple[Signal, ...]
    must_process: int
    best_effort_kept: int
    best_effort_shed: int

    @property
    def overflowed(self) -> bool:
        """True if the frame overflowed the cap and best_effort signals were shed."""
        return self.best_effort_shed > 0


def apply_backpressure(
    signals: Sequence[Signal], *, max_best_effort: int = MAX_BEST_EFFORT
) -> IntakeResult:
    """Classify *signals* and coalesce the best_effort class to *max_best_effort*.

    Keeps every ``must_process`` signal; when the best_effort count exceeds the cap,
    keeps the ``max_best_effort`` HIGHEST-salience best_effort signals (ties broken by
    earliest frame position, so the fold is deterministic) and sheds the rest. The
    returned :attr:`IntakeResult.signals` preserves the original frame order.
    """
    frame = tuple(signals)
    best_positions = [i for i, s in enumerate(frame) if priority_class(s.kind) == BEST_EFFORT]
    must_count = len(frame) - len(best_positions)

    if len(best_positions) <= max_best_effort:
        survivors = set(best_positions)
    else:
        # Shed by salience (spec §7): rank the best_effort signals by descending
        # salience, ties by earliest position; keep the top ``max_best_effort``.
        ranked = sorted(best_positions, key=lambda i: (-frame[i].salience, i))
        survivors = set(ranked[:max_best_effort])

    kept = tuple(
        s for i, s in enumerate(frame) if priority_class(s.kind) != BEST_EFFORT or i in survivors
    )
    return IntakeResult(
        signals=kept,
        must_process=must_count,
        best_effort_kept=len(survivors),
        best_effort_shed=len(best_positions) - len(survivors),
    )

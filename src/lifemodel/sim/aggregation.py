"""The aggregation layer — the contact-desire lifecycle (spec §4, §5, §7).

The drive accumulates pressure; when a wake-eligible urge crosses the threshold
the aggregation layer turns it into *one* desire and wakes cognition. This layer
owns the desire's whole life: it **dedups** every further urge against the live
desire (the anti-drum guarantee — no duplicate wakes), **holds** a desire that
cognition chose to *defer* until a release condition re-presents it, and
**clears** the desire on a resolving verdict or a real user exchange.

Project convention: this decision logic lives here, in a dedicated layer — never
smeared into the drive-component (which only measures) or a neuron.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DesireStatus(enum.Enum):
    """Where a contact-desire is in its lifecycle."""

    NONE = "none"  # no live desire
    ACTIVE = "active"  # created, cognition woken, awaiting a verdict
    DEFERRED = "deferred"  # cognition deferred it; held until a release condition


class Verdict(enum.Enum):
    """Cognition's decision on a woken desire (scripted in the sim, LLM live)."""

    FULFILL = "fulfill"  # a message is sent → contact happened
    DEFER = "defer"  # wrong moment → hold the intention, do not drop it
    REJECT = "reject"  # nothing to say → clear it (+ growing backoff, wake-decision's job)


@dataclass
class Aggregator:
    """The desire-lifecycle state machine (one desire per lane)."""

    status: DesireStatus = DesireStatus.NONE

    def on_urge(self) -> bool:
        """A wake-eligible urge arrived. Create a desire and wake **once**.

        Returns ``True`` if this urge created a new wake (``NONE → ACTIVE``);
        ``False`` if a desire is already live (active or deferred) and the urge
        is deduped — the ``ack`` that stops the being drumming.
        """
        if self.status is DesireStatus.NONE:
            self.status = DesireStatus.ACTIVE
            return True
        return False

    def on_release(self) -> bool:
        """A release condition holds for a *deferred* desire — re-present it.

        Returns ``True`` if a held desire was re-woken (``DEFERRED → ACTIVE``);
        ``False`` otherwise (nothing deferred to release).
        """
        if self.status is DesireStatus.DEFERRED:
            self.status = DesireStatus.ACTIVE
            return True
        return False

    def apply_verdict(self, verdict: Verdict) -> None:
        """Resolve a woken desire by cognition's verdict.

        ``FULFILL``/``REJECT`` clear the desire; ``DEFER`` holds it. Satiation,
        duration reset, and backoff bookkeeping are the drive's / wake-decision's
        jobs — this layer only advances the desire's status.
        """
        self.status = DesireStatus.DEFERRED if verdict is Verdict.DEFER else DesireStatus.NONE

    def on_exchange(self) -> None:
        """A real user exchange clears any live desire (active or held)."""
        self.status = DesireStatus.NONE

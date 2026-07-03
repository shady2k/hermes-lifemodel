"""``ActGate`` — the restraint extension point on outgoing acts (HLA §1/§7).

Cognition may *want* to speak or act; the act-gate is the "говорить или молчать"
check that sits between the wish and the world (HLA §1). It weighs receptivity,
quiet-hours, budget and cooldown, then returns a
:class:`~lifemodel.domain.act.Decision`. Enforcement of actions in the world is
Hermes' sandbox+approvals (HLA §7); this gate is *our* restraint over speech.

Contract only. The minimal Phase-1.4 gate (author/home channel · ≤1 message per
threshold cycle · cooldown · text-only) and the fuller Phase-2.3 gate implement
this — they do not redesign it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..domain.act import Decision


class ActGate(ABC):
    """Decide whether an outgoing act is allowed right now (HLA §7).

    ``ctx`` is typed ``Any`` so each concrete gate accepts the context it needs
    (current state, the wake-packet, the proposed message) without the base
    forcing a shape — the gate's inputs firm up as the gating logic lands.
    """

    @abstractmethod
    def allow(self, ctx: Any) -> Decision:
        """Return an allow/suppress decision for the proposed act (HLA §5)."""
        raise NotImplementedError

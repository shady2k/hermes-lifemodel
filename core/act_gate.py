"""``ActGate`` — the restraint extension point on outgoing acts (HLA §1/§7).

Cognition may *want* to speak or act; the act-gate is the "говорить или молчать"
check that sits between the wish and the world (HLA §1). It weighs receptivity,
quiet-hours, budget and cooldown, then returns a
:class:`~lifemodel.domain.act.Decision`. Enforcement of actions in the world is
Hermes' sandbox+approvals (HLA §7); this gate is *our* restraint over speech.

Contract only. The minimal-safety rails do **not** run through this gate: the
proactive turn is injected as a native reach-in from the supervised platform
adapter's tick (:mod:`lifemodel.adapters.being_platform`), so restraint is enforced
*structurally* — the home-lane target and the global backstop (≤ N/day + min
interval, :func:`lifemodel.core.backstop.allow_send`). This ``ActGate`` is the seam
for the fuller Phase-2.3 per-turn speak/silent gate (the ``[SILENT]`` path), which
will implement this contract rather than redesign it.
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

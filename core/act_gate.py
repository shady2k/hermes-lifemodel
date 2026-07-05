"""``ActGate`` — the restraint extension point on outgoing acts (HLA §1/§7).

Cognition may *want* to speak or act; the act-gate is the "говорить или молчать"
check that sits between the wish and the world (HLA §1). It weighs receptivity,
quiet-hours, budget and cooldown, then returns a
:class:`~lifemodel.domain.act.Decision`. Enforcement of actions in the world is
Hermes' sandbox+approvals (HLA §7); this gate is *our* restraint over speech.

Contract only. Phase 1.4's minimal-safety rails do **not** run through this gate:
in the wake-gate architecture (HLA D1/D4) the proactive turn is a Hermes cron
agent, so there is no in-process seam to call ``allow`` at delivery time. Those
rails are instead enforced *structurally* — author/home channel + text-only via
the cron job's ``deliver`` / ``enabled_toolsets`` (see :mod:`lifemodel.heartbeat`),
and ≤ 1 message per cycle + cooldown via the tick's drain
(:func:`lifemodel.tick.run_tick`, which gates the wake itself). This ``ActGate``
is the seam for the fuller Phase-2.3 per-turn speak/silent gate (the ``[SILENT]``
path), which will implement this contract rather than redesign it.
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

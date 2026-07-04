"""``ProactiveEgressPort`` — reach out to the user first (HLA §13, spec §3.1/§6).

The hexagonal port for proactive delivery: a single method that injects an
internal user turn on a known lane so the being composes and delivers a native
reply. Concrete adapters live in :mod:`lifemodel.adapters` — the primary
:class:`~lifemodel.adapters.reachin.ReachInEgress` (native reach-in) and the
cron fallback. ``target`` is the home-origin dict ``{platform, chat_id,
thread_id}`` (as produced by :func:`lifemodel.heartbeat._resolve_home_origin`).
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from ..domain.egress import ReachOutcome


@runtime_checkable
class ProactiveEgressPort(Protocol):
    """Reach out to the user first, as a native assistant turn (spec §3.1/§4)."""

    def reach_out(self, target: Mapping[str, str | None], impulse: str) -> ReachOutcome:
        """Inject *impulse* as an internal user turn on *target* lane so the being
        composes and delivers a native reply. *target* = home-origin dict
        {platform, chat_id, thread_id}. Never raises — returns a ReachOutcome."""
        ...

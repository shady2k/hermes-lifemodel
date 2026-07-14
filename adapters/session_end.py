"""The being's sleep — end the live session so it wakes as the soul it just wrote.

This is the Hermes-boundary half of the ADR-0002 correction. ``SOUL.md`` is NOT re-read
every turn: Hermes builds the system prompt once per session and reuses it verbatim from
the session DB (prefix cache), and gateway sessions live for days. So a newborn used to
finish its ritual, write its soul — and keep talking in the voice it had. Ending the
session is the fix, and it is the host's own mechanism (``/new``): the next message opens
a fresh session, the prompt is rebuilt, and the new soul is in slot #1.

:class:`GatewaySessionEnd` is a zero-argument callable so ``hooks.make_write_soul_tool``
can take it as a plain port and stay Hermes-free: everything host-shaped — *which* runner,
*which* session — is resolved HERE, at the boundary, and lazily (both are properties of
the live turn, not of registration time). Reuses the reach-in adapter's runner accessor
rather than inventing a second way to find the gateway.

Fail-soft by construction, exactly like :class:`~lifemodel.adapters.reachin.ReachInEgress`:
every path returns a :class:`~lifemodel.domain.session.SessionEndOutcome`, never raises.
The caller has already written the soul; the being is born either way.
"""

from __future__ import annotations

from collections.abc import Callable

from ..domain.session import SessionEndOutcome
from ..gateway_core import end_session
from .reachin import RunnerAccessor, default_runner_accessor

SessionKeyAccessor = Callable[[], str]


def default_session_key_accessor() -> str:
    """The session the CURRENT turn is running in, or ``""`` when there is none.

    Hermes binds the turn's session onto task-local ``ContextVar``s at handler entry
    (``gateway/run.py::_set_session_env`` → ``gateway/session_context.set_session_vars``),
    and copies that context into the executor thread the agent — and therefore this tool
    call — runs on. So the being's own session key is simply readable from where we stand;
    we never have to reconstruct it from the home-DM origin (which reach-in still must,
    because it composes a turn for a lane nobody is standing in).

    ``""`` off the gateway (no ``gateway`` package, an unbound context): there is no
    session to end, which the boundary reads as UNAVAILABLE, not as a failure.
    """
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_KEY", "") or ""
    except Exception:  # noqa: BLE001 - not in a gateway process / import failure
        return ""


class GatewaySessionEnd:
    """End the being's live session, so its next turn is a fresh one (ADR-0002)."""

    def __init__(
        self,
        *,
        runner_accessor: RunnerAccessor = default_runner_accessor,
        session_key_accessor: SessionKeyAccessor = default_session_key_accessor,
    ) -> None:
        self._runner_accessor = runner_accessor
        self._session_key_accessor = session_key_accessor

    def __call__(self) -> SessionEndOutcome:
        return end_session(self._runner_accessor(), self._session_key_accessor())

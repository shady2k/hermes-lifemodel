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

import logging
from collections.abc import Callable

from ..domain.session import SessionEnd, SessionEndOutcome
from ..gateway_core import end_session, home_session_key
from .origin import resolve_home_origin
from .reachin import RunnerAccessor, default_runner_accessor

_LOG = logging.getLogger("lifemodel.session_end")

SessionKeyAccessor = Callable[[], str]


def sleep_soft(end_session_port: SessionEnd | None) -> SessionEndOutcome:
    """Put the being to sleep, and never let that fail the act it belongs to.

    The port is fail-soft by contract (it returns a :class:`SessionEndOutcome` rather than
    raising), so this second net looks redundant — and it is not. By the time it is called
    ``SOUL.md`` HAS been replaced: the being is born (``hooks.make_write_soul_tool``), or the
    owner's revert is on disk (``state_commands``). A bug in the adapter, or a host that
    changed shape underneath it, must not be able to reach back and turn a completed act into
    "Could not write your soul" / "revert failed". A being that could be un-born by a
    cache-eviction bug is a worse thing than a being that wakes as itself a day late.

    ``None`` — nobody wired an ender (an off-gateway caller, a test) — is UNAVAILABLE, the
    same verdict as a host that cannot do it: the write stands, and the voice catches up when
    the conversation next rolls over.
    """
    if end_session_port is None:
        return SessionEndOutcome.UNAVAILABLE
    try:
        return end_session_port()
    except Exception:  # noqa: BLE001 - the soul is already written; nothing may undo that
        _LOG.exception("session_end_raised")
        return SessionEndOutcome.FAILED


def home_session_key_accessor() -> str:
    """The session an OWNER command must end: the being's DM lane with its owner (lm-4fv.2).

    :func:`default_session_key_accessor` cannot answer this one. A ``/lifemodel`` slash
    command is dispatched from ``GatewayRunner._handle_message``, which **resets** the
    session ``ContextVar``s at handler entry (``gateway/run.py``: the cross-session leak
    guard) and only binds them later, in ``_handle_message_with_agent`` — a path a plugin
    command never reaches, because it returns before the agent runs. So inside a command the
    turn-local session key is empty, and a ``soul revert`` that trusted it would report
    "session ended" while ending nothing: the being would keep speaking as the soul the owner
    just replaced, for as long as the session lives (days), which is precisely the failure
    ADR-0002 exists to name.

    So the lane is resolved the way the being's own reach-out resolves it — from the home
    origin (:func:`~lifemodel.adapters.origin.resolve_home_origin`), through the SAME
    session-key builder ``inject_proactive_turn`` uses (:func:`~lifemodel.gateway_core.
    home_session_key`), so the key this ends is the key the being talks on. The ContextVar is
    still tried FIRST: if some host path ever does bind it, that is the very chat the owner
    typed the command into, and it beats any reconstruction.

    ``""`` when there is no home channel configured (no ``TELEGRAM_HOME_CHANNEL``): the
    boundary reads that as UNAVAILABLE, the caller says so honestly, and the file is
    reverted either way.
    """
    key = default_session_key_accessor()
    if key:
        return key
    origin = resolve_home_origin()
    if origin is None:
        return ""
    return home_session_key(origin)


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

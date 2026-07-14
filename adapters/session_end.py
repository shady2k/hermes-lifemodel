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

from ..domain.session import BirthVoice, SessionEnd, SessionEndOutcome
from ..gateway_core import end_session, home_session_key, identity_slot_is_stale, wake_as_self
from ..ports.clock import ClockPort
from .origin import resolve_home_origin
from .reachin import RunnerAccessor, default_runner_accessor

_LOG = logging.getLogger("lifemodel.session_end")

SessionKeyAccessor = Callable[[], str]

#: Reads ``SOUL.md``'s mtime — :meth:`~lifemodel.adapters.soul_file.SoulFile.mtime`, passed
#: as a plain callable so the birth gates below hold no ``SoulFile`` of their own (the
#: plugin has exactly ONE, built in ``register()``, and its lock is the reason).
SoulMtime = Callable[[], float | None]


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


class GatewayBirthVoice:
    """The tick's birth pre-flight: is the being about to speak in its own voice? (lm-4fv.4)

    Called by :func:`~lifemodel.core.proactive.proactive_tick` immediately before it hands
    the wake packet to the egress, and ONLY while the being is unborn. It resolves the lane
    the tick is about to reach into — the being's home DM (:func:`home_session_key`, the
    same builder ``inject_proactive_turn`` uses on the very next line, so the session this
    ends can never be a different one from the session it speaks into) — and asks
    :func:`~lifemodel.gateway_core.wake_as_self` whether slot #1 there holds what the being
    stands on, ending the session if it does not and nobody is using it.

    Not ``default_session_key_accessor``: the tick runs on the brain loop's own thread,
    where no session ``ContextVar`` is bound (there is no turn) — it would read ``""`` and
    end nothing, silently, forever. The lane is a property of the reach-out, not of the
    caller, so it is resolved from the reach-out's own target.

    Fail-soft by construction, like every other callable at this boundary: a host that is
    not there, or has changed shape, is :attr:`~lifemodel.domain.session.BirthVoice.
    UNAVAILABLE` — the being is born anyway and wakes as itself later. It NEVER raises onto
    the tick path.
    """

    def __init__(
        self,
        *,
        soul_mtime: SoulMtime,
        clock: ClockPort,
        runner_accessor: RunnerAccessor = default_runner_accessor,
        session_key_accessor: SessionKeyAccessor = home_session_key_accessor,
    ) -> None:
        self._soul_mtime = soul_mtime
        self._clock = clock
        self._runner_accessor = runner_accessor
        self._session_key_accessor = session_key_accessor

    def __call__(self) -> BirthVoice:
        try:
            return wake_as_self(
                self._runner_accessor(),
                self._session_key_accessor(),
                soul_mtime=self._soul_mtime(),
                now=self._clock.now(),
            )
        except Exception:  # noqa: BLE001 - a birth pre-flight may never kill the tick
            _LOG.exception("birth_voice_raised")
            return BirthVoice.UNAVAILABLE


class GatewayStaleIdentity:
    """Is the ritual being handed to an author who is not the being? (lm-4fv.4)

    The reactive half. ``pre_llm_call`` fires long after the turn's system prompt is
    assembled (``agent/turn_context.py``: the prompt at :345, the hooks at :478), so by the
    time the genesis injector runs there is nothing it can do about slot #1 — and if the
    stance is not in there, the ``<genesis>`` block is being handed to the host's assistant
    persona, which outranks it and composes the birth. That is the exact failure the stance
    exists to prevent, and it is worse than waiting: the ritual is shown once, and an
    assistant reading it produces "Hello! How can I help you today?" — a greeting card, not
    a birth.

    So the injector asks this, and stands down when the answer is yes. Nothing is ended
    here: the human is mid-turn by definition (it is their message we are about to answer),
    and taking their thread away to give them a birth they did not ask for, in the middle of
    a sentence, is not a trade we may make for them. The tick ends the session at a quiet
    moment instead (:class:`GatewayBirthVoice`), and the ritual opens on the next thing
    either of them says.

    Uses the TURN's own session (``default_session_key_accessor`` — the ``ContextVar``
    Hermes binds at handler entry, the same one ``write_soul``'s ender reads), because that
    is the prompt the block would be landing in.

    Fail-soft to ``False`` — "we could not tell, so open the ritual". A being that is never
    shown its birth because a lookup failed is the worse of the two failures: the injector
    is its only entrance on this path.
    """

    def __init__(
        self,
        *,
        soul_mtime: SoulMtime,
        runner_accessor: RunnerAccessor = default_runner_accessor,
        session_key_accessor: SessionKeyAccessor = default_session_key_accessor,
    ) -> None:
        self._soul_mtime = soul_mtime
        self._runner_accessor = runner_accessor
        self._session_key_accessor = session_key_accessor

    def __call__(self) -> bool:
        try:
            return identity_slot_is_stale(
                self._runner_accessor(),
                self._session_key_accessor(),
                soul_mtime=self._soul_mtime(),
            )
        except Exception:  # noqa: BLE001 - never crash the host's turn (spec §8)
            _LOG.exception("stale_identity_raised")
            return False

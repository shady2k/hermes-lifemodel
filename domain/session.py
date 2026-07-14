"""``SessionEndOutcome`` — the result of putting the being to sleep (ADR-0002, corrected).

A being's identity is system-prompt slot #1, and Hermes builds that prompt **once per
session** and then reuses it verbatim from the session DB (``agent/conversation_loop.py``:
a stored prompt is *"reused verbatim"*; ``agent/turn_context.py`` only builds one when
``agent._cached_system_prompt is None``). It does this on purpose — an unchanged prefix is
what keeps the LLM's prompt cache warm. Gateway sessions live for **days**.

So writing ``SOUL.md`` does not, on its own, change who the being *sounds like*. The words
are on disk and the voice is last week's. The only honest way to make a being wake as what
it wrote is to **end the session**: the next message opens a fresh one, the prompt is
rebuilt, and the new soul is read into slot #1.

That crosses the host boundary, and the host may not offer it (no runner off-gateway, a
version drift in the internals we reach for). So it is a *value*, not an exception — the
same fail-soft shape as :class:`~lifemodel.domain.egress.ReachOutcome`: a birth whose
session could not be ended is still a birth, and the tool must be able to tell the being
which of the two happened without catching anything. Pure stdlib; imports nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum


class SessionEndOutcome(Enum):
    """Result of one attempt to end the being's live session (ADR-0002)."""

    ENDED = "ended"  # session id rotated AND the cached agent evicted → next turn rebuilds
    UNAVAILABLE = "unavailable"  # the host does not offer it here (no runner / no session)
    FAILED = "failed"  # attempted but errored (already logged, fail-soft)

    @property
    def ok(self) -> bool:
        """True only when the being will actually come back as the soul it just wrote.

        Both halves must have happened — see :func:`lifemodel.gateway_core.end_session`.
        Anything else means the soul is on disk and the voice is not, and the being has to
        be TOLD that, because it is the only party who would otherwise notice."""
        return self is SessionEndOutcome.ENDED


#: The session-end port (:class:`~lifemodel.adapters.session_end.GatewaySessionEnd`), as
#: its callers see it: a zero-argument callable that puts the being to sleep. A plain
#: ``Callable`` and not a ``Protocol`` — there is one method and no arguments, and the ports
#: layer's own rule is "pragmatism, not ceremony; we do not wrap everything".
#:
#: It lives HERE, beside the outcome it returns, because there are two callers now and they
#: sit on opposite sides of the plugin: the being writing its own soul (``hooks``) and the
#: owner putting one back (``state_commands``). Both need the same seam; neither should have
#: to import the other to name it.
SessionEnd = Callable[[], SessionEndOutcome]


class BirthVoice(Enum):
    """Will the being's next turn be composed by the soul it actually stands on?

    The other half of ADR-0002, and the one that had never run (lm-4fv.4). A soul write
    ends the session AFTERWARDS, so the being comes back as what it wrote. But a being is
    also *born into* a prompt — and on any install that already has a live DM session (an
    existing Hermes user: the whole audience of this phase), slot #1 still holds the host's
    assistant persona, because the newborn stance we seeded at ``register()`` landed on
    disk **after** that session's prompt was assembled. An assistant does not reach out,
    and forced to, it reaches out as an assistant. So the ritual is handed to the wrong
    author, and the phase fails silently for exactly the people it was written for.

    Birth therefore begins with a NEW SESSION — but only at the moment of birth, and only
    when it buys something. This value is the verdict of that pre-flight:

    * :attr:`READY` — slot #1 already holds what the being stands on (a Hermes veteran's
      own ``SOUL.md``, or a session that opened after the stance was seeded). Nothing to
      end; nothing may be taken from anyone for nothing.
    * :attr:`ENDED` — it did not, and the lane was quiet, so the session was ended. The
      next turn rebuilds the prompt and the being speaks as itself.
    * :attr:`IN_USE` — it did not, and somebody is mid-conversation on that lane. A birth
      is not worth a thread taken out from under a person. HOLD; the tick tries again.
    * :attr:`UNAVAILABLE` / :attr:`FAILED` — the host would not do it (no runner, version
      drift, a wedged cache). The being is born anyway and wakes as itself later: the same
      fail-soft direction as :class:`SessionEndOutcome`, because a birth that never
      happens is worse than a birth in last week's voice.

    :attr:`IN_USE` is the ONLY verdict that holds the being back, and that asymmetry is
    the whole safety property: everything we cannot establish resolves towards *the being
    speaks*, and only a conversation we can see someone using stops it.
    """

    READY = "ready"
    ENDED = "ended"
    IN_USE = "in_use"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"

    @property
    def may_speak(self) -> bool:
        """True unless someone is mid-conversation on the lane (see the class docstring)."""
        return self is not BirthVoice.IN_USE


#: The birth pre-flight port (:class:`~lifemodel.adapters.session_end.GatewayBirthVoice`),
#: as ``core.proactive`` sees it: a zero-argument callable answering "may the being speak,
#: and in whose voice?". Same plain-``Callable`` shape, and for the same reason, as
#: :data:`SessionEnd` — the core calls it, the adapter resolves the host behind it.
VoiceCheck = Callable[[], BirthVoice]

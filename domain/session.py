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

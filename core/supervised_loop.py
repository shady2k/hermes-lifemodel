"""``SupervisedLoop`` — a periodic loop that reports its own death (Hermes-free).

The being's autonomic brain must tick forever, but a self-spawned asyncio task
that dies (an unhandled exception) is invisible to its host — that was the exact
cause of the silent proactivity outage. This loop closes that gap: if the tick
raises, the loop stops and calls an injected ``on_death`` callback exactly once,
so the Hermes boundary (the platform adapter) can turn that into a
``_set_fatal_error(retryable=True)`` + ``_notify_fatal_error()`` and let the
gateway's reconnect watcher restart it.

Clean shutdown (:meth:`stop` or task cancellation) is NOT a death — ``on_death``
only fires on an unexpected tick failure. Stdlib only; imports no Hermes, so the
loop's lifecycle logic unit-tests off-host with a fake ``sleep``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

Tick = Callable[[], None]
OnDeath = Callable[[BaseException | None], None]
Sleep = Callable[[float], Awaitable[None]]


class SupervisedLoop:
    """Run *tick* every *interval_sec* until stopped, cancelled, or it raises.

    On a tick exception the loop reports it via *on_death* (once) and returns.
    :meth:`stop` and cancellation exit cleanly without reporting a death.
    """

    def __init__(
        self,
        *,
        tick: Tick,
        interval_sec: float,
        on_death: OnDeath,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._tick = tick
        self._interval = interval_sec
        self._on_death = on_death
        self._sleep = sleep
        self._alive = True
        self._death_reported = False

    async def run(self) -> None:
        """Loop ``tick()`` + ``sleep(interval)`` until stop/cancel/exception."""
        try:
            while self._alive:
                self._tick()
                if not self._alive:  # stop() called from within the tick
                    break
                await self._sleep(self._interval)
        except asyncio.CancelledError:
            raise  # clean shutdown — never a death
        except BaseException as exc:  # noqa: BLE001 - any tick failure is a reportable death
            self._report_death(exc)

    def stop(self) -> None:
        """Request a clean exit; the loop finishes without reporting a death."""
        self._alive = False

    def _report_death(self, exc: BaseException | None) -> None:
        if self._death_reported:
            return
        self._death_reported = True
        self._on_death(exc)

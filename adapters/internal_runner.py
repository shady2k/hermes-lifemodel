"""``InternalCognitionRunner`` — the adapter-owned, gateway-loop task lifecycle
for the non-delivered internal-cognition seam (lm-705.6, design §3.1).

``LaunchProactive`` is not a general async-job mechanism — it schedules a native
Hermes turn, and Hermes owns that turn's background execution + finalizer→
``post_llm``. A direct aux call bypasses all of that, so WE own the
orchestration here: create and retain an asyncio task per launch, await the
:class:`~lifemodel.core.llm_port.LlmPort` call OFF the state-actor lock, handle
timeout/cancel/exception as a typed failure, and on completion run the
completion frame (:func:`~lifemodel.core.internal_cognition.run_internal_completion`)
which clears :attr:`~lifemodel.state.model.State.pending_internal_id` no matter
how the call ended.

Built by :func:`lifemodel.adapters.being_platform.BeingAdapter.connect` (Task 7),
on the SAME asyncio loop the gateway drives the brain loop on — passed in
explicitly (``gateway_loop``) rather than discovered ambiently, so this class
stays unit-testable with a bare event loop and no gateway runner at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping

from ..composition import LifeModel
from ..core.budget import reserve_internal_call
from ..core.component import Component
from ..core.frame import state_actor_lock
from ..core.intents import UpdateState
from ..core.internal_cognition import run_internal_completion
from ..core.llm_port import InternalCognitionRequest, InternalCognitionResult, LlmPort
from ..core.timeutil import to_iso
from ..domain.session import VoiceCheck
from ..ports.proactive import ProactiveEgressPort

_LOG = logging.getLogger("lifemodel.internal_runner")

#: Default aux-call timeout (design §3.1 — "handles timeout"). A cheap side-model
#: pass has no business running longer than this; the runner treats an overrun
#: exactly like any other call failure (typed empty result, pending still clears).
DEFAULT_TIMEOUT_SECONDS = 30.0


class InternalCognitionRunner:
    """Launch, track, and complete non-delivered internal-cognition aux calls."""

    def __init__(
        self,
        build_lm: Callable[[], LifeModel],
        llm: LlmPort,
        egress: ProactiveEgressPort,
        target: Mapping[str, str | None],
        *,
        daily_ceiling: int,
        gateway_loop: asyncio.AbstractEventLoop,
        apply: Component,
        voice: VoiceCheck | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._build_lm = build_lm
        self._llm = llm
        self._egress = egress
        self._target = target
        self._daily_ceiling = daily_ceiling
        self._gateway_loop = gateway_loop
        self._apply = apply
        self._voice = voice
        self._timeout = timeout
        #: The tracked task set (design §3.1) — retained so a launched task is
        #: never garbage-collected mid-flight, and so :meth:`cancel_all` can find
        #: every in-flight call at shutdown.
        self._tasks: set[asyncio.Task[None]] = set()

    def launch(
        self,
        request: InternalCognitionRequest,
        correlation_id: str,
        *,
        subject_id: str | None = None,
    ) -> bool:
        """Single-flight gate + reserve FR20 + set ``pending_internal_id``/subject/
        interval (one frame, under the lock); on success, create + retain the async
        task that runs the aux call OFF the lock.

        Returns ``False`` (no task created, budget untouched) when EITHER an
        internal pass is already in flight (``pending_internal_id`` is already set
        — single-flight, prereq #2) OR the daily ceiling denies the reservation.
        Single-flight is checked FIRST, before ``reserve_internal_call`` — a
        denied-by-single-flight launch must never consume a day's budget slot for
        a call that never runs."""
        lm = self._build_lm()
        assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
        now = lm.clock.now()
        with state_actor_lock():
            state = lm.state_actor.state
            if state.pending_internal_id is not None:
                return False  # single-flight: an internal pass is already in flight
            reserved = reserve_internal_call(state, now=now, daily_ceiling=self._daily_ceiling)
            if reserved is None:
                return False
            lm.state_actor.apply(
                [
                    UpdateState(
                        {
                            "internal_calls_today": reserved.internal_calls_today,
                            "internal_calls_day": reserved.internal_calls_day,
                            "pending_internal_id": correlation_id,
                            "pending_internal_subject_id": subject_id,
                            "last_internal_call_at": to_iso(now),
                        }
                    )
                ]
            )
        task = self._gateway_loop.create_task(self._run(request, correlation_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run(self, request: InternalCognitionRequest, correlation_id: str) -> None:
        """Await the aux call OFF the lock, then run the completion frame.

        Fail-loud but never crash the loop (design §3.1): a plain exception from
        the aux call is mapped to an empty result INLINE (the completion frame
        still runs, so ``apply``/launch-dispatch/pending-clear all happen
        normally); a raise from the completion frame ITSELF is caught by the
        outer guard, which explicitly clears ``pending_internal_id`` so a bug in
        ``apply`` can never strand every future internal launch. Cancellation
        (``cancel_all`` at disconnect) is left to propagate — the state is
        touched only by :meth:`~lifemodel.adapters.being_platform.BeingAdapter`'s
        next ``recover_stale`` call, never mid-shutdown.
        """
        try:
            result = await self._call(request, correlation_id)
            lm = self._build_lm()
            run_internal_completion(
                lm,
                self._egress,
                self._target,
                correlation_id=correlation_id,
                result=result,
                apply=self._apply,
                voice=self._voice,
            )
        except asyncio.CancelledError:
            raise  # clean shutdown — recover_stale cleans up the pending marker at next connect
        except Exception as exc:  # noqa: BLE001 - fail-loud last-resort guard, never crash the loop
            _LOG.exception(
                "internal_cognition_run_failed correlation_id=%s error=%r", correlation_id, exc
            )
            self._clear_pending_fail_loud(correlation_id)

    async def _call(
        self, request: InternalCognitionRequest, correlation_id: str
    ) -> InternalCognitionResult:
        """Await the aux call OFF the lock; map timeout/exception to an empty result."""
        try:
            return await asyncio.wait_for(
                self._llm.complete_structured(request), timeout=self._timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a call failure is a typed empty result, not a crash
            _LOG.warning(
                "internal_cognition_call_failed correlation_id=%s error=%r", correlation_id, exc
            )
            return InternalCognitionResult(raw="", parsed=None)

    def _clear_pending_fail_loud(self, correlation_id: str) -> None:
        """Last-resort pending clear when the completion frame itself raised.

        Best-effort: if even this fails (e.g. the store is down), the pending
        marker is left for :meth:`recover_stale` to clean up at the next connect
        — never a second exception escaping the task."""
        try:
            lm = self._build_lm()
            assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
            with state_actor_lock():
                lm.state_actor.apply(
                    [
                        UpdateState(
                            {"pending_internal_id": None, "pending_internal_subject_id": None}
                        )
                    ]
                )
        except Exception as exc:  # noqa: BLE001 - recover_stale is the final backstop
            _LOG.exception(
                "internal_cognition_pending_clear_failed correlation_id=%s error=%r",
                correlation_id,
                exc,
            )

    def recover_stale(self, lm: LifeModel) -> None:
        """At connect: clear a leftover ``pending_internal_id`` from a task that
        died with the previous process (this runner is freshly constructed, so
        it tracks no live task — "stale" is simply "still set")."""
        assert lm.state_actor is not None, "state_actor must be wired by build_lifemodel"
        if lm.state_actor.state.pending_internal_id is None:
            return
        with state_actor_lock():
            lm.state_actor.apply(
                [UpdateState({"pending_internal_id": None, "pending_internal_subject_id": None})]
            )

    async def cancel_all(self) -> None:
        """Cancel + await every tracked task (called from ``disconnect``)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

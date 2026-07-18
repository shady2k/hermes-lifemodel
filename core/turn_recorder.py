"""``TurnRecorder`` — a turn is a first-class traced unit (lm-hg7).

Process-lifetime service (built once in ``register()``) that makes a Hermes turn
observable the way the tick already is: a per-turn ROOT span + a CHILD span per
``pre_llm_call`` injector and per tool, into the SAME ``observability.sqlite`` as
the tick, through the already-acquired live :class:`~lifemodel.state.trace_store.TraceWriter`.

It is deliberately NOT an :class:`~lifemodel.core.frame.ExecutionFrame`: a turn is
an asynchronous observability scope spanning host work across threads, so this
never takes the state-actor lock and never touches ``State``. It only writes to the
trace sink + metric registry + its own small in-memory ledger.

Every public method is fail-soft (a tracing hiccup must never crash the host turn).
The ledger is keyed by the host's ``(session_id, turn_id)`` and bounded (TTL + max
entries, lazy cleanup) — a restart simply loses it, and an open root left by a
crash/interrupt ages out or is reconciled ``failed`` by the next turn of the session.

This task adds :meth:`TurnRecorder.injector_span` (a per-``pre_llm_call``-injector
CHILD span + the ``turn_injector_total`` metric) on top of Task 3's construction +
:meth:`TurnRecorder.ensure_turn`; a later task adds per-tool spans and ``close_turn``
to this SAME class.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..ports.clock import ClockPort
from ..ports.tracer import TraceContext, TracerPort
from ..state.trace_store import TraceSink
from .metrics import MetricRegistry
from .timeutil import to_iso
from .turn_metrics import TURN_INJECTOR_TOTAL, register_turn_metrics

_LOG = logging.getLogger("lifemodel.turn_recorder")

#: The ``trace_spans.component`` value for a turn's ROOT span (spec §4.3 shape,
#: mirroring the tick's own root component naming).
_TURN_ROOT_COMPONENT = "turn"


class InjectorSpan:
    """The mutable handle an injector stamps its verdict onto (set-then-close).

    Deliberately NOT the tracer's :class:`~lifemodel.ports.tracer.ActiveSpan` (no
    ``context``/``tick``/``end`` surface to satisfy) — an injector only ever needs
    to record its per-call verdict (``outcome``) plus a few closed-vocabulary
    attrs before :meth:`TurnRecorder.injector_span` closes the child span itself.
    The last :meth:`set` wins if an injector calls it more than once.
    """

    __slots__ = ("_outcome", "_attrs")

    def __init__(self) -> None:
        self._outcome: str = "unknown"
        self._attrs: dict[str, Any] = {}

    def set(self, *, outcome: str, **attrs: Any) -> None:
        """Record this call's verdict — ``outcome`` plus any extra attrs."""
        self._outcome = outcome
        self._attrs.update(attrs)


@dataclass
class _Entry:
    """One open turn's ledger row — enough to reconcile or persist it later."""

    ctx: TraceContext
    opened_at_iso: str
    opened_mono: float
    session_id: str
    turn_id: str


class TurnRecorder:
    """Opens/reconciles/bounds per-turn trace roots + per-injector child spans; a
    later task adds per-tool spans + ``close_turn``.

    Constructed once (in ``register()``) over the live being's already-acquired
    :class:`~lifemodel.state.trace_store.TraceWriter`,
    :class:`~lifemodel.core.metrics.MetricRegistry`, a
    :class:`~lifemodel.ports.tracer.TracerPort` and a
    :class:`~lifemodel.ports.clock.ClockPort` — the SAME instances the tick
    already uses, so a turn's spans land in the SAME ``observability.sqlite`` file.
    """

    def __init__(
        self,
        *,
        tracer: TracerPort,
        writer: TraceSink,
        metrics: MetricRegistry,
        clock: ClockPort,
        ledger_ttl_s: float = 900.0,
        max_entries: int = 256,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tracer = tracer
        self._writer = writer
        self._metrics = metrics
        self._clock = clock
        self._ttl_s = ledger_ttl_s
        self._max = max_entries
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._ledger: dict[tuple[str, str], _Entry] = {}
        # Declare the turn metric into the shared registry (idempotent) so a later
        # injector close lands on a real metric rather than failing open as unknown.
        try:
            register_turn_metrics(self._metrics)
        except Exception:  # noqa: BLE001 - a metric-declare hiccup must not sink the recorder
            _LOG.exception("turn metric registration failed")

    # --- the open door ------------------------------------------------------ #
    def ensure_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        model: str = "",
        platform: str = "",
        origin: str = "reactive",
        upstream_traceparent: str | None = None,
    ) -> None:
        """Idempotently open the ROOT span for ``(session_id, turn_id)``.

        The FIRST caller for a key (an injector or a tool, whichever runs first
        this turn) mints the root and persists it open (``ended_at``/``status``
        both ``None``) so a crash mid-turn still leaves a discoverable parent;
        every later call for the SAME key is a no-op. Before minting, any OTHER
        still-open turn of this session is reconciled to ``failed``/``abandoned``
        (``post_llm_call`` is not a guaranteed close) and the ledger is bounded.

        ``origin="reactive"`` mints a brand-new trace; ``origin="proactive"`` with
        an ``upstream_traceparent`` CONTINUES that parsed trace (a reach-in turn
        joins the caller's trace). Never raises — a tracing hiccup, including a
        broken sink, must never crash the host turn.
        """
        try:
            key = (session_id, turn_id)
            now_iso = to_iso(self._clock.now())
            with self._lock:
                if key in self._ledger:  # idempotent — the first injector already opened it
                    return
                self._reconcile_session_locked(session_id, now_iso)
                self._evict_locked()
                ctx = self._tracer.start_root(upstream_traceparent=upstream_traceparent)
                self._ledger[key] = _Entry(ctx, now_iso, self._monotonic(), session_id, turn_id)
            # Persist OUTSIDE the lock (the sink is thread-safe + async): eager, with
            # ended_at/status NULL, so a crash still leaves a discoverable parent.
            self._submit(
                ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=now_iso,
                ended_at=None,
                status=None,
                attrs={
                    "frame_kind": "turn",
                    "turn_id": turn_id,
                    "session_id": session_id,
                    "origin": origin,
                    "model": model,
                    "platform": platform,
                },
            )
        except Exception:  # noqa: BLE001 - opening a turn trace must never crash the host turn
            _LOG.exception("ensure_turn failed session=%s turn=%s", session_id, turn_id)

    # --- the per-injector door ----------------------------------------------- #
    @contextmanager
    def injector_span(
        self, session_id: str, turn_id: str, component: str
    ) -> Iterator[InjectorSpan]:
        """Wrap one ``pre_llm_call`` injector's run in a CHILD span of the turn root.

        Yields an :class:`InjectorSpan` the injector stamps its verdict onto with
        :meth:`~InjectorSpan.set` before returning. On a clean exit the child is
        persisted ``status="ok"`` and :data:`~lifemodel.core.turn_metrics.TURN_INJECTOR_TOTAL`
        is incremented with ``outcome`` (``"unknown"`` if the injector never called
        :meth:`~InjectorSpan.set`). On a body exception the child is persisted
        ``status="failed"``/``outcome="error"`` instead, and the exception is
        RE-RAISED (this context manager never swallows it — the injector's own
        fail-soft ``except`` around its call site is what actually contains it).

        If ``(session_id, turn_id)`` has no open turn (``ensure_turn`` was never
        called, or the ledger already evicted it), the span degrades to a bare
        parentless root rather than raising — best-effort tracing, never a crash.
        """
        parent = self._ledger_ctx(session_id, turn_id)
        child = self._child_span_id(parent) if parent is not None else self._tracer.start_root()
        span = InjectorSpan()
        started = to_iso(self._clock.now())
        comp = f"turn.injector.{component}"
        try:
            yield span
        except Exception:
            self._close_injector(child, comp, started, status="failed", outcome="error", span=span)
            raise  # re-propagate so the injector's own fail-soft except runs
        else:
            self._close_injector(
                child, comp, started, status="ok", outcome=span._outcome, span=span
            )

    # --- internals ---------------------------------------------------------- #
    def _reconcile_session_locked(self, session_id: str, now_iso: str) -> None:
        """Close any OTHER still-open turn of this session as failed/abandoned — the
        next turn is the only reliable signal a prior one died (post_llm_call is not
        guaranteed). Caller holds the lock; the submit is best-effort after."""
        dead = [e for (sid, _), e in self._ledger.items() if sid == session_id]
        for entry in dead:
            self._ledger.pop((entry.session_id, entry.turn_id), None)
            self._submit(
                entry.ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=entry.opened_at_iso,
                ended_at=now_iso,
                status="failed",
                attrs={"frame_kind": "turn", "turn_id": entry.turn_id, "outcome": "abandoned"},
            )

    def _evict_locked(self) -> None:
        # TTL first (age out silent leaks), then cap the map. Lazy — no sweeper thread.
        cutoff = self._monotonic() - self._ttl_s
        for key in [k for k, e in self._ledger.items() if e.opened_mono < cutoff]:
            self._ledger.pop(key, None)
        while len(self._ledger) > self._max:
            oldest = min(self._ledger, key=lambda k: self._ledger[k].opened_mono)
            self._ledger.pop(oldest, None)

    def _submit(self, ctx: TraceContext, **kw: Any) -> None:
        try:
            self._writer.submit_span(
                trace_id=ctx.trace_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                tick=None,
                **kw,
            )
        except Exception:  # noqa: BLE001 - fail-open, exactly like the tick's _persist_span
            _LOG.exception("submit_span failed")

    def _ledger_ctx(self, session_id: str, turn_id: str) -> TraceContext | None:
        """The open turn's root :class:`TraceContext` for this key, or ``None`` if
        no turn is currently open (never opened, already closed, or evicted)."""
        with self._lock:
            entry = self._ledger.get((session_id, turn_id))
            return entry.ctx if entry is not None else None

    def _child_span_id(self, parent: TraceContext) -> TraceContext:
        return self._tracer.child_of(parent)

    def _close_injector(
        self,
        ctx: TraceContext,
        comp: str,
        started: str,
        *,
        status: str,
        outcome: str,
        span: InjectorSpan,
    ) -> None:
        """Persist the injector's child span + emit the shared outcome counter.

        Wrapped so a tracing/metric hiccup on CLOSE never masks or replaces the
        body's own exception — :meth:`injector_span` already re-raises around
        this call.
        """
        try:
            self._submit(
                ctx,
                component=comp,
                started_at=started,
                ended_at=to_iso(self._clock.now()),
                status=status,
                attrs={"outcome": outcome, **span._attrs},
            )
            self._metrics.inc(
                TURN_INJECTOR_TOTAL, component=comp.rsplit(".", 1)[-1], outcome=outcome
            )
        except Exception:  # noqa: BLE001 - never let the tracing close mask/replace the body
            _LOG.exception("injector span close failed component=%s", comp)

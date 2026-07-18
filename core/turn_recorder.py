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

Task 4 added :meth:`TurnRecorder.injector_span` (a per-``pre_llm_call``-injector
CHILD span + the ``turn_injector_total`` metric) on top of Task 3's construction +
:meth:`TurnRecorder.ensure_turn`. Task 5 added :meth:`TurnRecorder.tool_open` /
:meth:`TurnRecorder.tool_close` (per-tool CHILD spans keyed by ``tool_call_id``).
This task adds :meth:`TurnRecorder.close_turn` — a ``turn.completion`` CHILD span
plus the root span's terminal close — completing the service.

Task 13's real-code end-to-end harness (``tests/test_turn_observability_harness.py``)
caught a real composition gap the per-method unit tests never exercised: the store's
``submit_span`` upserts ``attrs_json`` wholesale (no partial merge), so
:meth:`close_turn`'s own attrs dict was silently erasing the
``origin``/``model``/``platform`` :meth:`ensure_turn` had already persisted at open —
every turn's closed root would read back with no origin at all. :class:`_Entry` now
carries those three forward so the closing/reconciling writes re-emit them.
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

#: Hard cap on any free-text attr value persisted onto a span (``final_output``) —
#: a turn's text is host-supplied and unbounded; the trace store is not a
#: transcript archive, so it is sliced rather than stored whole.
_MAX_TEXT = 4000


def _map_host_status(status: str) -> str:
    """Map a Hermes-supplied status string onto the recorder's own closed span
    vocabulary (:data:`~lifemodel.ports.tracer.SpanStatus` — ``ok``/``suppressed``/
    ``failed``) — FAIL-CLOSED: an unrecognized value NEVER reads back ``"ok"`` (a
    failed/blocked tool call, or a bug in the caller, must never look like success).

    Hermes' own ``post_tool_call`` vocabulary (``agent/shell_hooks.py``: ``status
    "ok" | "error" | "blocked"``) maps ``"ok"`` → ``"ok"``, ``"blocked"`` (a plugin
    veto — a deliberate no-act) → ``"suppressed"``, ``"error"`` → ``"failed"``.
    Anything else — an unrecognized host value, or a caller passing some OTHER
    string entirely — also maps to ``"failed"`` rather than being trusted verbatim;
    the raw value is preserved separately (``host_status`` attr) so it is never
    actually lost, just never allowed to read back as a false ``"ok"``.
    """
    if status == "ok":
        return "ok"
    if status == "blocked":
        return "suppressed"
    return "failed"


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
    """One open turn's ledger row — enough to reconcile or persist it later.

    ``origin``/``model``/``platform`` are the OPEN-time attrs :meth:`ensure_turn`
    already persisted onto the root span — stashed here too so a later
    ``close_turn``/reconcile can carry them forward. ``submit_span`` UPSERTS by
    ``(trace_id, span_id)`` and overwrites ``attrs_json`` wholesale on every
    call (there is no partial-attrs merge at the store layer), so a close that
    persisted only its OWN new attrs would silently erase the open's — the
    root's origin, otherwise unrecoverable once closed, is exactly what
    ``activity.py``'s timeline line reads back.
    """

    ctx: TraceContext
    opened_at_iso: str
    opened_mono: float
    session_id: str
    turn_id: str
    origin: str
    model: str
    platform: str


class TurnRecorder:
    """Opens/reconciles/bounds per-turn trace roots, per-injector child spans,
    per-tool child spans (keyed by ``tool_call_id``), and the turn's own close
    (a ``turn.completion`` child + the root span's terminal status).

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
        self._tools: dict[str, tuple[TraceContext, str, str]] = {}
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

        The open write is submitted (enqueued to the writer) BEFORE the ledger
        entry is inserted, and BOTH happen under the SAME lock :meth:`close_turn`
        acquires just to pop it (M1 fix): a racing ``close_turn`` can only see this
        key once the open write is already enqueued, and the writer applies its
        single queue strictly FIFO (state/trace_store.py) — so the open write can
        never land at the sink AFTER a subsequent close and stomp its
        ``ended_at``/``status`` back to open. The writer's own ``submit_span`` is
        a non-blocking ``put_nowait``, so holding the lock across it never risks
        a stall.
        """
        try:
            key = (session_id, turn_id)
            now_iso = to_iso(self._clock.now())
            with self._lock:
                if key in self._ledger:  # idempotent — the first injector already opened it
                    return
                self._reconcile_session_locked(session_id, now_iso)
                ctx = self._tracer.start_root(upstream_traceparent=upstream_traceparent)
                # Persist BEFORE the ledger entry becomes visible (M1 fix) — eager,
                # with ended_at/status NULL, so a crash still leaves a discoverable
                # parent.
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
                self._ledger[key] = _Entry(
                    ctx, now_iso, self._monotonic(), session_id, turn_id, origin, model, platform
                )
                # Evict AFTER inserting (M4 fix): the post-insert ledger size is
                # bounded to max_entries, never max_entries + 1.
                self._evict_locked()
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

        The SETUP (ledger lookup, ``child_of``/``start_root``, the clock read) is
        itself wrapped in its own ``try`` (I4 fix): if the tracer/clock/ledger
        blows up here, this degrades to a NO-OP span — the injector's body still
        runs and returns its ordinary result — rather than letting the setup
        failure propagate up and be mistaken for a BODY failure by the injector's
        own fail-soft ``except``, which would silently suppress the whole
        injection (felt/genesis/belief/commitment) on nothing worse than a broken
        tracer. Only the body's OWN exception (raised after a successful setup)
        is persisted ``failed``/``error`` and re-raised.
        """
        try:
            parent = self._ledger_ctx(session_id, turn_id)
            child = self._child_span_id(parent) if parent is not None else self._tracer.start_root()
            started = to_iso(self._clock.now())
        except Exception:  # noqa: BLE001 - a broken tracer must never suppress the injection
            _LOG.exception("injector_span setup failed component=%s", component)
            yield InjectorSpan()  # no-op: the body still runs; nothing gets persisted
            return
        span = InjectorSpan()
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

    # --- the per-tool door ---------------------------------------------------- #
    def tool_open(self, session_id: str, turn_id: str, *, tool: str, tool_call_id: str) -> None:
        """Stash an open ``turn.tool.<tool>`` child, keyed by ``tool_call_id``.

        Tool calls can run concurrently within a turn, so — unlike the single
        per-turn root — spans are tracked in a small separate map keyed by the
        host's own call id rather than ``(session_id, turn_id)``. Degrades to a
        bare parentless root if no turn is open for this key. Never raises — a
        tracing hiccup must never crash the host tool call.
        """
        try:
            parent = self._ledger_ctx(session_id, turn_id)
            child = self._child_span_id(parent) if parent is not None else self._tracer.start_root()
            with self._lock:
                self._tools[tool_call_id] = (child, tool, to_iso(self._clock.now()))
        except Exception:  # noqa: BLE001 - opening a tool span must never crash the host call
            _LOG.exception("tool_open failed tool=%s", tool)

    def tool_close(self, tool_call_id: str, *, status: str = "ok", **attrs: Any) -> None:
        """Persist the ``turn.tool.<tool>`` child opened by :meth:`tool_open` and
        drop it from the ledger. An unknown ``tool_call_id`` (never opened, or
        already closed) is a best-effort no-op — never raises.

        *status* is the RAW host value (``make_tool_span_close`` passes Hermes'
        own ``post_tool_call`` status straight through: ``"ok"``/``"error"``/
        ``"blocked"``) and is mapped through :func:`_map_host_status` onto the
        span's closed vocabulary — FAIL-CLOSED (I1 fix): an unrecognized status
        NEVER persists as ``"ok"``, so a failed or blocked tool can no longer
        read back as a success. The raw value survives as the ``host_status``
        attr regardless of the mapped outcome.

        ``host_status`` is placed AFTER ``**attrs`` (C-M1 fix) so the real raw
        status always wins: a caller-supplied ``attrs`` dict happening to carry
        its own ``host_status`` key (e.g. echoed from the tool's own return
        payload) can no longer clobber the one this method is actually
        recording.
        """
        try:
            with self._lock:
                found = self._tools.pop(tool_call_id, None)
            if found is None:
                return
            child, tool, started = found
            self._submit(
                child,
                component=f"turn.tool.{tool}",
                started_at=started,
                ended_at=to_iso(self._clock.now()),
                status=_map_host_status(status),
                attrs={**attrs, "host_status": status},
            )
        except Exception:  # noqa: BLE001 - closing a tool span must never crash the host call
            _LOG.exception("tool_close failed id=%s", tool_call_id)

    # --- the close door ------------------------------------------------------- #
    def close_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        final_output: str = "",
        status: str = "ok",
        model: str = "",
        platform: str = "",
    ) -> None:
        """Close ``(session_id, turn_id)`` — a ``turn.completion`` child persisting
        the bounded final text, then the root span itself (``ended_at``=now,
        terminal ``status``, carrying forward the ``origin``/``model``/``platform``
        :meth:`ensure_turn` stamped at open — the store's ``submit_span`` upserts
        ``attrs_json`` wholesale, so re-emitting them here is what stops the close
        from erasing what open already persisted) — and drop the ledger entry.

        ``model``/``platform`` (M5 fix): ``pre_llm_call`` (where :meth:`ensure_turn`
        runs) does NOT reliably carry them, but ``post_llm_call`` (where this runs)
        ALWAYS does — verified against ``agent/turn_finalizer.py``'s
        ``invoke_hook("post_llm_call", ..., model=agent.model, platform=...)``. A
        non-empty value passed here WINS over whatever :meth:`ensure_turn` stashed
        (which is ``""`` on every real reactive turn today); an empty one (the
        default, e.g. a caller that never learned them) falls back to the entry's
        own stashed value, so a close never REGRESSES a value open already had.

        An unknown key (never opened via :meth:`ensure_turn`, or already closed by
        an earlier call) is a best-effort no-op: the ledger pop returns ``None``
        and this returns immediately — idempotent, so a duplicate close (e.g. a
        retried ``post_llm_call``) never double-closes the root or raises. *status*
        is mapped through :func:`_map_host_status` — FAIL-CLOSED (I1 fix): an
        out-of-vocabulary value now persists ``"failed"``, never ``"ok"``. The raw
        pre-map value is ALSO kept as the root's own ``host_status`` attr (C-M1
        fix — :meth:`tool_close` already did this; the root close used to map-and-
        discard it, so a turn closed on an out-of-vocabulary host status had no
        way to recover what that raw value actually was). Never raises — a
        tracing hiccup on the way out must never crash the host turn.
        """
        try:
            key = (session_id, turn_id)
            with self._lock:
                entry = self._ledger.pop(key, None)
            if entry is None:
                return
            child = self._child_span_id(entry.ctx)
            now_iso = to_iso(self._clock.now())
            self._submit(
                child,
                component="turn.completion",
                started_at=now_iso,
                ended_at=now_iso,
                status="ok",
                attrs={"final_output": final_output[:_MAX_TEXT]},
            )
            self._submit(
                entry.ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=entry.opened_at_iso,
                ended_at=now_iso,
                status=_map_host_status(status),
                attrs={
                    "frame_kind": "turn",
                    "turn_id": turn_id,
                    "session_id": session_id,
                    "origin": entry.origin,
                    "model": model or entry.model,
                    "platform": platform or entry.platform,
                    "host_status": status,
                },
            )
        except Exception:  # noqa: BLE001 - closing a turn must never crash the host turn
            _LOG.exception("close_turn failed session=%s turn=%s", session_id, turn_id)

    # --- internals ---------------------------------------------------------- #
    def _reconcile_session_locked(self, session_id: str, now_iso: str) -> None:
        """Close any OTHER still-open turn of this session as failed/abandoned — the
        next turn is the only reliable signal a prior one died (post_llm_call is not
        guaranteed). Caller holds the lock; the submit is best-effort after.

        Re-emits the FULL opening attr set (I2 fix) — ``session_id``/``model``/
        ``platform`` alongside ``frame_kind``/``turn_id``/``origin`` — not just the
        four this used to send: the store's ``submit_span`` upserts ``attrs_json``
        WHOLESALE (no partial merge), so a reconcile that persisted only its own
        new keys would silently ERASE what :meth:`ensure_turn` already wrote,
        leaving an abandoned root unreadable by ``session_id``/``model``/
        ``platform`` forever (:class:`_Entry` already stashes all three for
        exactly this reason).
        """
        dead = [e for (sid, _), e in self._ledger.items() if sid == session_id]
        for entry in dead:
            self._ledger.pop((entry.session_id, entry.turn_id), None)
            self._submit(
                entry.ctx,
                component=_TURN_ROOT_COMPONENT,
                started_at=entry.opened_at_iso,
                ended_at=now_iso,
                status="failed",
                attrs={
                    "frame_kind": "turn",
                    "turn_id": entry.turn_id,
                    "session_id": entry.session_id,
                    "origin": entry.origin,
                    "model": entry.model,
                    "platform": entry.platform,
                    "outcome": "abandoned",
                },
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

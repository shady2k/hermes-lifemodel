"""Span-bound logging for the lifemodel plugin (spec §4.1/§4.2/§4.5).

The plugin loads inside Hermes' own interpreter, so runtime logging is **stdlib
only** — no structlog/loguru (spec §v1.2: every Hermes plugin logs through bare
``logging.getLogger(__name__)`` and Hermes' ``hermes_logging.setup_logging`` owns
routing/rotation/redaction into ``agent.log``). Two, and only two, logging
surfaces exist:

1. :class:`SpanLogger` — the tick path's "no-log-without-span" main lock (§4.1).
   Every ``.info``/``.debug``/… SELF-stamps its active span's
   ``{trace_id, span_id, tick}`` (a component cannot forget to, nor reach a bare
   logger — §1.1 is inexpressible) and fans one record out to three projections
   of the SAME stream: the durable trace store (queryable source of truth), the
   in-memory :class:`~lifemodel.events.EventRing` (freshness), and the human tail
   via stdlib ``logging.getLogger("lifemodel")`` → Hermes' ``agent.log``.
2. plain ``logging.getLogger("lifemodel.<sub>")`` — for lifecycle / registration
   / boundary code with no ambient span (spec §4.5 allowlist). We only ever
   ``getLogger`` it; NEVER ``basicConfig`` or (de)register handlers — setup is
   Hermes' job.

This module owns surface 1 (:class:`SpanLogger` / :class:`SpanBoundLogger`) plus
the log-level helpers the ``/lifemodel loglevel`` command uses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from .core.timeutil import to_iso
from .state.trace_store import next_record_id

if TYPE_CHECKING:
    from .ports.tracer import ActiveSpan


def _system_now() -> datetime:
    """Fallback "now" when a :class:`SpanLogger` is built without an injected clock.

    Routes through the ONE sanctioned system-time read
    (:class:`~lifemodel.adapters.clock.SystemClock`) rather than ``datetime.now``
    directly (spec §3.1) — imported lazily so core's import graph never eagerly
    pulls the adapters package. The live tick injects its clock; only bare
    test/CLI construction falls here.
    """
    from .adapters.clock import SystemClock

    return SystemClock().now()


#: The full standard Python logging level name set, in ascending severity order.
#: Exposed for reuse by the command layer (``/lifemodel loglevel``).
LOG_LEVEL_NAMES: tuple[str, ...] = ("debug", "info", "warning", "error", "critical")

_LEVEL_BY_NAME: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

#: The human live-tail logger — native stdlib ``logging`` routed by Hermes to
#: ``agent.log`` (spec §4.2/§v1.2). We only ever ``getLogger`` it; never
#: ``basicConfig`` or (de)register handlers — setup is Hermes' job.
_HUMAN_LOGGER = logging.getLogger("lifemodel")


def parse_log_level(name: str) -> int:
    """Parse one of the 5 standard level names (case-insensitive) to its int level.

    Raises :class:`ValueError` — with the valid names listed — for anything else.
    """
    key = name.strip().lower()
    try:
        return _LEVEL_BY_NAME[key]
    except KeyError:
        valid = ", ".join(LOG_LEVEL_NAMES)
        raise ValueError(f"invalid log level {name!r}; expected one of: {valid}") from None


def log_level_name(level: int) -> str:
    """Return the canonical lowercase name for a standard level int (e.g. 20 -> 'info')."""
    return logging.getLevelName(level).lower()


def apply_log_level(level: int) -> None:
    """Apply *level* to the plugin's ``lifemodel`` logger tree (spec §4.2, codex #3).

    This is the whole of "set the log level": ``setLevel`` on the single
    ``lifemodel`` logger every submodule (``lifemodel.<sub>``) descends from, so
    it gates both the human ``agent.log`` tail and the SpanLogger's human
    projection at once. We NEVER ``basicConfig`` or add/remove handlers — Hermes
    owns handler setup and routing (spec §4.2/§v1.2).
    """
    _HUMAN_LOGGER.setLevel(level)


class SpanBoundLogger(Protocol):
    """The ONLY logger surface a tick component receives (spec §4.1).

    A span-bound logger that also exposes its :attr:`span` — the live
    :class:`~lifemodel.ports.tracer.ActiveSpan` the component drops decision
    values onto (``span.set(u=…, effective_pressure=…)``) so the persisted span is
    *self-explaining*, not just a bag of ``reason`` codes. The concrete
    :class:`SpanLogger` and the test :class:`~lifemodel.testing.fakes.FakeSpanLogger`
    both satisfy it, so :class:`~lifemodel.core.component.TickContext` depends on
    this seam rather than a concrete class. A "bare" logger cannot appear in the
    tick path — §1.1 (a log without a span) is inexpressible by type.
    """

    @property
    def span(self) -> ActiveSpan: ...
    def debug(self, event: str, **fields: Any) -> None: ...
    def info(self, event: str, **fields: Any) -> None: ...
    def warning(self, event: str, **fields: Any) -> None: ...
    def error(self, event: str, **fields: Any) -> None: ...
    def critical(self, event: str, **fields: Any) -> None: ...


class _TraceEventWriter(Protocol):
    """The slice of :class:`~lifemodel.state.trace_store.TraceWriter` SpanLogger needs.

    Non-blocking, fail-open: returns ``True`` when the record was enqueued (the
    durable write will happen on the writer thread) and ``False`` when the queue
    was full and the record was dropped — the signal SpanLogger uses to enforce
    durable-first (§4.2).
    """

    def submit_event(
        self,
        *,
        record_id: int,
        trace_id: str,
        span_id: str | None,
        tick: int | None,
        event: str,
        ts: str,
        fields: Mapping[str, Any] | None = None,
    ) -> bool: ...


class _EventRingLike(Protocol):
    """The append surface of :class:`~lifemodel.events.EventRing` SpanLogger projects onto."""

    def append(self, record: Mapping[str, Any]) -> None: ...


class SpanLogger:
    """A span-bound logger — the "no-log-without-span" main lock (§4.1).

    Every ``.info``/``.debug``/… SELF-stamps the active span's
    ``{trace_id, span_id, tick}`` (a component cannot forget to, nor reach a
    "bare" logger — §1.1 becomes inexpressible), then fans one record out to
    three projections of the SAME stream:

    1. the durable :class:`~lifemodel.state.trace_store.TraceWriter` (async — the
       queryable source of truth);
    2. the in-memory :class:`~lifemodel.events.EventRing` (freshness / read-your-writes);
    3. the human tail via stdlib ``logging.getLogger("lifemodel")`` → ``agent.log``.

    **Durable-first ordering (spec §4.2, codex fix #2).** The durable enqueue is
    attempted FIRST; the ring + human projections happen ONLY if it succeeded, so
    ``agent.log ⊆ sqlite`` even under overload. On a full queue the writer bumps
    its ``dropped_count`` and this logger writes NO projections — it never
    fabricates a line the durable store is missing.

    Level gating is delegated: the durable + ring projections capture EVERY level
    (sqlite is the complete trace, DEBUG detail included), while the human
    ``logging`` call self-filters on the ``lifemodel`` logger's own level.
    """

    def __init__(
        self,
        span: ActiveSpan,
        *,
        writer: _TraceEventWriter,
        ring: _EventRingLike,
        now: Callable[[], datetime] | None = None,
        human_logger: logging.Logger | None = None,
    ) -> None:
        self._span = span
        self._writer = writer
        self._ring = ring
        self._now = now or _system_now
        self._human = human_logger or _HUMAN_LOGGER

    @property
    def span(self) -> ActiveSpan:
        """The active span this logger is bound to (for ``.set``/``.end`` at the site)."""
        return self._span

    def _emit(self, level: int, event: str, fields: dict[str, Any]) -> None:
        ctx = self._span.context
        tick = self._span.tick
        record_id = next_record_id()
        ts = to_iso(self._now())
        enqueued = self._writer.submit_event(
            record_id=record_id,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            tick=tick,
            event=event,
            ts=ts,
            fields=fields,
        )
        if not enqueued:
            return  # dropped on a full queue — write NO projections (durable-first)
        # Stamped ids are authoritative: they go last so a stray field cannot clobber them.
        self._ring.append(
            {
                "event": event,
                **fields,
                "record_id": record_id,
                "trace_id": ctx.trace_id,
                "span_id": ctx.span_id,
                "tick": tick,
                "ts": ts,
            }
        )
        self._human.log(
            level,
            "%s trace_id=%s span_id=%s tick=%s %s",
            event,
            ctx.trace_id,
            ctx.span_id,
            tick,
            fields,
        )

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, fields)

    def critical(self, event: str, **fields: Any) -> None:
        self._emit(logging.CRITICAL, event, fields)

"""Structured logging for the lifemodel plugin.

Prefers **structlog** — JSON events over the stdlib :mod:`logging` module, so
the plugin's logs integrate with Hermes' logging. structlog is an *optional*
runtime backend: the plugin is loaded inside Hermes' own interpreter, which may
not have structlog installed. When it is absent, :func:`get_logger` returns a
small stdlib-backed shim exposing the same ``.info(event, **fields)`` surface,
so the plugin still loads and its events stay observable.

This module only wires the pipeline and exposes a logger factory; the richer
debug events (``tick``, ``wake_decision``, ``act_gate``, ``dream_run``, ...)
described in HLA §13 land in task 0.3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, cast

from .events import EventSink

try:
    import structlog

    _HAVE_STRUCTLOG = True
except ModuleNotFoundError:  # pragma: no cover - only in a host lacking structlog
    _HAVE_STRUCTLOG = False


#: The full standard Python logging level name set, in ascending severity order.
#: Exposed for reuse by the command layer (e.g. `/lifemodel` log-level commands).
LOG_LEVEL_NAMES: tuple[str, ...] = ("debug", "info", "warning", "error", "critical")

_LEVEL_BY_NAME: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

#: The plugin's current effective log level. Set by :func:`configure`; read by
#: :class:`EventTee`/:class:`_StdlibEventLogger` on every call to decide whether
#: an event is recorded at all (forwarded to the base logger *and* written to
#: the :class:`~lifemodel.events.EventSink`). Kept as a plain module global
#: (not captured at logger-construction time) so `configure()` can change it at
#: runtime and every already-constructed logger picks it up immediately.
_effective_level: int = logging.INFO


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


class EventLogger(Protocol):
    """Minimal structlog-style logger surface used by the plugin.

    Exposes the full standard level set (debug/info/warning/error/critical) so
    callers can log at whatever severity fits, with the effective level (see
    :func:`configure`) gating what actually gets recorded.
    """

    def debug(self, event: str, **fields: Any) -> Any: ...
    def info(self, event: str, **fields: Any) -> Any: ...
    def warning(self, event: str, **fields: Any) -> Any: ...
    def error(self, event: str, **fields: Any) -> Any: ...
    def critical(self, event: str, **fields: Any) -> Any: ...


class EventTee:
    """An :class:`EventLogger` that also records every event to an :class:`EventSink`.

    This is the extension point that makes structured events *queryable* (HLA
    §12/§13): each level call (``.debug``/``.info``/``.warning``/``.error``/
    ``.critical``) is both forwarded to the wrapped logger (operator logs,
    unchanged) and appended to the bounded on-disk sink the debug command reads.
    The sink write is best-effort and comes first, so a sink hiccup never blocks
    — nor is masked by — the real log call.

    Level gating happens here explicitly (against the module-level
    :data:`_effective_level`, set by :func:`configure`): an event below the
    effective level is dropped entirely — not forwarded to the base logger, and
    not written to the sink. structlog's own filtering bound logger handles this
    for the base logger on its own, but the sink write is a separate code path
    that bypasses it, so without this check debug events would flood
    events.jsonl regardless of the configured level.
    """

    def __init__(self, base: EventLogger, sink: EventSink) -> None:
        self._base = base
        self._sink = sink

    def _emit(self, level: int, event: str, fields: dict[str, Any]) -> Any:
        if level < _effective_level:
            return None
        self._sink.emit(event, fields)  # best-effort; never raises
        method = getattr(self._base, log_level_name(level))
        return cast(Any, method(event, **fields))

    def debug(self, event: str, **fields: Any) -> Any:
        return self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> Any:
        return self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> Any:
        return self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> Any:
        return self._emit(logging.ERROR, event, fields)

    def critical(self, event: str, **fields: Any) -> Any:
        return self._emit(logging.CRITICAL, event, fields)


class _StdlibEventLogger:
    """Fallback :class:`EventLogger` used when structlog is not installed.

    Folds structlog-style keyword fields onto a stdlib logger so the same call
    sites work whether or not structlog is available in the host interpreter.
    Gates against the module-level :data:`_effective_level` explicitly (rather
    than relying solely on the stdlib logger's own level), so behavior stays
    consistent with :class:`EventTee` regardless of ambient root-logger state.
    """

    def __init__(self, name: str | None, **initial: Any) -> None:
        self._log = logging.getLogger(name or "lifemodel")
        self._initial = initial

    def _emit(self, level: int, event: str, fields: dict[str, Any]) -> None:
        if level < _effective_level:
            return
        method = getattr(self._log, log_level_name(level))
        method("%s %s", event, {**self._initial, **fields})

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


def configure(level: int = logging.INFO) -> None:
    """Configure the logging pipeline. Idempotent.

    With structlog present, renders JSON events over stdlib logging. Without it,
    configures plain stdlib logging so the fallback shim still emits. Either
    way, sets the module-level effective level that :class:`EventTee` and
    :class:`_StdlibEventLogger` gate on.
    """
    global _effective_level
    _effective_level = level
    logging.basicConfig(format="%(message)s", level=level)
    if not _HAVE_STRUCTLOG:
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def current_level() -> int:
    """Return the module's current effective log level (see :func:`configure`).

    Purely additive read accessor (bead lm-j2w B2) so callers — e.g. plugin
    startup deciding whether a persisted level actually needs applying — can
    check the current level without reaching into the private
    ``_effective_level`` global directly.
    """
    return _effective_level


def get_logger(name: str | None = None, **initial_values: Any) -> EventLogger:
    """Return an :class:`EventLogger` — structlog when available, else a shim."""
    if _HAVE_STRUCTLOG:
        return cast(EventLogger, structlog.get_logger(name, **initial_values))
    return _StdlibEventLogger(name, **initial_values)


@contextmanager
def bound_log_context(**fields: Any) -> Iterator[None]:
    """Bind *fields* onto every log line emitted inside the ``with`` block, then reset.

    The CoreLoop wraps every tick (and each component's child span) in this (spec §5)
    so every ``.info()`` auto-carries its span's ``{trace_id, span_id,
    parent_span_id, tick}`` and RESETS on block exit — no stale bind leaks across
    ticks or components. Wrapped here so the CoreLoop never imports structlog directly.

    The caller always supplies the active span's fields (tracing is mandatory — the
    tracer is a required CoreLoop dependency), so there is no untraced branch here:
    a log without an active span is structurally impossible at the call site, not
    papered over by an empty-bind no-op. With structlog present this uses the
    contextvars API (bind on enter, reset on exit even under an exception). Without
    structlog (the Hermes-host fallback) there are no contextvars, so the bind is a
    no-op — the trace CONTEXT is still threaded explicitly via ``TracerPort``/
    ``TickContext.trace`` and suppression spans carry their ids as explicit fields;
    only the per-line structlog decoration is absent in that degraded host.
    """
    if not _HAVE_STRUCTLOG:
        yield
        return
    tokens = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)

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
from typing import Any, Protocol, cast

from .events import EventSink

try:
    import structlog

    _HAVE_STRUCTLOG = True
except ModuleNotFoundError:  # pragma: no cover - only in a host lacking structlog
    _HAVE_STRUCTLOG = False


class EventLogger(Protocol):
    """Minimal structlog-style logger surface used by the plugin."""

    def info(self, event: str, **fields: Any) -> Any: ...


class EventTee:
    """An :class:`EventLogger` that also records every event to an :class:`EventSink`.

    This is the extension point that makes structured events *queryable* (HLA
    §12/§13): each ``.info(event, **fields)`` is both forwarded to the wrapped
    logger (operator logs, unchanged) and appended to the bounded on-disk sink
    the debug command reads. The sink write is best-effort and comes first, so a
    sink hiccup never blocks — nor is masked by — the real log call.
    """

    def __init__(self, base: EventLogger, sink: EventSink) -> None:
        self._base = base
        self._sink = sink

    def info(self, event: str, **fields: Any) -> Any:
        self._sink.emit(event, fields)  # best-effort; never raises
        return self._base.info(event, **fields)


class _StdlibEventLogger:
    """Fallback :class:`EventLogger` used when structlog is not installed.

    Folds structlog-style keyword fields onto a stdlib logger so the same call
    sites work whether or not structlog is available in the host interpreter.
    """

    def __init__(self, name: str | None, **initial: Any) -> None:
        self._log = logging.getLogger(name or "lifemodel")
        self._initial = initial

    def info(self, event: str, **fields: Any) -> None:
        self._log.info("%s %s", event, {**self._initial, **fields})


def configure(level: int = logging.INFO) -> None:
    """Configure the logging pipeline. Idempotent.

    With structlog present, renders JSON events over stdlib logging. Without it,
    configures plain stdlib logging so the fallback shim still emits.
    """
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


def get_logger(name: str | None = None, **initial_values: Any) -> EventLogger:
    """Return an :class:`EventLogger` — structlog when available, else a shim."""
    if _HAVE_STRUCTLOG:
        return cast(EventLogger, structlog.get_logger(name, **initial_values))
    return _StdlibEventLogger(name, **initial_values)

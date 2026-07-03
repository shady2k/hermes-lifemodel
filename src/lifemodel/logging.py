"""Structured logging for the lifemodel plugin.

Configures structlog to emit JSON events over the stdlib :mod:`logging` module
so the plugin's logs integrate with Hermes' logging. This module only wires the
JSON pipeline and exposes a logger factory; the richer debug events (``tick``,
``wake_decision``, ``act_gate``, ``dream_run``, ...) described in HLA §13 land
in task 0.3.
"""

from __future__ import annotations

import logging

import structlog
from structlog.typing import FilteringBoundLogger


def configure(level: int = logging.INFO) -> None:
    """Configure structlog to render JSON events over stdlib logging.

    Idempotent: calling it again re-applies the same configuration.
    """
    logging.basicConfig(format="%(message)s", level=level)
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


def get_logger(name: str | None = None, **initial_values: object) -> FilteringBoundLogger:
    """Return a structlog logger, optionally named and bound to context."""
    logger: FilteringBoundLogger = structlog.get_logger(name, **initial_values)
    return logger

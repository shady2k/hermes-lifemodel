"""``EventRing`` — the in-memory, bounded freshness ring of trace events (spec §4.2).

Where the durable :class:`~lifemodel.state.trace_store.TraceWriter` is the
queryable source of truth, this in-process ``collections.deque`` is the
ephemeral, not-yet-merged tail: read-your-writes for the ``/lifemodel trace``
viewer (before a flush lands) and ergonomic for test assertions. Synchronous,
O(1) append, no I/O — the tick projects onto it without ever touching disk.

A projection, never a source: :class:`~lifemodel.log.SpanLogger` appends here
ONLY after the durable enqueue succeeds (durable-first), and each record carries
the trace store's monotonic ``record_id`` so the viewer can overlay the ring on
the flushed rows and dedup by it (spec §4.2, codex fix #5).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

#: Default ring depth — plenty of history for the viewer/tests, trivially bounded.
_DEFAULT_MAX_RECORDS = 512


class EventRing:
    """A thread-safe, in-memory, bounded ring of structured event records.

    The freshness half of the trace pipeline (spec §4.2). The ``emit``/``read``
    surface mirrors what callers expect; :class:`~lifemodel.log.SpanLogger`
    projects onto it via :meth:`append`.
    """

    def __init__(self, *, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        if max_records < 1:
            raise ValueError(f"max_records must be >= 1, got {max_records}")
        self._lock = threading.Lock()
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one already-formed record (e.g. a SpanLogger projection).

        Best-effort and never raises: the ring is an aid, not a liability. The
        record is copied so a later caller mutation cannot reach back into it.
        """
        with self._lock:
            self._records.append(dict(record))

    def emit(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        """Append a ``{"event": ..., **fields}`` record."""
        record: dict[str, Any] = {"event": event, **(dict(fields) if fields else {})}
        self.append(record)

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the ring's records oldest→newest (a snapshot copy).

        ``limit`` returns at most that many most-recent records (``0`` → none).
        """
        with self._lock:
            records = [dict(record) for record in self._records]
        if limit is None:
            return records
        return records[-limit:] if limit > 0 else []

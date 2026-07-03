"""``EventSink`` — a bounded, best-effort ring file of structured events (§12/§13).

The structured events described in HLA §13 (``tick``, ``wake_decision``,
``act_gate``, ``dream_run``, ...) are emitted to the logger for operator
observability, but logs are not *queryable* by the plugin itself. The debug
command (HLA §12, NFR9) needs to answer "what was the last tick / wake / act?"
without scraping operator logs, so we also **tee** every event into a small
on-disk sink it can read back.

Design constraints:

* **Bounded.** The file is a ring capped to ``max_records`` most-recent lines,
  so a long-running being never grows an unbounded event file. Old records are
  dropped, newest kept.
* **Best-effort.** Emitting an event must *never* crash the caller — a full disk
  or a bad path is swallowed. Observability is not worth taking the engine down.
* **Read-only reads.** :meth:`read` only reads; the debug path (HLA §9) touches
  nothing else.
* **Stdlib only.** One JSON object per line, ``json`` alone round-trips it — no
  Hermes, no third-party dependency (the plugin runs in Hermes' interpreter).

This is the event *store*; the tee that feeds it lives with the logger
(:class:`lifemodel.logging.EventTee`), and the reader is the debug command
(:mod:`lifemodel.debug`).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Filename of the ring sink under the profile state dir.
EVENTS_FILENAME = "events.jsonl"

#: The canonical structured-event vocabulary (HLA §13). Emitters (later tasks)
#: and the debug reader share these names so "last tick / wake / act / dream"
#: line up. Kept here because the sink is the events' home.
EVENT_TICK = "tick"
EVENT_WAKE_DECISION = "wake_decision"
EVENT_ACT_GATE = "act_gate"
EVENT_DREAM_RUN = "dream_run"

#: Default ring depth — plenty of history for debugging, trivially bounded.
_DEFAULT_MAX_RECORDS = 512
_TMP_PREFIX = ".events-"
_TMP_SUFFIX = ".tmp"


class EventSink:
    """A bounded, best-effort append-only ring of structured events.

    Each :meth:`emit` appends one ``{"event": ..., **fields}`` JSON line and
    trims the file back to the newest ``max_records`` lines. Both write and read
    swallow their own I/O errors — the sink is an aid, never a liability.
    """

    def __init__(self, path: Path, *, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        if max_records < 1:
            raise ValueError(f"max_records must be >= 1, got {max_records}")
        self._path = path
        self._max_records = max_records

    def emit(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        """Append one event record; never raise (best-effort, §12/NFR9).

        Any failure — an unserializable field, an unwritable path, a full disk —
        is swallowed so the emitting caller (the engine) is never taken down by
        the observability sink.
        """
        try:
            record: dict[str, Any] = {"event": event, **(dict(fields) if fields else {})}
            # ``default=str`` degrades an odd value to its string form rather
            # than raising; ``allow_nan=False`` rejects poison floats (caught).
            line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)
        except (TypeError, ValueError):
            return
        try:
            self._append_and_trim(line)
        except OSError:
            return

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the ring's records oldest→newest, tolerant of a corrupt line.

        Read-only (HLA §9): opens the file for reading and nothing else. A
        missing file yields ``[]``; a malformed or non-object line is skipped
        rather than raising, so a single torn record never blinds the debugger.
        ``limit`` returns at most that many most-recent records (``0`` → none).
        """
        records: list[dict[str, Any]] = []
        for raw in self._read_lines():
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        if limit is None:
            return records
        return records[-limit:] if limit > 0 else []

    def _append_and_trim(self, line: str) -> None:
        """Append *line* then rewrite to the newest ``max_records`` if over cap.

        Append is the fast path; the trim only rewrites when the ring is full.
        The whole method runs under :meth:`emit`'s best-effort guard.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
        lines = self._read_lines()
        if len(lines) > self._max_records:
            self._rewrite(lines[-self._max_records :])

    def _read_lines(self) -> list[str]:
        """Return non-empty lines from the ring file, or ``[]`` if unreadable.

        Swallows every read failure (missing file, a non-directory parent, bad
        bytes): the sink is best-effort on read too, so a broken path degrades
        to "no events" rather than raising into the debug command.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        return [line for line in text.split("\n") if line]

    def _rewrite(self, lines: list[str]) -> None:
        """Atomically replace the ring with *lines* (tmp file + ``os.replace``).

        Same tmp+rename discipline as the state store: a crash mid-trim leaves
        the previous full ring intact rather than a torn file.
        """
        payload = "".join(f"{line}\n" for line in lines)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

"""``FileSignalBus`` — the durable append-only signal log (HLA §2/§10/§13).

A :class:`~lifemodel.core.signal_bus.SignalBus` backed by two append-only files
under an injected *base_dir* (the composition root wires it to the profile state
dir; tests inject a ``tmp_path``):

* ``signals.log`` — one JSON object per line, every published signal in order.
  The **durable** record: signals survive a crash between publish and consume.
* ``signals.consumed`` — one already-returned origin id per line. The dedup
  ledger that makes ``consume_unprocessed`` idempotent across calls *and*
  restarts (HLA §10).

**Why a dedicated ledger, not ``State.processed_signal_ids``.** The port's
``consume_unprocessed()`` takes no arguments, so the bus must persist "consumed"
itself. If it did so by loading and committing the shared ``State``, that write
would race the tick's own ``State`` commit (the tick loads state *before* it
consumes, then commits *after* — clobbering any ids the bus wrote in between).
Keeping the bus's dedup ledger independent of ``State`` sidesteps that entirely
and lets the aggregator (1.3) consume without managing dedup at all. Higher-level
message dedup (HLA §10, gateway-turn vs. next-tick) rides the *same* origin ids,
so nothing downstream is forced to track them twice.

**Durability.** Each append is ``flush``ed and ``fsync``ed. Reads count only
newline-terminated records, so a torn final line from a crash mid-append is
ignored rather than misread — the append-only analogue of the state store's
tmp+rename atomicity. Log/ledger compaction (unbounded growth) is a later
concern; Phase 1 keeps it simple. Stdlib + our domain/logging only; no Hermes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.signal_bus import SignalBus
from ..domain.signal import Signal
from ..logging import EventLogger, get_logger

_LOG_FILENAME = "signals.log"
_CONSUMED_FILENAME = "signals.consumed"


class FileSignalBus(SignalBus):
    """A durable append-only :class:`SignalBus` over two files in *base_dir*."""

    def __init__(self, base_dir: Path, *, logger: EventLogger | None = None) -> None:
        self._base_dir = base_dir
        self._log_path = base_dir / _LOG_FILENAME
        self._consumed_path = base_dir / _CONSUMED_FILENAME
        self._log = logger or get_logger("lifemodel.signal_bus")

    def publish(self, signal: Signal) -> None:
        """Append *signal* to the durable log (``flush`` + ``fsync``)."""
        line = json.dumps(signal.to_dict(), ensure_ascii=False, allow_nan=False)
        self._append_line(self._log_path, line)
        self._log.info("signal_published", origin_id=signal.origin_id, kind=signal.kind)

    def consume_unprocessed(self) -> list[Signal]:
        """Return not-yet-consumed signals, deduped by origin id, and mark them.

        Reads the whole log, drops any origin id already in the consumed ledger
        or seen earlier in this batch, records the survivors' ids to the ledger
        durably, and returns them in publish order. Idempotent across calls and
        restarts (HLA §10).
        """
        consumed = self._read_committed_lines(self._consumed_path)
        already: set[str] = set(consumed)

        fresh: list[Signal] = []
        newly_consumed: list[str] = []
        seen: set[str] = set(already)
        for raw in self._read_committed_lines(self._log_path):
            signal = Signal.from_dict(json.loads(raw))
            if signal.origin_id in seen:
                continue
            seen.add(signal.origin_id)
            fresh.append(signal)
            newly_consumed.append(signal.origin_id)

        if newly_consumed:
            self._append_lines(self._consumed_path, newly_consumed)
        self._log.info("signals_consumed", count=len(fresh))
        return fresh

    def _append_line(self, path: Path, line: str) -> None:
        self._append_lines(path, [line])

    def _append_lines(self, path: Path, lines: list[str]) -> None:
        """Append newline-terminated *lines* to *path* durably (``fsync``)."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        payload = "".join(f"{line}\n" for line in lines)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read_committed_lines(path: Path) -> list[str]:
        """Return only fully newline-terminated, non-empty lines from *path*.

        A missing file yields ``[]``. A trailing line without its newline is a
        torn append (crash mid-write) and is dropped, never parsed.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        if not text.endswith("\n"):
            # Drop the torn final record; keep everything committed before it.
            text = text[: text.rfind("\n") + 1]
        return [line for line in text.split("\n") if line]

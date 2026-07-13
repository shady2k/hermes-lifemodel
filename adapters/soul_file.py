"""The ONLY thing in the plugin that touches ``SOUL.md`` (spec §4, ADR-0002).

The soul is one prose document with no fences. It is **always its own base**: every
write reads what is on disk right now, and whatever is there — ours, or the human's
hand-edit — is the input. So there is no clobber, no merge, and no arbitration. The
sha is not a fence against the human; it is a compare-and-swap so that an edit landing
*during* our LLM turn is not eaten, and a way to NOTICE that the human rewrote the
being between our writes.

We never delete this file and never roll it back to a default. Destroying a soul is an
act that belongs to the human.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from ..core.soul_guard import validate_soul


class SoulError(Exception):
    """Base for every refusal to write a soul."""


class SoulRejected(SoulError):
    """The candidate soul would erase or damage the being (see ``core.soul_guard``)."""


class SoulConflict(SoulError):
    """The file changed under us — the human saved while the being was writing."""


class SoulFile:
    """``$HERMES_HOME/SOUL.md`` — read, hash, and atomically replace."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Serialises our OWN writers (a tool call and a startup reconcile can overlap).
        # It cannot exclude the human's editor — nothing can — which is exactly why the
        # compare-and-swap below exists as well.
        self._lock = threading.Lock()

    def read(self) -> str:
        """The soul as it is on disk right now. A missing file reads as ``""``."""
        try:
            return self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def sha(self) -> str:
        """Digest of the current content — the compare-and-swap token."""
        return hashlib.sha256(self.read().encode("utf-8")).hexdigest()

    def is_pristine_default(self, *, default_text: str) -> bool:
        """True when the soul is still the host's untouched seed.

        A stranger has Hermes's ``DEFAULT_SOUL_MD``; a veteran has prose they wrote
        themselves. The ritual opens differently for each (spec §6.4), and this is the
        only way to tell — the file ALWAYS exists (``hermes_cli/config.py:893``).
        """
        return self.read().strip() == default_text.strip()

    def write(self, text: str, *, expect_sha: str | None) -> str:
        """Validate, compare-and-swap, replace atomically. Returns the new sha.

        Raises :class:`SoulRejected` when the soul would erase the being, and
        :class:`SoulConflict` when the human saved during the turn — in which case the
        caller re-runs the write on the FRESH text (their edit is the new base, never a
        casualty).
        """
        reason = validate_soul(text)
        if reason is not None:
            raise SoulRejected(reason)
        with self._lock:
            if expect_sha is not None and self.sha() != expect_sha:
                raise SoulConflict(
                    "SOUL.md changed while the being was writing it — re-read and retry"
                )
            self._replace_atomically(text)
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _replace_atomically(self, text: str) -> None:
        # tmp + os.replace: a crash never leaves a half-written soul, and Hermes — which
        # re-reads this file on EVERY turn — never observes a torn document.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=".soul-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

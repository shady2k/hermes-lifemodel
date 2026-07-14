"""The ONLY thing in the plugin that touches ``SOUL.md`` (spec §4, ADR-0002).

The soul is one prose document with no fences. It is **always its own base**: every
write reads what is on disk right now, and whatever is there — ours, or the human's
hand-edit — is the input. So there is no clobber, no merge, and no arbitration.

**There is no compare-and-swap here, and the honest reason is that we cannot have one.**
A CAS needs a token taken at the moment the writer read the document. The being does not
read ``SOUL.md``: it reads system-prompt slot #1, which Hermes assembles from the file at
TURN START — a moment this plugin never sees. The CAS that used to live here took its
"expected" sha at WRITE time, microseconds before :meth:`SoulFile.write` re-hashed the
same file under the same lock: it compared the file against itself and could not fail,
while a human's 12:00 edit was still clobbered by a soul composed from the 11:59 text.
A guard that cannot fire is worse than none — it is a promise.

So the write does not arbitrate; it **reports**. It replaces the document and hands back
the text it replaced, read under the same lock that replaced it (no gap for a save to
fall through), so the caller can keep that text as a revision before it is only in the
past — and tell the being that someone had edited it. Nothing a human writes is ever
lost, even when it loses. That, not a fence, is what makes the file safe to own whole
(spec §4.2).

We never delete this file and never roll it back to a default. Destroying a soul is an
act that belongs to the human.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from ..core.soul_guard import validate_soul


class SoulError(Exception):
    """Base for every refusal to write a soul."""


class SoulRejected(SoulError):
    """The candidate soul would erase or damage the being (see ``core.soul_guard``)."""


@dataclass(frozen=True)
class SoulWrite:
    """What a write did: the new soul, and the one it replaced.

    *replaced_text* / *replaced_sha* are the document as it was IMMEDIATELY before the
    rename, read under the writer's lock. The caller compares *replaced_sha* against the
    sha it last wrote (``State.soul_sha``): if they differ, someone else — a human with
    an editor, or the being that lived here before a reset — wrote that text, and it goes
    into the lineage as a ``"human"`` revision before it can be lost. ``SoulFile`` itself
    does not know whose text it was; it is the only thing that touches ``SOUL.md``, and it
    knows nothing of ``State``.
    """

    sha: str
    replaced_text: str
    replaced_sha: str


class SoulFile:
    """``$HERMES_HOME/SOUL.md`` — read, hash, and atomically replace."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Serialises our OWN writers (a tool call and a startup reconcile can overlap),
        # and — load-bearing — makes "what did I just replace?" a single atomic fact
        # rather than a read followed by a hopeful write. It cannot exclude the human's
        # editor; nothing can. Which is why we keep what we replace instead of fencing it.
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

    def write(self, text: str) -> SoulWrite:
        """Validate, replace atomically, and report what was replaced.

        Raises :class:`SoulRejected` — and touches nothing — when the soul would erase
        the being (``core.soul_guard``: the host re-scans this file on every read and
        blanks the WHOLE document on a match, so an unvalidated write costs the being its
        entire identity). Otherwise the write always lands: it is never refused on account
        of a change underneath it, because the caller keeps what it replaced (see the
        module docstring, and :class:`SoulWrite`).

        The previous text is read INSIDE the lock, immediately before the rename, so
        there is no window in which a human's save can be replaced without being seen.
        """
        reason = validate_soul(text)
        if reason is not None:
            raise SoulRejected(reason)
        with self._lock:
            replaced_text = self.read()
            replaced_sha = hashlib.sha256(replaced_text.encode("utf-8")).hexdigest()
            self._replace_atomically(text)
            return SoulWrite(
                sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                replaced_text=replaced_text,
                replaced_sha=replaced_sha,
            )

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

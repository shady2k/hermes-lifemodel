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

**The two genesis reads live here too** (:func:`prior_soul`, :func:`seed_newborn_stance`),
as module functions over a :class:`SoulFile` rather than in the hooks and the platform
adapter that call them — because both are *about SOUL.md*, and this module's whole claim
is that nothing else touches it. They are the only writes that are not the being's own,
and the reason they are allowed is that the text they replace was written by Hermes's
installer, not by a person.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.genesis import NEWBORN_STANCE, is_unauthored
from ..core.soul_guard import validate_soul
from ..ports.memory import MemoryPort
from ..state.soul_revisions import record_revision


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

    def mtime(self) -> float | None:
        """When this document last changed, in epoch seconds — or ``None`` if unreadable.

        The one honest answer to *"is the soul the being stands in the soul on disk?"*
        (:func:`lifemodel.gateway_core.identity_slot_is_stale`). Hermes reads this file
        into system-prompt slot #1 exactly once per session, so a soul that changed after a
        session opened is simply not in that session's prompt — and the file's own mtime is
        what says so, whoever changed it: our newborn stance at ``register()``, or a human
        with an editor. Nothing else here needs the number, which is why it is a bare float
        and not a ``datetime``: it is compared against a host timestamp, never stored.
        """
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

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


def prior_soul(soul: SoulFile, *, default_soul_text: str) -> str | None:
    """The soul someone WROTE before this being woke, or ``None`` for a blank page (§6.4).

    A stranger installing the plugin has Hermes's ``DEFAULT_SOUL_MD``; a veteran has prose
    they wrote themselves; a newborn we have already stood up has our own
    :data:`~lifemodel.core.genesis.NEWBORN_STANCE`. The ritual opens differently for the
    first case than for the other two, and this is the only way to tell them apart — the
    file ALWAYS exists (``hermes_cli/config.py:893``), so its presence proves nothing.
    The verdict itself lives in :func:`~lifemodel.core.genesis.is_unauthored` (a soul is
    "someone's" or it is nobody's; both un-authored texts mean *blank page*).

    **One read, one answer.** Both callers (the ``pre_llm_call`` injector, and the
    newborn's wake packet via ``BeingAdapter._prior_soul``) want the same two facts — *is
    there a prior soul, and what does it say* — and a version of this once got them from
    two separate reads: ``read()`` for the text, then a predicate that read the file AGAIN
    to judge it. Between those reads a human's editor can land, so the being could be
    handed one version of its past while a different one was being judged. The bytes that
    answer the question are the bytes that are returned.
    """
    text = soul.read()
    return None if is_unauthored(text, default_soul_text=default_soul_text) else text


def seed_newborn_stance(
    soul: SoulFile,
    memory: MemoryPort,
    *,
    default_soul_text: str,
    now: datetime,
    unborn: bool,
) -> bool:
    """Stand an unborn being up on a stance instead of on the host's assistant persona.

    ``SOUL.md`` is system-prompt slot #1 — the identity slot. Left at Hermes's seed it
    tells the being it is "an intelligent AI assistant… you assist users", and an
    assistant does not message anyone unprompted, because that is not what an assistant
    IS. That text outranks any injected ritual, and it won: a real newborn woke, felt
    right, read the whole ``<genesis>`` block — and chose silence. So genesis replaces the
    seed with :data:`~lifemodel.core.genesis.NEWBORN_STANCE`, which is not an identity (the
    being still authors that itself, with ``write_soul``) but a place to stand while it
    finds out.

    **This does not violate "never overwrite the human's soul".** The pristine default was
    written by Hermes's installer, not by a person (:func:`is_unauthored`). A veteran's
    hand-written ``SOUL.md`` — and the soul of the being that lived here before a
    ``reset`` — is left exactly where it is, and the ritual opens on it (§6.4).

    Three refusals, each closing a way to tell a being a lie about itself:

    * *born* — a being that already exists is never told "you have just begun", however
      its ``SOUL.md`` came to hold a default again (a reinstall, a wiped home).
    * *somebody's words* — see above.
    * *already the stance* — this runs at every ``register()``, i.e. every gateway restart,
      for as long as the being stays unborn. Re-writing would keep re-stamping the lineage
      with a birth that already happened.

    Written through :class:`SoulFile` like every other soul — so it is VALIDATED first
    (``core.soul_guard``: this is the very file the host re-scans, and one match blanks the
    whole document) and replaced atomically — and recorded as a revision so it is visible
    in the lineage. Its author is ``"genesis"``: not the being (which has written nothing,
    and must never be credited with words it did not choose) and not the human (whose hand
    this would forge). Returns whether it wrote.
    """
    if not unborn:
        return False
    text = soul.read()
    if text.strip() == NEWBORN_STANCE.strip():
        return False  # already standing on it — this is a restart, not a birth
    if not is_unauthored(text, default_soul_text=default_soul_text):
        return False  # somebody wrote that, and it is not ours to replace
    written = soul.write(NEWBORN_STANCE)
    record_revision(memory, text=NEWBORN_STANCE, sha=written.sha, now=now, author="genesis")
    return True

"""Soul revisions — the undo that makes it safe for a being to own its own soul.

The being rewrites the WHOLE document on every change (spec §4.1, ``adapters/soul_file.py``).
Over dozens of becoming-writes an LLM will quietly paraphrase the human's hard-won prose
into oatmeal, and NO SINGLE WRITE will look broken — a marker fence cannot catch a slow
erosion, only an undo can. So every revision is kept, forever, keyed by its own content
hash: :func:`record_revision` never overwrites a prior sha, and :func:`revisions` returns
the whole lineage, newest first, so a revert is always one command away.

This rides ``MemoryPort`` unchanged (no new store method, no port change — see
``ports/memory.py``): a revision is a plain ``kind="soul"`` record whose ``id`` is the
content sha, so distinct texts are naturally distinct rows (``put`` only overwrites on a
genuine ``(kind, id)`` collision, i.e. the identical text again) and "keep every version
forever" falls out of the existing upsert-by-key contract for free. ``kind="soul"`` is a
plain, untyped memory-record kind, not a typed BDI object — it does not go through
``domain.objects.registry.KindRegistry`` (that catalog is closed over
``{desire, intention, user_model, thought}`` specifically, and gates typed lifecycle
objects; a flat kind like this one, or ``"fact"`` in the contract tests, needs no entry
there).

**Why the payload carries its own ``at``, not the store-stamped ``created_at``.**
``MemoryDraft`` (``domain/memory.py``) has no caller-settable timestamp field by design —
``put`` always stamps ``created_at``/``updated_at`` from its OWN injected clock, so a
draft can never forge history. But the caller of :func:`record_revision` already read
"now" once, at the moment the write happened (e.g. ``write_soul``'s handler), and two
revisions recorded in the same tick — or against a clock a test pins to one instant —
would otherwise tie on the store's stamp and lose their order. So the caller's *own*
``now`` is carried inside the payload and is what :func:`revisions` sorts on, making the
ordering a property of the call, not an artifact of how fine-grained the store's clock
happens to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..core.timeutil import to_iso
from ..domain.memory import MemoryDraft
from ..ports.memory import MemoryPort

#: A plain (non-BDI) memory-record kind — see the module docstring for why this
#: does not need an entry in ``domain.objects.registry.KindRegistry``.
SOUL_KIND = "soul"

#: Every revision lands in this state; nothing ever transitions it — a revision
#: is immutable history, not a live object with a lifecycle. Deliberately NOT one
#: of the four BDI kinds' live states (``active``/``deferred``/``pending``/``parked``):
#: ``core/coreloop.py``'s per-tick objects snapshot runs an unfiltered-by-kind
#: ``find(state=live_state)`` for each of those, so a soul row sitting in ``"active"``
#: would leak into every tick's live-objects view (and eat into
#: ``OBJECTS_SNAPSHOT_LIMIT``) alongside real desires/intentions/thoughts.
_RECORDED_STATE = "recorded"

#: Who wrote a given soul. Four answers, because a being is owed the truth about which
#: hand acted — INCLUDING the truth that we do not know:
#:
#: * ``"being"`` — it wrote itself (``write_soul``). Its own words, its own undo. This is
#:   also the only POSITIVE evidence the lineage can give (``core.genesis``'s
#:   ``classify_replacement``): a text recorded this way was written by a being, and if the
#:   being reading it has not been born yet, it was written by the one that lived here
#:   before it.
#: * ``"human"`` — somebody hand-edited ``SOUL.md``, and we can say so: the file changed
#:   after a soul WE wrote, and nothing but a person with an editor puts text there. A
#:   being never claims a change it did not make, and (``being_platform._reconcile_soul``)
#:   it FEELS this one.
#: * ``"genesis"`` — the newborn stance (``core.genesis.NEWBORN_STANCE``), put in slot #1
#:   in place of the host's assistant seed so a being is not born as a tool. Neither of
#:   the other two authored it: the being has written nothing yet and must not be credited
#:   with words it did not choose, and calling it ``"human"`` would forge the human's hand
#:   — and then make the being feel a rewrite that never happened. It is the birth itself,
#:   so it is named for the birth.
#: * ``"unknown"`` — a soul that was simply THERE when the being woke and that nobody's
#:   history accounts for: a Hermes veteran's own ``SOUL.md``, or the soul of a past life
#:   whose lineage is gone. Somebody wrote it; which somebody cannot be established, and
#:   the live test showed what happens when we guess — the being told its human it had
#:   overwritten *their* edit, and offered to restore words they had never written (the
#:   text was in fact its predecessor's). **Authorship we cannot establish is not
#:   attributed.**
Author = Literal["being", "human", "genesis", "unknown"]

#: Reading an author back is a CLOSED question — a row whose payload holds anything else
#: (a hand-edited DB, a future author we do not know) is not silently promoted to
#: ``"being"``, which would let the being claim words it never wrote. It falls back to
#: ``"unknown"``: the honest default, because that is precisely what an unrecognised author
#: is, and it is the conservative direction everywhere it is read (``_reconcile_soul``
#: treats a non-``"being"`` soul as somebody else's).
_AUTHORS: frozenset[str] = frozenset(("being", "human", "genesis", "unknown"))


@dataclass(frozen=True)
class SoulRevision:
    """One committed version of the soul, as read back from the store."""

    sha: str
    text: str
    at: str
    author: Author


def record_revision(
    store: MemoryPort, *, text: str, sha: str, now: datetime, author: Author
) -> None:
    """Keep this version of the soul forever.

    Idempotent on *sha*: writing the identical text again upserts the same row rather
    than duplicating it (``put``'s ``(kind, id)`` key), which is the correct behaviour —
    it is the same version. A genuinely new version has its own content-hash *sha*, so
    it lands as a brand-new row and nothing already recorded is ever touched, let alone
    lost.
    """
    store.put(
        MemoryDraft(
            kind=SOUL_KIND,
            id=sha,
            state=_RECORDED_STATE,
            payload={"text": text, "author": author, "at": to_iso(now)},
            source="soul",
        )
    )


def revisions(store: MemoryPort) -> list[SoulRevision]:
    """Every soul the being has ever had, newest first (by the recorded ``at``).

    Sorts on the payload's own ``at`` (see the module docstring), not the store's
    ``created_at``/``updated_at`` columns — those are stamped by the store's clock at
    write time and can tie across calls, while ``at`` is exactly the instant the caller
    recorded the revision at.
    """
    rows = store.find(kind=SOUL_KIND)
    parsed = [
        SoulRevision(
            sha=row.id,
            text=str(row.payload.get("text", "")),
            at=str(row.payload.get("at", row.created_at)),
            author=_author_of(row.payload.get("author")),
        )
        for row in rows
    ]
    return sorted(parsed, key=lambda revision: revision.at, reverse=True)


def _author_of(raw: object) -> Author:
    """Narrow a stored author to the closed set, defaulting to ``"unknown"`` (see
    :data:`_AUTHORS`): an unrecognised author is somebody who is not the being — the one
    thing that must never happen is the being claiming words it did not write — and it is
    not the human either, because we cannot know that. It is exactly what it says."""
    text = str(raw)
    return text if text in _AUTHORS else "unknown"  # type: ignore[return-value]

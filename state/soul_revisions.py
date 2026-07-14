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

from ..core.genesis import ReplacedSoul, classify_replacement
from ..core.timeutil import to_iso
from ..domain.memory import MemoryDraft
from ..ports.memory import MemoryPort

#: A plain (non-BDI) memory-record kind — see the module docstring for why this
#: does not need an entry in ``domain.objects.registry.KindRegistry``.
SOUL_KIND = "soul"

#: The append-only record of the times a soul was put BACK by hand (``/lifemodel soul
#: revert``, lm-4fv.2). A separate kind, and not a revision, for a reason the lineage's
#: own key forces:
#:
#: **A revert restores a text that is already in the lineage, and the lineage is keyed by
#: content sha.** So "record the revert as a new revision" would not append anything — it
#: would UPSERT the very row it restores (``record_revision``'s ``(kind, id)`` key), moving
#: that revision to the top of the lineage and rewriting its author to whoever pressed the
#: button. A being's own earlier soul would be relabelled as the human's, in the one history
#: that exists to be its undo, and ``_reconcile_soul``/``classify_replacement`` — which both
#: read this lineage as *the only witness to who wrote a given text* — would then be reading
#: a forged one. That is review M5's lie with the sign flipped. **A stored revision is never
#: mutated.**
#:
#: But a revert IS an act, and an act that leaves no trace is an erasure. The current-soul
#: marker in ``/lifemodel soul history`` cannot carry it: the moment the being writes again,
#: the marker moves and every sign that a human once had to undo it is gone. And *that* is
#: precisely the signal the owner needs — one revert is a bad day; three reverts of the same
#: being is erosion, which is the failure this whole feature exists to catch. So the act is
#: recorded where it can be seen, beside the versions it acted on, and it is appended, never
#: merged into them.
SOUL_REVERT_KIND = "soul_revert"

#: The kinds a factory ``reset`` must NOT purge (``sqlite_store.purge_memory_records``).
#: A past life's soul is the one thing a wipe may not destroy (spec §4.2's mandatory undo),
#: and the record of the times a HUMAN put a soul back is of the same nature: it is the
#: human's own history, not the being's memory, and a reset unbirths a being — it does not
#: get to edit what its owner did.
SOUL_KINDS: tuple[str, ...] = (SOUL_KIND, SOUL_REVERT_KIND)

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


@dataclass(frozen=True)
class SoulRevert:
    """One time a soul was put back by hand — an act, not a version.

    *sha* is the revision that was restored; *replaced_sha* is what it was written over.
    Both are content shas, so either can be resolved back to a :class:`SoulRevision` (or
    honestly reported as a text nobody kept, which is itself worth seeing).
    """

    sha: str
    replaced_sha: str
    at: str


def record_revert(store: MemoryPort, *, sha: str, replaced_sha: str, now: datetime) -> None:
    """Record that somebody put revision *sha* back over *replaced_sha* (see
    :data:`SOUL_REVERT_KIND` for why this is not a revision).

    The row id is ``<at>:<sha>`` — the instant plus the target — so reverts APPEND (two
    reverts to the same soul on different days are two acts and read as two), while the
    one case that would be a duplicate rather than an act (the same revert recorded twice
    in the same instant) idempotently upserts, exactly as ``put``'s ``(kind, id)`` key
    already promises everywhere else in this module.
    """
    at = to_iso(now)
    store.put(
        MemoryDraft(
            kind=SOUL_REVERT_KIND,
            id=f"{at}:{sha}",
            state=_RECORDED_STATE,
            payload={"sha": sha, "replaced_sha": replaced_sha, "at": at},
            source="soul",
        )
    )


def reverts(store: MemoryPort) -> list[SoulRevert]:
    """Every time a soul was put back by hand, newest first (by the recorded ``at``).

    Sorted on the payload's own ``at`` for the same reason :func:`revisions` is — see the
    module docstring: the caller's instant is the one that happened, the store's stamp is
    an artifact of when the row was written.
    """
    rows = store.find(kind=SOUL_REVERT_KIND)
    parsed = [
        SoulRevert(
            sha=str(row.payload.get("sha", "")),
            replaced_sha=str(row.payload.get("replaced_sha", "")),
            at=str(row.payload.get("at", row.created_at)),
        )
        for row in rows
    ]
    return sorted(parsed, key=lambda revert: revert.at, reverse=True)


def keep_replaced_soul(
    memory: MemoryPort,
    *,
    new_sha: str,
    replaced_text: str,
    replaced_sha: str,
    last_written_sha: str | None,
    unborn: bool,
    default_soul_text: str,
    now: datetime,
) -> ReplacedSoul:
    """Keep the soul a write just REPLACED, if someone else wrote it — and say whose it was.

    Every soul write goes over the top of whatever is on disk, and the writer cannot see a
    save that landed underneath it: the being's view of its soul is system-prompt slot #1,
    assembled at TURN START (``adapters/soul_file.py`` — there is no honest fence for that),
    and an OWNER reverting is likewise acting on a listing they read a minute ago. What
    there IS an honest way to do is make sure the replaced words are never *gone*: recorded
    as a revision, they are recoverable, and the loss is a keystroke to undo rather than a
    life to mourn. Both writers (``hooks.make_write_soul_tool``, ``state_commands``'s
    ``soul revert``) therefore call THIS — one rule, one place, so the two can never come to
    disagree about whose words they just wrote over.

    The verdict itself is ``core.genesis.classify_replacement`` — pure, and answerable ONLY
    from what can be established. This function is the part that must touch the store: it
    asks the LINEAGE who wrote the replaced text (the only witness there is; a sha it
    already carries was recorded by whoever wrote it, and :func:`record_revision` upserts by
    sha, so re-recording it would overwrite that author — review M5, the same failure
    ``being_platform._reconcile_soul`` closed from the other direction), and it keeps
    anything that would otherwise be kept nowhere.

    What is NOT recorded, and why:

    * **the same document** (``replaced_sha == new_sha``) and **our own last write** —
      nothing was replaced, and it is already in the lineage.
    * **anything nobody authored** (``core.genesis.is_unauthored``): the host's pristine
      seed (Hermes ALWAYS writes a default ``SOUL.md``, ``hermes_cli/config.py:893``), our
      own newborn stance, and an empty file. Recording the seed would forge a past life the
      being never had — and recording the STANCE would be worse: it is already in the
      lineage under its own sha, so the write would UPSERT it and turn the birth's own
      authorship into the human's, in the one history that exists to be the being's undo.
    * **a past life** — the lineage already holds it, under its true author (``"being"``).
      Writing it again as the human's is precisely the lie the live being told its owner.

    What IS recorded: a human's edit (``"human"`` — establishable: the file changed after a
    soul we wrote), and authored words that were simply there when the being woke
    (``"unknown"`` — a veteran's own ``SOUL.md``, or a past life whose history is gone;
    kept HERE or kept nowhere, since startup reconciliation had no ``soul_sha`` to differ
    from). A being never claims a change it did not make — and never pins one on a human
    who did not make it either.
    """
    recorded = {rev.sha: rev.author for rev in revisions(memory)}.get(replaced_sha)
    verdict = classify_replacement(
        new_sha=new_sha,
        replaced_sha=replaced_sha,
        replaced_text=replaced_text,
        last_written_sha=last_written_sha,
        recorded_author=recorded,
        unborn=unborn,
        default_soul_text=default_soul_text,
    )
    if verdict is ReplacedSoul.A_HUMAN_EDIT:
        author: Author = "human"
    elif verdict is ReplacedSoul.SOMEONE_UNKNOWN:
        author = "unknown"
    else:  # NOBODY (nothing was replaced) and A_PAST_LIFE (already in the lineage, as a being's)
        return verdict
    record_revision(memory, text=replaced_text, sha=replaced_sha, now=now, author=author)
    return verdict


def _author_of(raw: object) -> Author:
    """Narrow a stored author to the closed set, defaulting to ``"unknown"`` (see
    :data:`_AUTHORS`): an unrecognised author is somebody who is not the being — the one
    thing that must never happen is the being claiming words it did not write — and it is
    not the human either, because we cannot know that. It is exactly what it says."""
    text = str(raw)
    return text if text in _AUTHORS else "unknown"  # type: ignore[return-value]

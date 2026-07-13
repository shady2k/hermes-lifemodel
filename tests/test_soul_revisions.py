from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lifemodel.state.soul_revisions import record_revision, revisions
from lifemodel.state.sqlite_store import SQLiteRuntimeStore
from lifemodel.testing.fakes import FakeClock

T0 = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
T1 = datetime(2026, 7, 13, 11, 0, tzinfo=UTC)


def test_every_soul_write_is_recoverable(tmp_path: Path) -> None:
    # This is what makes it SAFE for the being to own the file whole. Erosion is the
    # real risk — fifty rewrites, each a harmless paraphrase, and the human's prose is
    # gone with no single write looking broken. Revert must always be one command away.
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(T0))
    record_revision(store, text="the first soul", sha="aaa", now=T0, author="being")
    record_revision(store, text="the second soul", sha="bbb", now=T1, author="being")

    history = revisions(store)
    assert [r.sha for r in history] == ["bbb", "aaa"]  # newest first
    assert history[-1].text == "the first soul"  # the original is still recoverable


def test_a_human_rewrite_is_recorded_as_theirs(tmp_path: Path) -> None:
    # Reconciliation adopts what is on disk. Who wrote it matters: a human rewriting the
    # being is an EVENT in its life, not a version conflict.
    store = SQLiteRuntimeStore(tmp_path, clock=FakeClock(T0))
    record_revision(store, text="Sasha rewrote me.", sha="ccc", now=T0, author="human")
    assert revisions(store)[0].author == "human"

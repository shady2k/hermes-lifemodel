"""Shared test fixtures for the ``tests/`` suite.

Kept separate from the root ``conftest.py`` (which only solves the flat-layout
import binding for a non-canonically-named checkout) — this one holds actual
pytest fixtures shared across test modules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lifemodel.composition import LifeModel, build_lifemodel


@pytest.fixture
def build_lm(tmp_path: Path) -> Callable[[], LifeModel]:
    """A :class:`LifeModel` factory over ONE ``tmp_path``-backed SQLite store.

    Mirrors how the real wiring builds a graph: every call assembles a FRESH
    ``LifeModel`` (a tool handler must never hold a stale in-memory graph across
    calls — see ``hooks.py``'s afferent builders), but every call points at the
    SAME on-disk store (``tmp_path``), so a commit made through one ``build_lm()``
    call is visible to the next — exactly the durability a tool handler relies on.
    """

    def _build() -> LifeModel:
        return build_lifemodel(base_dir=tmp_path)

    return _build
